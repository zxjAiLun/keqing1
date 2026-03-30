"""特征编码器：将 GameState.snapshot() 字典编码为神经网络输入。

输出：
  tile_features : np.ndarray shape (C_TILE, 34)  float32
  scalar_features: np.ndarray shape (N_SCALAR,)   float32

通道布局 (C_TILE = 54)：
  0-3   : 自家手牌计数 planes（plane k = 该牌种数量 >= k+1）
  4-7   : 自家副露 presence（4 个副露槽，有牌 = 1.0）
  8-9   : 他家1舍牌（+0 立直宣言牌, +1 手切flag）
  10-11 : 他家2舍牌（同上）
  12-13 : 他家3舍牌（同上）
  14    : 他家1副露 presence
  15    : 他家2副露 presence
  16    : 他家3副露 presence
  17    : 自家舍牌 presence
  18    : dora 实际牌（累加，上限不限）
  19    : 听牌进张位置（waits，注入时有效，= 1.0）
  20-51 : 舍牌巡目分段编码（4家 × 8段，ch = 20 + slot*8 + seg）
          slot: 他家 0/1/2（与 ch 8-13 相同相对顺序），自家 slot=3
          seg = min(discard_turn // 3, 7)，每3巡一段，共8段
  52    : tsumo 牌位置（当前摸到的牌 = 1.0，其余为 0）
  53    : 最新立直家立直后其他家打过的牌（无立直时全0）

N_SCALAR = 48：
  0  : bakaze E（1/0）
  1  : bakaze S（1/0）
  2  : bakaze W（1/0）  ← N场时三者均为0
  3  : kyoku (1-4) / 4.0
  4  : honba / 8.0
  5  : kyotaku / 8.0
  6  : actor score / 50000
  7  : score rank / 3.0
  8  : 向听数 / 8.0
  9  : 有效进张数 / 34.0
  10 : 自家副露数 / 4.0
  11 : 自家舍牌数 / 18.0
  12 : actor 立直 flag（相对 slot=0）
  13 : 下家 立直 flag（相对 slot=1）
  14 : 对家 立直 flag（相对 slot=2）
  15 : 上家 立直 flag（相对 slot=3）
  16 : 赤5总数（手牌+副露）/ 4.0
  17 : 断幺九（0/1）
  18 : 赤5m flag（手牌+副露）
  19 : 赤5p flag（手牌+副露）
  20 : 赤5s flag（手牌+副露）
  21 : actor 副露数 / 4.0（相对 slot=0）
  22 : 下家 副露数 / 4.0（相对 slot=1）
  23 : 对家 副露数 / 4.0（相对 slot=2）
  24 : 上家 副露数 / 4.0（相对 slot=3）
  25 : 下家 分数差 / 30000（相对 slot=1）
  26 : 对家 分数差 / 30000（相对 slot=2）
  27 : 上家 分数差 / 30000（相对 slot=3）
  28 : jikaze（自风，0-3归一化 /3.0）
  29 : 场风 E（1/0）
  30 : 场风 S（1/0）
  31 : 场风 W（1/0）
  32 : dora+赤5总数 / 10.0
  33 : 纯全带幺（0/1）
  34 : 混全带幺（0/1）
  35 : 混老头（0/1）
  36 : 花色占比 万
  37 : 花色占比 饼
  38 : 花色占比 索
  39 : 花色占比 字
  40 : 混一色（0/1）
  41 : 清一色（0/1）
  42 : 一杯口可能性（0/1）
  43 : 三色同顺可能性（0/1）
  44 : 对对胡可能性（0/1）
  45 : 三暗刻可能性（0/1）
  46 : 振听 flag（0/1）
  47 : 确定番数下限 / 8.0
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from riichienv import calculate_shanten, HandEvaluator, Meld as RiichiMeld, MeldType

from mahjong_env.tiles import tile_to_136 as _to_136, tile_to_34 as _to_34, tile_is_aka as _is_aka


_DORA_NEXT = {
    '1m':'2m','2m':'3m','3m':'4m','4m':'5m','5m':'6m','6m':'7m','7m':'8m','8m':'9m','9m':'1m',
    '1p':'2p','2p':'3p','3p':'4p','4p':'5p','5p':'6p','6p':'7p','7p':'8p','8p':'9p','9p':'1p',
    '1s':'2s','2s':'3s','3s':'4s','4s':'5s','5s':'6s','6s':'7s','7s':'8s','8s':'9s','9s':'1s',
    'E':'S','S':'W','W':'N','N':'E','P':'F','F':'C','C':'P',
}

# ---------------------------------------------------------------------------
# 34 种牌名顺序（与 action_space.py 保持一致，tile34 index 对应）
# ---------------------------------------------------------------------------
N_TILES = 34
C_TILE = 54
N_SCALAR = 48

_WIND_ORDER = ["E", "S", "W", "N"]
_BAKAZE_IDX = {"E": 0, "S": 1, "W": 2, "N": 3}


def _snap_melds_to_mahjong(melds: List[dict]) -> List[RiichiMeld]:
    """将 snap meld dict 列表转为 riichienv.Meld 列表（供 HandEvaluator 使用）。"""
    _TYPE_MAP = {
        'chi': MeldType.Chi,
        'pon': MeldType.Pon,
        'daiminkan': MeldType.Daiminkan,
        'ankan': MeldType.Ankan,
        'kakan': MeldType.Kakan,
    }
    result = []
    for m in melds:
        mt = _TYPE_MAP.get(m.get('type', ''))
        if mt is None:
            continue
        tiles136 = [_to_136(t) for t in m.get('consumed', []) if _to_136(t) >= 0]
        pai = m.get('pai')
        if pai and _to_136(pai) >= 0:
            tiles136.append(_to_136(pai))
        opened = m.get('type') != 'ankan'
        result.append(RiichiMeld(mt, tiles136, opened))
    return result


def encode(state: Dict, actor: int):
    """返回 (tile_features, scalar_features)。"""
    tile_feat = np.zeros((C_TILE, N_TILES), dtype=np.float32)
    scalar = np.zeros(N_SCALAR, dtype=np.float32)

    hand: List[str] = state.get("hand", [])
    discards: List[List] = state.get("discards", [[], [], [], []])

    def _disc_pai(d) -> str:
        return d["pai"] if isinstance(d, dict) else d

    def _disc_tsumogiri(d) -> bool:
        return d.get("tsumogiri", False) if isinstance(d, dict) else False

    def _disc_reach_declared(d) -> bool:
        return d.get("reach_declared", False) if isinstance(d, dict) else False
    melds: List[List[dict]] = state.get("melds", [[], [], [], []])
    scores: List[int] = state.get("scores", [25000] * 4)
    dora_markers: List[str] = state.get("dora_markers", [])
    reached: List[bool] = state.get("reached", [False] * 4)
    bakaze: str = state.get("bakaze", "E")
    kyoku: int = state.get("kyoku", 1)
    honba: int = state.get("honba", 0)
    kyotaku: int = state.get("kyotaku", 0)
    oya: int = state.get("oya", 0)

    # ---- ch 0-3: 自家手牌计数 planes ----
    from collections import Counter
    hand_count: Counter = Counter(_to_34(t) for t in hand if _to_34(t) >= 0)
    for tile34, cnt in hand_count.items():
        for k in range(min(cnt, 4)):
            tile_feat[k, tile34] = 1.0

    # ---- ch 4-7: 自家副露 presence ----
    actor_melds = melds[actor] if actor < len(melds) else []
    for meld_slot, meld in enumerate(actor_melds[:4]):
        ch = 4 + meld_slot
        pais = meld.get("consumed", []) + ([meld.get("pai")] if meld.get("pai") else [])
        for p in pais:
            idx = _to_34(p)
            if idx >= 0:
                tile_feat[ch, idx] = 1.0

    # ---- ch 8-13: 他家舍牌（3家 × 2通道，ch_base+0: 立直宣言牌, ch_base+1: 手切flag）----
    other_pids = [pid for pid in range(4) if pid != actor]
    for slot, pid in enumerate(other_pids):  # slot 0,1,2
        ch_base = 8 + slot * 2
        pid_discs = discards[pid] if pid < len(discards) else []
        for d in pid_discs:
            idx = _to_34(_disc_pai(d))
            if idx < 0:
                continue
            if _disc_reach_declared(d):
                tile_feat[ch_base,     idx] = 1.0  # 立直宣言牌
            if not _disc_tsumogiri(d):
                tile_feat[ch_base + 1, idx] = 1.0  # 手切

    # ---- ch 14-16: 他家副露 presence（3家 × 1通道）----
    for slot, pid in enumerate(other_pids):  # slot 0,1,2
        pid_melds_list = melds[pid] if pid < len(melds) else []
        for meld in pid_melds_list:
            pais = meld.get('consumed', []) + ([meld.get('pai')] if meld.get('pai') else [])
            for p in pais:
                idx = _to_34(p)
                if idx >= 0:
                    tile_feat[14 + slot, idx] = 1.0
    # ---- ch 17: 自家舍牌 presence ----
    for disc in discards[actor] if actor < len(discards) else []:
        idx = _to_34(_disc_pai(disc))
        if idx >= 0:
            tile_feat[17, idx] = 1.0

    # ---- ch 18: dora 实际牌（指示牌下一张，累加）----
    for dm in dora_markers:
        dt = _DORA_NEXT.get(dm)
        if dt:
            idx = _to_34(dt)
            if idx >= 0:
                tile_feat[18, idx] += 1.0

    # ---- ch 19: 听牌进张位置（注入 waits_tiles 时有效） ----
    waits_tiles = state.get("waits_tiles")
    if waits_tiles is not None:
        for i, w in enumerate(waits_tiles[:34]):
            if w:
                tile_feat[19, i] = 1.0

    # ---- ch 20-51: 舍牌巡目分段编码（4家 × 8段，ch = 20 + slot*8 + seg）----
    # slot: 他家 0/1/2（与 ch 8-13 相同相对顺序），自家 slot=3
    # 每3巡一段（0-2巡→seg0, 3-5→seg1, ..., 21+→seg7）
    slot_for_pid = {pid: slot for slot, pid in enumerate(other_pids)}
    slot_for_pid[actor] = 3
    for pid in range(4):
        pid_discards = discards[pid] if pid < len(discards) else []
        slot = slot_for_pid[pid]
        for turn, d in enumerate(pid_discards):
            idx = _to_34(_disc_pai(d))
            if idx < 0:
                continue
            seg = min(turn // 3, 7)
            ch = 20 + slot * 8 + seg
            tile_feat[ch, idx] = 1.0

    # ch 52: tsumo 牌位置
    tsumo_pai = state.get("tsumo_pai")
    if tsumo_pai:
        idx = _to_34(tsumo_pai)
        if idx >= 0:
            tile_feat[52, idx] = 1.0

    # ch 53: 最新立直家立直后其他家打过的牌（对该立直家的真安全牌）
    # 找最后一个立直的他家 pid，取其立直宣言牌之后其他家的舍牌
    latest_riichi_pid = None
    latest_riichi_turn = -1
    for pid in other_pids:
        if pid < len(reached) and reached[pid]:
            pid_discs = discards[pid] if pid < len(discards) else []
            for turn, d in enumerate(pid_discs):
                if _disc_reach_declared(d) and turn > latest_riichi_turn:
                    latest_riichi_turn = turn
                    latest_riichi_pid = pid
    if latest_riichi_pid is not None:
        # 收集立直后所有其他家（含 actor）的舍牌
        after_riichi_tiles: set = set()
        for pid in range(4):
            if pid == latest_riichi_pid:
                continue
            pid_discs = discards[pid] if pid < len(discards) else []
            for d in pid_discs:
                idx = _to_34(_disc_pai(d))
                if idx >= 0:
                    after_riichi_tiles.add(idx)
        for idx in after_riichi_tiles:
            tile_feat[53, idx] = 1.0

    actor_disc_n = len(discards[actor]) if actor < len(discards) else 0
    actor_score = scores[actor] if actor < len(scores) else 25000
    jikaze = (actor - oya) % 4

    # ===== 役种特征计算基础 =====
    is_open = len(actor_melds) > 0

    dora_set: Counter = Counter()
    for dm in dora_markers:
        dt = _DORA_NEXT.get(dm)
        if dt:
            dora_set[_to_34(dt)] += 1
    # 手牌+副露的所有牌（tile34 list），仅构建一次，供 dora 计数和役种判断共用
    all34_list: List[int] = list(hand_count.elements())
    for meld in actor_melds:
        for p in meld.get('consumed', []) + ([meld.get('pai')] if meld.get('pai') else []):
            t34 = _to_34(p)
            if t34 >= 0:
                all34_list.append(t34)
    dora_count = sum(dora_set[t] for t in all34_list)

    _YAOCHUU_34 = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
    _MAN_34   = set(range(0, 9))
    _PIN_34   = set(range(9, 18))
    _SOU_34   = set(range(18, 27))
    _HONOR_34 = set(range(27, 34))
    _SUUPAI_34 = set(range(0, 27))

    all34_set = set(all34_list)
    total_tiles = len(all34_list)

    man_cnt   = sum(1 for t in all34_list if t in _MAN_34)
    pin_cnt   = sum(1 for t in all34_list if t in _PIN_34)
    sou_cnt   = sum(1 for t in all34_list if t in _SOU_34)
    honor_cnt = sum(1 for t in all34_list if t in _HONOR_34)
    yaochuu_cnt = sum(1 for t in all34_list if t in _YAOCHUU_34)

    has_tanyao = len(all34_set & _YAOCHUU_34) == 0 and total_tiles > 0
    suupai_yaochuu_cnt = sum(1 for t in all34_list if t in {0,8,9,17,18,26})
    has_junchan = (not is_open) and (honor_cnt == 0) and total_tiles > 0 and (suupai_yaochuu_cnt * 3 >= total_tiles)
    has_chanta = total_tiles > 0 and (yaochuu_cnt * 3 >= total_tiles)
    has_honroutou = total_tiles > 0 and all(t in _YAOCHUU_34 for t in all34_list)
    suit_counts = [c for c in (man_cnt, pin_cnt, sou_cnt) if c > 0]
    has_honitsu = len(suit_counts) == 1 and total_tiles > 0
    has_chinitsu = len(suit_counts) == 1 and honor_cnt == 0 and total_tiles > 0
    has_iipeiko = False
    if not is_open:
        for t34, cnt in hand_count.items():
            if t34 in _SUUPAI_34 and cnt >= 2:
                has_iipeiko = True
                break
    has_sanshoku = False
    for num in range(0, 7):
        has_m = (num in all34_set) or (num+1 in all34_set) or (num+2 in all34_set)
        has_p = (num+9 in all34_set) or (num+10 in all34_set) or (num+11 in all34_set)
        has_s = (num+18 in all34_set) or (num+19 in all34_set) or (num+20 in all34_set)
        if has_m and has_p and has_s:
            has_sanshoku = True
            break
    meld_all_koutsu = all(
        m.get('type') in ('pon', 'daiminkan', 'ankan', 'kakan')
        for m in actor_melds
    ) if actor_melds else True
    has_no_shuntsu = True
    if meld_all_koutsu:
        for suit_start in (0, 9, 18):
            for n in range(suit_start, suit_start + 7):
                if hand_count.get(n,0)>=1 and hand_count.get(n+1,0)>=1 and hand_count.get(n+2,0)>=1:
                    has_no_shuntsu = False
                    break
    has_toitoi = meld_all_koutsu and has_no_shuntsu
    ankoutsu_cnt = sum(1 for cnt in hand_count.values() if cnt >= 3)

    # ===== Scalar features =====
    # [0-2] bakaze 3-bit（E/S/W，N场三者均为0）
    scalar[0] = 1.0 if bakaze == "E" else 0.0
    scalar[1] = 1.0 if bakaze == "S" else 0.0
    scalar[2] = 1.0 if bakaze == "W" else 0.0
    scalar[3] = kyoku / 4.0
    scalar[4] = honba / 8.0
    scalar[5] = kyotaku / 8.0
    scalar[6] = actor_score / 50000.0

    # [7] score rank (0=1st, 3=4th)
    rank = sum(1 for s in scores if s > actor_score)
    scalar[7] = rank / 3.0

    # [8-9] 向听数 / 有效进张数
    # 优先使用 snap 注入的精确值（build_supervised_samples 用 riichi 库算好），
    # 推理时 snap 无注入则 fallback 到自算
    if "shanten" in state:
        scalar[8] = int(state["shanten"]) / 8.0
        scalar[9] = int(state.get("waits_count", 0)) / 34.0
    else:
        try:
            hand_ids = [_to_136(t) for t in hand if _to_136(t) >= 0]
            mahjong_melds = _snap_melds_to_mahjong(actor_melds)
            he = HandEvaluator(hand_ids, mahjong_melds if mahjong_melds else None)
            waits = he.get_waits()
            scalar[8] = (0 if he.is_tenpai() else calculate_shanten(hand_ids)) / 8.0
            scalar[9] = len(waits) / 34.0
        except Exception:
            pass

    scalar[10] = len(actor_melds) / 4.0
    scalar[11] = actor_disc_n / 18.0

    # [12-15] 立直 flag，相对 slot（slot=0: actor, 1: 下家, 2: 对家, 3: 上家）
    all_pids_by_slot = [actor] + other_pids  # slot 0=actor, 1/2/3=他家
    for slot, pid in enumerate(all_pids_by_slot):
        scalar[12 + slot] = 1.0 if (pid < len(reached) and reached[pid]) else 0.0

    # [16] 赤5总数（手牌+副露）/ 4.0
    all_tiles_str: List[str] = list(hand)
    for meld in actor_melds:
        all_tiles_str += meld.get('consumed', [])
        if meld.get('pai'):
            all_tiles_str.append(meld['pai'])
    aka_m = sum(1 for t in all_tiles_str if _is_aka(t) and 'm' in t)
    aka_p = sum(1 for t in all_tiles_str if _is_aka(t) and 'p' in t)
    aka_s = sum(1 for t in all_tiles_str if _is_aka(t) and 's' in t)
    aka_total = aka_m + aka_p + aka_s
    scalar[16] = aka_total / 4.0
    # [17] 断幺九
    scalar[17] = 1.0 if has_tanyao else 0.0
    # [18-20] 赤5 flag（手牌+副露）
    scalar[18] = 1.0 if aka_m > 0 else 0.0
    scalar[19] = 1.0 if aka_p > 0 else 0.0
    scalar[20] = 1.0 if aka_s > 0 else 0.0
    # [21-24] 各家副露数，相对 slot（slot=0: actor, 1: 下家, 2: 对家, 3: 上家）
    for slot, pid in enumerate(all_pids_by_slot):
        pid_melds = melds[pid] if pid < len(melds) else []
        scalar[21 + slot] = len(pid_melds) / 4.0
    # [25-27] 他家分数差（相对 actor），slot=1 下家, 2 对家, 3 上家（actor slot=0 跳过）
    for slot, pid in enumerate(all_pids_by_slot):
        if slot == 0:
            continue
        pid_score = scores[pid] if pid < len(scores) else 25000
        scalar[25 + slot - 1] = (pid_score - actor_score) / 30000.0
    # [28] jikaze（自风，0-3归一化）
    scalar[28] = jikaze / 3.0
    # [29-31] 场风 3-bit
    scalar[29] = 1.0 if bakaze == "E" else 0.0
    scalar[30] = 1.0 if bakaze == "S" else 0.0
    scalar[31] = 1.0 if bakaze == "W" else 0.0
    # [32] dora数量（含赤5，上限clip 10）
    scalar[32] = min((dora_count + aka_total) / 10.0, 1.0)
    # [33] 纯全带幺
    scalar[33] = 1.0 if has_junchan else 0.0
    # [34] 混全带幺
    scalar[34] = 1.0 if has_chanta else 0.0
    # [35] 混老头
    scalar[35] = 1.0 if has_honroutou else 0.0
    # [36-39] 花色占比
    if total_tiles > 0:
        scalar[36] = man_cnt   / total_tiles
        scalar[37] = pin_cnt   / total_tiles
        scalar[38] = sou_cnt   / total_tiles
        scalar[39] = honor_cnt / total_tiles
    # [40] 混一色
    scalar[40] = 1.0 if has_honitsu else 0.0
    # [41] 清一色
    scalar[41] = 1.0 if has_chinitsu else 0.0
    # [42] 一杯口可能性
    scalar[42] = 1.0 if has_iipeiko else 0.0
    # [43] 三色同顺可能性
    scalar[43] = 1.0 if has_sanshoku else 0.0
    # [44] 对对胡可能性
    scalar[44] = 1.0 if has_toitoi else 0.0
    # [45] 三暗刻可能性
    scalar[45] = 1.0 if (ankoutsu_cnt >= 3) else 0.0
    # [46] 振听 flag
    furiten_list = state.get("furiten", [False] * 4)
    scalar[46] = 1.0 if (actor < len(furiten_list) and furiten_list[actor]) else 0.0
    # [47] 确定番数下限（已确定能得到的番，不含进张后可能役种）
    _WIND_34 = [27, 28, 29, 30]  # E/S/W/N
    _SANGEN_34 = [31, 32, 33]    # 白/发/中
    bakaze_34 = 27 + _BAKAZE_IDX.get(bakaze, 0)
    jikaze_34 = 27 + jikaze
    confirmed_han = 0
    # 役牌刻子（场风/自风/三元）— 手牌刻子或副露刻子/杠
    honor_koutsu = set()
    for meld in actor_melds:
        if meld.get('type') in ('pon', 'daiminkan', 'ankan', 'kakan'):
            t34 = _to_34(meld.get('pai', ''))
            if t34 in _HONOR_34:
                honor_koutsu.add(t34)
    for t34, cnt in hand_count.items():
        if t34 in _HONOR_34 and cnt >= 3:
            honor_koutsu.add(t34)
    if bakaze_34 in honor_koutsu:
        confirmed_han += 1
    if jikaze_34 in honor_koutsu and jikaze_34 != bakaze_34:
        confirmed_han += 1
    confirmed_han += sum(1 for t in _SANGEN_34 if t in honor_koutsu)
    # 断幺九（确定无幺九）
    if has_tanyao:
        confirmed_han += 1
    # 混一色/清一色（确定）
    if has_chinitsu:
        confirmed_han += 6 if not is_open else 5
    elif has_honitsu:
        confirmed_han += 3 if not is_open else 2
    # dora + 赤5
    confirmed_han += dora_count + aka_total
    scalar[47] = min(confirmed_han / 8.0, 1.0)

    return tile_feat, scalar
