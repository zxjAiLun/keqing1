"""特征编码器：将 GameState.snapshot() 字典编码为神经网络输入。

输出：
  tile_features : np.ndarray shape (C_TILE, 34)  float32
  scalar_features: np.ndarray shape (N_SCALAR,)   float32

通道布局 (C_TILE = 128)：
  0-3   : 自家手牌计数 planes（plane k = 该牌种数量 >= k+1）
  4-7   : 自家副露 presence（4 个副露槽，有牌 = 1.0）
  8-11  : 对家1 舍牌（+0 presence, +1 归一化巡目, +2 立直宣言牌, +3 手切flag）
  12-15 : 对家2 舍牌（同上）
  16-19 : 对家3 舍牌（同上）
  20    : 自家舍牌 presence
  21-24 : 各家立直状态（broadcast，全列同值）
  25-29 : dora 指示牌 planes（最多 5 个，累加）
  30    : 自家听牌 flag（broadcast）
  31    : 自家听牌进张（waits，有效进张 = 1.0）
  32-35 : 4家副露数量（归一化，broadcast）
  36-39 : [空闲]
  40-43 : 场风 one-hot（E/S/W/N，broadcast）
  44    : 宝牌指示牌数量（归一化，broadcast）
  45    : 本场数（归一化）
  46    : 供托数（归一化）
  47    : [空闲]
  48-51 : 各家分数相对差（4 通道，每通道 = (score_i - actor_score)/30000，broadcast）
  52    : open hand flag（有副露=1，broadcast）
  53    : 自家宝牌数（手牌+副露，归一化 /4，broadcast）
  54    : 断幺九可能性（手牌+副露无幺九牌，broadcast）
  55    : 平胡可能性（门清+无幺九牌近似，broadcast）
  56    : 自家手牌含赤5m（aka 0m）flag（broadcast）
  57    : 自家手牌含赤5p（aka 0p）flag（broadcast）
  58    : 自家手牌含赤5s（aka 0s）flag（broadcast）
  59-90 : 舍牌巡目分段编码（4家 × 8段，ch = 59 + pid*8 + seg）
          seg = min(discard_turn // 3, 7)，即每3巡一段，共8段
          第 k 张舍牌在对应段的对应牌位置置 1
  91-94 : 自风 one-hot（E/S/W/N，broadcast）
  95    : 纯全带幺可能性（门清+无字牌+幺九数牌占比>=1/3，broadcast）
  96    : 混全带幺可能性（幺九牌占比>=1/3，broadcast）
  97    : 混老头可能性（全部牌为幺九牌，broadcast）
  98-101: 花色占比（万/饼/索/字，broadcast）
  102   : 混一色可能性（只含1种数牌花色，可含字牌，broadcast）
  103   : 清一色可能性（只含1种数牌花色，不含字牌，broadcast）
  104   : 一杯口可能性（门清+手牌中同种牌>=2张，broadcast）
  105   : 三色同顺可能性（手牌+副露中m/p/s含相同数字顺子雏形，broadcast）
  106   : 对对胡可能性（副露全刻/杠+手牌无顺子块，broadcast）
  107   : 三暗刻可能性（手牌中>=3种牌数量>=3，broadcast）
  108   : [空闲]
  109-127: [空闲]

N_SCALAR = 20：
  0  : bakaze one-hot bit 0（E=1, else=0）
  1  : bakaze one-hot bit 1（S=1, else=0）
  2  : kyoku (1-4) / 4.0
  3  : honba / 8.0
  4  : kyotaku / 8.0
  5  : actor score / 50000
  6  : score rank（自家在4家中的排名，0-3，归一化）
  7  : 向听数 / 8.0（需从手牌推算，暂用 0）
  8  : 有效进张数 / 34.0（暂用 0）
  9  : 自家立直 flag
  10 : 自家副露数 / 4.0
  11 : 自家舍牌数 / 24.0
  12-15: 其余各家立直 flag（pid != actor）
  16 : style_speed   [-1,+1]  速攻(+1) vs 慢打(-1)，训练时=0
  17 : style_riichi  [-1,+1]  立直优先(+1) vs 默听(-1)，训练时=0
  18 : style_value   [-1,+1]  高打点(+1) vs 注重胡率(-1)，训练时=0
  19 : style_defense [-1,+1]  铁壁防守(+1) vs 对攻(-1)，训练时=0
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from riichienv import calculate_shanten, HandEvaluator, Meld as RiichiMeld, MeldType

from mahjong_env.tiles import tile_to_136 as _to_136, tile_to_34 as _to_34, tile_is_aka as _is_aka


# ---------------------------------------------------------------------------
# 34 种牌名顺序（与 action_space.py 保持一致，tile34 index 对应）
# ---------------------------------------------------------------------------
N_TILES = 34
C_TILE = 128
N_SCALAR = 20

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

    # ---- ch 8-19: 他家舍牌（3家 × 4通道） ----
    # ch_base+0: presence(0/1)  ch_base+1: 归一化巡目(turn/24)  ch_base+2: 立直宣言牌(0/1)  ch_base+3: 手切(1)/摸切(0)
    other_pids = [pid for pid in range(4) if pid != actor]
    for slot, pid in enumerate(other_pids):  # slot 0,1,2
        ch_base = 8 + slot * 4
        pid_discs = discards[pid] if pid < len(discards) else []
        for turn, d in enumerate(pid_discs):
            idx = _to_34(_disc_pai(d))
            if idx < 0:
                continue
            tile_feat[ch_base,     idx] = 1.0                          # presence
            tile_feat[ch_base + 1, idx] = max(tile_feat[ch_base + 1, idx], turn / 24.0)  # 最后巡目
            if _disc_reach_declared(d):
                tile_feat[ch_base + 2, idx] = 1.0                      # 立直宣言牌
            if not _disc_tsumogiri(d):
                tile_feat[ch_base + 3, idx] = 1.0                      # 手切
    # ---- ch 20: 自家舍牌 presence ----
    for disc in discards[actor] if actor < len(discards) else []:
        idx = _to_34(_disc_pai(disc))
        if idx >= 0:
            tile_feat[20, idx] = 1.0

    # ---- ch 21-24: 各家立直状态（broadcast） ----
    for pid in range(4):
        if pid < len(reached) and reached[pid]:
            tile_feat[21 + pid, :] = 1.0

    # ---- ch 25-29: dora 指示牌（累加，最多5个） ----
    for di, dm in enumerate(dora_markers[:5]):
        idx = _to_34(dm)
        if idx >= 0:
            tile_feat[25 + di, idx] = 1.0

    # ---- ch 30, 31: 听牌 flag / 进张位置 ----
    waits_tiles = state.get("waits_tiles")  # length-34 bool list injected by bot/replay
    if waits_tiles is not None:
        if any(waits_tiles):
            tile_feat[30, :] = 1.0
            for i, w in enumerate(waits_tiles[:34]):
                if w:
                    tile_feat[31, i] = 1.0
    else:
        try:
            from mahjong_env.replay import _calc_shanten_waits
            _, _, waits34_fb, _ = _calc_shanten_waits(hand, actor_melds)
            if any(waits34_fb):
                tile_feat[30, :] = 1.0
                # ch31 只在注入 waits_tiles 时填写，fallback 不填
        except Exception:
            pass

    # ---- ch 32-35: 各家副露数（归一化，broadcast） ----
    for pid in range(4):
        n_melds = len(melds[pid]) if pid < len(melds) else 0
        tile_feat[32 + pid, :] = n_melds / 4.0

    # ch 36-39: 已删除（各家舍牌数 broadcast，与 scalar 重复）

    # ---- ch 40-43: 场风 one-hot（E/S/W/N，broadcast） ----
    bk_idx = _BAKAZE_IDX.get(bakaze, 0)
    tile_feat[40 + bk_idx, :] = 1.0

    # ---- ch 44: dora 数量（归一化，broadcast） ----
    tile_feat[44, :] = len(dora_markers) / 5.0

    # ---- ch 45: 本场数 ----
    tile_feat[45, :] = honba / 8.0

    # ---- ch 46: 供托数 ----
    tile_feat[46, :] = kyotaku / 8.0

    # ch 47: 已删除（巡目估计，与 scalar[11] 重复）
    actor_disc_n = len(discards[actor]) if actor < len(discards) else 0

    # ---- ch 48-51: 各家分数相对差（broadcast） ----
    actor_score = scores[actor] if actor < len(scores) else 25000
    for pid in range(4):
        s = scores[pid] if pid < len(scores) else 25000
        tile_feat[48 + pid, :] = (s - actor_score) / 30000.0

    # ---- ch 56-58: 自家手牌赤宝牌 flag（broadcast） ----
    for tile in hand:
        if tile == '5mr':
            tile_feat[56, :] = 1.0
        elif tile == '5pr':
            tile_feat[57, :] = 1.0
        elif tile == '5sr':
            tile_feat[58, :] = 1.0

    # ---- ch 59-90: 舍牌巡目分段编码（4家 × 8段）----
    # 每3巡一段（0-2巡→seg0, 3-5→seg1, ..., 21+→seg7）
    for pid in range(4):
        pid_discards = discards[pid] if pid < len(discards) else []
        for turn, d in enumerate(pid_discards):
            idx = _to_34(_disc_pai(d))
            if idx < 0:
                continue
            seg = min(turn // 3, 7)
            ch = 59 + pid * 8 + seg
            tile_feat[ch, idx] = 1.0

    # ---- ch 91-94: 自风 one-hot（E/S/W/N，broadcast） ----
    jikaze = (actor - oya) % 4
    tile_feat[91 + jikaze, :] = 1.0

    # ---- ch 52: open hand flag（有副露=1，broadcast） ----
    is_open = len(actor_melds) > 0
    tile_feat[52, :] = 1.0 if is_open else 0.0

    # ---- ch 53: 自家宝牌数（手牌+副露，归一化，broadcast） ----
    # dora marker -> 下一张牌为宝牌
    _DORA_NEXT = {
        '1m':'2m','2m':'3m','3m':'4m','4m':'5m','5m':'6m','6m':'7m','7m':'8m','8m':'9m','9m':'1m',
        '1p':'2p','2p':'3p','3p':'4p','4p':'5p','5p':'6p','6p':'7p','7p':'8p','8p':'9p','9p':'1p',
        '1s':'2s','2s':'3s','3s':'4s','4s':'5s','5s':'6s','6s':'7s','7s':'8s','8s':'9s','9s':'1s',
        'E':'S','S':'W','W':'N','N':'E','P':'F','F':'C','C':'P',
    }
    dora_set: Counter = Counter()
    for dm in dora_markers:
        dt = _DORA_NEXT.get(dm)
        if dt:
            dora_set[_to_34(dt)] += 1
    all_tiles_34 = list(hand_count.elements())
    for meld in actor_melds:
        for p in meld.get('consumed', []) + ([meld.get('pai')] if meld.get('pai') else []):
            t34 = _to_34(p)
            if t34 >= 0:
                all_tiles_34.append(t34)
    dora_count = sum(dora_set[t] for t in all_tiles_34)
    tile_feat[53, :] = min(dora_count / 4.0, 1.0)

    # ---- 役种特征计算基础：手牌+副露所有牌的 tile34 集合 ----
    _YAOCHUU_34 = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}  # 1m9m1p9p1s9s ESWN字
    _MAN_34   = set(range(0, 9))
    _PIN_34   = set(range(9, 18))
    _SOU_34   = set(range(18, 27))
    _HONOR_34 = set(range(27, 34))
    _SUUPAI_34 = set(range(0, 27))  # 数牌

    all34_list: List[int] = list(hand_count.elements())
    for meld in actor_melds:
        for p in meld.get('consumed', []) + ([meld.get('pai')] if meld.get('pai') else []):
            t34 = _to_34(p)
            if t34 >= 0:
                all34_list.append(t34)
    all34_set = set(all34_list)
    total_tiles = len(all34_list)

    man_cnt   = sum(1 for t in all34_list if t in _MAN_34)
    pin_cnt   = sum(1 for t in all34_list if t in _PIN_34)
    sou_cnt   = sum(1 for t in all34_list if t in _SOU_34)
    honor_cnt = sum(1 for t in all34_list if t in _HONOR_34)
    yaochuu_cnt = sum(1 for t in all34_list if t in _YAOCHUU_34)

    # ---- ch 54: 断幺九可能性（手牌+副露无幺九牌，broadcast） ----
    has_tanyao = (all34_set - _YAOCHUU_34 == all34_set) if all34_set else False
    has_tanyao = len(all34_set & _YAOCHUU_34) == 0 and total_tiles > 0
    tile_feat[54, :] = 1.0 if has_tanyao else 0.0

    # ---- ch 55: 平胡可能性（门清+手牌无幺九牌近似，broadcast） ----
    has_pinfu = (not is_open) and (len(all34_set & _YAOCHUU_34) == 0) and total_tiles > 0
    tile_feat[55, :] = 1.0 if has_pinfu else 0.0

    # ---- ch 95: 纯全带幺可能性（门清+无字牌+幺九数牌占比高，broadcast） ----
    # 近似：门清 + 无字牌 + 幺九数牌(1/9)占比 >= 1/3
    suupai_yaochuu_cnt = sum(1 for t in all34_list if t in {0,8,9,17,18,26})
    has_junchan = (not is_open) and (honor_cnt == 0) and total_tiles > 0 and (suupai_yaochuu_cnt * 3 >= total_tiles)
    tile_feat[95, :] = 1.0 if has_junchan else 0.0

    # ---- ch 96: 混全带幺可能性（幺九牌占比 >= 1/3，broadcast） ----
    has_chanta = total_tiles > 0 and (yaochuu_cnt * 3 >= total_tiles)
    tile_feat[96, :] = 1.0 if has_chanta else 0.0

    # ---- ch 97: 混老头可能性（全部牌都是幺九牌，broadcast） ----
    has_honroutou = total_tiles > 0 and all(t in _YAOCHUU_34 for t in all34_list)
    tile_feat[97, :] = 1.0 if has_honroutou else 0.0

    # ---- ch 98-101: 花色占比（万/饼/索/字，broadcast） ----
    if total_tiles > 0:
        tile_feat[98, :] = man_cnt   / total_tiles
        tile_feat[99, :] = pin_cnt   / total_tiles
        tile_feat[100, :] = sou_cnt  / total_tiles
        tile_feat[101, :] = honor_cnt / total_tiles

    # ---- ch 102: 混一色可能性（只含1种数牌花色，可含字牌，broadcast） ----
    suit_counts = [c for c in (man_cnt, pin_cnt, sou_cnt) if c > 0]
    has_honitsu = len(suit_counts) == 1 and total_tiles > 0
    tile_feat[102, :] = 1.0 if has_honitsu else 0.0

    # ---- ch 103: 清一色可能性（只含1种数牌花色，不含字牌，broadcast） ----
    has_chinitsu = len(suit_counts) == 1 and honor_cnt == 0 and total_tiles > 0
    tile_feat[103, :] = 1.0 if has_chinitsu else 0.0

    # ---- ch 104: 一杯口可能性（门清+手牌中有2组相同顺子雏形，broadcast） ----
    # 近似：门清 + 手牌中同一花色内有2张相同数字的牌（顺子共用牌）
    has_iipeiko = False
    if not is_open:
        for t34, cnt in hand_count.items():
            if t34 in _SUUPAI_34 and cnt >= 2:
                has_iipeiko = True
                break
    tile_feat[104, :] = 1.0 if has_iipeiko else 0.0

    # ---- ch 105: 三色同顺可能性（手牌+副露中m/p/s含相同数字的顺子雏形，broadcast） ----
    # 近似：对每个数字n(1-7)，检查手牌+副露中m/p/s均含n号数牌
    has_sanshoku = False
    for num in range(0, 7):  # 0-6 对应 1-7
        has_m = (num in all34_set) or (num+1 in all34_set) or (num+2 in all34_set)
        has_p = (num+9 in all34_set) or (num+10 in all34_set) or (num+11 in all34_set)
        has_s = (num+18 in all34_set) or (num+19 in all34_set) or (num+20 in all34_set)
        if has_m and has_p and has_s:
            has_sanshoku = True
            break
    tile_feat[105, :] = 1.0 if has_sanshoku else 0.0

    # ---- ch 106: 对对胡可能性（副露全是刻/杠+手牌无连续3张不同数字，broadcast） ----
    # 近似：副露全是刻/杠 + 手牌中同一花色内无3张连续不同数字
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
    tile_feat[106, :] = 1.0 if has_toitoi else 0.0

    # ---- ch 107: 三暗刻可能性（手牌中有>=3组刻子，broadcast） ----
    # 近似：手牌中有>=3种牌数量>=3
    ankoutsu_cnt = sum(1 for cnt in hand_count.values() if cnt >= 3)
    tile_feat[107, :] = 1.0 if ankoutsu_cnt >= 3 else 0.0

    # ch 108: 已删除（门清 flag，与 ch52 互补冗余）

    # ===== Scalar features =====
    scalar[0] = 1.0 if bakaze == "E" else 0.0
    scalar[1] = 1.0 if bakaze == "S" else 0.0
    scalar[2] = kyoku / 4.0
    scalar[3] = honba / 8.0
    scalar[4] = kyotaku / 8.0
    scalar[5] = actor_score / 50000.0

    # score rank (0=1st, 3=4th)
    rank = sum(1 for s in scores if s > actor_score)
    scalar[6] = rank / 3.0

    # scalar[7]: 向听数，scalar[8]: 有效进张数
    # 优先使用 snap 注入的精确值（build_supervised_samples 用 riichi 库算好），
    # 推理时 snap 无注入则 fallback 到自算
    if "shanten" in state:
        scalar[7] = int(state["shanten"]) / 8.0
        scalar[8] = int(state.get("waits_count", 0)) / 34.0
    else:
        try:
            hand_ids = [_to_136(t) for t in hand if _to_136(t) >= 0]
            mahjong_melds = _snap_melds_to_mahjong(actor_melds)
            he = HandEvaluator(hand_ids, mahjong_melds if mahjong_melds else None)
            waits = he.get_waits()
            scalar[7] = (0 if he.is_tenpai() else calculate_shanten(hand_ids)) / 8.0
            scalar[8] = len(waits) / 34.0
        except Exception:
            pass

    actor_reached = reached[actor] if actor < len(reached) else False
    scalar[9] = 1.0 if actor_reached else 0.0
    scalar[10] = len(actor_melds) / 4.0
    scalar[11] = actor_disc_n / 24.0

    other_reach_slot = 12
    for pid in range(4):
        if pid == actor:
            continue
        scalar[other_reach_slot] = 1.0 if (pid < len(reached) and reached[pid]) else 0.0
        other_reach_slot += 1

    return tile_feat, scalar
