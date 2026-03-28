"""特征编码器：将 GameState.snapshot() 字典编码为神经网络输入。

输出：
  tile_features : np.ndarray shape (C_TILE, 34)  float32
  scalar_features: np.ndarray shape (N_SCALAR,)   float32

通道布局 (C_TILE = 128，前 59 个通道有语义，其余补 0 供未来扩展)：
  0-3   : 自家手牌计数 planes（plane k = 该牌种数量 >= k+1）
  4-7   : 自家副露 presence（4 个副露槽，有牌 = 1.0）
  8-11  : 对家1 舍牌 presence
  12-15 : 对家2 舍牌 presence
  16-19 : 对家3 舍牌 presence
  20    : 自家舍牌 presence
  21-24 : 各家立直状态（broadcast，全列同值）
  25-29 : dora 指示牌 planes（最多 5 个，累加）
  30    : 自家听牌 flag（broadcast）
  31    : 自家听牌进张（waits，有效进张 = 1.0）
  32-35 : 4家副露数量（归一化，broadcast）
  36-39 : 4家舍牌数量（归一化，broadcast）
  40-43 : 场风 one-hot（E/S/W/N，broadcast）
  91-94 : 自风 one-hot（E/S/W/N，broadcast）
  44    : 宝牌指示牌数量（归一化，broadcast）
  45    : 本场数（归一化）
  46    : 供托数（归一化）
  47    : 巡目估计（归一化，= 自家舍牌数/24）
  48-51 : 各家分数相对差（4 通道，每通道 = (score_i - actor_score)/30000，broadcast）
  52-55 : 保留为 0（对齐旧版索引）
  56    : 自家手牌含赤5m（aka 0m）flag（broadcast）
  57    : 自家手牌含赤5p（aka 0p）flag（broadcast）
  58    : 自家手牌含赤5s（aka 0s）flag（broadcast）
  59-90 : 舍牌巡目分段编码（4家 × 8段，ch = 59 + pid*8 + seg）
          seg = min(discard_turn // 3, 7)，即每3巡一段，共8段
          第 k 张舍牌在对应段的对应牌位置置 1
  91-94 : 自风 one-hot（E/S/W/N，broadcast）
  95-127: 补 0（保留）

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

from mahjong.tile import TilesConverter, FIVE_RED_MAN, FIVE_RED_PIN, FIVE_RED_SOU
from riichienv import calculate_shanten, HandEvaluator, Meld as RiichiMeld, MeldType

# ---------------------------------------------------------------------------
# 牌格式转换（项目字符串格式 → tile136 / tile34）
# 项目格式: "1m".."9m", "5mr", "1p".."9p", "5pr", "1s".."9s", "5sr",
#           "E","S","W","N","P","F","C"
# ---------------------------------------------------------------------------
_SUIT_KW = {'m': 'man', 'p': 'pin', 's': 'sou'}
_HONOR_Z  = {'E': '1', 'S': '2', 'W': '3', 'N': '4', 'P': '5', 'F': '6', 'C': '7'}

_STR_TO_136: Dict[str, int] = {}
for _suit, _kw in _SUIT_KW.items():
    for _n in range(1, 10):
        _key = f'{_n}{_suit}'
        _t = TilesConverter.string_to_136_array(**{_kw: str(_n)}, has_aka_dora=True)
        _STR_TO_136[_key] = _t[0]
_STR_TO_136['5mr'] = FIVE_RED_MAN
_STR_TO_136['5pr'] = FIVE_RED_PIN
_STR_TO_136['5sr'] = FIVE_RED_SOU
for _name, _z in _HONOR_Z.items():
    _STR_TO_136[_name] = TilesConverter.string_to_136_array(honors=_z)[0]

_AKA_136 = frozenset({FIVE_RED_MAN, FIVE_RED_PIN, FIVE_RED_SOU})


def _to_136(tile: str) -> int:
    """项目字符串 → tile136；未知牌返回 -1。"""
    return _STR_TO_136.get(tile, -1)


def _to_34(tile: str) -> int:
    """项目字符串 → tile34（0-33）；未知牌返回 -1。"""
    t136 = _STR_TO_136.get(tile, -1)
    return t136 // 4 if t136 >= 0 else -1


def _is_aka(tile: str) -> bool:
    return _STR_TO_136.get(tile, -1) in _AKA_136


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
    discards: List[List[str]] = state.get("discards", [[], [], [], []])
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

    # ---- ch 8-20: 各家舍牌 presence ----
    other_pids = [pid for pid in range(4) if pid != actor]
    for slot, pid in enumerate(other_pids):  # slot 0,1,2
        ch_base = 8 + slot * 4
        for disc in discards[pid] if pid < len(discards) else []:
            idx = _to_34(disc)
            if idx >= 0:
                tile_feat[ch_base, idx] = 1.0  # presence（不区分顺序）
    # 自家舍牌 ch 20
    for disc in discards[actor] if actor < len(discards) else []:
        idx = _to_34(disc)
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
            hand_ids = [_to_136(t) for t in hand if _to_136(t) >= 0]
            mahjong_melds = _snap_melds_to_mahjong(actor_melds)
            he = HandEvaluator(hand_ids, mahjong_melds if mahjong_melds else None)
            if he.is_tenpai():
                tile_feat[30, :] = 1.0
                # ch31 fallback: leave 0 (get_waits() returns discardable tiles, not waits)
        except Exception:
            pass

    # ---- ch 32-35: 各家副露数（归一化，broadcast） ----
    for pid in range(4):
        n_melds = len(melds[pid]) if pid < len(melds) else 0
        tile_feat[32 + pid, :] = n_melds / 4.0

    # ---- ch 36-39: 各家舍牌数（归一化，broadcast） ----
    for pid in range(4):
        n_disc = len(discards[pid]) if pid < len(discards) else 0
        tile_feat[36 + pid, :] = n_disc / 24.0

    # ---- ch 40-43: 场风 one-hot（E/S/W/N，broadcast） ----
    bk_idx = _BAKAZE_IDX.get(bakaze, 0)
    tile_feat[40 + bk_idx, :] = 1.0

    # ---- ch 91-94: 自风 one-hot（E/S/W/N，broadcast） ----
    jikaze = (actor - oya) % 4
    tile_feat[91 + jikaze, :] = 1.0

    # ---- ch 44: dora 数量（归一化，broadcast） ----
    tile_feat[44, :] = len(dora_markers) / 5.0

    # ---- ch 45: 本场数 ----
    tile_feat[45, :] = honba / 8.0

    # ---- ch 46: 供托数 ----
    tile_feat[46, :] = kyotaku / 8.0

    # ---- ch 47: 巡目估计 ----
    actor_disc_n = len(discards[actor]) if actor < len(discards) else 0
    tile_feat[47, :] = actor_disc_n / 24.0

    # ---- ch 48-51: 各家分数相对差（broadcast） ----
    actor_score = scores[actor] if actor < len(scores) else 25000
    for pid in range(4):
        s = scores[pid] if pid < len(scores) else 25000
        tile_feat[48 + pid, :] = (s - actor_score) / 30000.0

    # ---- ch 56-58: 自家手牌赤宝牌 flag（broadcast） ----
    for tile in hand:
        t136 = _to_136(tile)
        if t136 == FIVE_RED_MAN:
            tile_feat[56, :] = 1.0
        elif t136 == FIVE_RED_PIN:
            tile_feat[57, :] = 1.0
        elif t136 == FIVE_RED_SOU:
            tile_feat[58, :] = 1.0

    # ---- ch 59-90: 舍牌巡目分段编码（4家 × 8段）----
    # 每3巡一段（0-2巡→seg0, 3-5→seg1, ..., 21+→seg7）
    for pid in range(4):
        pid_discards = discards[pid] if pid < len(discards) else []
        for turn, tile in enumerate(pid_discards):
            idx = _to_34(tile)
            if idx < 0:
                continue
            seg = min(turn // 3, 7)
            ch = 59 + pid * 8 + seg
            tile_feat[ch, idx] = 1.0

    # ch 91-127: 保留为 0

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
