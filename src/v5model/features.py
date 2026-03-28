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
  40-43 : 场风/自风 one-hot（broadcast）
  44    : 宝牌指示牌数量（归一化，broadcast）
  45    : 本场数（归一化）
  46    : 供托数（归一化）
  47    : 巡目估计（归一化，= 自家舍牌数/24）
  48-51 : 各家分数相对差（4 通道，每通道 = (score_i - actor_score)/30000，broadcast）
  52-55 : 保留为 0（对齐旧版索引）
  56    : 自家手牌含赤5m（aka 0m）flag（broadcast）
  57    : 自家手牌含赤5p（aka 0p）flag（broadcast）
  58    : 自家手牌含赤5s（aka 0s）flag（broadcast）
  59-127: 补 0（保留）

N_SCALAR = 16：
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
"""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from riichienv import calculate_shanten, HandEvaluator
import riichienv.convert as _cvt

from mahjong_env.tiles import normalize_tile, AKA_DORA_TILES

# ---------------------------------------------------------------------------
# 34 种牌名顺序（与 action_space.py 保持一致）
# ---------------------------------------------------------------------------
_SUITS = ("m", "p", "s")
_TILE_NAMES: List[str] = []
for _suit in _SUITS:
    for _n in range(1, 10):
        _TILE_NAMES.append(f"{_n}{_suit}")
_TILE_NAMES.extend(["E", "S", "W", "N", "P", "F", "C"])

TILE_TO_IDX: Dict[str, int] = {t: i for i, t in enumerate(_TILE_NAMES)}
N_TILES = 34

C_TILE = 128
N_SCALAR = 16

_WIND_ORDER = ["E", "S", "W", "N"]
_BAKAZE_IDX = {"E": 0, "S": 1, "W": 2, "N": 3}


def _deaka(tile: str) -> str:
    if tile in AKA_DORA_TILES:
        return normalize_tile(tile)
    return tile


def _tile_idx(tile: str) -> int:
    return TILE_TO_IDX.get(_deaka(tile), -1)


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
    hand_count: Counter = Counter(_deaka(t) for t in hand)
    for tile, cnt in hand_count.items():
        idx = _tile_idx(tile)
        if idx < 0:
            continue
        for k in range(min(cnt, 4)):
            tile_feat[k, idx] = 1.0

    # ---- ch 4-7: 自家副露 presence ----
    actor_melds = melds[actor] if actor < len(melds) else []
    for meld_slot, meld in enumerate(actor_melds[:4]):
        ch = 4 + meld_slot
        pais = meld.get("consumed", []) + ([meld.get("pai")] if meld.get("pai") else [])
        for p in pais:
            idx = _tile_idx(p)
            if idx >= 0:
                tile_feat[ch, idx] = 1.0

    # ---- ch 8-20: 各家舍牌 presence ----
    other_pids = [pid for pid in range(4) if pid != actor]
    for slot, pid in enumerate(other_pids):  # slot 0,1,2
        ch_base = 8 + slot * 4
        for disc in discards[pid] if pid < len(discards) else []:
            idx = _tile_idx(disc)
            if idx >= 0:
                tile_feat[ch_base, idx] = 1.0  # presence（不区分顺序）
    # 自家舍牌 ch 20
    for disc in discards[actor] if actor < len(discards) else []:
        idx = _tile_idx(disc)
        if idx >= 0:
            tile_feat[20, idx] = 1.0

    # ---- ch 21-24: 各家立直状态（broadcast） ----
    for pid in range(4):
        if pid < len(reached) and reached[pid]:
            tile_feat[21 + pid, :] = 1.0

    # ---- ch 25-29: dora 指示牌（累加，最多5个） ----
    for di, dm in enumerate(dora_markers[:5]):
        idx = _tile_idx(dm)
        if idx >= 0:
            tile_feat[25 + di, idx] = 1.0

    # ch 30, 31: 听牌信息（暂时留 0，需 shanten 计算库）

    # ---- ch 32-35: 各家副露数（归一化，broadcast） ----
    for pid in range(4):
        n_melds = len(melds[pid]) if pid < len(melds) else 0
        tile_feat[32 + pid, :] = n_melds / 4.0

    # ---- ch 36-39: 各家舍牌数（归一化，broadcast） ----
    for pid in range(4):
        n_disc = len(discards[pid]) if pid < len(discards) else 0
        tile_feat[36 + pid, :] = n_disc / 24.0

    # ---- ch 40-43: 场风/自风 one-hot（broadcast） ----
    bk_idx = _BAKAZE_IDX.get(bakaze, 0)
    jikaze = (actor - oya) % 4
    tile_feat[40 + bk_idx, :] = 1.0
    tile_feat[40 + jikaze, :] = 1.0  # 自风（可与场风重叠）

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
    aka_flags = {"0m": 56, "0p": 57, "0s": 58}
    for tile in hand:
        if tile in aka_flags:
            tile_feat[aka_flags[tile], :] = 1.0

    # ch 59-127: 保留为 0

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
    try:
        tile_ids = _cvt.mjai_to_tid_list(hand)
        shanten = calculate_shanten(tile_ids)
        scalar[7] = shanten / 8.0
        if shanten == 0 and len(tile_ids) == 13:
            he = HandEvaluator(tile_ids)
            scalar[8] = len(he.get_waits()) / 34.0
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
