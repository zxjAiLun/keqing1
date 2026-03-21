from __future__ import annotations

from collections import Counter
from typing import Dict, List

from mahjong_env.tiles import all_discardable_tiles_with_aka


TILES = all_discardable_tiles_with_aka()
TILE_INDEX = {t: i for i, t in enumerate(TILES)}
OBS_DIM = 270  # + shanten(1) + waits_count(1)


def vectorize_state_py(state: Dict, actor: int) -> List[float]:
    """
    Pure-python version of `train.dataset.vectorize_state`.
    mjai Docker runtime may not have numpy, so inference must stay numpy-less.
    """
    hand = Counter(state["hand"])
    x = [0.0] * OBS_DIM

    # [0:34] own hand tile histogram
    # [34:170] all players discards histogram (4 * 34)
    # [170:174] one-hot actor id
    # [174:178] reached flags
    # [178:182] normalized scores
    # [182:188] round features
    # [188:222] last discard tile one-hot histogram
    # [222:256] dora marker histogram
    # [256:260] meld count per player
    # [260:264] normalized discard count per player
    # [264:268] one-hot oya id

    for t, c in hand.items():
        idx = TILE_INDEX.get(t)
        if idx is not None:
            x[idx] = float(c)

    for pid in range(4):
        disc = Counter(state["discards"][pid])
        base = 34 + pid * 34
        for t, c in disc.items():
            idx = TILE_INDEX.get(t)
            if idx is not None:
                x[base + idx] = float(c)

    x[170 + actor] = 1.0
    for pid, reached in enumerate(state["reached"]):
        x[174 + pid] = 1.0 if reached else 0.0

    for pid, score in enumerate(state["scores"]):
        x[178 + pid] = float(score) / 50000.0

    x[182] = float(state["kyoku"]) / 4.0
    x[183] = float(state["honba"]) / 10.0
    x[184] = float(state["kyotaku"]) / 10.0
    x[185] = 1.0 if state["actor_to_move"] == actor else 0.0
    x[186] = 1.0 if state.get("last_discard") is not None else 0.0
    x[187] = float(len(state["hand"])) / 14.0

    last_discard = state.get("last_discard")
    if last_discard and last_discard.get("pai") in TILE_INDEX:
        x[188 + TILE_INDEX[last_discard["pai"]]] = 1.0

    for d in state.get("dora_markers", []):
        idx = TILE_INDEX.get(d)
        if idx is not None:
            x[222 + idx] += 1.0

    for pid in range(4):
        x[256 + pid] = float(len(state["melds"][pid])) / 4.0
        x[260 + pid] = float(len(state["discards"][pid])) / 24.0

    x[264 + int(state["oya"])] = 1.0

    # Extra shanten/waits proxy features (computed from libriichi oracle state).
    # Defaults keep compatibility with old snapshots.
    shanten = float(state.get("shanten", 0.0))
    waits_count = float(state.get("waits_count", 0.0))
    x[268] = shanten / 8.0
    x[269] = waits_count / 34.0
    return x

