"""Xmodel1 runtime/state feature helpers."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np

from keqingv3.features import C_TILE, N_SCALAR, encode as _encode_state, encode_with_timings
from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.tiles import tile_to_34
from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_MAX_SPECIAL_CANDIDATES,
    XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
)
from xmodel1.candidate_quality import build_candidate_features, build_special_candidate_arrays, iter_legal_discards

EVENT_HISTORY_LEN = 48
EVENT_HISTORY_FEATURE_DIM = 5
EVENT_TYPE_PAD = 0
EVENT_NO_ACTOR = 4
EVENT_NO_TILE = -1
EVENT_MAX_TURN_IDX = 24
_EVENT_TYPE_ID = {
    "tsumo": 1,
    "dahai": 2,
    "pon": 3,
    "chi": 4,
    "daiminkan": 5,
    "ankan": 6,
    "kakan": 7,
    "reach": 8,
    "dora": 9,
    "hora": 10,
    "ryukyoku": 11,
}


def encode(state: dict, actor: int, *, state_scalar_dim: int = 64):
    tile_feat, scalar = _encode_state(state, actor)
    if scalar.shape[0] < state_scalar_dim:
        padded = np.zeros((state_scalar_dim,), dtype=np.float32)
        padded[: scalar.shape[0]] = scalar
        scalar = padded
    elif scalar.shape[0] > state_scalar_dim:
        scalar = scalar[:state_scalar_dim]
    return tile_feat, scalar.astype(np.float32)


def empty_event_history() -> np.ndarray:
    out = np.zeros((EVENT_HISTORY_LEN, EVENT_HISTORY_FEATURE_DIM), dtype=np.int16)
    out[:, 0] = EVENT_NO_ACTOR
    out[:, 1] = EVENT_TYPE_PAD
    out[:, 2] = EVENT_NO_TILE
    return out


def compute_event_history(all_events: list[dict], event_index: int) -> np.ndarray:
    out = empty_event_history()
    if event_index <= 0:
        return out
    end = min(int(event_index), len(all_events))
    if end <= 0:
        return out
    kyoku_start = 0
    for idx in range(end - 1, -1, -1):
        if all_events[idx].get("type") == "start_kyoku":
            kyoku_start = idx + 1
            break
    if kyoku_start >= end:
        return out
    slice_start = max(kyoku_start, end - EVENT_HISTORY_LEN)
    dahai_count_so_far = sum(1 for item in all_events[kyoku_start:slice_start] if item.get("type") == "dahai")
    token_count = end - slice_start
    pad_len = EVENT_HISTORY_LEN - token_count
    for offset, idx in enumerate(range(slice_start, end)):
        event = all_events[idx]
        etype = str(event.get("type", ""))
        actor = int(event.get("actor", EVENT_NO_ACTOR))
        if actor < 0 or actor > 3:
            actor = EVENT_NO_ACTOR
        pai = event.get("pai")
        tile_id = EVENT_NO_TILE if pai is None else int(tile_to_34(pai))
        if tile_id < 0:
            tile_id = EVENT_NO_TILE
        slot = pad_len + offset
        out[slot, 0] = actor
        out[slot, 1] = _EVENT_TYPE_ID.get(etype, 15)
        out[slot, 2] = tile_id
        out[slot, 3] = min(EVENT_MAX_TURN_IDX, max(0, dahai_count_so_far // 4))
        out[slot, 4] = 0 if etype != "dahai" or bool(event.get("tsumogiri", False)) else 1
        if etype == "dahai":
            dahai_count_so_far += 1
    return out


def resolve_runtime_event_history(snap: dict) -> np.ndarray:
    value = snap.get("event_history")
    if value is None:
        return empty_event_history()
    arr = np.asarray(value, dtype=np.int16)
    if arr.shape != (EVENT_HISTORY_LEN, EVENT_HISTORY_FEATURE_DIM):
        raise ValueError(
            f"event_history shape {arr.shape} != ({EVENT_HISTORY_LEN}, {EVENT_HISTORY_FEATURE_DIM})"
        )
    return arr


def build_runtime_candidate_arrays(
    state: dict,
    actor: int,
    legal_actions: Iterable[dict] | None = None,
    *,
    max_candidates: int = XMODEL1_MAX_CANDIDATES,
    candidate_feature_dim: int = XMODEL1_CANDIDATE_FEATURE_DIM,
    candidate_flag_dim: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if legal_actions is None:
        legal_actions = [a.to_mjai() for a in enumerate_legal_actions(state, actor)]
    else:
        legal_actions = [dict(a) for a in legal_actions]
    discard_actions = iter_legal_discards(legal_actions)
    candidate_feat = np.zeros((max_candidates, candidate_feature_dim), dtype=np.float32)
    candidate_tile_id = np.full((max_candidates,), -1, dtype=np.int16)
    candidate_mask = np.zeros((max_candidates,), dtype=np.uint8)
    candidate_flags = np.zeros((max_candidates, candidate_flag_dim), dtype=np.uint8)
    for slot, action in enumerate(discard_actions[:max_candidates]):
        feat, flags, _quality, _rank, _hard_bad = build_candidate_features(state, actor, action)
        candidate_feat[slot] = feat.astype(np.float32, copy=False)
        candidate_tile_id[slot] = int(tile_to_34(action["pai"]))
        candidate_mask[slot] = 1
        candidate_flags[slot] = flags.astype(np.uint8, copy=False)
    return candidate_feat, candidate_tile_id, candidate_mask, candidate_flags


def build_runtime_special_candidate_arrays(
    state: dict,
    actor: int,
    legal_actions: Iterable[dict] | None = None,
    *,
    max_candidates: int = XMODEL1_MAX_SPECIAL_CANDIDATES,
    feature_dim: int = XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if legal_actions is None:
        legal_actions = [a.to_mjai() for a in enumerate_legal_actions(state, actor)]
    else:
        legal_actions = [dict(a) for a in legal_actions]
    feat, type_id, mask, _quality, _rank_bucket, _hard_bad, _chosen_idx = build_special_candidate_arrays(
        state,
        actor,
        legal_actions,
        chosen_action=None,
        max_candidates=max_candidates,
        feature_dim=feature_dim,
        include_terminal_actions=True,
    )
    return feat.astype(np.float32, copy=False), type_id.astype(np.int16, copy=False), mask.astype(np.uint8, copy=False)


__all__ = [
    "C_TILE",
    "N_SCALAR",
    "EVENT_HISTORY_FEATURE_DIM",
    "EVENT_HISTORY_LEN",
    "EVENT_NO_ACTOR",
    "EVENT_NO_TILE",
    "EVENT_TYPE_PAD",
    "compute_event_history",
    "empty_event_history",
    "resolve_runtime_event_history",
    "encode",
    "encode_with_timings",
    "build_runtime_candidate_arrays",
    "build_runtime_special_candidate_arrays",
]
