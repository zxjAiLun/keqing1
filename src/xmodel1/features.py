"""Xmodel1 runtime/state feature helpers."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Tuple

import numpy as np

from keqingv3.feature_tracker import SnapshotFeatureTracker
from keqingv3.features import C_TILE, N_SCALAR, encode as _encode_state, encode_with_timings
from keqingv3.progress_oracle import analyze_normal_progress_from_counts
from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.replay import _calc_shanten_waits
from mahjong_env.tiles import normalize_tile, tile_to_34


def encode(state: dict, actor: int, *, state_scalar_dim: int = 64):
    tile_feat, scalar = _encode_state(state, actor)
    if scalar.shape[0] < state_scalar_dim:
        padded = np.zeros((state_scalar_dim,), dtype=np.float32)
        padded[: scalar.shape[0]] = scalar
        scalar = padded
    elif scalar.shape[0] > state_scalar_dim:
        scalar = scalar[:state_scalar_dim]
    return tile_feat, scalar.astype(np.float32)


def _combined_hand(state: dict) -> list[str]:
    hand = list(state.get("hand", []))
    tsumo = state.get("tsumo_pai")
    if tsumo:
        hand.append(tsumo)
    return hand


def _counts34_from_hand(hand: list[str]) -> list[int]:
    counts = [0] * 34
    for tile in hand:
        idx = tile_to_34(normalize_tile(tile))
        if idx >= 0:
            counts[idx] += 1
    return counts


def _seat_and_round_yakuhai_ids(state: dict, actor: int) -> set[int]:
    bakaze = state.get("bakaze", "E")
    oya = int(state.get("oya", 0))
    wind_map = {"E": 27, "S": 28, "W": 29, "N": 30}
    ids = {31, 32, 33}
    if bakaze in wind_map:
        ids.add(wind_map[bakaze])
    ids.add(27 + ((actor - oya) % 4))
    return ids


def _simulate_discard_hand(state: dict, discard34: int) -> list[str]:
    hand = _combined_hand(state)
    removed = False
    out: list[str] = []
    for tile in hand:
        if not removed and tile_to_34(normalize_tile(tile)) == discard34:
            removed = True
            continue
        out.append(tile)
    return out if removed else hand


def _candidate_order_tile_ids(state: dict, legal_actions: Iterable[dict] | None = None) -> list[int]:
    if legal_actions is not None:
        tile_ids: list[int] = []
        for action in legal_actions:
            if action.get("type") != "dahai":
                continue
            pai = action.get("pai")
            if not pai:
                continue
            idx = tile_to_34(normalize_tile(pai))
            if idx >= 0 and idx not in tile_ids:
                tile_ids.append(idx)
        if tile_ids:
            return tile_ids
    counts = _counts34_from_hand(_combined_hand(state))
    return [idx for idx, count in enumerate(counts) if count > 0]


def build_runtime_candidate_arrays(
    state: dict,
    actor: int,
    legal_actions: Iterable[dict] | None = None,
    *,
    max_candidates: int,
    candidate_feature_dim: int,
    candidate_flag_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if legal_actions is None:
        legal_actions = [a.to_mjai() for a in enumerate_legal_actions(state, actor)]
    tile_ids = _candidate_order_tile_ids(state, legal_actions)

    candidate_feat = np.zeros((max_candidates, candidate_feature_dim), dtype=np.float32)
    candidate_tile_id = np.full((max_candidates,), -1, dtype=np.int16)
    candidate_mask = np.zeros((max_candidates,), dtype=np.uint8)
    candidate_flags = np.zeros((max_candidates, candidate_flag_dim), dtype=np.uint8)

    tracker = SnapshotFeatureTracker.from_state(state, actor)
    yakuhai_ids = _seat_and_round_yakuhai_ids(state, actor)
    visible_before = list(tracker.visible_counts34)
    current_shanten = int(state.get("shanten", 8))
    current_waits_count = int(state.get("waits_count", 0))

    for slot, discard34 in enumerate(tile_ids[:max_candidates]):
        after_hand = _simulate_discard_hand(state, discard34)
        after_counts34 = _counts34_from_hand(after_hand)
        visible_after = list(visible_before)
        visible_after[discard34] = min(4, visible_after[discard34] + 1)
        progress = analyze_normal_progress_from_counts(tuple(after_counts34), tuple(visible_after))
        after_shanten, after_waits_cnt, after_waits_tiles, _ = _calc_shanten_waits(
            after_hand,
            (state.get("melds") or [[], [], [], []])[actor],
        )
        wait_live_count = sum(
            max(0, 4 - visible_after[t34])
            for t34, flag in enumerate(after_waits_tiles)
            if flag
        )
        pair_count = sum(1 for c in after_counts34 if c >= 2)
        taatsu_count = 0
        for base in (0, 9, 18):
            suit = after_counts34[base : base + 9]
            for i in range(8):
                if suit[i] > 0 and suit[i + 1] > 0:
                    taatsu_count += 1
            for i in range(7):
                if suit[i] > 0 and suit[i + 2] > 0:
                    taatsu_count += 1

        yakuhai_pair_preserved = 1.0 if any(after_counts34[idx] >= 2 for idx in yakuhai_ids) else 0.0
        dual_pon_value = 1.0 if any(after_counts34[idx] >= 2 and idx in yakuhai_ids for idx in yakuhai_ids) else 0.0
        tile_counter = Counter(idx for idx, c in enumerate(after_counts34) for _ in range(c))
        tanyao_path = 1.0 if tile_counter and all(idx not in {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33} for idx in tile_counter) else 0.0
        suit_presence = [sum(after_counts34[base : base + 9]) > 0 for base in (0, 9, 18)]
        flush_path = 1.0 if sum(1 for v in suit_presence if v) <= 1 else 0.0
        confirmed_han_floor = float(yakuhai_pair_preserved + tanyao_path)

        break_tenpai = 1 if current_shanten == 0 and after_shanten != 0 else 0
        break_best_wait = 1 if current_shanten == 0 and after_waits_cnt < current_waits_count else 0
        break_meld_structure = 1 if current_shanten <= 1 and after_shanten > current_shanten else 0
        drop_open_yakuhai_pair = 1 if tracker.meld_count > 0 and discard34 in yakuhai_ids and yakuhai_pair_preserved == 0.0 else 0
        drop_dual_pon_value = 1 if tracker.meld_count > 0 and discard34 in yakuhai_ids and dual_pon_value == 0.0 else 0

        is_honor = 1 if discard34 >= 27 else 0
        is_terminal = 1 if discard34 in {0, 8, 9, 17, 18, 26} else 0
        is_yakuhai = 1 if discard34 in yakuhai_ids else 0

        feat = np.array(
            [
                after_shanten / 8.0,
                1.0 if after_shanten == 0 else 0.0,
                after_waits_cnt / 34.0,
                wait_live_count / 136.0,
                progress.ukeire_type_count / 34.0,
                progress.ukeire_live_count / 136.0,
                progress.good_shape_ukeire_live_count / 136.0,
                wait_live_count / 136.0,
                pair_count / 7.0,
                taatsu_count / 6.0,
                max(0.0, min(1.0, (pair_count + taatsu_count) / 8.0)),
                min(confirmed_han_floor / 8.0, 1.0),
                1.0 if tracker.meld_count == 0 or confirmed_han_floor > 0 else 0.0,
                yakuhai_pair_preserved,
                dual_pon_value,
                tanyao_path,
                flush_path,
                1.0 if any(flag for idx, flag in enumerate(state.get("reached", [False] * 4)) if idx != actor) else 0.0,
                0.0,
                0.0,
                1.0 if visible_after[discard34] >= 3 else 0.0,
            ],
            dtype=np.float32,
        )
        flags = np.array(
            [
                break_tenpai,
                break_best_wait,
                break_meld_structure,
                drop_open_yakuhai_pair,
                drop_dual_pon_value,
                is_honor,
                is_terminal,
                0,
                0,
                is_yakuhai,
            ],
            dtype=np.uint8,
        )

        candidate_feat[slot] = feat
        candidate_tile_id[slot] = discard34
        candidate_mask[slot] = 1
        candidate_flags[slot] = flags

    return candidate_feat, candidate_tile_id, candidate_mask, candidate_flags


__all__ = [
    "C_TILE",
    "N_SCALAR",
    "encode",
    "encode_with_timings",
    "build_runtime_candidate_arrays",
]
