"""Candidate feature and quality helpers for Xmodel1."""

from __future__ import annotations

from collections import Counter
from typing import Iterable

import numpy as np

from keqingv1.action_space import TILE_NAME_TO_IDX
from keqingv3.feature_tracker import SnapshotFeatureTracker
from keqingv3.progress_oracle import analyze_normal_progress_from_counts
from mahjong_env.replay import _calc_shanten_waits
from mahjong_env.tiles import normalize_tile, tile_is_aka, tile_to_34


def iter_legal_discards(legal_actions: Iterable[dict]) -> list[dict]:
    return sorted(
        [a for a in legal_actions if a.get("type") == "dahai"],
        key=lambda a: (TILE_NAME_TO_IDX.get(normalize_tile(a.get("pai", "")), 99), a.get("pai", "")),
    )


def simulate_discard_snapshot(snap: dict, actor: int, pai: str) -> dict:
    hand = list(snap.get("hand", []))
    removed = False
    norm_pai = normalize_tile(pai)
    new_hand = []
    for tile in hand:
        if not removed and normalize_tile(tile) == norm_pai:
            removed = True
        else:
            new_hand.append(tile)
    out = dict(snap)
    out["hand"] = new_hand if removed else hand
    discards = [list(d) for d in snap.get("discards", [[], [], [], []])]
    if actor < len(discards):
        discards[actor] = discards[actor] + [pai]
    out["discards"] = discards
    out["tsumo_pai"] = None
    return out


def _visible_counts34_from_tracker(snap: dict, actor: int) -> tuple[int, ...]:
    tracker = SnapshotFeatureTracker.from_state(snap, actor)
    return tuple(int(v) for v in tracker.visible_counts34)


def _hand_counts34(hand: list[str]) -> tuple[int, ...]:
    counts = [0] * 34
    for tile in hand:
        idx = tile_to_34(tile)
        if 0 <= idx < 34:
            counts[idx] += 1
    return tuple(counts)


def _pair_taatsu_metrics(hand: list[str]) -> tuple[int, int]:
    counts = Counter(tile_to_34(t) for t in hand if tile_to_34(t) >= 0)
    pair_count = sum(1 for c in counts.values() if c >= 2)
    taatsu_count = 0
    for base in (0, 9, 18):
        suit = [counts.get(base + i, 0) for i in range(9)]
        for i in range(8):
            if suit[i] > 0 and suit[i + 1] > 0:
                taatsu_count += 1
        for i in range(7):
            if suit[i] > 0 and suit[i + 2] > 0:
                taatsu_count += 1
    return pair_count, taatsu_count


def build_candidate_features(before_snap: dict, actor: int, discard_action: dict):
    hand_before = list(before_snap.get("hand", []))
    melds_before = (before_snap.get("melds") or [[], [], [], []])[actor]
    before_shanten, before_waits_count, before_waits_tiles, _ = _calc_shanten_waits(hand_before, melds_before)
    before_visible = _visible_counts34_from_tracker(before_snap, actor)
    before_waits_live = sum(
        max(0, 4 - before_visible[t34]) for t34, flag in enumerate(before_waits_tiles) if flag
    )

    after_snap = simulate_discard_snapshot(before_snap, actor, discard_action["pai"])
    hand_after = list(after_snap.get("hand", []))
    melds_after = (after_snap.get("melds") or [[], [], [], []])[actor]
    after_shanten, after_waits_count, after_waits_tiles, _ = _calc_shanten_waits(hand_after, melds_after)
    after_visible = _visible_counts34_from_tracker(after_snap, actor)
    after_counts34 = _hand_counts34(hand_after)
    after_progress = analyze_normal_progress_from_counts(after_counts34, after_visible)
    after_waits_live = sum(
        max(0, 4 - after_visible[t34]) for t34, flag in enumerate(after_waits_tiles) if flag
    )
    pair_count, taatsu_count = _pair_taatsu_metrics(hand_after)

    pai = discard_action["pai"]
    pai34 = tile_to_34(pai)
    norm_pai = normalize_tile(pai)
    wind_order = ["E", "S", "W", "N"]
    actor_wind = wind_order[(actor - before_snap.get("oya", 0)) % 4]
    yakuhai_tiles = {"P", "F", "C", before_snap.get("bakaze", "E"), actor_wind}
    before_norm = [normalize_tile(t) for t in hand_before]
    after_norm = [normalize_tile(t) for t in hand_after]

    break_tenpai = int(before_shanten == 0 and after_shanten > 0)
    break_best_wait = int(before_shanten == 0 and after_waits_live < before_waits_live)
    break_meld_structure = int(before_shanten <= 1 and after_shanten > before_shanten)
    drop_open_yakuhai_pair = int(
        len(melds_before) > 0 and norm_pai in yakuhai_tiles and before_norm.count(norm_pai) >= 2 and after_norm.count(norm_pai) < 2
    )
    drop_dual_pon = int(before_norm.count(norm_pai) >= 2 and after_norm.count(norm_pai) < 2)
    is_honor = int(pai34 >= 27)
    is_terminal = int(pai34 in {0, 8, 9, 17, 18, 26})
    is_aka = int(tile_is_aka(pai))
    is_dora = 0
    is_yakuhai = int(norm_pai in yakuhai_tiles)

    feat = np.array(
        [
            after_shanten / 8.0,
            1.0 if after_shanten == 0 else 0.0,
            after_waits_count / 34.0,
            min(after_waits_live / 20.0, 1.0),
            min(after_progress.ukeire_type_count / 34.0, 1.0),
            min(after_progress.ukeire_live_count / 34.0, 1.0),
            min(after_progress.good_shape_ukeire_live_count / 20.0, 1.0),
            min(after_waits_live / 20.0, 1.0),
            pair_count / 7.0,
            taatsu_count / 6.0,
            1.0 - min(abs(after_shanten - before_shanten) / 2.0, 1.0),
            0.0,
            1.0 - float(drop_open_yakuhai_pair),
            1.0 - float(drop_open_yakuhai_pair),
            1.0 - float(drop_dual_pon),
            0.0,
            0.0,
            float(any(before_snap.get("reached", [False] * 4)[pid] for pid in range(4) if pid != actor)),
            0.0,
            0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    flags = np.array(
        [
            break_tenpai,
            break_best_wait,
            break_meld_structure,
            drop_open_yakuhai_pair,
            drop_dual_pon,
            is_honor,
            is_terminal,
            is_aka,
            is_dora,
            is_yakuhai,
        ],
        dtype=np.uint8,
    )
    quality = (
        1.5 * float(after_shanten == 0)
        - 0.8 * float(after_shanten)
        + 0.04 * float(after_progress.ukeire_live_count)
        + 0.06 * float(after_waits_live)
        - 2.0 * float(break_tenpai)
        - 1.0 * float(break_meld_structure)
        - 1.2 * float(drop_open_yakuhai_pair)
        - 0.7 * float(drop_dual_pon)
    )
    hard_bad = int(break_tenpai or drop_open_yakuhai_pair or break_meld_structure)
    rank_bucket = 0 if hard_bad else (3 if quality >= 1.0 else 2 if quality >= 0.0 else 1)
    return feat, flags, float(quality), int(rank_bucket), int(hard_bad)


__all__ = ["iter_legal_discards", "simulate_discard_snapshot", "build_candidate_features"]
