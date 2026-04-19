"""Candidate feature and quality helpers for Xmodel1 v2."""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from mahjong_env.action_space import TILE_NAME_TO_IDX
from mahjong_env.feature_tracker import SnapshotFeatureTracker
from mahjong_env.progress_oracle import analyze_normal_progress_from_counts
from mahjong_env.replay import _calc_shanten_waits
from mahjong_env.tiles import normalize_tile, tile_is_aka, tile_to_34
from xmodel1.schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CHI_SPECIAL_TYPES,
    XMODEL1_KAN_SPECIAL_TYPES,
    XMODEL1_MAX_SPECIAL_CANDIDATES,
    XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
    XMODEL1_SPECIAL_TYPE_ANKAN,
    XMODEL1_SPECIAL_TYPE_CHI_HIGH,
    XMODEL1_SPECIAL_TYPE_CHI_LOW,
    XMODEL1_SPECIAL_TYPE_CHI_MID,
    XMODEL1_SPECIAL_TYPE_DAIMINKAN,
    XMODEL1_SPECIAL_TYPE_DAMA,
    XMODEL1_SPECIAL_TYPE_HORA,
    XMODEL1_SPECIAL_TYPE_KAKAN,
    XMODEL1_SPECIAL_TYPE_NONE,
    XMODEL1_SPECIAL_TYPE_PON,
    XMODEL1_SPECIAL_TYPE_REACH,
    XMODEL1_SPECIAL_TYPE_RYUKYOKU,
)

_TERMINAL_HONOR_IDS = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}
_TERMINAL_IDS = {0, 8, 9, 17, 18, 26}


def _dora_target_ids(state: dict) -> set[int]:
    markers = []
    if state.get("dora_markers"):
        markers.extend(state.get("dora_markers", []))
    elif state.get("dora_marker"):
        markers.append(state["dora_marker"])
    targets: set[int] = set()
    for marker in markers:
        norm = normalize_tile(marker)
        if len(norm) == 2 and norm[0].isdigit():
            rank = int(norm[0])
            suit = norm[1]
            next_rank = 1 if rank == 9 else rank + 1
            target = f"{next_rank}{suit}"
        else:
            honor_cycle = {"E": "S", "S": "W", "W": "N", "N": "E", "P": "F", "F": "C", "C": "P"}
            target = honor_cycle.get(norm)
        if target is None:
            continue
        idx = tile_to_34(target)
        if idx >= 0:
            targets.add(idx)
    return targets


def _yakuhai_ids(state: dict, actor: int) -> set[int]:
    bakaze = state.get("bakaze", "E")
    oya = int(state.get("oya", 0))
    actor_wind = ["E", "S", "W", "N"][(actor - oya) % 4]
    tiles = {"P", "F", "C", bakaze, actor_wind}
    return {tile_to_34(tile) for tile in tiles if tile_to_34(tile) >= 0}


def _hand_counts34(hand: Sequence[str]) -> tuple[int, ...]:
    counts = [0] * 34
    for tile in hand:
        idx = tile_to_34(normalize_tile(tile))
        if idx >= 0:
            counts[idx] += 1
    return tuple(counts)


def _pair_taatsu_metrics(hand_counts34: Sequence[int]) -> tuple[int, int]:
    pair_count = sum(1 for c in hand_counts34 if c >= 2)
    taatsu_count = 0
    for base in (0, 9, 18):
        suit = hand_counts34[base : base + 9]
        for i in range(8):
            if suit[i] > 0 and suit[i + 1] > 0:
                taatsu_count += 1
        for i in range(7):
            if suit[i] > 0 and suit[i + 2] > 0:
                taatsu_count += 1
    return pair_count, taatsu_count


def _tanyao_path(hand_counts34: Sequence[int]) -> float:
    return 1.0 if all(hand_counts34[idx] == 0 for idx in _TERMINAL_HONOR_IDS) else 0.0


def _flush_path(hand_counts34: Sequence[int]) -> float:
    suit_presence = [sum(hand_counts34[base : base + 9]) > 0 for base in (0, 9, 18)]
    return 1.0 if sum(1 for flag in suit_presence if flag) <= 1 else 0.0


def _pinfu_like_path(hand_counts34: Sequence[int], *, yakuhai_ids: set[int], is_open_hand: bool) -> float:
    if is_open_hand or any(hand_counts34[idx] >= 3 for idx in range(34)):
        return 0.0
    pair_tiles = [idx for idx, count in enumerate(hand_counts34) if count >= 2]
    return 1.0 if len(pair_tiles) == 1 and pair_tiles[0] not in yakuhai_ids else 0.0


def _iipeikou_like_path(hand_counts34: Sequence[int], *, is_open_hand: bool) -> float:
    if is_open_hand:
        return 0.0
    for base in (0, 9, 18):
        suit = hand_counts34[base : base + 9]
        for idx in range(7):
            if min(suit[idx], suit[idx + 1], suit[idx + 2]) >= 2:
                return 1.0
    return 0.0


def _max_hand_value_norm(
    *,
    confirmed_han_floor: float,
    dora_count: float,
    aka_count: float,
    waits_count: int,
    waits_live: int,
    is_tenpai: bool,
) -> float:
    value_proxy = confirmed_han_floor + 0.5 * dora_count + 0.35 * aka_count
    if is_tenpai:
        value_proxy += 0.5 + min(waits_count, 5) * 0.15 + min(waits_live, 12) * 0.03
    return float(min(value_proxy / 8.0, 1.0))


def _after_state_path_metrics(
    hand_after: Sequence[str],
    hand_counts34: Sequence[int],
    *,
    yakuhai_ids: set[int],
    dora_target_ids: set[int],
    is_open_hand: bool,
    waits_count: int,
    waits_live: int,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    yakuhai_pair_preserved = 1.0 if any(hand_counts34[idx] >= 2 for idx in yakuhai_ids) else 0.0
    dual_yakuhai_pair_value = 1.0 if sum(1 for idx in yakuhai_ids if hand_counts34[idx] >= 2) >= 2 else 0.0
    tanyao_path = _tanyao_path(hand_counts34)
    flush_path = _flush_path(hand_counts34)
    after_aka_count = float(sum(tile_is_aka(tile) for tile in hand_after))
    after_dora_count = float(sum(hand_counts34[idx] for idx in dora_target_ids) + after_aka_count)
    yakuhai_triplet_count = float(sum(1 for idx in yakuhai_ids if hand_counts34[idx] >= 3))
    confirmed_han_floor = min(8.0, yakuhai_triplet_count + tanyao_path + after_dora_count)
    max_hand_value_norm = _max_hand_value_norm(
        confirmed_han_floor=confirmed_han_floor,
        dora_count=after_dora_count,
        aka_count=after_aka_count,
        waits_count=waits_count,
        waits_live=waits_live,
        is_tenpai=waits_count > 0,
    )
    hand_value_survives = 1.0 if (not is_open_hand) or confirmed_han_floor > 0.0 else 0.0
    return (
        yakuhai_pair_preserved,
        dual_yakuhai_pair_value,
        tanyao_path,
        flush_path,
        after_dora_count,
        after_aka_count,
        yakuhai_triplet_count,
        confirmed_han_floor,
        max_hand_value_norm,
        hand_value_survives,
    )


def _visible_counts34_from_tracker(snap: dict, actor: int) -> tuple[int, ...]:
    tracker = SnapshotFeatureTracker.from_state(snap, actor)
    return tuple(int(v) for v in tracker.visible_counts34)


def _combined_hand(state: dict) -> list[str]:
    hand = list(state.get("hand", []))
    tsumo = state.get("tsumo_pai")
    if tsumo:
        hand.append(tsumo)
    return hand


def _per_opponent_dealin_risks(
    state: dict,
    actor: int,
    tile34: int,
    *,
    visible_counts34: Sequence[int],
) -> tuple[float, float, float]:
    live_factor = max(0.0, min(1.0, (4.0 - float(visible_counts34[tile34])) / 4.0))
    reached = list(state.get("reached", [False, False, False, False]))
    melds = state.get("melds") or [[], [], [], []]
    risks = []
    for rel in range(1, 4):
        opp = (actor + rel) % 4
        open_pressure = 1.0 if any(m.get("type") != "ankan" for m in melds[opp]) else 0.0
        base = 1.0 if reached[opp] else 0.2 + 0.15 * open_pressure
        if tile34 >= 27:
            base *= 0.9
        risks.append(float(max(0.0, min(1.0, live_factor * base))))
    return tuple(risks)  # type: ignore[return-value]


def iter_legal_discards(legal_actions: Iterable[dict]) -> list[dict]:
    return sorted(
        [a for a in legal_actions if a.get("type") == "dahai"],
        key=lambda a: (TILE_NAME_TO_IDX.get(normalize_tile(a.get("pai", "")), 99), a.get("pai", "")),
    )


def simulate_discard_snapshot(snap: dict, actor: int, pai: str) -> dict:
    hand = list(_combined_hand(snap))
    removed = False
    norm_pai = normalize_tile(pai)
    new_hand = []
    for tile in hand:
        if not removed and normalize_tile(tile) == norm_pai:
            removed = True
            continue
        new_hand.append(tile)
    out = dict(snap)
    out["hand"] = new_hand if removed else hand
    discards = [list(d) for d in snap.get("discards", [[], [], [], []])]
    if actor < len(discards):
        discards[actor] = discards[actor] + [pai]
    out["discards"] = discards
    out["tsumo_pai"] = None
    return out


def build_candidate_features(before_snap: dict, actor: int, discard_action: dict):
    hand_before = _combined_hand(before_snap)
    melds_before = (before_snap.get("melds") or [[], [], [], []])[actor]
    before_shanten, before_waits_count, before_waits_tiles, _ = _calc_shanten_waits(hand_before, melds_before)
    before_visible = _visible_counts34_from_tracker(before_snap, actor)
    before_waits_live = sum(max(0, 4 - before_visible[t34]) for t34, flag in enumerate(before_waits_tiles) if flag)
    before_counts34 = _hand_counts34(hand_before)
    yakuhai_ids = _yakuhai_ids(before_snap, actor)
    dora_target_ids = _dora_target_ids(before_snap)
    before_tanyao = _tanyao_path(before_counts34)
    before_pinfu = _pinfu_like_path(before_counts34, yakuhai_ids=yakuhai_ids, is_open_hand=len(melds_before) > 0)
    before_iipeikou = _iipeikou_like_path(before_counts34, is_open_hand=len(melds_before) > 0)

    after_snap = simulate_discard_snapshot(before_snap, actor, discard_action["pai"])
    hand_after = list(after_snap.get("hand", []))
    melds_after = (after_snap.get("melds") or [[], [], [], []])[actor]
    after_shanten, after_waits_count, after_waits_tiles, _ = _calc_shanten_waits(hand_after, melds_after)
    after_visible = _visible_counts34_from_tracker(after_snap, actor)
    after_counts34 = _hand_counts34(hand_after)
    after_progress = analyze_normal_progress_from_counts(after_counts34, after_visible)
    after_waits_live = sum(max(0, 4 - after_visible[t34]) for t34, flag in enumerate(after_waits_tiles) if flag)
    pair_count, taatsu_count = _pair_taatsu_metrics(after_counts34)
    (
        yakuhai_pair_preserved,
        dual_yakuhai_pair_value,
        tanyao_path,
        flush_path,
        after_dora_count,
        after_aka_count,
        yakuhai_triplet_count,
        confirmed_han_floor,
        after_max_hand_value_norm,
        hand_value_survives,
    ) = _after_state_path_metrics(
        hand_after,
        after_counts34,
        yakuhai_ids=yakuhai_ids,
        dora_target_ids=dora_target_ids,
        is_open_hand=len(melds_before) > 0,
        waits_count=after_waits_count,
        waits_live=after_waits_live,
    )

    pai34 = tile_to_34(normalize_tile(discard_action["pai"]))
    after_risks = _per_opponent_dealin_risks(before_snap, actor, pai34, visible_counts34=before_visible)
    wait_density = 0.0 if after_waits_count <= 0 else min(after_waits_live / max(1.0, 4.0 * after_waits_count), 1.0)
    structure_density = min((pair_count + taatsu_count) / 8.0, 1.0)
    break_tenpai = int(before_shanten == 0 and after_shanten != 0)
    break_best_wait = int(before_shanten == 0 and after_waits_live < before_waits_live)
    break_meld_structure = int(before_shanten <= 1 and after_shanten > before_shanten)
    yaku_break_tanyao = int(before_tanyao > 0.5 and tanyao_path < 0.5)
    yaku_break_pinfu = int(before_pinfu > 0.5 and _pinfu_like_path(after_counts34, yakuhai_ids=yakuhai_ids, is_open_hand=len(melds_before) > 0) < 0.5)
    yaku_break_iipeikou = int(before_iipeikou > 0.5 and _iipeikou_like_path(after_counts34, is_open_hand=len(melds_before) > 0) < 0.5)
    is_honor = int(pai34 >= 27)
    is_terminal = int(pai34 in _TERMINAL_IDS)
    is_dora = int(pai34 in dora_target_ids or tile_is_aka(discard_action["pai"]))
    is_yakuhai = int(pai34 in yakuhai_ids)
    discard_dead = float(after_visible[pai34] >= 3)

    feat = np.array(
        [
            after_shanten / 8.0,
            1.0 if after_shanten == 0 else 0.0,
            after_waits_count / 34.0,
            min(after_progress.ukeire_live_count / 136.0, 1.0),
            after_max_hand_value_norm,
            np.clip((after_shanten - before_shanten) / 4.0, -1.0, 1.0),
            min(after_dora_count / 4.0, 1.0),
            min(after_aka_count / 2.0, 1.0),
            float(yaku_break_tanyao),
            float(yaku_break_pinfu),
            float(yaku_break_iipeikou),
            after_risks[0],
            after_risks[1],
            after_risks[2],
            wait_density,
            pair_count / 7.0,
            taatsu_count / 6.0,
            structure_density,
            hand_value_survives,
            yakuhai_pair_preserved,
            dual_yakuhai_pair_value,
            tanyao_path,
            flush_path,
            float(any(before_snap.get("reached", [False] * 4)[pid] for pid in range(4) if pid != actor)),
            discard_dead,
            min(getattr(after_progress, "good_shape_ukeire_live_count", 0) / 136.0, 1.0),
            min(getattr(after_progress, "improvement_live_count", 0) / 136.0, 1.0),
            min(confirmed_han_floor / 8.0, 1.0),
            min(yakuhai_triplet_count / 3.0, 1.0),
            float(break_tenpai),
            float(break_best_wait),
            float(break_meld_structure),
            float(is_dora),
            float(is_yakuhai),
            float(before_waits_live > after_waits_live),
        ],
        dtype=np.float32,
    )
    assert feat.shape[0] == XMODEL1_CANDIDATE_FEATURE_DIM

    flags = np.array(
        [
            break_tenpai,
            break_best_wait,
            break_meld_structure,
            yaku_break_tanyao,
            yaku_break_pinfu,
            yaku_break_iipeikou,
            is_honor,
            is_terminal,
            is_dora,
            is_yakuhai,
        ],
        dtype=np.uint8,
    )
    quality = (
        1.5 * float(after_shanten == 0)
        - 0.8 * float(after_shanten)
        + 0.5 * after_max_hand_value_norm
        + 0.2 * feat[25]
        + 0.15 * feat[26]
        + 0.15 * feat[27]
        - 1.2 * float(break_tenpai)
        - 0.5 * float(yaku_break_tanyao or yaku_break_pinfu or yaku_break_iipeikou)
        - 0.7 * float(np.mean(after_risks))
        - 0.15 * discard_dead
    )
    hard_bad = int(break_tenpai or break_meld_structure or np.mean(after_risks) > 0.75)
    rank_bucket = 0 if hard_bad else (3 if quality >= 1.0 else 2 if quality >= 0.0 else 1)
    return feat, flags, float(quality), int(rank_bucket), int(hard_bad)


def _current_hand_summary(before_snap: dict, actor: int) -> dict[str, object]:
    hand = _combined_hand(before_snap)
    melds = (before_snap.get("melds") or [[], [], [], []])[actor]
    shanten, waits_count, waits_tiles, _ = _calc_shanten_waits(hand, melds)
    visible = _visible_counts34_from_tracker(before_snap, actor)
    hand_counts34 = _hand_counts34(hand)
    yakuhai_ids = _yakuhai_ids(before_snap, actor)
    dora_target_ids = _dora_target_ids(before_snap)
    waits_live = sum(max(0, 4 - visible[t34]) for t34, flag in enumerate(waits_tiles) if flag)
    scores = list(before_snap.get("scores", [25000, 25000, 25000, 25000]))
    actor_score = float(scores[actor]) if actor < len(scores) else 25000.0
    mean_score = float(sum(scores) / max(1, len(scores)))
    threat_by_opponent = _per_opponent_dealin_risks(before_snap, actor, 27, visible_counts34=visible)
    (
        _yakuhai_pair_preserved,
        _dual_yakuhai_pair_value,
        _tanyao_path,
        _flush_path,
        current_dora_count,
        _current_aka_count,
        _current_yakuhai_triplet_count,
        current_han_floor,
        current_max_value_norm,
        current_hand_value_survives,
    ) = _after_state_path_metrics(
        hand,
        hand_counts34,
        yakuhai_ids=yakuhai_ids,
        dora_target_ids=dora_target_ids,
        is_open_hand=len(melds) > 0,
        waits_count=waits_count,
        waits_live=waits_live,
    )
    best_discard_quality = 0.0
    for action in iter_legal_discards(before_snap.get("legal_actions", [])):
        _, _, quality, _, _ = build_candidate_features(before_snap, actor, action)
        best_discard_quality = max(best_discard_quality, float(quality))
    return {
        "shanten": float(shanten),
        "tenpai": 1.0 if shanten == 0 else 0.0,
        "waits_count": float(waits_count),
        "waits_live_norm": min(waits_live / 136.0, 1.0),
        "round_progress": min(1.0, sum(len(items) for items in (before_snap.get("discards") or [[], [], [], []])) / 60.0),
        "score_gap": (actor_score - mean_score) / 30000.0,
        "threat_proxy_any_reached": 1.0 if any(before_snap.get("reached", [False] * 4)[pid] for pid in range(4) if pid != actor) else 0.0,
        "threat_by_opponent": threat_by_opponent,
        "is_open": 1.0 if len(melds) > 0 else 0.0,
        "best_discard_quality_norm": min(best_discard_quality / 3.0, 1.0),
        "current_han_floor_norm": min(current_han_floor / 8.0, 1.0),
        "current_dora_count_norm": min(current_dora_count / 4.0, 1.0),
        "current_max_value_norm": current_max_value_norm,
        "current_hand_value_survives": current_hand_value_survives,
        "yakuhai_ids": yakuhai_ids,
        "dora_target_ids": dora_target_ids,
    }


def _chi_special_type(action: dict) -> int:
    pai = normalize_tile(action.get("pai", ""))
    pai_rank = int(pai[0]) if len(pai) == 2 and pai[0].isdigit() else 0
    consumed = sorted(
        int(tile[0]) for tile in [normalize_tile(value) for value in action.get("consumed", [])] if len(tile) == 2 and tile[0].isdigit()
    )
    if len(consumed) < 2 or pai_rank < consumed[0]:
        return XMODEL1_SPECIAL_TYPE_CHI_LOW
    if pai_rank < consumed[1]:
        return XMODEL1_SPECIAL_TYPE_CHI_MID
    return XMODEL1_SPECIAL_TYPE_CHI_HIGH


def _special_action_bonus_metrics(action: dict, *, yakuhai_ids: set[int], dora_target_ids: set[int]) -> tuple[float, float]:
    action_tiles = list(action.get("consumed") or [])
    pai = action.get("pai")
    if pai:
        action_tiles.append(pai)
    tile_ids = [tile_to_34(normalize_tile(tile)) for tile in action_tiles]
    tile_ids = [idx for idx in tile_ids if idx >= 0]
    action_dora_bonus = min((sum(idx in dora_target_ids for idx in tile_ids) + sum(tile_is_aka(tile) for tile in action_tiles)) / 4.0, 1.0)
    action_yakuhai_bonus = min(sum(idx in yakuhai_ids for idx in tile_ids) / 4.0, 1.0)
    return action_dora_bonus, action_yakuhai_bonus


def _special_exposure_factor(special_type: int, *, ankan_preserves_tenpai: float) -> float:
    if special_type in XMODEL1_CHI_SPECIAL_TYPES:
        return 0.65
    if special_type == XMODEL1_SPECIAL_TYPE_PON:
        return 0.55
    if special_type == XMODEL1_SPECIAL_TYPE_DAIMINKAN:
        return 0.8
    if special_type == XMODEL1_SPECIAL_TYPE_KAKAN:
        return 0.7
    if special_type == XMODEL1_SPECIAL_TYPE_ANKAN:
        return 0.2 if ankan_preserves_tenpai > 0.5 else 0.4
    if special_type in {XMODEL1_SPECIAL_TYPE_REACH, XMODEL1_SPECIAL_TYPE_DAMA}:
        return 0.35
    return 0.0


def _special_action_after_value(summary: dict[str, object], special_type: int, action: dict) -> float:
    current_max_value_norm = float(summary["current_max_value_norm"])
    best_discard_quality_norm = float(summary["best_discard_quality_norm"])
    action_dora_bonus, action_yakuhai_bonus = _special_action_bonus_metrics(
        action,
        yakuhai_ids=summary["yakuhai_ids"],  # type: ignore[arg-type]
        dora_target_ids=summary["dora_target_ids"],  # type: ignore[arg-type]
    )
    if special_type in {XMODEL1_SPECIAL_TYPE_REACH, XMODEL1_SPECIAL_TYPE_DAMA}:
        reach_bonus = 0.15 if special_type == XMODEL1_SPECIAL_TYPE_REACH else 0.0
        return float(min(1.0, current_max_value_norm + reach_bonus + 0.2 * action_dora_bonus))
    if special_type in XMODEL1_CHI_SPECIAL_TYPES or special_type in XMODEL1_KAN_SPECIAL_TYPES or special_type == XMODEL1_SPECIAL_TYPE_PON:
        return float(min(1.0, 0.6 * best_discard_quality_norm + 0.25 * action_dora_bonus + 0.25 * action_yakuhai_bonus))
    if special_type == XMODEL1_SPECIAL_TYPE_HORA:
        return float(max(current_max_value_norm, 0.8))
    return 0.0


def _special_call_family_score(action: dict) -> float:
    bonus = 0.0
    if action.get("type") == "pon":
        bonus = 0.2
    elif action.get("type") in {"daiminkan", "ankan", "kakan"}:
        bonus = 0.1
    return float(len(action.get("consumed", []))) + bonus


def build_special_candidate_arrays(
    before_snap: dict,
    actor: int,
    legal_actions: Iterable[dict],
    chosen_action: dict | None = None,
    *,
    max_candidates: int = XMODEL1_MAX_SPECIAL_CANDIDATES,
    feature_dim: int = XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
    include_terminal_actions: bool = False,
):
    legal_actions = [dict(action) for action in legal_actions]
    before_snap = dict(before_snap)
    before_snap["legal_actions"] = legal_actions
    summary = _current_hand_summary(before_snap, actor)

    feat = np.zeros((max_candidates, feature_dim), dtype=np.float32)
    type_id = np.full((max_candidates,), -1, dtype=np.int16)
    mask = np.zeros((max_candidates,), dtype=np.uint8)
    quality = np.zeros((max_candidates,), dtype=np.float32)
    rank_bucket = np.zeros((max_candidates,), dtype=np.int8)
    hard_bad = np.zeros((max_candidates,), dtype=np.uint8)
    chosen_idx = -1

    grouped: dict[int, dict | None] = {
        XMODEL1_SPECIAL_TYPE_REACH: None,
        XMODEL1_SPECIAL_TYPE_DAMA: None,
        XMODEL1_SPECIAL_TYPE_HORA: None,
        XMODEL1_SPECIAL_TYPE_CHI_LOW: None,
        XMODEL1_SPECIAL_TYPE_CHI_MID: None,
        XMODEL1_SPECIAL_TYPE_CHI_HIGH: None,
        XMODEL1_SPECIAL_TYPE_PON: None,
        XMODEL1_SPECIAL_TYPE_DAIMINKAN: None,
        XMODEL1_SPECIAL_TYPE_ANKAN: None,
        XMODEL1_SPECIAL_TYPE_KAKAN: None,
        XMODEL1_SPECIAL_TYPE_RYUKYOKU: None,
        XMODEL1_SPECIAL_TYPE_NONE: None,
    }

    if any(action.get("type") == "reach" for action in legal_actions):
        grouped[XMODEL1_SPECIAL_TYPE_REACH] = {"type": "reach"}
        grouped[XMODEL1_SPECIAL_TYPE_DAMA] = {"type": "dama"}

    for action in legal_actions:
        action_type = action.get("type")
        if action_type == "chi":
            special_type = _chi_special_type(action)
        elif action_type == "pon":
            special_type = XMODEL1_SPECIAL_TYPE_PON
        elif action_type == "daiminkan":
            special_type = XMODEL1_SPECIAL_TYPE_DAIMINKAN
        elif action_type == "ankan":
            special_type = XMODEL1_SPECIAL_TYPE_ANKAN
        elif action_type == "kakan":
            special_type = XMODEL1_SPECIAL_TYPE_KAKAN
        elif include_terminal_actions and action_type == "hora":
            special_type = XMODEL1_SPECIAL_TYPE_HORA
        elif include_terminal_actions and action_type == "ryukyoku":
            special_type = XMODEL1_SPECIAL_TYPE_RYUKYOKU
        elif action_type == "none":
            special_type = XMODEL1_SPECIAL_TYPE_NONE
        else:
            continue
        previous = grouped.get(special_type)
        if previous is None or _special_call_family_score(action) > _special_call_family_score(previous):
            grouped[special_type] = action

    if any(grouped[special_type] is not None for special_type in (*XMODEL1_CHI_SPECIAL_TYPES, XMODEL1_SPECIAL_TYPE_PON, *XMODEL1_KAN_SPECIAL_TYPES)):
        grouped[XMODEL1_SPECIAL_TYPE_NONE] = grouped[XMODEL1_SPECIAL_TYPE_NONE] or {"type": "none"}

    order = [
        XMODEL1_SPECIAL_TYPE_REACH,
        XMODEL1_SPECIAL_TYPE_DAMA,
        XMODEL1_SPECIAL_TYPE_HORA,
        XMODEL1_SPECIAL_TYPE_CHI_LOW,
        XMODEL1_SPECIAL_TYPE_CHI_MID,
        XMODEL1_SPECIAL_TYPE_CHI_HIGH,
        XMODEL1_SPECIAL_TYPE_PON,
        XMODEL1_SPECIAL_TYPE_DAIMINKAN,
        XMODEL1_SPECIAL_TYPE_ANKAN,
        XMODEL1_SPECIAL_TYPE_KAKAN,
        XMODEL1_SPECIAL_TYPE_RYUKYOKU,
        XMODEL1_SPECIAL_TYPE_NONE,
    ]
    for slot, special_type in enumerate(order[:max_candidates]):
        action = grouped[special_type]
        if action is None:
            continue
        type_id[slot] = special_type
        mask[slot] = 1
        action_dora_bonus, action_yakuhai_bonus = _special_action_bonus_metrics(
            action,
            yakuhai_ids=summary["yakuhai_ids"],  # type: ignore[arg-type]
            dora_target_ids=summary["dora_target_ids"],  # type: ignore[arg-type]
        )
        ankan_preserves_tenpai = 1.0 if special_type == XMODEL1_SPECIAL_TYPE_ANKAN and float(summary["tenpai"]) > 0.5 else 0.0
        chi_low = 1.0 if special_type == XMODEL1_SPECIAL_TYPE_CHI_LOW else 0.0
        chi_mid = 1.0 if special_type == XMODEL1_SPECIAL_TYPE_CHI_MID else 0.0
        chi_high = 1.0 if special_type == XMODEL1_SPECIAL_TYPE_CHI_HIGH else 0.0
        rinshan_bonus = 0.25 if special_type in XMODEL1_KAN_SPECIAL_TYPES else 0.0
        after_value_norm = _special_action_after_value(summary, special_type, action)
        if special_type == XMODEL1_SPECIAL_TYPE_REACH:
            speed_gain_norm = 0.4
            value_loss_norm = 0.05
        elif special_type == XMODEL1_SPECIAL_TYPE_DAMA:
            speed_gain_norm = 0.0
            value_loss_norm = 0.0
        elif special_type in XMODEL1_CHI_SPECIAL_TYPES:
            speed_gain_norm = 0.35
            value_loss_norm = 0.35
        elif special_type == XMODEL1_SPECIAL_TYPE_PON:
            speed_gain_norm = 0.45
            value_loss_norm = 0.2
        elif special_type in XMODEL1_KAN_SPECIAL_TYPES:
            speed_gain_norm = 0.3
            value_loss_norm = 0.45 if special_type != XMODEL1_SPECIAL_TYPE_ANKAN else 0.2
        elif special_type == XMODEL1_SPECIAL_TYPE_HORA:
            speed_gain_norm = 1.0
            value_loss_norm = 0.0
        else:
            speed_gain_norm = 0.0
            value_loss_norm = 0.0
        hand_value_survives = 1.0 if special_type in {XMODEL1_SPECIAL_TYPE_HORA, XMODEL1_SPECIAL_TYPE_NONE, XMODEL1_SPECIAL_TYPE_RYUKYOKU} else float(bool(summary["current_hand_value_survives"]) or after_value_norm > 0.0 or float(summary["current_han_floor_norm"]) > 0.0)
        exposure = _special_exposure_factor(special_type, ankan_preserves_tenpai=ankan_preserves_tenpai)
        threat_by_opponent = summary["threat_by_opponent"]  # type: ignore[assignment]
        risk_proxy = tuple(float(min(1.0, exposure * float(threat_by_opponent[idx]))) for idx in range(3))
        feat[slot] = np.array(
            [
                float(summary["shanten"]) / 8.0,
                float(summary["tenpai"]),
                float(summary["waits_count"]) / 34.0,
                float(summary["waits_live_norm"]),
                float(summary["round_progress"]),
                float(np.clip(summary["score_gap"], -1.0, 1.0)),
                float(summary["threat_proxy_any_reached"]),
                float(summary["is_open"]),
                after_value_norm,
                speed_gain_norm,
                value_loss_norm,
                float(summary["current_han_floor_norm"]),
                action_dora_bonus,
                action_yakuhai_bonus,
                risk_proxy[0],
                risk_proxy[1],
                risk_proxy[2],
                hand_value_survives,
                ankan_preserves_tenpai,
                chi_low,
                chi_mid,
                chi_high,
                rinshan_bonus,
                float(summary["current_han_floor_norm"]),
                float(summary["current_dora_count_norm"]),
            ],
            dtype=np.float32,
        )
        quality[slot] = (
            0.6 * after_value_norm
            + 0.35 * speed_gain_norm
            - 0.4 * value_loss_norm
            - 0.35 * float(np.mean(risk_proxy))
            + 0.15 * action_dora_bonus
            + 0.15 * action_yakuhai_bonus
            + (1.0 if special_type == XMODEL1_SPECIAL_TYPE_HORA else 0.0)
        )
        hard_bad[slot] = int(
            (special_type == XMODEL1_SPECIAL_TYPE_REACH and float(summary["threat_proxy_any_reached"]) > 0.5 and float(summary["waits_live_norm"]) < 0.03)
            or (special_type in (*XMODEL1_CHI_SPECIAL_TYPES, *XMODEL1_KAN_SPECIAL_TYPES) and value_loss_norm >= 0.35 and float(summary["current_han_floor_norm"]) <= 0.0)
        )
        rank_bucket[slot] = 0 if hard_bad[slot] else (3 if quality[slot] >= 1.0 else 2 if quality[slot] >= 0.25 else 1)

        if chosen_action is not None:
            chosen_type = chosen_action.get("type")
            if chosen_type == "reach" and special_type == XMODEL1_SPECIAL_TYPE_REACH:
                chosen_idx = slot
            elif chosen_type == "dahai" and special_type == XMODEL1_SPECIAL_TYPE_DAMA and grouped[XMODEL1_SPECIAL_TYPE_REACH] is not None:
                chosen_idx = slot
            elif chosen_type == "hora" and special_type == XMODEL1_SPECIAL_TYPE_HORA:
                chosen_idx = slot
            elif chosen_type == "ryukyoku" and special_type == XMODEL1_SPECIAL_TYPE_RYUKYOKU:
                chosen_idx = slot
            elif chosen_type == "pon" and special_type == XMODEL1_SPECIAL_TYPE_PON:
                chosen_idx = slot
            elif chosen_type == "chi" and special_type == _chi_special_type(chosen_action):
                chosen_idx = slot
            elif chosen_type == "daiminkan" and special_type == XMODEL1_SPECIAL_TYPE_DAIMINKAN:
                chosen_idx = slot
            elif chosen_type == "ankan" and special_type == XMODEL1_SPECIAL_TYPE_ANKAN:
                chosen_idx = slot
            elif chosen_type == "kakan" and special_type == XMODEL1_SPECIAL_TYPE_KAKAN:
                chosen_idx = slot
            elif chosen_type == "none" and special_type == XMODEL1_SPECIAL_TYPE_NONE:
                chosen_idx = slot

    return feat, type_id, mask, quality, rank_bucket, hard_bad, np.int16(chosen_idx)


__all__ = [
    "_after_state_path_metrics",
    "_dora_target_ids",
    "_per_opponent_dealin_risks",
    "iter_legal_discards",
    "simulate_discard_snapshot",
    "build_candidate_features",
    "build_special_candidate_arrays",
]
