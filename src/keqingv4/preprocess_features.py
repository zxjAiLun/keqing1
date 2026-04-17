"""keqingv4 typed action-summary builders for preprocess alpha."""

from __future__ import annotations

from collections import Counter

import numpy as np

from keqingv1.action_space import (
    ANKAN_IDX,
    CHI_HIGH_IDX,
    CHI_LOW_IDX,
    CHI_MID_IDX,
    DAIMINKAN_IDX,
    KAKAN_IDX,
    NONE_IDX,
    PON_IDX,
    REACH_IDX,
    RYUKYOKU_IDX,
    HORA_IDX,
    action_to_idx,
)
from mahjong_env.replay import _calc_normal_progress as _replay_calc_normal_progress
from mahjong_env.legal_actions import _reach_discard_candidates
from keqing_core import (
    build_keqingv4_typed_summaries as _rust_build_keqingv4_typed_summaries,
)
from training.cache_schema import (
    KEQINGV4_CALL_SUMMARY_SLOTS,
    KEQINGV4_SPECIAL_SUMMARY_SLOTS,
    KEQINGV4_SUMMARY_DIM,
)
from mahjong_env.tiles import normalize_tile, tile_to_34

_CALL_ACTION_IDS = (
    CHI_LOW_IDX,
    CHI_MID_IDX,
    CHI_HIGH_IDX,
    PON_IDX,
    DAIMINKAN_IDX,
    ANKAN_IDX,
    KAKAN_IDX,
    NONE_IDX,
)
_CALL_ACTION_TO_SLOT = {action_id: slot for slot, action_id in enumerate(_CALL_ACTION_IDS)}

_SPECIAL_ACTION_IDS = (REACH_IDX, HORA_IDX, RYUKYOKU_IDX)
_SPECIAL_ACTION_TO_SLOT = {action_id: slot for slot, action_id in enumerate(_SPECIAL_ACTION_IDS)}

_WIND_TILE_IDS = {"E": 27, "S": 28, "W": 29, "N": 30}
_TILE34_STR = (
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
)
_TERMINAL_HONOR_IDS = {0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33}


def _combined_hand(state: dict) -> list[str]:
    hand = list(state.get("hand", []))
    tsumo = state.get("tsumo_pai")
    if tsumo:
        hand.append(tsumo)
    return hand


def _counts34(tiles: list[str]) -> tuple[int, ...]:
    counts = [0] * 34
    for tile in tiles:
        idx = tile_to_34(normalize_tile(tile))
        if idx >= 0:
            counts[idx] += 1
    return tuple(counts)


def _remove_tiles_once(tiles: list[str], remove_tiles: list[str]) -> list[str]:
    out = list(tiles)
    for tile in remove_tiles:
        target = normalize_tile(tile)
        removed = False
        next_out: list[str] = []
        for current in out:
            if not removed and normalize_tile(current) == target:
                removed = True
                continue
            next_out.append(current)
        out = next_out
    return out


def _visible_counts34_from_state(state: dict) -> tuple[int, ...]:
    visible = [0] * 34
    for tile in _combined_hand(state):
        idx = tile_to_34(normalize_tile(tile))
        if idx >= 0:
            visible[idx] += 1
    for meld_group in state.get("melds", [[], [], [], []]):
        for meld in meld_group:
            for tile in meld.get("consumed", []):
                idx = tile_to_34(normalize_tile(tile))
                if idx >= 0:
                    visible[idx] += 1
            pai = meld.get("pai")
            if pai:
                idx = tile_to_34(normalize_tile(pai))
                if idx >= 0:
                    visible[idx] += 1
    for disc_group in state.get("discards", [[], [], [], []]):
        for discard in disc_group:
            pai = discard.get("pai") if isinstance(discard, dict) else discard
            if not pai:
                continue
            idx = tile_to_34(normalize_tile(pai))
            if idx >= 0:
                visible[idx] += 1
    for marker in state.get("dora_markers", []):
        idx = tile_to_34(normalize_tile(marker))
        if idx >= 0:
            visible[idx] += 1
    return tuple(visible)


def _actor_melds(state: dict, actor: int) -> list[dict]:
    meld_groups = state.get("melds", [[], [], [], []])
    return [dict(meld) for meld in (meld_groups[actor] if actor < len(meld_groups) else [])]


def _collect_actor_all_tiles(hand_tiles: list[str], melds: list[dict]) -> list[str]:
    tiles = list(hand_tiles)
    for meld in melds:
        tiles.extend(meld.get("consumed", []))
        pai = meld.get("pai")
        if pai:
            tiles.append(pai)
    return [normalize_tile(tile) for tile in tiles if tile]


def _pair_taatsu_metrics(counts34: tuple[int, ...]) -> tuple[int, int]:
    pair_count = sum(1 for count in counts34 if count >= 2)
    taatsu_count = 0
    for base in (0, 9, 18):
        suit = counts34[base : base + 9]
        for idx in range(8):
            if suit[idx] > 0 and suit[idx + 1] > 0:
                taatsu_count += 1
        for idx in range(7):
            if suit[idx] > 0 and suit[idx + 2] > 0:
                taatsu_count += 1
    return pair_count, taatsu_count


def _yakuhai_pair_flag(counts34: tuple[int, ...], state: dict, actor: int) -> float:
    oya = int(state.get("oya", 0))
    bakaze = str(state.get("bakaze", "E"))
    yakuhai_ids = {31, 32, 33}
    yakuhai_ids.add(_WIND_TILE_IDS.get(bakaze, 27))
    yakuhai_ids.add(27 + ((actor - oya) % 4))
    return 1.0 if any(counts34[idx] >= 2 for idx in yakuhai_ids if 0 <= idx < 34) else 0.0


def _opponent_reach_flag(state: dict, actor: int) -> float:
    return 1.0 if any(flag for idx, flag in enumerate(state.get("reached", [False] * 4)) if idx != actor) else 0.0


def _terminal_honor_unique_count(hand_tiles: list[str]) -> int:
    uniq: set[int] = set()
    for tile in hand_tiles:
        idx = tile_to_34(normalize_tile(tile))
        if idx in _TERMINAL_HONOR_IDS:
            uniq.add(idx)
    return len(uniq)


def _dora_from_marker_idx(marker_idx: int) -> int:
    if 0 <= marker_idx <= 8:
        return 0 if marker_idx == 8 else marker_idx + 1
    if 9 <= marker_idx <= 17:
        return 9 if marker_idx == 17 else marker_idx + 1
    if 18 <= marker_idx <= 26:
        return 18 if marker_idx == 26 else marker_idx + 1
    if 27 <= marker_idx <= 30:
        return 27 if marker_idx == 30 else marker_idx + 1
    if 31 <= marker_idx <= 33:
        return 31 if marker_idx == 33 else marker_idx + 1
    return -1


def _after_state_bonus_metrics(state: dict, actor: int, hand_tiles: list[str], melds: list[dict]) -> tuple[float, float, float, float, float, float]:
    all_tiles = _collect_actor_all_tiles(hand_tiles, melds)
    counts34 = _counts34(all_tiles)
    dora_count = 0
    for marker in state.get("dora_markers", []):
        marker_idx = tile_to_34(normalize_tile(marker))
        dora_idx = _dora_from_marker_idx(marker_idx)
        if dora_idx >= 0:
            dora_count += counts34[dora_idx]
    aka_count = sum(1 for tile in all_tiles if tile.endswith("r"))
    terminal_honor_unique = _terminal_honor_unique_count(all_tiles)
    meld_count = len(melds)
    remaining_wall = float(max(0, int(state.get("remaining_wall", 70))))
    tanyao_keep_flag = 1.0 if terminal_honor_unique == 0 else 0.0
    return (
        dora_count / 10.0,
        aka_count / 3.0,
        terminal_honor_unique / 13.0,
        meld_count / 4.0,
        min(1.0, remaining_wall / 70.0),
        tanyao_keep_flag,
    )


def _yakuhai_triplet_flag(counts34: tuple[int, ...], state: dict, actor: int) -> float:
    oya = int(state.get("oya", 0))
    bakaze = str(state.get("bakaze", "E"))
    yakuhai_ids = {31, 32, 33}
    yakuhai_ids.add(_WIND_TILE_IDS.get(bakaze, 27))
    yakuhai_ids.add(27 + ((actor - oya) % 4))
    return 1.0 if any(counts34[idx] >= 3 for idx in yakuhai_ids if 0 <= idx < 34) else 0.0


def _chiitoi_path_flag(counts34: tuple[int, ...], open_hand_flag: float) -> float:
    if open_hand_flag > 0.5:
        return 0.0
    pair_count = sum(1 for count in counts34 if count >= 2)
    unique_count = sum(1 for count in counts34 if count > 0)
    return 1.0 if pair_count >= 6 and unique_count >= 7 else 0.0


def _iipeiko_path_flag(counts34: tuple[int, ...], open_hand_flag: float) -> float:
    if open_hand_flag > 0.5:
        return 0.0
    for base in (0, 9, 18):
        suit = counts34[base : base + 9]
        for idx in range(7):
            if suit[idx] >= 2 and suit[idx + 1] >= 2 and suit[idx + 2] >= 2:
                return 1.0
    return 0.0


def _pinfu_like_path_flag(counts34: tuple[int, ...], state: dict, actor: int, open_hand_flag: float) -> float:
    if open_hand_flag > 0.5:
        return 0.0
    yakuhai_ids = {31, 32, 33, _WIND_TILE_IDS.get(str(state.get("bakaze", "E")), 27), 27 + ((actor - int(state.get("oya", 0))) % 4)}
    non_value_pair = any(counts34[idx] >= 2 for idx in range(34) if idx not in yakuhai_ids)
    if not non_value_pair:
        return 0.0
    sequence_heads = 0
    for base in (0, 9, 18):
        suit = counts34[base : base + 9]
        for idx in range(7):
            if suit[idx] > 0 and suit[idx + 1] > 0 and suit[idx + 2] > 0:
                sequence_heads += 1
    return 1.0 if sequence_heads >= 3 else 0.0


def _common_yaku_path_metrics(
    counts34: tuple[int, ...],
    state: dict,
    actor: int,
    open_hand_flag: float,
) -> tuple[float, float, float, float, float]:
    tanyao_keep = 1.0 if _terminal_honor_unique_count([_TILE34_STR[idx] for idx, count in enumerate(counts34) for _ in range(count)]) == 0 else 0.0
    yakuhai_pair = _yakuhai_pair_flag(counts34, state, actor)
    chiitoi_path = _chiitoi_path_flag(counts34, open_hand_flag)
    iipeiko_path = _iipeiko_path_flag(counts34, open_hand_flag)
    pinfu_like_path = _pinfu_like_path_flag(counts34, state, actor, open_hand_flag)
    return tanyao_keep, yakuhai_pair, chiitoi_path, iipeiko_path, pinfu_like_path


def _max_hand_value_norm(
    *,
    confirmed_han_floor: float,
    waits_count: int,
    waits_live: int,
    is_tenpai: bool,
) -> float:
    value_proxy = float(confirmed_han_floor)
    if is_tenpai:
        value_proxy += 0.5 + min(waits_count, 5) * 0.15 + min(waits_live, 12) * 0.03
    return min(value_proxy / 8.0, 1.0)


def _value_proxy_metrics(
    *,
    progress,
    dora_norm: float,
    aka_norm: float,
    open_hand_flag: float,
    yakuhai_pair: float,
    yakuhai_triplet: float,
    current_tanyao_keep: float,
    after_tanyao_keep: float,
    current_yakuhai_pair: float,
    current_chiitoi_path: float,
    after_chiitoi_path: float,
    current_iipeiko_path: float,
    after_iipeiko_path: float,
    current_pinfu_like_path: float,
    after_pinfu_like_path: float,
) -> tuple[float, float, float, float]:
    tenpai_flag = 1.0 if progress.shanten == 0 else 0.0
    confirmed_han_floor = min(8.0, yakuhai_triplet + after_tanyao_keep + dora_norm * 10.0 + aka_norm * 3.0)
    max_hand_value_proxy = tenpai_flag * _max_hand_value_norm(
        confirmed_han_floor=confirmed_han_floor,
        waits_count=int(progress.waits_count),
        waits_live=int(progress.ukeire_live_count),
        is_tenpai=bool(tenpai_flag),
    )
    reach_bonus = (
        0.12
        + 0.05 * after_pinfu_like_path
        + 0.05 * after_iipeiko_path
        + 0.04 * after_chiitoi_path
    )
    reach_hand_value_proxy = tenpai_flag * (1.0 - open_hand_flag) * min(1.0, max_hand_value_proxy + reach_bonus)
    yaku_break_flag = 1.0 if any(
        (
            current_tanyao_keep > after_tanyao_keep,
            current_yakuhai_pair > yakuhai_pair,
            current_chiitoi_path > after_chiitoi_path,
            current_iipeiko_path > after_iipeiko_path,
            current_pinfu_like_path > after_pinfu_like_path,
        )
    ) else 0.0
    closed_route_bonus = max(after_pinfu_like_path, after_iipeiko_path, after_chiitoi_path)
    closed_value_proxy = (1.0 - open_hand_flag) * min(
        1.0,
        0.45 * max_hand_value_proxy
        + 0.25 * reach_hand_value_proxy
        + 0.2 * (confirmed_han_floor / 8.0)
        + 0.1 * closed_route_bonus,
    )
    return max_hand_value_proxy, reach_hand_value_proxy, yaku_break_flag, closed_value_proxy


def _shape_value_metrics(
    counts34: tuple[int, ...],
    pair_count: int,
    open_hand_flag: float,
    yakuhai_triplet: float,
) -> tuple[float, float, float, float]:
    suit_counts = [
        sum(counts34[0:9]),
        sum(counts34[9:18]),
        sum(counts34[18:27]),
    ]
    honor_count = sum(counts34[27:34])
    total_tiles = max(1, sum(suit_counts) + honor_count)
    max_suit = max(suit_counts) if suit_counts else 0
    honitsu_tendency = (max_suit + honor_count) / total_tiles
    chinitsu_tendency = 0.0 if honor_count > 0 else max_suit / max(1, sum(suit_counts))
    chiitoi_tendency = (1.0 - open_hand_flag) * min(1.0, pair_count / 7.0)
    yakuhai_value_proxy = min(1.0, 0.6 * yakuhai_triplet + 0.4 * chiitoi_tendency)
    return honitsu_tendency, chinitsu_tendency, chiitoi_tendency, yakuhai_value_proxy


def _summary_vector(
    *,
    hand_tiles: list[str],
    melds: list[dict],
    visible_counts34: tuple[int, ...],
    state: dict,
    actor: int,
    current_shanten: int,
    open_hand_flag: float,
    current_tanyao_keep: float = 0.0,
    current_yakuhai_pair: float = 0.0,
    current_chiitoi_path: float = 0.0,
    current_iipeiko_path: float = 0.0,
    current_pinfu_like_path: float = 0.0,
) -> np.ndarray:
    counts34 = _counts34(hand_tiles)
    progress = _replay_calc_normal_progress(hand_tiles, melds, list(visible_counts34))
    pair_count, taatsu_count = _pair_taatsu_metrics(counts34)
    delta_shanten = np.clip((float(current_shanten) - float(progress.shanten)) / 4.0, -1.0, 1.0)
    dora_norm, aka_norm, terminal_norm, meld_norm, wall_norm, tanyao_keep = _after_state_bonus_metrics(
        state, actor, hand_tiles, melds
    )
    after_tanyao_keep, yakuhai_pair, after_chiitoi_path, after_iipeiko_path, after_pinfu_like_path = _common_yaku_path_metrics(
        counts34, state, actor, open_hand_flag
    )
    yakuhai_triplet = _yakuhai_triplet_flag(counts34, state, actor)
    max_hand_value_proxy, reach_hand_value_proxy, yaku_break_flag, closed_value_proxy = _value_proxy_metrics(
        progress=progress,
        dora_norm=dora_norm,
        aka_norm=aka_norm,
        open_hand_flag=open_hand_flag,
        yakuhai_pair=yakuhai_pair,
        yakuhai_triplet=yakuhai_triplet,
        current_tanyao_keep=current_tanyao_keep,
        after_tanyao_keep=after_tanyao_keep,
        current_yakuhai_pair=current_yakuhai_pair,
        current_chiitoi_path=current_chiitoi_path,
        after_chiitoi_path=after_chiitoi_path,
        current_iipeiko_path=current_iipeiko_path,
        after_iipeiko_path=after_iipeiko_path,
        current_pinfu_like_path=current_pinfu_like_path,
        after_pinfu_like_path=after_pinfu_like_path,
    )
    honitsu_tendency, chinitsu_tendency, chiitoi_tendency, yakuhai_value_proxy = _shape_value_metrics(
        counts34, pair_count, open_hand_flag, yakuhai_triplet
    )
    return np.array(
        [
            progress.shanten / 8.0,
            1.0 if progress.shanten == 0 else 0.0,
            progress.waits_count / 34.0,
            progress.ukeire_type_count / 34.0,
            progress.ukeire_live_count / 34.0,
            progress.good_shape_ukeire_live_count / 34.0,
            progress.improvement_live_count / 34.0,
            pair_count / 7.0,
            taatsu_count / 6.0,
            yakuhai_pair,
            _opponent_reach_flag(state, actor),
            open_hand_flag,
            delta_shanten,
            1.0,
            dora_norm,
            aka_norm,
            terminal_norm,
            meld_norm,
            wall_norm,
            after_tanyao_keep,
            max_hand_value_proxy,
            reach_hand_value_proxy,
            yaku_break_flag,
            closed_value_proxy,
            honitsu_tendency,
            chinitsu_tendency,
            chiitoi_tendency,
            yakuhai_value_proxy,
        ],
        dtype=np.float32,
    )


def _summary_score(vec: np.ndarray) -> float:
    return float(
        -100.0 * vec[0]
        + 12.0 * vec[1]
        + 10.0 * vec[4]
        + 7.0 * vec[5]
        + 5.0 * vec[6]
        + 3.0 * vec[2]
        + 1.0 * vec[8]
        + 0.5 * vec[7]
        + 2.0 * vec[14]
        + 1.0 * vec[19]
        + 3.0 * vec[20]
        + 2.0 * vec[21]
        - 1.5 * vec[22]
        + 1.0 * vec[23]
        + 0.8 * vec[24]
        + 0.8 * vec[25]
        + 0.6 * vec[26]
        + 1.2 * vec[27]
    )


def _call_action_slot(action: dict) -> int | None:
    action_type = action.get("type", "")
    if action_type == "none":
        return 7
    if action_type == "pon":
        return 3
    if action_type == "daiminkan":
        return 4
    if action_type == "ankan":
        return 5
    if action_type == "kakan":
        return 6
    if action_type == "chi":
        pai_rank = int(normalize_tile(action.get("pai", "0m"))[0]) if action.get("pai") else 0
        ranks = sorted(
            int(normalize_tile(tile)[0])
            for tile in action.get("consumed", [])
            if normalize_tile(tile) and normalize_tile(tile)[0].isdigit()
        )
        if len(ranks) == 2:
            if pai_rank < ranks[0]:
                return 0
            if pai_rank < ranks[1]:
                return 1
            return 2
        return 0
    return None


def _current_summary_context(state: dict, actor: int) -> tuple[list[str], tuple[int, ...], object, float, np.ndarray]:
    current_hand = _combined_hand(state)
    visible_counts34 = _visible_counts34_from_state(state)
    actor_melds = _actor_melds(state, actor)
    current_progress = _replay_calc_normal_progress(current_hand, actor_melds, list(visible_counts34))
    current_open_hand = 1.0 if any(meld.get("type") != "ankan" for meld in actor_melds) else 0.0
    current_counts34 = _counts34(_collect_actor_all_tiles(current_hand, actor_melds))
    (
        current_tanyao_keep,
        current_yakuhai_pair,
        current_chiitoi_path,
        current_iipeiko_path,
        current_pinfu_like_path,
    ) = _common_yaku_path_metrics(current_counts34, state, actor, current_open_hand)
    current_vector = _summary_vector(
        hand_tiles=current_hand,
        melds=actor_melds,
        visible_counts34=visible_counts34,
        state=state,
        actor=actor,
        current_shanten=current_progress.shanten,
        open_hand_flag=current_open_hand,
        current_tanyao_keep=current_tanyao_keep,
        current_yakuhai_pair=current_yakuhai_pair,
        current_chiitoi_path=current_chiitoi_path,
        current_iipeiko_path=current_iipeiko_path,
        current_pinfu_like_path=current_pinfu_like_path,
    )
    return current_hand, visible_counts34, current_progress, current_open_hand, current_vector


def _current_yaku_guards(state: dict, actor: int, current_hand: list[str], actor_melds: list[dict]) -> tuple[float, float, float, float, float]:
    all_tiles = _collect_actor_all_tiles(current_hand, actor_melds)
    current_counts34 = _counts34(all_tiles)
    current_open_hand = 1.0 if any(meld.get("type") != "ankan" for meld in actor_melds) else 0.0
    return _common_yaku_path_metrics(current_counts34, state, actor, current_open_hand)


def _upgrade_pon_melds_for_kakan(melds: list[dict], pai: str) -> list[dict]:
    pai_norm = normalize_tile(pai)
    upgraded: list[dict] = []
    upgraded_one = False
    for meld in melds:
        next_meld = dict(meld)
        if (
            not upgraded_one
            and next_meld.get("type") == "pon"
            and normalize_tile(next_meld.get("pai", "")) == pai_norm
        ):
            next_meld["type"] = "kakan"
            consumed = list(next_meld.get("consumed", []))
            consumed.append(pai)
            next_meld["consumed"] = consumed
            next_meld["pai"] = pai_norm
            upgraded_one = True
        upgraded.append(next_meld)
    if not upgraded_one:
        upgraded.append(
            {
                "type": "kakan",
                "pai": pai_norm,
                "consumed": [pai],
                "target": None,
            }
        )
    return upgraded


def _project_call_state(state: dict, actor: int, action: dict) -> tuple[list[str], list[dict], float, bool] | None:
    action_type = action.get("type", "")
    hand = _combined_hand(state)
    remove_tiles = list(action.get("consumed", []))
    actor_melds = _actor_melds(state, actor)
    current_open_hand = 1.0 if any(meld.get("type") != "ankan" for meld in actor_melds) else 0.0
    pai = action.get("pai")

    if action_type == "kakan":
        if not pai:
            return None
        remove_tiles = [pai]

    after_hand = _remove_tiles_once(hand, remove_tiles)
    if len(after_hand) >= len(hand):
        return None

    if action_type == "chi":
        projected_melds = [*actor_melds, {"type": "chi", "pai": pai, "consumed": list(action.get("consumed", [])), "target": action.get("target")}]
        return after_hand, projected_melds, 1.0, False
    if action_type == "pon":
        projected_melds = [*actor_melds, {"type": "pon", "pai": pai, "consumed": list(action.get("consumed", [])), "target": action.get("target")}]
        return after_hand, projected_melds, 1.0, False
    if action_type == "daiminkan":
        projected_melds = [*actor_melds, {"type": "daiminkan", "pai": pai, "consumed": list(action.get("consumed", [])), "target": action.get("target")}]
        return after_hand, projected_melds, 1.0, True
    if action_type == "ankan":
        projected_melds = [*actor_melds, {"type": "ankan", "pai": pai or (remove_tiles[0] if remove_tiles else None), "consumed": list(action.get("consumed", [])), "target": actor}]
        return after_hand, projected_melds, current_open_hand, True
    if action_type == "kakan":
        projected_melds = _upgrade_pon_melds_for_kakan(actor_melds, pai)
        return after_hand, projected_melds, current_open_hand, True
    return None


def _best_discard_summary_vector(
    *,
    hand_tiles: list[str],
    melds: list[dict],
    visible_counts34: tuple[int, ...],
    state: dict,
    actor: int,
    current_shanten: int,
    open_hand_flag: float,
) -> np.ndarray:
    (
        current_tanyao_keep,
        current_yakuhai_pair,
        current_chiitoi_path,
        current_iipeiko_path,
        current_pinfu_like_path,
    ) = _current_yaku_guards(state, actor, hand_tiles, melds)
    best_vec: np.ndarray | None = None
    best_score = -1e18
    seen: set[int] = set()
    for tile in hand_tiles:
        tile34 = tile_to_34(normalize_tile(tile))
        if tile34 < 0 or tile34 in seen:
            continue
        seen.add(tile34)
        after_hand = _remove_tiles_once(hand_tiles, [tile])
        vec = _summary_vector(
            hand_tiles=after_hand,
            melds=melds,
            visible_counts34=visible_counts34,
            state=state,
            actor=actor,
            current_shanten=current_shanten,
            open_hand_flag=open_hand_flag,
            current_tanyao_keep=current_tanyao_keep,
            current_yakuhai_pair=current_yakuhai_pair,
            current_chiitoi_path=current_chiitoi_path,
            current_iipeiko_path=current_iipeiko_path,
            current_pinfu_like_path=current_pinfu_like_path,
        )
        score = _summary_score(vec)
        if score > best_score:
            best_score = score
            best_vec = vec
    if best_vec is None:
        return _summary_vector(
            hand_tiles=hand_tiles,
            melds=melds,
            visible_counts34=visible_counts34,
            state=state,
            actor=actor,
            current_shanten=current_shanten,
            open_hand_flag=open_hand_flag,
            current_tanyao_keep=current_tanyao_keep,
            current_yakuhai_pair=current_yakuhai_pair,
            current_chiitoi_path=current_chiitoi_path,
            current_iipeiko_path=current_iipeiko_path,
            current_pinfu_like_path=current_pinfu_like_path,
        )
    return best_vec


def _best_discard_summary_from_state(
    *,
    state: dict,
    actor: int,
) -> np.ndarray:
    current_hand, visible_counts34, current_progress, current_open_hand, current_vector = _current_summary_context(state, actor)
    (
        current_tanyao_keep,
        current_yakuhai_pair,
        current_chiitoi_path,
        current_iipeiko_path,
        current_pinfu_like_path,
    ) = _current_yaku_guards(state, actor, current_hand, _actor_melds(state, actor))
    best_vec: np.ndarray | None = None
    best_score = -1e18
    seen: set[int] = set()
    for tile in current_hand:
        tile34 = tile_to_34(normalize_tile(tile))
        if tile34 < 0 or tile34 in seen:
            continue
        seen.add(tile34)
        after_hand = _remove_tiles_once(current_hand, [tile])
        vec = _summary_vector(
            hand_tiles=after_hand,
            melds=_actor_melds(state, actor),
            visible_counts34=visible_counts34,
            state=state,
            actor=actor,
            current_shanten=current_progress.shanten,
            open_hand_flag=current_open_hand,
            current_tanyao_keep=current_tanyao_keep,
            current_yakuhai_pair=current_yakuhai_pair,
            current_chiitoi_path=current_chiitoi_path,
            current_iipeiko_path=current_iipeiko_path,
            current_pinfu_like_path=current_pinfu_like_path,
        )
        score = _summary_score(vec)
        if score > best_score:
            best_score = score
            best_vec = vec
    return current_vector if best_vec is None else best_vec


def _rinshan_weighted_summary_vector(
    *,
    hand_tiles: list[str],
    melds: list[dict],
    visible_counts34: tuple[int, ...],
    state: dict,
    actor: int,
    current_shanten: int,
    open_hand_flag: float,
) -> np.ndarray:
    (
        current_tanyao_keep,
        current_yakuhai_pair,
        current_chiitoi_path,
        current_iipeiko_path,
        current_pinfu_like_path,
    ) = _current_yaku_guards(state, actor, hand_tiles, melds)
    acc = np.zeros((KEQINGV4_SUMMARY_DIM,), dtype=np.float32)
    total_weight = 0.0
    for tile34, visible in enumerate(visible_counts34):
        live = max(0, 4 - int(visible))
        if live <= 0:
            continue
        draw_tile = _TILE34_STR[tile34]
        draw_visible = list(visible_counts34)
        draw_visible[tile34] += 1
        vec = _best_discard_summary_vector(
            hand_tiles=[*hand_tiles, draw_tile],
            melds=melds,
            visible_counts34=tuple(draw_visible),
            state=state,
            actor=actor,
            current_shanten=current_shanten,
            open_hand_flag=open_hand_flag,
        )
        acc += live * vec
        total_weight += float(live)
    if total_weight <= 0.0:
        return _summary_vector(
            hand_tiles=hand_tiles,
            melds=melds,
            visible_counts34=visible_counts34,
            state=state,
            actor=actor,
            current_shanten=current_shanten,
            open_hand_flag=open_hand_flag,
            current_tanyao_keep=current_tanyao_keep,
            current_yakuhai_pair=current_yakuhai_pair,
            current_chiitoi_path=current_chiitoi_path,
            current_iipeiko_path=current_iipeiko_path,
            current_pinfu_like_path=current_pinfu_like_path,
        )
    return acc / total_weight


def _projected_call_summary_vector(
    *,
    state: dict,
    actor: int,
    action: dict,
) -> np.ndarray | None:
    projected = _project_call_state(state, actor, action)
    if projected is None:
        return None
    _projected_hand, projected_melds, _projected_open_hand, needs_rinshan = projected
    projected_state = dict(state)
    projected_state["hand"] = list(_projected_hand)
    meld_groups = [list(group) for group in state.get("melds", [[], [], [], []])]
    while len(meld_groups) <= actor:
        meld_groups.append([])
    meld_groups[actor] = list(projected_melds)
    projected_state["melds"] = meld_groups
    projected_state["last_discard"] = None
    projected_state["last_kakan"] = None
    projected_state["tsumo_pai"] = None
    projected_state["actor_to_move"] = actor

    if not needs_rinshan:
        return _best_discard_summary_from_state(state=projected_state, actor=actor)

    weighted = np.zeros((KEQINGV4_SUMMARY_DIM,), dtype=np.float32)
    visible_counts34 = _visible_counts34_from_state(projected_state)
    total_weight = 0.0
    for tile34, visible in enumerate(visible_counts34):
        live = max(0, 4 - int(visible))
        if live <= 0:
            continue
        draw_tile = _TILE34_STR[tile34]
        draw_state = dict(projected_state)
        draw_state["hand"] = [*projected_state["hand"], draw_tile]
        draw_state["tsumo_pai"] = draw_tile
        last_tsumo = list(projected_state.get("last_tsumo", [None, None, None, None]))
        while len(last_tsumo) <= actor:
            last_tsumo.append(None)
        last_tsumo[actor] = normalize_tile(draw_tile)
        draw_state["last_tsumo"] = last_tsumo
        last_tsumo_raw = list(projected_state.get("last_tsumo_raw", [None, None, None, None]))
        while len(last_tsumo_raw) <= actor:
            last_tsumo_raw.append(None)
        last_tsumo_raw[actor] = draw_tile
        draw_state["last_tsumo_raw"] = last_tsumo_raw
        weighted += float(live) * _best_discard_summary_from_state(state=draw_state, actor=actor)
        total_weight += float(live)
    if total_weight <= 0.0:
        return _current_summary_context(projected_state, actor)[4]
    return weighted / total_weight


def _resolve_reach_tsumo_tile(state: dict, actor: int) -> tuple[str | None, str | None]:
    tsumo_pai = state.get("tsumo_pai")
    if tsumo_pai:
        return normalize_tile(tsumo_pai), tsumo_pai
    last_tsumo = list(state.get("last_tsumo", [None, None, None, None]))
    last_tsumo_raw = list(state.get("last_tsumo_raw", [None, None, None, None]))
    normalized = last_tsumo[actor] if actor < len(last_tsumo) else None
    raw = last_tsumo_raw[actor] if actor < len(last_tsumo_raw) else normalized
    if normalized is None and raw is not None:
        normalized = normalize_tile(raw)
    return normalized, raw


def _best_reach_summary_vector(
    *,
    state: dict,
    actor: int,
    visible_counts34: tuple[int, ...],
    current_shanten: int,
) -> np.ndarray | None:
    hand = _combined_hand(state)
    if not hand:
        return None
    actor_melds = _actor_melds(state, actor)
    (
        current_tanyao_keep,
        current_yakuhai_pair,
        current_chiitoi_path,
        current_iipeiko_path,
        current_pinfu_like_path,
    ) = _current_yaku_guards(state, actor, hand, actor_melds)
    hand_counter = Counter(hand)
    last_tsumo, last_tsumo_raw = _resolve_reach_tsumo_tile(state, actor)
    candidates = _reach_discard_candidates(hand_counter, last_tsumo, last_tsumo_raw)
    if not candidates:
        return None

    best_vec: np.ndarray | None = None
    best_score = -1e18
    for pai_out, _tsumogiri in candidates:
        after_hand = _remove_tiles_once(hand, [pai_out])
        vec = _summary_vector(
            hand_tiles=after_hand,
            melds=actor_melds,
            visible_counts34=visible_counts34,
            state=state,
            actor=actor,
            current_shanten=current_shanten,
            open_hand_flag=0.0,
            current_tanyao_keep=current_tanyao_keep,
            current_yakuhai_pair=current_yakuhai_pair,
            current_chiitoi_path=current_chiitoi_path,
            current_iipeiko_path=current_iipeiko_path,
            current_pinfu_like_path=current_pinfu_like_path,
        )
        score = _summary_score(vec)
        if score > best_score:
            best_score = score
            best_vec = vec
    return best_vec


def _build_hora_summary_vector(
    *,
    state: dict,
    actor: int,
    current_vector: np.ndarray,
    action: dict,
) -> np.ndarray:
    vec = current_vector.copy()
    target = action.get("target", actor)
    is_tsumo = float(target == actor)
    is_ron = 1.0 - is_tsumo
    rinshan = float(bool(action.get("is_rinshan")) or bool(state.get("_hora_is_rinshan")))
    chankan = float(bool(action.get("is_chankan")) or bool(state.get("_hora_is_chankan")))
    haitei = float(bool(action.get("is_haitei")) or bool(state.get("_hora_is_haitei")))
    houtei = float(bool(action.get("is_houtei")) or bool(state.get("_hora_is_houtei")))

    vec[0] = -0.125
    vec[1] = 1.0
    vec[2] = is_tsumo
    vec[3] = is_ron
    vec[4] = rinshan
    vec[5] = chankan
    vec[6] = haitei
    vec[7] = houtei
    vec[12] = 1.0
    vec[13] = 1.0
    return vec


def _build_ryukyoku_summary_vector(
    *,
    hand_tiles: list[str],
    current_vector: np.ndarray,
) -> np.ndarray:
    vec = current_vector.copy()
    unique_yaochu = _terminal_honor_unique_count(hand_tiles)
    vec[2] = unique_yaochu / 13.0
    vec[3] = 1.0 if unique_yaochu >= 9 else 0.0
    vec[4] = unique_yaochu / 9.0
    vec[12] = 1.0 if unique_yaochu >= 9 else 0.0
    vec[13] = 1.0
    return vec


def build_typed_action_summaries(
    state: dict,
    actor: int,
    legal_actions: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    discard_summary = np.zeros((34, KEQINGV4_SUMMARY_DIM), dtype=np.float32)
    call_summary = np.zeros((KEQINGV4_CALL_SUMMARY_SLOTS, KEQINGV4_SUMMARY_DIM), dtype=np.float32)
    special_summary = np.zeros((KEQINGV4_SPECIAL_SUMMARY_SLOTS, KEQINGV4_SUMMARY_DIM), dtype=np.float32)

    current_hand, visible_counts34, current_progress, current_open_hand, current_vector = _current_summary_context(state, actor)
    actor_melds = _actor_melds(state, actor)
    (
        current_tanyao_keep,
        current_yakuhai_pair,
        current_chiitoi_path,
        current_iipeiko_path,
        current_pinfu_like_path,
    ) = _current_yaku_guards(state, actor, current_hand, actor_melds)
    rust_summaries_ready = False
    try:
        discard_summary, call_summary, special_summary = _rust_build_keqingv4_typed_summaries(
            state,
            actor,
            legal_actions,
        )
        rust_summaries_ready = True
    except Exception:
        rust_summaries_ready = False

    for action in legal_actions:
        try:
            action_idx = action_to_idx(action)
        except Exception:
            continue

        action_type = action.get("type", "")
        if action_type == "dahai":
            if rust_summaries_ready:
                continue
            pai = action.get("pai")
            if not pai:
                continue
            discard34 = tile_to_34(normalize_tile(pai))
            if discard34 < 0:
                continue
            after_hand = _remove_tiles_once(current_hand, [pai])
            discard_summary[discard34] = _summary_vector(
                hand_tiles=after_hand,
                melds=actor_melds,
                visible_counts34=visible_counts34,
                state=state,
                actor=actor,
                current_shanten=current_progress.shanten,
                open_hand_flag=current_open_hand,
                current_tanyao_keep=current_tanyao_keep,
                current_yakuhai_pair=current_yakuhai_pair,
                current_chiitoi_path=current_chiitoi_path,
                current_iipeiko_path=current_iipeiko_path,
                current_pinfu_like_path=current_pinfu_like_path,
            )
            continue

        if action_idx in _CALL_ACTION_TO_SLOT:
            if rust_summaries_ready:
                continue
            slot = _call_action_slot(action)
            if slot is None:
                continue
            if action_type == "none":
                call_summary[slot] = current_vector
                continue
            projected = _project_call_state(state, actor, action)
            if projected is None:
                call_summary[slot] = current_vector
                continue
            projected_vec = _projected_call_summary_vector(
                state=state,
                actor=actor,
                action=action,
            )
            call_summary[slot] = current_vector if projected_vec is None else projected_vec
            continue

        if action_idx in _SPECIAL_ACTION_TO_SLOT:
            if rust_summaries_ready:
                continue
            slot = _SPECIAL_ACTION_TO_SLOT[action_idx]
            if action_type == "reach":
                reach_vec = _best_reach_summary_vector(
                    state=state,
                    actor=actor,
                    visible_counts34=visible_counts34,
                    current_shanten=current_progress.shanten,
                )
                special_summary[slot] = reach_vec if reach_vec is not None else current_vector
            elif action_type == "hora":
                special_summary[slot] = _build_hora_summary_vector(
                    state=state,
                    actor=actor,
                    current_vector=current_vector,
                    action=action,
                )
            elif action_type == "ryukyoku":
                special_summary[slot] = _build_ryukyoku_summary_vector(
                    hand_tiles=current_hand,
                    current_vector=current_vector,
                )
            else:
                special_summary[slot] = current_vector

    return discard_summary, call_summary, special_summary


__all__ = [
    "KEQINGV4_SUMMARY_DIM",
    "KEQINGV4_CALL_SUMMARY_SLOTS",
    "KEQINGV4_SPECIAL_SUMMARY_SLOTS",
    "build_typed_action_summaries",
]
