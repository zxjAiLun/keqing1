from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Optional, Sequence, Set, Tuple

from mahjong.shanten import Shanten
from riichienv import calculate_shanten as _calculate_standard_shanten


_TILE34_TO_STR: List[str] = [
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
]

_COUNT34_TO_136_IDS: tuple[tuple[tuple[int, ...], ...], ...] = tuple(
    tuple(tuple(t34 * 4 + copy_idx for copy_idx in range(cnt)) for cnt in range(5))
    for t34 in range(34)
)


@dataclass(frozen=True)
class NormalProgressInfo:
    shanten: int
    waits_count: int
    waits_tiles: List[bool]
    tehai_count: int
    ukeire_type_count: int
    ukeire_live_count: int
    ukeire_tiles: List[bool]
    good_shape_ukeire_type_count: int
    good_shape_ukeire_live_count: int
    good_shape_ukeire_tiles: List[bool]
    improvement_type_count: int
    improvement_live_count: int
    improvement_tiles: List[bool]


def counts_to_hand(counts: Sequence[int]) -> List[str]:
    hand: List[str] = []
    for t34, cnt in enumerate(counts):
        if cnt > 0:
            hand.extend([_TILE34_TO_STR[t34]] * cnt)
    return hand


@lru_cache(maxsize=400000)
def _counts34_to_tile136_ids(counts34: tuple[int, ...]) -> tuple[int, ...]:
    total = sum(counts34)
    if total == 0:
        return ()
    ids = [0] * total
    pos = 0
    for t34, cnt in enumerate(counts34):
        if cnt <= 0:
            continue
        vals = _COUNT34_TO_136_IDS[t34][cnt]
        n = len(vals)
        ids[pos:pos + n] = vals
        pos += n
    return tuple(ids)


def _is_suited_sequence_start(tile34: int) -> bool:
    return tile34 < 27 and (tile34 % 9) <= 6


def _is_complete_regular_counts(
    counts: tuple[int, ...],
    cache: Dict[tuple[tuple[int, ...], int, bool], bool],
    melds_needed: Optional[int] = None,
    need_pair: bool = True,
) -> bool:
    if melds_needed is None:
        tile_count = sum(counts)
        if tile_count % 3 != 2:
            return False
        melds_needed = (tile_count - 2) // 3

    key = (counts, melds_needed, need_pair)
    cached = cache.get(key)
    if cached is not None:
        return cached

    first = next((i for i, cnt in enumerate(counts) if cnt > 0), None)
    if first is None:
        result = melds_needed == 0 and not need_pair
        cache[key] = result
        return result

    work = list(counts)
    result = False

    if need_pair and work[first] >= 2:
        work[first] -= 2
        if _is_complete_regular_counts(tuple(work), cache, melds_needed, False):
            result = True
        work[first] += 2

    if not result and melds_needed > 0 and work[first] >= 3:
        work[first] -= 3
        if _is_complete_regular_counts(tuple(work), cache, melds_needed - 1, need_pair):
            result = True
        work[first] += 3

    if (
        not result
        and melds_needed > 0
        and _is_suited_sequence_start(first)
        and work[first + 1] > 0
        and work[first + 2] > 0
    ):
        work[first] -= 1
        work[first + 1] -= 1
        work[first + 2] -= 1
        if _is_complete_regular_counts(tuple(work), cache, melds_needed - 1, need_pair):
            result = True

    cache[key] = result
    return result


def find_regular_waits(counts: tuple[int, ...]) -> List[bool]:
    waits = [False] * 34
    if sum(counts) % 3 != 1:
        return waits

    complete_cache: Dict[tuple[tuple[int, ...], int, bool], bool] = {}
    for tile34, cnt in enumerate(counts):
        if cnt >= 4:
            continue
        work = list(counts)
        work[tile34] += 1
        if _is_complete_regular_counts(tuple(work), complete_cache):
            waits[tile34] = True
    return waits


@lru_cache(maxsize=400000)
def calc_shanten_waits_from_counts(
    counts34: tuple[int, ...],
) -> tuple[int, int, tuple[bool, ...], int]:
    tehai_count = sum(counts34)
    if tehai_count == 0:
        return 8, 0, tuple([False] * 34), 0
    shanten = int(Shanten().calculate_shanten_for_regular_hand(list(counts34)))
    waits34 = tuple(find_regular_waits(counts34)) if shanten == 0 else tuple([False] * 34)
    return shanten, sum(waits34), waits34, tehai_count


@lru_cache(maxsize=400000)
def calc_standard_shanten_from_counts(counts34: tuple[int, ...]) -> int:
    return int(_calculate_standard_shanten(_counts34_to_tile136_ids(counts34)))


def calc_shanten_waits_from_hand(hand: List[str], counts_builder) -> tuple[int, int, List[bool], int]:
    counts = tuple(counts_builder(hand))
    shanten, waits_count, waits, tehai_count = calc_shanten_waits_from_counts(counts)
    return shanten, waits_count, list(waits), tehai_count


def _shape_score(counts: tuple[int, ...]) -> int:
    pair_count = sum(1 for c in counts if c >= 2)
    ryanmen_count = 0
    for base in (0, 9, 18):
        suit = counts[base:base + 9]
        for i in range(1, 7):
            if suit[i] > 0 and suit[i + 1] > 0:
                ryanmen_count += 1
    return ryanmen_count * 8 + pair_count * 4


def _tenpai_live_wait_count(
    counts13: tuple[int, ...],
    visible_counts_local: tuple[int, ...],
) -> int:
    shanten, _waits_count, waits_tiles, _tehai_count = calc_shanten_waits_from_counts(counts13)
    if shanten != 0:
        return 0
    return sum(max(0, 4 - visible_counts_local[t34]) for t34, flag in enumerate(waits_tiles) if flag)


def _best_tenpai_wait_live_after_draw(
    counts14: tuple[int, ...],
    visible_counts_local: tuple[int, ...],
) -> int:
    best_wait_live = 0
    seen_discards: Set[int] = set()
    for discard34, cnt in enumerate(counts14):
        if cnt <= 0 or discard34 in seen_discards:
            continue
        seen_discards.add(discard34)
        after_counts13 = list(counts14)
        after_counts13[discard34] -= 1
        after_counts13_t = tuple(after_counts13)
        after_shanten, _w_cnt, _w_tiles, _tehai = calc_shanten_waits_from_counts(after_counts13_t)
        if after_shanten != 0:
            continue
        wait_live = _tenpai_live_wait_count(after_counts13_t, visible_counts_local)
        if wait_live > best_wait_live:
            best_wait_live = wait_live
    return best_wait_live


def _is_good_shape_draw_for_one_shanten(
    counts13: tuple[int, ...],
    draw_tile34: int,
    visible_counts_local: tuple[int, ...],
) -> bool:
    counts14 = list(counts13)
    counts14[draw_tile34] += 1
    return _best_tenpai_wait_live_after_draw(tuple(counts14), visible_counts_local) > 4


def _make_progress(
    shanten: int,
    waits_count: int,
    waits_tiles: tuple[bool, ...] | List[bool],
    tehai_count: int,
    ukeire_tiles: List[bool],
    good_shape_ukeire_tiles: List[bool],
    improvement_tiles: List[bool],
    visible_counts_local: tuple[int, ...],
) -> NormalProgressInfo:
    return NormalProgressInfo(
        shanten=shanten,
        waits_count=waits_count,
        waits_tiles=list(waits_tiles),
        tehai_count=tehai_count,
        ukeire_type_count=sum(ukeire_tiles),
        ukeire_live_count=sum(max(0, 4 - visible_counts_local[t34]) for t34, flag in enumerate(ukeire_tiles) if flag),
        ukeire_tiles=ukeire_tiles,
        good_shape_ukeire_type_count=sum(good_shape_ukeire_tiles),
        good_shape_ukeire_live_count=sum(max(0, 4 - visible_counts_local[t34]) for t34, flag in enumerate(good_shape_ukeire_tiles) if flag),
        good_shape_ukeire_tiles=good_shape_ukeire_tiles,
        improvement_type_count=sum(improvement_tiles),
        improvement_live_count=sum(max(0, 4 - visible_counts_local[t34]) for t34, flag in enumerate(improvement_tiles) if flag),
        improvement_tiles=improvement_tiles,
    )


@lru_cache(maxsize=400000)
def _summarize_3n1_cached(
    counts_3n1: tuple[int, ...],
    visible_counts_local: tuple[int, ...],
) -> NormalProgressInfo:
    standard_shanten = calc_standard_shanten_from_counts(counts_3n1)
    shanten = standard_shanten
    tehai_count = sum(counts_3n1)
    waits_count = 0
    waits_tiles: tuple[bool, ...] = tuple([False] * 34)
    ukeire_tiles = [False] * 34
    good_shape_ukeire_tiles = [False] * 34
    improvement_tiles = [False] * 34
    base_shape: Optional[int] = _shape_score(counts_3n1) if shanten == 2 else None
    regular_shanten: Optional[int] = None
    use_regular_detail = False
    if shanten <= 1:
        regular_shanten, waits_count, waits_tiles, tehai_count = calc_shanten_waits_from_counts(counts_3n1)
        use_regular_detail = regular_shanten == standard_shanten
        if not use_regular_detail:
            waits_count = 0
            waits_tiles = tuple([False] * 34)

    for tile34 in range(34):
        live = max(0, 4 - visible_counts_local[tile34])
        if live <= 0:
            continue
        counts_3n2 = list(counts_3n1)
        counts_3n2[tile34] += 1

        if shanten > 2:
            best_after_shanten = shanten
            seen_discards: Set[int] = set()
            for discard34, cnt in enumerate(counts_3n2):
                if cnt <= 0 or discard34 in seen_discards:
                    continue
                seen_discards.add(discard34)
                after_counts_3n1 = list(counts_3n2)
                after_counts_3n1[discard34] -= 1
                after_counts_3n1_t = tuple(after_counts_3n1)
                after_shanten = calc_standard_shanten_from_counts(after_counts_3n1_t)
                if after_shanten < best_after_shanten:
                    best_after_shanten = after_shanten
                    if best_after_shanten < shanten:
                        break
            if best_after_shanten < shanten:
                ukeire_tiles[tile34] = True
            continue

        is_good_shape_draw = use_regular_detail and shanten == 1 and _is_good_shape_draw_for_one_shanten(
            counts_3n1,
            tile34,
            visible_counts_local,
        )
        best_after = None
        seen_discards: Set[int] = set()
        for discard34, cnt in enumerate(counts_3n2):
            if cnt <= 0 or discard34 in seen_discards:
                continue
            seen_discards.add(discard34)
            after_counts_3n1 = list(counts_3n2)
            after_counts_3n1[discard34] -= 1
            after_counts_3n1_t = tuple(after_counts_3n1)
            after_shanten = calc_standard_shanten_from_counts(after_counts_3n1_t)
            after_shape = _shape_score(after_counts_3n1_t) if shanten == 2 else 0
            after_wait_live = 0
            after_good_shape = False
            if shanten == 1 and after_shanten == 0 and use_regular_detail:
                after_regular_shanten, _w_cnt, after_waits_tiles, _tehai = calc_shanten_waits_from_counts(after_counts_3n1_t)
                if after_regular_shanten == 0:
                    after_wait_live = sum(
                        max(0, 4 - visible_counts_local[t34]) for t34, flag in enumerate(after_waits_tiles) if flag
                    )
                    after_good_shape = after_wait_live > 4
            key = (-after_shanten, after_wait_live, after_shape)
            if best_after is None or key > best_after[0]:
                best_after = (key, after_shanten, after_shape, after_good_shape)
        if best_after is None:
            continue
        _, after_shanten, after_shape, after_good_shape = best_after
        if after_shanten < shanten:
            ukeire_tiles[tile34] = True
            if shanten == 1 and is_good_shape_draw:
                good_shape_ukeire_tiles[tile34] = True
            elif after_shanten == 0 and after_good_shape:
                good_shape_ukeire_tiles[tile34] = True
        elif shanten == 2 and after_shanten == shanten and base_shape is not None and after_shape > base_shape:
            improvement_tiles[tile34] = True

    return _make_progress(
        shanten, waits_count, waits_tiles, tehai_count,
        ukeire_tiles, good_shape_ukeire_tiles, improvement_tiles, visible_counts_local,
    )


@lru_cache(maxsize=400000)
def _summarize_3n2_cached(
    counts_3n2: tuple[int, ...],
    visible_counts_local: tuple[int, ...],
) -> NormalProgressInfo:
    best = None
    seen_discards: Set[int] = set()
    for discard34, cnt in enumerate(counts_3n2):
        if cnt <= 0 or discard34 in seen_discards:
            continue
        seen_discards.add(discard34)
        counts_3n1 = list(counts_3n2)
        counts_3n1[discard34] -= 1
        progress_3n1 = _summarize_3n1_cached(tuple(counts_3n1), visible_counts_local)
        key = (
            -progress_3n1.shanten,
            progress_3n1.ukeire_live_count,
            progress_3n1.good_shape_ukeire_live_count,
            progress_3n1.improvement_live_count,
            progress_3n1.ukeire_type_count,
            progress_3n1.improvement_type_count,
        )
        if best is None or key > best[0]:
            best = (key, progress_3n1)
    if best is not None:
        return best[1]
    return _summarize_3n1_cached(tuple(counts_3n2), visible_counts_local)


@lru_cache(maxsize=400000)
def analyze_normal_progress_from_counts(
    hand_counts34: tuple[int, ...],
    visible_counts34: tuple[int, ...],
) -> NormalProgressInfo:
    tile_count = sum(hand_counts34)
    mod = tile_count % 3
    if mod == 1:
        return _summarize_3n1_cached(hand_counts34, visible_counts34)
    if mod == 2:
        return _summarize_3n2_cached(hand_counts34, visible_counts34)
    return _make_progress(8, 0, tuple([False] * 34), tile_count, [False] * 34, [False] * 34, [False] * 34, visible_counts34)
