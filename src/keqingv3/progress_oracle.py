from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import time
from typing import Dict, List, Optional, Sequence, Set, Tuple

from mahjong.shanten import Shanten
from riichienv import calculate_shanten as _calculate_standard_shanten


_TILE136_CACHE_SIZE = 20000
_REGULAR_WAITS_CACHE_SIZE = 10000
_STANDARD_SHANTEN_CACHE_SIZE = 20000
_SUMMARY_CACHE_SIZE = 10000
_ACTIVE_PROGRESS_PROFILER: Optional[dict[str, float]] = None


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


def _progress_profiler_add(key: str, delta_s: float) -> None:
    global _ACTIVE_PROGRESS_PROFILER
    if _ACTIVE_PROGRESS_PROFILER is not None:
        _ACTIVE_PROGRESS_PROFILER[key] = _ACTIVE_PROGRESS_PROFILER.get(key, 0.0) + delta_s


def _progress_profiler_inc(key: str, delta: int = 1) -> None:
    global _ACTIVE_PROGRESS_PROFILER
    if _ACTIVE_PROGRESS_PROFILER is not None:
        _ACTIVE_PROGRESS_PROFILER[key] = _ACTIVE_PROGRESS_PROFILER.get(key, 0) + delta


@lru_cache(maxsize=_TILE136_CACHE_SIZE)
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


def _tile_in_obvious_meld(counts34: tuple[int, ...] | List[int], tile34: int) -> bool:
    cnt = counts34[tile34]
    if cnt <= 0:
        return False
    if cnt >= 3:
        return True
    if tile34 >= 27:
        return False
    pos = tile34 % 9
    base = tile34 - pos
    if pos >= 2 and counts34[base + pos - 1] > 0 and counts34[base + pos - 2] > 0:
        return True
    if 1 <= pos <= 7 and counts34[base + pos - 1] > 0 and counts34[base + pos + 1] > 0:
        return True
    if pos <= 6 and counts34[base + pos + 1] > 0 and counts34[base + pos + 2] > 0:
        return True
    return False


def _candidate_discards_no_meld_break(counts34: tuple[int, ...] | List[int]) -> tuple[int, ...]:
    preferred: list[int] = []
    fallback: list[int] = []
    seen: Set[int] = set()
    for tile34, cnt in enumerate(counts34):
        if cnt <= 0 or tile34 in seen:
            continue
        seen.add(tile34)
        fallback.append(tile34)
        if not _tile_in_obvious_meld(counts34, tile34):
            preferred.append(tile34)
    return tuple(preferred) if preferred else tuple(fallback)


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


@lru_cache(maxsize=_REGULAR_WAITS_CACHE_SIZE)
def calc_shanten_waits_from_counts(
    counts34: tuple[int, ...],
) -> tuple[int, int, tuple[bool, ...], int]:
    tehai_count = sum(counts34)
    if tehai_count == 0:
        return 8, 0, tuple([False] * 34), 0
    shanten = int(Shanten().calculate_shanten_for_regular_hand(list(counts34)))
    waits34 = tuple(find_regular_waits(counts34)) if shanten == 0 else tuple([False] * 34)
    return shanten, sum(waits34), waits34, tehai_count


@lru_cache(maxsize=_STANDARD_SHANTEN_CACHE_SIZE)
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
    for discard34 in _candidate_discards_no_meld_break(counts14):
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


@lru_cache(maxsize=_SUMMARY_CACHE_SIZE)
def _summarize_3n1_cached(
    counts_3n1: tuple[int, ...],
    visible_counts_local: tuple[int, ...],
) -> NormalProgressInfo:
    t0 = time.perf_counter()
    _progress_profiler_inc("standard_shanten_calls")
    standard_shanten = calc_standard_shanten_from_counts(counts_3n1)
    _progress_profiler_add("standard_shanten_s", time.perf_counter() - t0)
    shanten = standard_shanten
    tehai_count = sum(counts_3n1)
    waits_count = 0
    waits_tiles: tuple[bool, ...] = tuple([False] * 34)
    ukeire_tiles = [False] * 34
    good_shape_ukeire_tiles = [False] * 34
    improvement_tiles = [False] * 34
    regular_shanten: Optional[int] = None
    use_regular_detail = False
    if shanten <= 1:
        t0 = time.perf_counter()
        _progress_profiler_inc("waits_calls")
        regular_shanten, waits_count, waits_tiles, tehai_count = calc_shanten_waits_from_counts(counts_3n1)
        _progress_profiler_add("waits_s", time.perf_counter() - t0)
        use_regular_detail = regular_shanten == standard_shanten
        if not use_regular_detail:
            waits_count = 0
            waits_tiles = tuple([False] * 34)
    loop_t0 = time.perf_counter()
    for tile34 in range(34):
        live = max(0, 4 - visible_counts_local[tile34])
        if live <= 0:
            continue
        counts_3n2 = list(counts_3n1)
        counts_3n2[tile34] += 1

        if shanten > 2:
            # Fast path above 2-shanten:
            # use the drawn 14-tile standard shanten directly as a lightweight
            # proxy for whether this draw meaningfully improves the hand.
            t0 = time.perf_counter()
            _progress_profiler_inc("standard_shanten_calls")
            draw_shanten = calc_standard_shanten_from_counts(tuple(counts_3n2))
            _progress_profiler_add("standard_shanten_s", time.perf_counter() - t0)
            if draw_shanten < shanten:
                ukeire_tiles[tile34] = True
            continue

        if shanten == 2:
            improves = False
            for discard34 in _candidate_discards_no_meld_break(counts_3n2):
                after_counts_3n1 = list(counts_3n2)
                after_counts_3n1[discard34] -= 1
                after_counts_3n1_t = tuple(after_counts_3n1)
                t0 = time.perf_counter()
                _progress_profiler_inc("standard_shanten_calls")
                after_shanten = calc_standard_shanten_from_counts(after_counts_3n1_t)
                _progress_profiler_add("standard_shanten_s", time.perf_counter() - t0)
                if after_shanten < shanten:
                    improves = True
                    break
            if improves:
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
            t0 = time.perf_counter()
            _progress_profiler_inc("standard_shanten_calls")
            after_shanten = calc_standard_shanten_from_counts(after_counts_3n1_t)
            _progress_profiler_add("standard_shanten_s", time.perf_counter() - t0)
            after_wait_live = 0
            after_good_shape = False
            if shanten == 1 and after_shanten == 0 and use_regular_detail:
                t0 = time.perf_counter()
                _progress_profiler_inc("waits_calls")
                after_regular_shanten, _w_cnt, after_waits_tiles, _tehai = calc_shanten_waits_from_counts(after_counts_3n1_t)
                _progress_profiler_add("waits_s", time.perf_counter() - t0)
                if after_regular_shanten == 0:
                    after_wait_live = sum(
                        max(0, 4 - visible_counts_local[t34]) for t34, flag in enumerate(after_waits_tiles) if flag
                    )
                    after_good_shape = after_wait_live > 4
            key = (-after_shanten, after_wait_live)
            if best_after is None or key > best_after[0]:
                best_after = (key, after_shanten, after_good_shape)
        if best_after is None:
            continue
        _, after_shanten, after_good_shape = best_after
        if after_shanten < shanten:
            ukeire_tiles[tile34] = True
            if shanten == 1 and is_good_shape_draw:
                good_shape_ukeire_tiles[tile34] = True
            elif after_shanten == 0 and after_good_shape:
                good_shape_ukeire_tiles[tile34] = True
    _progress_profiler_add("ukeire_improvement_s", time.perf_counter() - loop_t0)

    return _make_progress(
        shanten, waits_count, waits_tiles, tehai_count,
        ukeire_tiles, good_shape_ukeire_tiles, improvement_tiles, visible_counts_local,
    )


@lru_cache(maxsize=_SUMMARY_CACHE_SIZE)
def _summarize_3n2_cached(
    counts_3n2: tuple[int, ...],
    visible_counts_local: tuple[int, ...],
) -> NormalProgressInfo:
    best = None
    loop_t0 = time.perf_counter()
    for discard34 in _candidate_discards_no_meld_break(counts_3n2):
        counts_3n1 = list(counts_3n2)
        counts_3n1[discard34] -= 1
        counts_3n1_t = tuple(counts_3n1)
        progress_3n1 = _summarize_3n1_cached(counts_3n1_t, visible_counts_local)
        shape_score = _shape_score(counts_3n1_t) if progress_3n1.shanten == 2 else 0
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
    _progress_profiler_add("ukeire_improvement_s", time.perf_counter() - loop_t0)
    if best is not None:
        return best[1]
    return _summarize_3n1_cached(tuple(counts_3n2), visible_counts_local)


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


def analyze_normal_progress_with_timings(
    hand_counts34: tuple[int, ...],
    visible_counts34: tuple[int, ...],
) -> tuple[NormalProgressInfo, dict[str, float]]:
    global _ACTIVE_PROGRESS_PROFILER
    timings = {
        "standard_shanten_s": 0.0,
        "waits_s": 0.0,
        "ukeire_improvement_s": 0.0,
        "shape_s": 0.0,
        "standard_shanten_calls": 0,
        "waits_calls": 0,
        "standard_shanten_hits": 0,
        "standard_shanten_misses": 0,
        "waits_hits": 0,
        "waits_misses": 0,
    }
    std_before = calc_standard_shanten_from_counts.cache_info()
    waits_before = calc_shanten_waits_from_counts.cache_info()
    prev = _ACTIVE_PROGRESS_PROFILER
    _ACTIVE_PROGRESS_PROFILER = timings
    try:
        progress = analyze_normal_progress_from_counts(hand_counts34, visible_counts34)
    finally:
        _ACTIVE_PROGRESS_PROFILER = prev
    std_after = calc_standard_shanten_from_counts.cache_info()
    waits_after = calc_shanten_waits_from_counts.cache_info()
    timings["standard_shanten_hits"] = std_after.hits - std_before.hits
    timings["standard_shanten_misses"] = std_after.misses - std_before.misses
    timings["waits_hits"] = waits_after.hits - waits_before.hits
    timings["waits_misses"] = waits_after.misses - waits_before.misses
    return progress, timings


def clear_progress_caches() -> None:
    _counts34_to_tile136_ids.cache_clear()
    calc_shanten_waits_from_counts.cache_clear()
    calc_standard_shanten_from_counts.cache_clear()
    _summarize_3n1_cached.cache_clear()
    _summarize_3n2_cached.cache_clear()
