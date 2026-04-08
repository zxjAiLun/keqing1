from __future__ import annotations

import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import keqing_core
from keqingv3.progress_oracle import (
    analyze_normal_progress_from_counts,
    calc_standard_shanten_from_counts,
    clear_progress_caches,
)


def _random_counts(total_tiles: int, rng: random.Random) -> tuple[int, ...]:
    counts = [0] * 34
    remaining = [4] * 34
    tiles = 0
    while tiles < total_tiles:
        idx = rng.randrange(34)
        if remaining[idx] <= 0:
            continue
        remaining[idx] -= 1
        counts[idx] += 1
        tiles += 1
    return tuple(counts)


def _random_visible_counts(hand_counts: tuple[int, ...], rng: random.Random) -> tuple[int, ...]:
    visible = list(hand_counts)
    for idx, cnt in enumerate(visible):
        extra_max = max(0, 4 - cnt)
        if extra_max > 0:
            visible[idx] += rng.randrange(extra_max + 1)
    return tuple(visible)


def _bench_standard_shanten(samples: list[tuple[int, ...]], use_rust: bool) -> float:
    keqing_core.enable_rust(use_rust)
    clear_progress_caches()
    start = time.perf_counter()
    for sample in samples:
        calc_standard_shanten_from_counts(sample)
    return time.perf_counter() - start


def _bench_progress(samples: list[tuple[tuple[int, ...], tuple[int, ...]]], use_rust: bool) -> float:
    keqing_core.enable_rust(use_rust)
    clear_progress_caches()
    start = time.perf_counter()
    for hand_counts, visible_counts in samples:
        analyze_normal_progress_from_counts(hand_counts, visible_counts)
    return time.perf_counter() - start


def main() -> None:
    if not keqing_core.is_available():
        raise RuntimeError(
            "keqing_core native extension is not available. Build/install the wheel before benchmarking."
        )
    rng = random.Random(20260408)
    shanten_13 = [_random_counts(13, rng) for _ in range(400)]
    shanten_14 = [_random_counts(14, rng) for _ in range(400)]
    shanten_samples = shanten_13 + shanten_14
    progress_samples_13 = [
        (hand_counts, _random_visible_counts(hand_counts, rng))
        for hand_counts in [_random_counts(13, rng) for _ in range(120)]
    ]
    progress_samples_14 = [
        (hand_counts, _random_visible_counts(hand_counts, rng))
        for hand_counts in [_random_counts(14, rng) for _ in range(120)]
    ]
    progress_samples = progress_samples_13 + progress_samples_14

    std_py = [_bench_standard_shanten(shanten_samples, use_rust=False) for _ in range(5)]
    std_rs = [_bench_standard_shanten(shanten_samples, use_rust=True) for _ in range(5)]
    prog_py = [_bench_progress(progress_samples, use_rust=False) for _ in range(5)]
    prog_rs = [_bench_progress(progress_samples, use_rust=True) for _ in range(5)]
    prog13_py = [_bench_progress(progress_samples_13, use_rust=False) for _ in range(5)]
    prog13_rs = [_bench_progress(progress_samples_13, use_rust=True) for _ in range(5)]
    prog14_py = [_bench_progress(progress_samples_14, use_rust=False) for _ in range(5)]
    prog14_rs = [_bench_progress(progress_samples_14, use_rust=True) for _ in range(5)]

    print("calc_standard_shanten_from_counts")
    print(f"  python mean: {statistics.mean(std_py):.6f}s")
    print(f"  rust   mean: {statistics.mean(std_rs):.6f}s")
    print(f"  speedup: {statistics.mean(std_py) / statistics.mean(std_rs):.3f}x")
    print()
    print("analyze_normal_progress_from_counts")
    print(f"  python mean: {statistics.mean(prog_py):.6f}s")
    print(f"  rust   mean: {statistics.mean(prog_rs):.6f}s")
    print(f"  speedup: {statistics.mean(prog_py) / statistics.mean(prog_rs):.3f}x")
    print()
    print("analyze_normal_progress_from_counts (3n+1 only)")
    print(f"  python mean: {statistics.mean(prog13_py):.6f}s")
    print(f"  rust   mean: {statistics.mean(prog13_rs):.6f}s")
    print(f"  speedup: {statistics.mean(prog13_py) / statistics.mean(prog13_rs):.3f}x")
    print()
    print("analyze_normal_progress_from_counts (3n+2 only)")
    print(f"  python mean: {statistics.mean(prog14_py):.6f}s")
    print(f"  rust   mean: {statistics.mean(prog14_rs):.6f}s")
    print(f"  speedup: {statistics.mean(prog14_py) / statistics.mean(prog14_rs):.3f}x")


if __name__ == "__main__":
    main()
