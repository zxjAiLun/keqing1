#!/usr/bin/env python3
"""Audit V1 model_v4 synthetic log pools before training."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Iterable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.eval_metrics import parse_rank_points
from scripts.mortal.stat_report import build_stat_report

DEFAULT_OUTPUT_ROOT = Path("artifacts/experiments/v4_synthetic_2026_06")
DEFAULT_DATA_ROOT = DEFAULT_OUTPUT_ROOT / "V1_data"

POOL_SPECS = {
    "selfplay_v4_legacy_3000h": {
        "relative_dir": "selfplay_v4_12000h_1v3",
        "file_pattern": "*_a.json.gz",
        "expected_games": 3000,
        "train_labels": ("challenger", "champion"),
        "all_labels": ("challenger", "champion"),
        "expected_trainable_seats_per_game": 4,
    },
    "selfplay_v4_unique_9000h": {
        "relative_dir": "selfplay_v4_unique_9000h",
        "file_pattern": "*.json.gz",
        "expected_games": 9000,
        "train_labels": ("v4",),
        "all_labels": ("v4",),
        "expected_trainable_seats_per_game": 4,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_ROOT / "V1_v4_synthetic_warmstart_2026_06" / "dataset_audit.json")
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--rank-points", default="90,45,0,-135")
    parser.add_argument("--min-coverage", type=float, default=0.95)
    parser.add_argument("--allow-partial", action="store_true", help="Do not fail when expected full-pool coverage is below --min-coverage.")
    return parser.parse_args()


LOG_NAME_RE = re.compile(r"^(?P<seed>\d+)_(?P<key>\d+)(?:_[a-d])?\.json\.gz$")


def iter_logs(log_dir: Path, pattern: str) -> list[Path]:
    return sorted(log_dir.glob(pattern))


def read_start_names(path: Path) -> list[str]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        start = json.loads(next(handle))
    names = start.get("names")
    if not isinstance(names, list) or len(names) != 4:
        raise ValueError(f"invalid start names in {path}: {names!r}")
    return [str(name) for name in names]


def count_trainable_seats(names: Iterable[str], train_labels: set[str]) -> int:
    return sum(1 for name in names if name in train_labels)


def canonical_log_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            event = json.loads(line)
            event.pop("meta", None)
            if event.get("type") == "start_game":
                event.pop("names", None)
            encoded = json.dumps(event, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
            digest.update(encoded.encode("utf-8"))
            digest.update(b"\n")
    return digest.hexdigest()


def parse_seed_key(path: Path) -> tuple[int, int]:
    match = LOG_NAME_RE.fullmatch(path.name)
    if match is None:
        raise ValueError(f"unrecognized native log filename: {path.name}")
    return int(match.group("seed")), int(match.group("key"))


def audit_pool(
    *,
    pool_id: str,
    data_root: Path,
    rank_points: tuple[float, float, float, float],
    mortal_root: Path,
) -> dict[str, Any]:
    spec = POOL_SPECS[pool_id]
    pool_dir = data_root / str(spec["relative_dir"])
    log_dir = pool_dir / "logs"
    file_pattern = str(spec["file_pattern"])
    expected_games = int(spec["expected_games"])
    train_labels = tuple(str(label) for label in spec["train_labels"])
    all_labels = tuple(str(label) for label in spec["all_labels"])
    expected_trainable_seats_per_game = int(spec["expected_trainable_seats_per_game"])
    train_label_set = set(train_labels)
    raw_files = iter_logs(log_dir, "*.json.gz")
    files = iter_logs(log_dir, file_pattern)

    malformed: list[dict[str, str]] = []
    start_name_counts: Counter[str] = Counter()
    seat_patterns: Counter[str] = Counter()
    canonical_hashes: dict[str, list[str]] = {}
    seed_keys: dict[tuple[int, int], list[str]] = {}
    trainable_seats = 0
    for path in files:
        try:
            names = read_start_names(path)
            digest = canonical_log_hash(path)
            seed_key = parse_seed_key(path)
        except Exception as exc:  # noqa: BLE001
            malformed.append({"path": str(path), "error": str(exc)})
            continue
        start_name_counts.update(names)
        seat_patterns.update(["|".join(names)])
        trainable_seats += count_trainable_seats(names, train_label_set)
        canonical_hashes.setdefault(digest, []).append(str(path))
        seed_keys.setdefault(seed_key, []).append(str(path))

    expected_trainable_seats = expected_games * expected_trainable_seats_per_game
    file_coverage = len(files) / expected_games if expected_games else 0.0
    trainable_coverage = trainable_seats / expected_trainable_seats if expected_trainable_seats else 0.0

    stat_report: dict[str, Any] | None = None
    stat_error: str | None = None
    if files and not malformed and file_pattern == "*.json.gz":
        try:
            stat_report = build_stat_report(
                log_dir=log_dir,
                players={label: label for label in all_labels},
                mortal_root=mortal_root,
                rank_pts=rank_points,
                rank_points_profile="custom",
            )
        except Exception as exc:  # noqa: BLE001
            stat_error = str(exc)

    return {
        "pool_id": pool_id,
        "pool_dir": str(pool_dir),
        "log_dir": str(log_dir),
        "file_pattern": file_pattern,
        "expected_games": expected_games,
        "raw_file_count": len(raw_files),
        "file_count": len(files),
        "file_coverage": file_coverage,
        "train_labels": list(train_labels),
        "expected_trainable_seats_per_game": expected_trainable_seats_per_game,
        "expected_trainable_v4_seats": expected_trainable_seats,
        "trainable_v4_seat_count": trainable_seats,
        "trainable_v4_seat_coverage": trainable_coverage,
        "malformed_count": len(malformed),
        "malformed_examples": malformed[:10],
        "canonical_unique_count": len(canonical_hashes),
        "canonical_duplicate_count": sum(len(paths) - 1 for paths in canonical_hashes.values()),
        "canonical_duplicate_rate": (
            (len(files) - len(canonical_hashes)) / len(files) if files else 0.0
        ),
        "canonical_duplicate_examples": [
            {"hash": digest, "paths": paths[:5]}
            for digest, paths in canonical_hashes.items()
            if len(paths) > 1
        ][:10],
        "canonical_hash_paths": [
            [digest, paths[0]] for digest, paths in canonical_hashes.items()
        ],
        "seed_key_count": len(seed_keys),
        "seed_key_duplicate_count": sum(len(paths) - 1 for paths in seed_keys.values()),
        "seed_keys": [[seed, key] for seed, key in sorted(seed_keys)],
        "start_name_counts": dict(sorted(start_name_counts.items())),
        "seat_pattern_examples": dict(seat_patterns.most_common(10)),
        "stat_error": stat_error,
        "stat_report": stat_report,
    }


def main() -> None:
    args = parse_args()
    rank_points = parse_rank_points(args.rank_points)
    pools = [
        audit_pool(
            pool_id=pool_id,
            data_root=args.data_root,
            rank_points=rank_points,
            mortal_root=args.mortal_root,
        )
        for pool_id in POOL_SPECS
    ]
    total_expected_games = sum(int(pool["expected_games"]) for pool in pools)
    total_file_count = sum(int(pool["file_count"]) for pool in pools)
    total_expected_trainable = sum(int(pool["expected_trainable_v4_seats"]) for pool in pools)
    total_trainable = sum(int(pool["trainable_v4_seat_count"]) for pool in pools)
    malformed_count = sum(int(pool["malformed_count"]) for pool in pools)
    canonical_hash_owners: dict[str, list[str]] = {}
    for pool in pools:
        for digest, path in pool.pop("canonical_hash_paths"):
            canonical_hash_owners.setdefault(str(digest), []).append(str(path))
    canonical_unique_count = len(canonical_hash_owners)
    canonical_duplicate_count = total_file_count - canonical_unique_count
    canonical_duplicate_examples = [
        {"hash": digest, "paths": paths[:5]}
        for digest, paths in canonical_hash_owners.items()
        if len(paths) > 1
    ][:10]
    seed_key_owners: dict[tuple[int, int], list[str]] = {}
    for pool in pools:
        for seed, key in pool.pop("seed_keys"):
            seed_key_owners.setdefault((int(seed), int(key)), []).append(str(pool["pool_id"]))
    cross_pool_seed_overlaps = {
        f"{seed}_{key}": owners
        for (seed, key), owners in seed_key_owners.items()
        if len(owners) > 1
    }
    trainable_coverage = total_trainable / total_expected_trainable if total_expected_trainable else 0.0
    file_coverage = total_file_count / total_expected_games if total_expected_games else 0.0
    integrity_passed = (
        malformed_count == 0
        and canonical_duplicate_count == 0
        and not cross_pool_seed_overlaps
    )
    coverage_passed = (
        file_coverage >= float(args.min_coverage)
        and trainable_coverage >= float(args.min_coverage)
    )
    passed = integrity_passed and (coverage_passed or bool(args.allow_partial))
    report = {
        "schema": "keqing.mortal.v1_v4_synthetic_dataset_audit.v2",
        "data_root": str(args.data_root),
        "rank_points": [float(value) for value in rank_points],
        "min_coverage": float(args.min_coverage),
        "allow_partial": bool(args.allow_partial),
        "summary": {
            "expected_games": total_expected_games,
            "file_count": total_file_count,
            "file_coverage": file_coverage,
            "expected_trainable_v4_seats": total_expected_trainable,
            "trainable_v4_seat_count": total_trainable,
            "trainable_v4_seat_coverage": trainable_coverage,
            "malformed_count": malformed_count,
            "canonical_unique_count": canonical_unique_count,
            "canonical_duplicate_count": canonical_duplicate_count,
            "canonical_duplicate_examples": canonical_duplicate_examples,
            "cross_pool_seed_overlap_count": len(cross_pool_seed_overlaps),
            "cross_pool_seed_overlap_examples": dict(list(cross_pool_seed_overlaps.items())[:10]),
            "integrity_passed": integrity_passed,
            "coverage_passed": coverage_passed,
            "passed": passed,
        },
        "pools": pools,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2), flush=True)
    if not report["summary"]["passed"]:
        raise SystemExit("V1 synthetic dataset audit failed")


if __name__ == "__main__":
    main()
