#!/usr/bin/env python3
"""Download categorized wait CSVs into matching converted_mjai subdirectories.

Usage:
    uv run python scripts/download_wait_categories.py
    uv run python scripts/download_wait_categories.py --wait-dir dataset/wait --output-base artifacts/converted_mjai
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from download_and_convert import process_csv


def collect_category_csvs(wait_dir: Path) -> list[tuple[str, Path]]:
    results: list[tuple[str, Path]] = []
    if not wait_dir.exists():
        return results
    for category_dir in sorted(p for p in wait_dir.iterdir() if p.is_dir()):
        for csv_path in sorted(category_dir.glob("*.csv")):
            results.append((category_dir.name, csv_path))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download replay CSVs under dataset/wait/<category>/ into matching output folders.",
    )
    parser.add_argument(
        "--wait-dir",
        type=Path,
        default=Path("dataset/wait"),
        help="Input category directory root (default: dataset/wait)",
    )
    parser.add_argument(
        "--output-base",
        type=Path,
        default=Path("artifacts/converted_mjai"),
        help="Output base directory (default: artifacts/converted_mjai)",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=1.5,
        help="Delay between downloads in seconds (default: 1.5)",
    )
    args = parser.parse_args()

    tasks = collect_category_csvs(args.wait_dir)
    if not tasks:
        print(f"ERROR: no category CSVs found under {args.wait_dir}")
        return 1

    total_success = 0
    total_skipped = 0
    total_failed = 0

    print(f"Found {len(tasks)} CSV file(s) under {args.wait_dir}")
    for category, csv_path in tasks:
        output_dir = args.output_base / category
        print(f"\n=== {category} / {csv_path.name} -> {output_dir} ===")
        success, skipped, failed = process_csv(csv_path, output_dir, args.delay)
        total_success += success
        total_skipped += skipped
        total_failed += failed

    print(
        f"\n=== Total: {total_success} downloaded+converted, "
        f"{total_skipped} skipped, {total_failed} failed ==="
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
