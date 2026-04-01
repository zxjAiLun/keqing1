#!/usr/bin/env python3
"""Compile CSV statistical tables into a runtime-friendly JSON bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from static_tables.builder import build_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description="Build runtime bundle for static mahjong tables")
    parser.add_argument(
        "--source_dir",
        type=Path,
        default=Path("dataset/mahjong_heratu_data/csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/static/tables/mahjong_book_stats.json"),
    )
    args = parser.parse_args()

    bundle = build_bundle(args.source_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {args.output} ({len(bundle['tables'])} tables)")


if __name__ == "__main__":
    main()
