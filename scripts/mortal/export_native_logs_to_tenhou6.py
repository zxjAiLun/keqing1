#!/usr/bin/env python3
"""Batch convert native libriichi mjai logs into Tenhou6 JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from tools.mjai_jsonl_to_tenhou6 import convert_mjai_jsonl_to_tenhou6, load_mjai_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Tenhou6 JSON files from native mjai logs")
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _source_key(path: Path) -> tuple[int, str]:
    try:
        seed = int(path.name.split("_", 1)[0])
    except ValueError:
        seed = 0
    return seed, path.name


def _convert_one(source: Path, output: Path) -> dict[str, Any]:
    events = load_mjai_jsonl(source)
    tenhou6 = convert_mjai_jsonl_to_tenhou6(events)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(tenhou6, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    return {
        "source_log": str(source),
        "tenhou6_path": str(output),
        "player_names": list(tenhou6.get("name", [])),
        "kyoku_count": len(tenhou6.get("log", [])),
    }


def main() -> None:
    args = _parse_args()
    sources = sorted(args.log_dir.glob("*.json.gz"), key=_source_key)
    if args.limit and args.limit > 0:
        sources = sources[: args.limit]
    if not sources:
        raise SystemExit(f"no .json.gz logs found in {args.log_dir}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.manifest or args.output_dir.parent / "tenhou6_manifest.jsonl"

    rows: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        output = args.output_dir / f"game_{index:05d}_{source.stem.removesuffix('.json')}.tenhou6.json"
        if output.exists() and not args.overwrite:
            tenhou6 = json.loads(output.read_text(encoding="utf-8"))
            row = {
                "source_log": str(source),
                "tenhou6_path": str(output),
                "player_names": list(tenhou6.get("name", [])),
                "kyoku_count": len(tenhou6.get("log", [])),
            }
        else:
            row = _convert_one(source, output)
        rows.append(row)
        if (index + 1) % 100 == 0 or index + 1 == len(sources):
            print(f"converted {index + 1}/{len(sources)}", flush=True)

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"saved tenhou6: {args.output_dir}", flush=True)
    print(f"saved manifest: {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
