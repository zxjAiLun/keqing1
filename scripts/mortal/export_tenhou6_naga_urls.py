#!/usr/bin/env python3
"""Export NAGA-friendly tenhou.net/6 URLs from Tenhou6 hanchan JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_TITLE_INFO = ["玉の間四人南", "2026/6/13 01:44:27"]
DEFAULT_RATING_INFO = "[125,60,-5,-240],[125,60,-5,-195],[125,60,-5,-195],[125,60,-5,-180],1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export NAGA-friendly tenhou.net/6 URLs")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-json-dir", type=Path, required=True)
    parser.add_argument("--output-urls", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--title-disp", default=DEFAULT_TITLE_INFO[0])
    parser.add_argument("--title-date", default=DEFAULT_TITLE_INFO[1])
    parser.add_argument("--rating-info", default=DEFAULT_RATING_INFO)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _naga_payload(source: dict[str, Any], kyoku_log: list[Any], *, title_disp: str, title_date: str, rating_info: str) -> dict[str, Any]:
    return {
        "title": [[title_disp, title_date], rating_info],
        "name": list(source.get("name", ["A", "B", "C", "D"]))[:4],
        "rule": {"disp": title_disp, "aka53": 1, "aka52": 1, "aka51": 1},
        "log": [kyoku_log],
    }


def _compact_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def main() -> None:
    args = _parse_args()
    sources = sorted(args.input_dir.glob("*.tenhou6.json"))
    if not sources:
        raise SystemExit(f"no .tenhou6.json files found in {args.input_dir}")

    args.output_json_dir.mkdir(parents=True, exist_ok=True)
    args.output_urls.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    url_lines: list[str] = []
    total_kyoku = 0
    for game_index, source_path in enumerate(sources):
        source = json.loads(source_path.read_text(encoding="utf-8"))
        logs = list(source.get("log") or [])
        for kyoku_index, kyoku_log in enumerate(logs):
            payload = _naga_payload(
                source,
                kyoku_log,
                title_disp=str(args.title_disp),
                title_date=str(args.title_date),
                rating_info=str(args.rating_info),
            )
            output_path = args.output_json_dir / f"game_{game_index:05d}_kyoku_{kyoku_index:02d}.naga.tenhou6.json"
            if args.overwrite or not output_path.exists():
                output_path.write_text(_compact_json(payload) + "\n", encoding="utf-8")
            url = "https://tenhou.net/6/#json=" + _compact_json(payload)
            row = {
                "source_tenhou6_path": str(source_path),
                "naga_tenhou6_path": str(output_path),
                "url": url,
                "game_index": game_index,
                "kyoku_index": kyoku_index,
                "player_names": payload["name"],
            }
            rows.append(row)
            url_lines.append(url)
            total_kyoku += 1
        if (game_index + 1) % 100 == 0 or game_index + 1 == len(sources):
            print(f"processed {game_index + 1}/{len(sources)} hanchans, {total_kyoku} kyoku", flush=True)

    args.output_urls.write_text("\n".join(url_lines) + "\n", encoding="utf-8")
    with args.manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")

    print(f"saved NAGA JSON: {args.output_json_dir}", flush=True)
    print(f"saved NAGA URLs: {args.output_urls}", flush=True)
    print(f"saved manifest: {args.manifest}", flush=True)
    print(f"total kyoku: {total_kyoku}", flush=True)


if __name__ == "__main__":
    main()
