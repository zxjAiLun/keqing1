#!/usr/bin/env python3
"""
将 mjai_log.json 分割为多个 tenhou6 JSON 文件

用法:
    # 按局分割（每个 start_kyoku -> end_kyoku 为一局）
    python -m tools.split_mjai_log_to_tenhou6 \
        --input artifacts/mjai_sim_logs/xxx/mjai_log.json \
        --output-dir artifacts/mjai_sim_logs/xxx/tenhou6_split \
        --split-by kyoku

    # 按完整对局分割（start_game -> end_game）
    python -m tools.split_mjai_log_to_tenhou6 \
        --input artifacts/mjai_sim_logs/xxx/mjai_log.json \
        --output-dir artifacts/mjai_sim_logs/xxx/tenhou6_split \
        --split-by game
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from tools.mjai_jsonl_to_tenhou6 import convert_mjai_jsonl_to_tenhou6


def split_events_by_kyoku(events: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Split mjai events by kyoku (start_kyoku to end_kyoku).
    Each kyoku is a complete tenhou6 JSON with single log entry.

    Returns:
        List of event lists, one per kyoku.
    """
    kyokus: List[List[Dict[str, Any]]] = []
    current_kyoku: List[Dict[str, Any]] = []

    for ev in events:
        current_kyoku.append(ev)
        if ev.get("type") == "end_kyoku":
            kyokus.append(current_kyoku)
            current_kyoku = []

    if current_kyoku:
        kyokus.append(current_kyoku)

    return kyokus


def split_events_by_game(events: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """
    Split mjai events by game (start_game to end_game).

    Returns:
        List of event lists, one per game.
    """
    games: List[List[Dict[str, Any]]] = []
    current_game: List[Dict[str, Any]] = []

    for ev in events:
        current_game.append(ev)
        if ev.get("type") == "end_game":
            games.append(current_game)
            current_game = []

    if current_game:
        games.append(current_game)

    return games


def main() -> None:
    parser = argparse.ArgumentParser(description="Split mjai_log.json into multiple tenhou6 JSON files")
    parser.add_argument("--input", "-i", required=True, help="Path to mjai_log.json (JSONL format)")
    parser.add_argument("--output-dir", "-o", required=True, help="Output directory for tenhou6 JSON files")
    parser.add_argument("--split-by", "-s", choices=["kyoku", "game"], default="kyoku",
                        help="Split by 'kyoku' (each round) or 'game' (full game)")
    parser.add_argument("--prefix", "-p", default="game", help="Filename prefix for output files")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    events: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))

    if args.split_by == "kyoku":
        splits = split_events_by_kyoku(events)
        print(f"Found {len(splits)} kyoku in {input_path}")
        label = "Kyoku"
        prefix = args.prefix
    else:
        splits = split_events_by_game(events)
        print(f"Found {len(splits)} games in {input_path}")
        label = "Game"
        prefix = args.prefix

    for i, split_events in enumerate(splits):
        try:
            tenhou6_data = convert_mjai_jsonl_to_tenhou6(split_events)

            num_kyoku = len(tenhou6_data.get("log", []))
            if num_kyoku == 0:
                print(f"  {label} {i}: 跳过 (空)")
                continue

            output_path = output_dir / f"{prefix}_{i:04d}.json"
            with output_path.open("w", encoding="utf-8") as f:
                json.dump(tenhou6_data, f, ensure_ascii=False)

            print(f"  {label} {i}: {num_kyoku} 局 -> {output_path.name}")

        except Exception as e:
            print(f"  {label} {i}: ERROR - {e}")

    print(f"\nDone! Output: {output_dir}")


if __name__ == "__main__":
    main()
