from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission-zip",
        required=True,
        help="mjai submission zip. Usually 1 zip for all 4 players.",
    )
    parser.add_argument("--logs-dir", default="artifacts/mjai_sim_logs")
    parser.add_argument(
        "--output-json",
        required=True,
        help="Output tenhou.net/6 JSON file path.",
    )
    parser.add_argument("--timeout", type=float, default=10.0)
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--seed-end", type=int, default=2000)
    parser.add_argument("--tsumogiri-threshold", type=float, default=0.9)
    parser.add_argument("--fail-on-high-tsumogiri", action="store_true")
    args = parser.parse_args()

    # Import locally so `pip install`/venv differences won't break CLI parsing.
    from mjai import Simulator  # type: ignore

    logs_dir = Path(args.logs_dir)
    logs_dir.mkdir(parents=True, exist_ok=True)

    submissions = [Path(args.submission_zip)] * 4
    sim: Simulator = Simulator(
        submissions=submissions,
        logs_dir=logs_dir,
        timeout=args.timeout,
        seed=(args.seed_start, args.seed_end),
    )
    sim.run()

    mjai_log_path = logs_dir / "mjai_log.json"
    if not mjai_log_path.exists():
        raise RuntimeError(f"missing simulator output: {mjai_log_path}")

    # Convert mjai_log.json (JSONL) -> tenhou6 JSON.
    from tools.mjai_jsonl_to_tenhou6 import convert_mjai_jsonl_to_tenhou6, load_jsonl

    events = load_jsonl(mjai_log_path)
    out = convert_mjai_jsonl_to_tenhou6(events)

    # Gate metric: monitor whether all players are almost always tsumogiri.
    stats: Dict[int, Dict[str, float]] = {}
    for pid in range(4):
        stats[pid] = {"dahai": 0.0, "tsumogiri": 0.0}
    for ev in events:
        if ev.get("type") != "dahai":
            continue
        actor = ev.get("actor")
        if not isinstance(actor, int) or actor not in stats:
            continue
        stats[actor]["dahai"] += 1.0
        if bool(ev.get("tsumogiri")):
            stats[actor]["tsumogiri"] += 1.0
    player_stats = {}
    high_players = []
    for pid, st in stats.items():
        total = st["dahai"]
        tsum = st["tsumogiri"]
        rate = (tsum / total) if total > 0 else 0.0
        player_stats[str(pid)] = {
            "dahai_count": int(total),
            "tsumogiri_count": int(tsum),
            "tsumogiri_rate": rate,
        }
        if total > 0 and rate >= args.tsumogiri_threshold:
            high_players.append(pid)
    gate_summary = {
        "tsumogiri_threshold": args.tsumogiri_threshold,
        "players": player_stats,
        "high_tsumogiri_players": high_players,
        "all_players_high": len(high_players) == 4,
    }
    gate_path = logs_dir / "policy_debug_summary.json"
    gate_path.write_text(json.dumps(gate_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.fail_on_high_tsumogiri and len(high_players) == 4:
        raise RuntimeError(f"all players tsumogiri_rate >= {args.tsumogiri_threshold}, see {gate_path}")

    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")

    print(
        json.dumps(
            {
                "ok": True,
                "logs_dir": str(logs_dir),
                "mjai_log": str(mjai_log_path),
                "tenhou6_out": str(output_path),
                "policy_debug_summary": str(gate_path),
                "high_tsumogiri_players": high_players,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

