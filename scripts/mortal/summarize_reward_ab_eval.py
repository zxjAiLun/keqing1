#!/usr/bin/env python3
"""Summarize matched-seed native evaluations for the reward-semantics A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


RANK_POINTS = (90.0, 45.0, 0.0, -135.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--expected-games", type=int, default=250)
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _raw_row(label: str, raw: dict[str, Any]) -> dict[str, Any]:
    games = int(raw["game"])
    rounds = int(raw["round"])
    rank_counts = [int(raw[f"rank_{rank}"]) for rank in range(1, 5)]
    rank_pt = sum(count * point for count, point in zip(rank_counts, RANK_POINTS, strict=True))
    if games <= 0 or rounds <= 0:
        raise ValueError(f"invalid game/round counts for {label}: {raw}")
    return {
        "label": label,
        "games": games,
        "rounds": rounds,
        "rank_counts": rank_counts,
        "avg_rank": sum((rank + 1) * count for rank, count in enumerate(rank_counts)) / games,
        "avg_rank_pt": rank_pt / games,
        "avg_score_delta": float(raw["point"]) / games,
        "agari_rate": int(raw["agari"]) / rounds,
        "houjuu_rate": int(raw["houjuu"]) / rounds,
        "fuuro_rate": int(raw["fuuro"]) / rounds,
        "riichi_rate": int(raw["riichi"]) / rounds,
        "ryukyoku_rate": int(raw["ryukyoku"]) / rounds,
        "tobi_rate": int(raw["tobi"]) / games,
        "agari": int(raw["agari"]),
        "houjuu": int(raw["houjuu"]),
        "fuuro": int(raw["fuuro"]),
        "riichi": int(raw["riichi"]),
    }


def _aggregate(rows: list[dict[str, Any]], label: str) -> dict[str, Any]:
    if not rows:
        raise ValueError(f"no rows for {label}")
    games = sum(int(row["games"]) for row in rows)
    rounds = sum(int(row["rounds"]) for row in rows)
    rank_counts = [sum(int(row["rank_counts"][index]) for row in rows) for index in range(4)]
    agari = sum(int(row["agari"]) for row in rows)
    houjuu = sum(int(row["houjuu"]) for row in rows)
    fuuro = sum(int(row["fuuro"]) for row in rows)
    riichi = sum(int(row["riichi"]) for row in rows)
    return {
        "label": label,
        "runs": len(rows),
        "games": games,
        "rounds": rounds,
        "rank_counts": rank_counts,
        "avg_rank": sum((rank + 1) * count for rank, count in enumerate(rank_counts)) / games,
        "avg_rank_pt": sum(
            count * point for count, point in zip(rank_counts, RANK_POINTS, strict=True)
        ) / games,
        "avg_score_delta": sum(float(row["avg_score_delta"]) * int(row["games"]) for row in rows) / games,
        "agari_rate": agari / rounds,
        "houjuu_rate": houjuu / rounds,
        "fuuro_rate": fuuro / rounds,
        "riichi_rate": riichi / rounds,
        "ryukyoku_rate": sum(int(row["ryukyoku_rate"] * row["rounds"]) for row in rows) / rounds,
        "tobi_rate": sum(float(row["tobi_rate"]) * int(row["games"]) for row in rows) / games,
    }


def _fmt_pct(value: float) -> str:
    return f"{value:.2%}"


def main() -> None:
    args = parse_args()
    run_dirs = sorted(path for path in args.eval_root.glob("F_G_*") if path.is_dir())
    if not run_dirs:
        raise SystemExit(f"no F_G_* evaluation directories under {args.eval_root}")

    per_seed: list[dict[str, Any]] = []
    source_checks: list[dict[str, Any]] = []
    for run_dir in run_dirs:
        metrics = _read_json(run_dir / "metrics.json")
        detailed = _read_json(run_dir / "detailed_stats.json")
        log_count = len(list((run_dir / "logs").glob("*.json.gz")))
        if log_count != args.expected_games:
            raise ValueError(f"{run_dir}: expected {args.expected_games} logs, got {log_count}")
        run = metrics.get("run", {})
        if run.get("games") != args.expected_games:
            raise ValueError(f"{run_dir}: metrics games mismatch: {run.get('games')}")
        if run.get("device") != "cuda":
            raise ValueError(f"{run_dir}: expected CUDA device, got {run.get('device')}")
        if run.get("seat_mode") != "random":
            raise ValueError(f"{run_dir}: expected random seat mode")
        labels = [label for label in detailed["players"] if label in {"70k", "ext_mortal"} or label.startswith(("F_", "G_"))]
        if len(labels) != 4:
            raise ValueError(f"{run_dir}: unexpected players: {list(detailed['players'])}")
        seed = next(label.split("_", 1)[1] for label in labels if label.startswith("F_"))
        rows = {
            label: _raw_row(label, detailed["players"][label]["raw"])
            for label in labels
        }
        per_seed.append({"run": run_dir.name, "seed": int(seed), "models": rows})
        source_checks.append(
            {
                "run": run_dir.name,
                "games": log_count,
                "metrics": str(run_dir / "metrics.json"),
                "detailed_stats": str(run_dir / "detailed_stats.json"),
                "platform_accounts": (run_dir / "platform_accounts").is_dir(),
                "seed_start": run.get("seed_start"),
                "seed_key": run.get("seed_key"),
                "device": run.get("device"),
                "seat_mode": run.get("seat_mode"),
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for item in per_seed:
        for label, row in item["models"].items():
            grouped.setdefault("F" if label.startswith("F_") else "G" if label.startswith("G_") else label, []).append(row)
    pooled = {label: _aggregate(rows, label) for label, rows in sorted(grouped.items())}
    pairwise: list[dict[str, Any]] = []
    for item in per_seed:
        f = next(row for label, row in item["models"].items() if label.startswith("F_"))
        g = next(row for label, row in item["models"].items() if label.startswith("G_"))
        pairwise.append(
            {
                "run": item["run"],
                "seed": item["seed"],
                "F_avg_rank": f["avg_rank"],
                "G_avg_rank": g["avg_rank"],
                "F_avg_rank_pt": f["avg_rank_pt"],
                "G_avg_rank_pt": g["avg_rank_pt"],
                "G_minus_F_avg_rank_pt": g["avg_rank_pt"] - f["avg_rank_pt"],
                "G_minus_F_avg_rank": g["avg_rank"] - f["avg_rank"],
                "G_minus_F_agari_pp": (g["agari_rate"] - f["agari_rate"]) * 100.0,
                "G_minus_F_houjuu_pp": (g["houjuu_rate"] - f["houjuu_rate"]) * 100.0,
                "G_minus_F_fuuro_pp": (g["fuuro_rate"] - f["fuuro_rate"]) * 100.0,
                "G_minus_F_riichi_pp": (g["riichi_rate"] - f["riichi_rate"]) * 100.0,
            }
        )

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    document = {
        "schema": "keqing.mortal.reward_ab_eval_summary.v1",
        "eval_root": str(args.eval_root),
        "expected_games_per_pair": args.expected_games,
        "run_count": len(per_seed),
        "rank_points": list(RANK_POINTS),
        "source_checks": source_checks,
        "per_seed": per_seed,
        "pairwise": pairwise,
        "pooled": pooled,
        "interpretation": {
            "scope": "250 hanchans per matched seed pair; screening evaluation, not final promotion evidence",
            "favorable_G_pairs_by_avg_rank_pt": sum(1 for row in pairwise if row["G_minus_F_avg_rank_pt"] > 0),
            "pair_count": len(pairwise),
        },
    }
    (output_dir / "reward_ab_eval_250h_summary.json").write_text(
        json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    lines = [
        "# Reward Semantics A/B: 250-Hanchan Screening",
        "",
        "This is a matched-seed native random-seat screening evaluation. It is not a promotion gate.",
        "All rows are reported as separate F/G results; no two-way aggregate is used.",
        "",
        "## Per Pair",
        "",
        "| Pair | F avg rank | G avg rank | F avg Pt | G avg Pt | G-F Pt | G-F agari | G-F houjuu | G-F fuuro | G-F riichi |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in pairwise:
        lines.append(
            f"| {row['run']} | {row['F_avg_rank']:.3f} | {row['G_avg_rank']:.3f} | "
            f"{row['F_avg_rank_pt']:.2f} | {row['G_avg_rank_pt']:.2f} | "
            f"{row['G_minus_F_avg_rank_pt']:+.2f} | {row['G_minus_F_agari_pp']:+.2f}pp | "
            f"{row['G_minus_F_houjuu_pp']:+.2f}pp | {row['G_minus_F_fuuro_pp']:+.2f}pp | "
            f"{row['G_minus_F_riichi_pp']:+.2f}pp |"
        )
    lines.extend(
        [
            "",
            "## Pooled Auxiliary View",
            "",
            "| Model | Games | Avg rank | Avg Pt | Agari | Houjuu | Fuuro | Riichi | Rank counts |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for label in ("70k", "ext_mortal", "F", "G"):
        row = pooled[label]
        lines.append(
            f"| {label} | {row['games']} | {row['avg_rank']:.3f} | {row['avg_rank_pt']:.2f} | "
            f"{_fmt_pct(row['agari_rate'])} | {_fmt_pct(row['houjuu_rate'])} | "
            f"{_fmt_pct(row['fuuro_rate'])} | {_fmt_pct(row['riichi_rate'])} | {row['rank_counts']} |"
        )
    lines.extend(
        [
            "",
            "## Reading",
            "",
            f"- G is ahead of F on average rank Pt in {document['interpretation']['favorable_G_pairs_by_avg_rank_pt']}/{len(pairwise)} matched pairs.",
            "- The three-pair screen is only a direction check. Any promotion or recipe change requires a longer fixed-seed evaluation.",
            "- The 70k and ext_mortal rows are controls for this lineup, not a claim that this screen replaces the final model-pool league.",
        ]
    )
    (output_dir / "reward_ab_eval_250h_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"runs": len(per_seed), "pooled": pooled, "pairwise": pairwise}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
