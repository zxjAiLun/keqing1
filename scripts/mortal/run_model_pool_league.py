#!/usr/bin/env python3
"""Run pending native league lineups and rebuild the persistent model-pool Pt/R report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.build_platform_account_report import build_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/experiments/model_pool_2026_07/league_manifest.json"))
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-lineups", type=int, default=0, help="0 means all pending lineups")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--report-only", action="store_true")
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def run_lineup(lineup: dict[str, Any], args: argparse.Namespace) -> None:
    output_dir = Path(str(lineup["output_dir"]))
    command = [
        sys.executable,
        str(_REPO_ROOT / "scripts/mortal/four_player_native.py"),
        "--require-cuda",
        "--device", str(args.device),
        "--seat-mode", "random",
        "--seed-start", str(int(lineup["seed_start"])),
        "--seed-key", str(int(lineup["seed_key"])),
        "--games", str(int(lineup["games"])),
        "--native-batch-games", str(int(lineup["native_batch_games"])),
        "--progress-every", str(int(lineup["native_batch_games"])),
        "--rank-points", "90,45,0,-135",
        "--no-platform-report",
        "--output-dir", str(output_dir),
    ]
    for label, checkpoint in zip(lineup["models"], lineup["checkpoint_paths"], strict=True):
        command.extend(["--model", f"{label}={checkpoint}"])
    if args.resume:
        command.append("--resume")
    print(f"[model_pool] starting {lineup['lineup_id']}: {' | '.join(lineup['models'])}", flush=True)
    subprocess.run(command, check=True)


def rebuild_report(manifest: dict[str, Any], args: argparse.Namespace) -> None:
    ordered_lineups = sorted(manifest["lineups"], key=lambda item: int(item["league_order"]))
    completed = [lineup for lineup in ordered_lineups if lineup.get("status") == "completed"]
    if not completed:
        return
    log_dirs = [Path(str(lineup["output_dir"])) / "logs" for lineup in completed]
    root = args.manifest.parent
    report = build_report(
        log_dirs=log_dirs,
        output_dir=root / "platform_accounts",
        mortal_root=args.mortal_root,
        platform_model_label=None,
        rank_points=(90.0, 45.0, 0.0, -135.0),
        preserve_log_dir_order=True,
        interleave_log_dirs=True,
    )
    summary = {
        "schema": "keqing.mortal.model_pool_summary.v1",
        "league_id": manifest["league_id"],
        "completed_lineups": [lineup["lineup_id"] for lineup in completed],
        "accounts": report["accounts"],
        "platform_accounts_dir": str(root / "platform_accounts"),
    }
    (root / "model_pool_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (root / "model_pool_summary.md").write_text(build_summary_markdown(summary), encoding="utf-8")


def build_summary_markdown(summary: dict[str, Any]) -> str:
    accounts = sorted(summary["accounts"], key=lambda row: float(row["avg_rank_pt"]), reverse=True)
    lines = [
        "# Model Pool League",
        "",
        f"- League: `{summary['league_id']}`",
        f"- Completed lineups: {', '.join(summary['completed_lineups'])}",
        "- Every account plays 1000 hanchan; every account pair shares 500 hanchan.",
        "- Pt and average rank are the primary cross-ecology strength readouts.",
        "- Tenhou-style R is replayed from an interleaved, rotating lineup order so early-game R correction is not assigned to a fixed lineup. It remains a sequential platform rating, not a replacement for average rank/pt.",
        "",
        "| Model | Games | Avg rank | Avg pt | Pt | R | 1st | 2nd | 3rd | 4th | Agari | Houjuu | Fuuro | Riichi | Win value |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in accounts:
        lines.append(
            "| {model} | {games} | {avg_rank:.3f} | {avg_pt:+.3f} | {pt:+.0f} | {rating:.1f} | {r1} | {r2} | {r3} | {r4} | {agari:.2%} | {houjuu:.2%} | {fuuro:.2%} | {riichi:.2%} | {win:.0f} |".format(
                model=row["model_label"],
                games=int(row["games"]),
                avg_rank=float(row["avg_rank"]),
                avg_pt=float(row["avg_rank_pt"]),
                pt=float(row["pt_current"]),
                rating=float(row["rating"]),
                r1=int(row["rank_1"]),
                r2=int(row["rank_2"]),
                r3=int(row["rank_3"]),
                r4=int(row["rank_4"]),
                agari=float(row["agari_rate"]),
                houjuu=float(row["houjuu_rate"]),
                fuuro=float(row["fuuro_rate"]),
                riichi=float(row["riichi_rate"]),
                win=float(row["avg_point_per_agari"]),
            )
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    if not args.report_only:
        pending = [lineup for lineup in manifest["lineups"] if lineup.get("status") == "pending"]
        if args.max_lineups > 0:
            pending = pending[: int(args.max_lineups)]
        for lineup in pending:
            missing = [path for path in lineup["checkpoint_paths"] if not Path(str(path)).exists()]
            if missing:
                print(f"[model_pool] skip {lineup['lineup_id']}: missing checkpoints {missing}", flush=True)
                continue
            lineup["status"] = "running"
            write_manifest(args.manifest, manifest)
            run_lineup(lineup, args)
            lineup["status"] = "completed"
            write_manifest(args.manifest, manifest)
            rebuild_report(manifest, args)
    rebuild_report(manifest, args)


if __name__ == "__main__":
    main()
