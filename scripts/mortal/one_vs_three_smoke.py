#!/usr/bin/env python3
"""Run a fixed-seed Mortal OneVsThree smoke evaluation."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.eval_metrics import (
    add_rank_point_args,
    build_metrics_document,
    resolve_rank_points,
    summarize_rank_counts_with_references,
    write_metrics,
)
from scripts.mortal.build_platform_account_report import build_report as build_platform_account_report
from scripts.mortal.stat_report import write_stat_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mortal native OneVsThree smoke evaluation")
    parser.add_argument("--challenger", type=Path, default=Path("artifacts/mortal_training/mortal.pth"))
    parser.add_argument("--champion", type=Path, default=None, help="defaults to --challenger")
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval/one_vs_three_smoke"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--seed-key", type=int, default=0x2000)
    parser.add_argument("--seed-count", type=int, default=1, help="1 seed produces 4 hanchans")
    parser.add_argument("--resume", action="store_true", help="resume from existing logs in output-dir/logs")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="emit stderr progress every N seeds; 0 runs the native arena in one batch",
    )
    parser.add_argument(
        "--native-batch-seeds",
        type=int,
        default=0,
        help="seed sets per Rust arena batch; 0 preserves the progress-sized batch behavior",
    )
    parser.add_argument("--profile", action="store_true", help="record per-engine inference batch and timing telemetry")
    parser.add_argument("--no-platform-report", action="store_true", help="skip platform account pt/rating report")
    parser.add_argument("--platform-model-label", default=None, help="force platform account labels to MODEL@01-04")
    parser.add_argument("--enable-amp", action="store_true")
    parser.add_argument("--challenger-label", default="challenger (x1)")
    parser.add_argument("--champion-label", default="champion (x3)")
    add_rank_point_args(parser)
    return parser.parse_args()


def _load_engine(
    *,
    state_file: Path,
    mortal_root: Path,
    device: str,
    name: str,
    enable_amp: bool,
    enable_profile: bool,
) -> Any:
    mortal_python_dir = (mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))

    from engine import MortalEngine  # noqa: PLC0415
    from model import Brain, DQN  # noqa: PLC0415

    state = torch.load(state_file, weights_only=True, map_location=torch.device("cpu"))
    cfg = state["config"]
    version = int(cfg["control"].get("version", 4))
    conv_channels = int(cfg["resnet"]["conv_channels"])
    num_blocks = int(cfg["resnet"]["num_blocks"])

    mortal = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
    dqn = DQN(version=version).eval()
    mortal.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    return MortalEngine(
        mortal,
        dqn,
        is_oracle=False,
        version=version,
        device=torch.device(device),
        enable_amp=bool(enable_amp),
        enable_rule_based_agari_guard=True,
        name=name,
        enable_profile=enable_profile,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    if bool(args.require_cuda) and not torch.cuda.is_available():
        raise SystemExit("CUDA required but torch.cuda.is_available() is False")
    mortal_python_dir = (args.mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))
    from libriichi.arena import OneVsThree  # noqa: PLC0415

    champion_path = args.champion or args.challenger
    rank_points_profile, rank_points = resolve_rank_points(
        rank_points=getattr(args, "rank_points", None),
        profile=str(getattr(args, "rank_points_profile", "tenhou_reference")),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"

    challenger = _load_engine(
        state_file=args.challenger,
        mortal_root=args.mortal_root,
        device=str(args.device),
        name="challenger",
        enable_amp=bool(args.enable_amp),
        enable_profile=bool(args.profile),
    )
    champion = _load_engine(
        state_file=champion_path,
        mortal_root=args.mortal_root,
        device=str(args.device),
        name="champion",
        enable_amp=bool(args.enable_amp),
        enable_profile=bool(args.profile),
    )

    env = OneVsThree(disable_progress_bar=True, log_dir=str(log_dir))
    total_seeds = int(args.seed_count)
    progress_every = int(getattr(args, "progress_every", 0) or 0)
    requested_batch_size = int(getattr(args, "native_batch_seeds", 0) or 0)
    batch_size = requested_batch_size or (total_seeds if progress_every <= 0 else max(1, progress_every))
    if batch_size <= 0:
        raise ValueError("--native-batch-seeds must be positive when provided")
    rank_counts = [0, 0, 0, 0]
    completed = 0
    if bool(getattr(args, "resume", False)) and log_dir.exists():
        existing_stat_report = write_stat_report(
            output_dir=args.output_dir,
            log_dir=log_dir,
            players={
                str(getattr(args, "challenger_label", "challenger (x1)")): "challenger",
                str(getattr(args, "champion_label", "champion (x3)")): "champion",
            },
            mortal_root=args.mortal_root,
            rank_pts=rank_points,
            rank_points_profile=rank_points_profile,
        )
        challenger_key = str(getattr(args, "challenger_label", "challenger (x1)"))
        raw = existing_stat_report["players"][challenger_key]["raw"]
        completed_games = int(raw["game"])
        completed = min(completed_games // 4, total_seeds)
        rank_counts = [int(raw[f"rank_{rank}"]) for rank in range(1, 5)]
        if completed:
            print(
                f"resuming from {completed}/{total_seeds} seeds ({completed * 4}/{total_seeds * 4} games) in {log_dir}",
                flush=True,
            )
    started_at = time.monotonic()
    while completed < total_seeds:
        count = min(batch_size, total_seeds - completed)
        batch_seed_start = int(args.seed_start) + completed
        if progress_every > 0:
            print(
                (
                    f"[one_vs_three] seeds {completed + 1}-{completed + count}/"
                    f"{total_seeds} start={batch_seed_start} device={args.device}"
                ),
                file=sys.stderr,
                flush=True,
            )
        batch_rank_counts = list(
            env.py_vs_py(
                challenger=challenger,
                champion=champion,
                seed_start=(batch_seed_start, int(args.seed_key)),
                seed_count=count,
            )
        )
        for i, value in enumerate(batch_rank_counts):
            rank_counts[i] += int(value)
        completed += count
        if progress_every > 0:
            elapsed = time.monotonic() - started_at
            seeds_per_sec = completed / elapsed if elapsed > 0 else 0.0
            remaining = (total_seeds - completed) / seeds_per_sec if seeds_per_sec > 0 else 0.0
            print(
                (
                    f"[one_vs_three] completed {completed}/{total_seeds} seeds "
                    f"({completed * 4}/{total_seeds * 4} games), "
                    f"elapsed={elapsed:.1f}s eta={remaining:.1f}s "
                    f"rank_counts={rank_counts}"
                ),
                file=sys.stderr,
                flush=True,
            )
    document = build_metrics_document(
        run={
            "kind": "one_vs_three_smoke",
            "backend": "libriichi.arena.OneVsThree",
            "challenger": str(args.challenger),
            "champion": str(champion_path),
            "seed_start": int(args.seed_start),
            "seed_key": int(args.seed_key),
            "seed_count": int(args.seed_count),
            "native_batch_seeds": int(batch_size),
            "device": str(args.device),
            "rank_points_profile": rank_points_profile,
            "rank_points_values": [float(value) for value in rank_points],
        },
        metrics={"challenger": summarize_rank_counts_with_references(rank_counts, rank_points=rank_points)},
        artifacts={"log_dir": str(log_dir)},
        rank_points_profile=rank_points_profile,
        rank_points_values=rank_points,
    )
    write_metrics(args.output_dir / "metrics.json", document)
    if bool(args.profile):
        inference_profile = {
            "challenger": challenger.profile_snapshot(),
            "champion": champion.profile_snapshot(),
        }
        profile_path = args.output_dir / "inference_profile.json"
        profile_path.write_text(json.dumps(inference_profile, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        document["artifacts"]["inference_profile_json"] = str(profile_path)
        document["inference_profile"] = inference_profile
    stat_report = write_stat_report(
        output_dir=args.output_dir,
        log_dir=log_dir,
        players={
            str(getattr(args, "challenger_label", "challenger (x1)")): "challenger",
            str(getattr(args, "champion_label", "champion (x3)")): "champion",
        },
        mortal_root=args.mortal_root,
        rank_pts=rank_points,
        rank_points_profile=rank_points_profile,
    )
    document["artifacts"]["detailed_stats_json"] = str(args.output_dir / "detailed_stats.json")
    document["artifacts"]["detailed_stats_md"] = str(args.output_dir / "detailed_stats.md")
    document["detailed_stats_schema"] = stat_report["schema"]
    if not bool(getattr(args, "no_platform_report", False)):
        platform_output_dir = args.output_dir / "platform_accounts"
        platform_report = build_platform_account_report(
            log_dirs=[log_dir],
            output_dir=platform_output_dir,
            mortal_root=args.mortal_root,
            platform_model_label=getattr(args, "platform_model_label", None),
            rank_points=rank_points,
        )
        document["artifacts"]["platform_accounts_dir"] = str(platform_output_dir)
        document["platform_accounts_schema"] = platform_report["schema"]
    write_metrics(args.output_dir / "metrics.json", document)
    print(json.dumps(document["metrics"], ensure_ascii=False, indent=2), flush=True)
    return document


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
