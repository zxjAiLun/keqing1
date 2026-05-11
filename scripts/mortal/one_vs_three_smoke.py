#!/usr/bin/env python3
"""Run a fixed-seed Mortal OneVsThree smoke evaluation."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.eval_metrics import build_metrics_document, summarize_rank_counts, write_metrics
from scripts.mortal.stat_report import write_stat_report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Mortal native OneVsThree smoke evaluation")
    parser.add_argument("--challenger", type=Path, default=Path("artifacts/mortal_training/mortal.pth"))
    parser.add_argument("--champion", type=Path, default=None, help="defaults to --challenger")
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval/one_vs_three_smoke"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed-start", type=int, default=10000)
    parser.add_argument("--seed-key", type=int, default=0x2000)
    parser.add_argument("--seed-count", type=int, default=1, help="1 seed produces 4 hanchans")
    parser.add_argument("--enable-amp", action="store_true")
    parser.add_argument("--challenger-label", default="challenger (x1)")
    parser.add_argument("--champion-label", default="champion (x3)")
    return parser.parse_args()


def _load_engine(
    *,
    state_file: Path,
    mortal_root: Path,
    device: str,
    name: str,
    enable_amp: bool,
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
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    mortal_python_dir = (args.mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))
    from libriichi.arena import OneVsThree  # noqa: PLC0415

    champion_path = args.champion or args.challenger
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = args.output_dir / "logs"

    challenger = _load_engine(
        state_file=args.challenger,
        mortal_root=args.mortal_root,
        device=str(args.device),
        name="challenger",
        enable_amp=bool(args.enable_amp),
    )
    champion = _load_engine(
        state_file=champion_path,
        mortal_root=args.mortal_root,
        device=str(args.device),
        name="champion",
        enable_amp=bool(args.enable_amp),
    )

    env = OneVsThree(disable_progress_bar=True, log_dir=str(log_dir))
    rank_counts = list(
        env.py_vs_py(
            challenger=challenger,
            champion=champion,
            seed_start=(int(args.seed_start), int(args.seed_key)),
            seed_count=int(args.seed_count),
        )
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
            "device": str(args.device),
        },
        metrics={"challenger": summarize_rank_counts(rank_counts)},
        artifacts={"log_dir": str(log_dir)},
    )
    write_metrics(args.output_dir / "metrics.json", document)
    stat_report = write_stat_report(
        output_dir=args.output_dir,
        log_dir=log_dir,
        players={
            str(getattr(args, "challenger_label", "challenger (x1)")): "challenger",
            str(getattr(args, "champion_label", "champion (x3)")): "champion",
        },
        mortal_root=args.mortal_root,
    )
    document["artifacts"]["detailed_stats_json"] = str(args.output_dir / "detailed_stats.json")
    document["artifacts"]["detailed_stats_md"] = str(args.output_dir / "detailed_stats.md")
    document["detailed_stats_schema"] = stat_report["schema"]
    write_metrics(args.output_dir / "metrics.json", document)
    print(json.dumps(document["metrics"], ensure_ascii=False, indent=2), flush=True)
    return document


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
