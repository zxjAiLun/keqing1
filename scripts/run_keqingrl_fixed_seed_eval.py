#!/usr/bin/env python3
"""Run KeqingRL-Lite fixed-seed seat-rotation eval smoke."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import torch

from keqingrl import DiscardOnlyMahjongEnv, RulePriorDeltaPolicy, run_fixed_seed_evaluation_smoke


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KeqingRL-Lite P10 fixed-seed seat-rotation eval smoke")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/keqingrl_fixed_seed_eval"))
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--max-kyokus", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-res-blocks", type=int, default=1)
    parser.add_argument("--device", default=None)
    parser.add_argument("--policy-mode", choices=("greedy", "sample"), default="greedy")
    parser.add_argument("--max-fourth-rate", type=float, default=0.75)
    parser.add_argument("--max-deal-in-rate", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    policy = RulePriorDeltaPolicy(
        hidden_dim=args.hidden_dim,
        num_res_blocks=args.num_res_blocks,
        dropout=0.0,
    )
    report = run_fixed_seed_evaluation_smoke(
        DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
        policy,
        num_games=args.eval_seeds,
        seed=args.seed,
        seed_stride=args.seed_stride,
        seat_rotation=(0, 1, 2, 3),
        max_steps=args.max_steps,
        greedy=args.policy_mode == "greedy",
        reuse_training_rollout=False,
        max_fourth_rate=args.max_fourth_rate,
        max_deal_in_rate=args.max_deal_in_rate,
        device=args.device,
    )
    _write_json(args.out_dir / "fixed_seed_eval.json", report)
    summary = _summary_markdown(args, report)
    (args.out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(value), indent=2, sort_keys=True), encoding="utf-8")


def _summary_markdown(args: argparse.Namespace, report) -> str:
    lines = [
        "# KeqingRL-Lite Fixed-Seed Eval Smoke",
        "",
        f"seed: `{args.seed}`",
        f"fixed_seed_count: `{report.fixed_seed_count}`",
        f"games_per_seed: `{report.games_per_seed}`",
        f"policy_mode: `{report.policy_mode}`",
        f"opponent: `{report.opponent_name}`",
        "",
        "This is a seat-rotation non-regression smoke. It does not claim model strength.",
        "A later duplicate strength eval should use a seed registry, frozen opponent snapshots, paired deltas, and confidence intervals.",
        "",
        "## Metrics",
        "",
        f"- passed_smoke_checks: `{report.passed_smoke_checks}`",
        f"- failure_reasons: `{list(report.failure_reasons)}`",
        f"- episode_count: `{report.episode_count}`",
        f"- average_rank: `{report.average_rank:.6g}`",
        f"- rank_pt: `{report.rank_pt:.6g}`",
        f"- fourth_rate: `{report.fourth_rate:.6g}`",
        f"- win_rate: `{report.win_rate:.6g}`",
        f"- deal_in_rate: `{report.deal_in_rate:.6g}`",
        f"- call_rate: `{report.call_rate:.6g}`",
        f"- riichi_rate: `{report.riichi_rate:.6g}`",
        f"- illegal_action_rate: `{report.illegal_action_rate:.6g}`",
        f"- fallback_rate: `{report.fallback_rate:.6g}`",
        f"- forced_terminal_missed: `{report.forced_terminal_missed}`",
        f"- terminal_reason_count: `{report.terminal_reason_count}`",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
