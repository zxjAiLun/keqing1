#!/usr/bin/env python3
"""Run KeqingRL-Lite smoke gates and write reproducible reports."""

from __future__ import annotations

import argparse
from dataclasses import asdict, is_dataclass
import json
from pathlib import Path
from typing import Any

import torch

from keqingrl import (
    DiscardOnlyMahjongEnv,
    RulePriorDeltaPolicy,
    measure_latency_smoke,
    run_critic_pretrain_smoke,
    run_discard_only_ppo_smoke,
    run_fixed_seed_evaluation_smoke,
    run_zero_delta_selfplay_smoke,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KeqingRL-Lite v0 smoke gates")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/keqingrl_smoke"))
    parser.add_argument("--seed", type=int, default=20260424)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-kyokus", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-res-blocks", type=int, default=1)
    parser.add_argument("--zero-delta-greedy-episodes", type=int, default=16)
    parser.add_argument("--zero-delta-sample-episodes", type=int, default=16)
    parser.add_argument("--critic-episodes", type=int, default=32)
    parser.add_argument("--critic-steps", type=int, default=10)
    parser.add_argument("--ppo-iterations", type=int, default=3)
    parser.add_argument("--ppo-rollout-episodes", type=int, default=8)
    parser.add_argument("--ppo-update-epochs", type=int, default=1)
    parser.add_argument("--eval-seeds", type=int, default=8)
    parser.add_argument("--latency-games", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    reports: dict[str, Any] = {}

    reports["zero_delta_selfplay"] = run_zero_delta_selfplay_smoke(
        DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
        greedy_episodes=args.zero_delta_greedy_episodes,
        sample_episodes=args.zero_delta_sample_episodes,
        seed=args.seed,
        device=args.device,
    )
    _write_json(args.out_dir / "zero_delta_selfplay.json", reports["zero_delta_selfplay"])

    reports["critic_pretrain"] = run_critic_pretrain_smoke(
        DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
        episodes=args.critic_episodes,
        pretrain_steps=args.critic_steps,
        seed=args.seed,
        device=args.device,
    )
    _write_json(args.out_dir / "critic_pretrain.json", reports["critic_pretrain"])

    reports["discard_only_ppo"] = run_discard_only_ppo_smoke(
        DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
        iterations=args.ppo_iterations,
        rollout_episodes_per_iter=args.ppo_rollout_episodes,
        update_epochs=args.ppo_update_epochs,
        seed=args.seed,
        hidden_dim=args.hidden_dim,
        num_res_blocks=args.num_res_blocks,
        device=args.device,
    )
    _write_json(args.out_dir / "discard_only_ppo.json", reports["discard_only_ppo"])

    eval_policy = RulePriorDeltaPolicy(
        hidden_dim=args.hidden_dim,
        num_res_blocks=args.num_res_blocks,
        dropout=0.0,
    )
    reports["fixed_seed_eval"] = run_fixed_seed_evaluation_smoke(
        DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
        eval_policy,
        num_games=args.eval_seeds,
        seed=args.seed,
        seat_rotation=(0, 1, 2, 3),
        greedy=True,
        device=args.device,
    )
    _write_json(args.out_dir / "fixed_seed_eval.json", reports["fixed_seed_eval"])

    latency_policy = RulePriorDeltaPolicy(
        hidden_dim=args.hidden_dim,
        num_res_blocks=args.num_res_blocks,
        dropout=0.0,
    )
    reports["latency"] = measure_latency_smoke(
        DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
        latency_policy,
        num_games=args.latency_games,
        seed=args.seed,
        device=args.device,
    )
    _write_json(args.out_dir / "latency.json", reports["latency"])

    summary = _summary_markdown(args, reports)
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


def _summary_markdown(args: argparse.Namespace, reports: dict[str, Any]) -> str:
    zero = reports["zero_delta_selfplay"]
    critic = reports["critic_pretrain"]
    ppo = reports["discard_only_ppo"]
    fixed = reports["fixed_seed_eval"]
    latency = reports["latency"]
    ppo_last = ppo.iterations[-1]
    lines = [
        "# KeqingRL-Lite Smoke Report",
        "",
        f"seed: `{args.seed}`",
        f"max_kyokus: `{args.max_kyokus}`",
        "",
        "## Scope",
        "",
        "This is a smoke/non-regression report. It does not claim model strength.",
        "Metric fidelity note: illegal/fallback/forced-terminal counters are hard failure gates until env/review expose recoverable counters.",
        "Call/riichi rates are coarse learner-step rates in discard-only scope; opportunity-denominator metrics are a later unlock requirement.",
        "Fixed-seed eval is seat-rotation smoke, not paired duplicate strength evaluation with confidence intervals.",
        "",
        "## Results",
        "",
        f"- zero_delta: learner_steps={zero.learner_step_count}, autopilot_steps={zero.autopilot_step_count}, rule_kl_mean={zero.rule_kl_mean:.6g}, delta_max={zero.neural_delta_abs_max:.6g}",
        f"- critic_pretrain: value_loss={critic.value_loss:.6g}, rank_loss={critic.rank_loss}, actor_logits_diff={critic.actor_logits_before_after_diff:.6g}, optimizer_actor_params={critic.optimizer_actor_param_count}",
        f"- discard_only_ppo: iterations={ppo.iteration_count}, stopped_early={ppo.stopped_early}, final_delta_max={ppo.final_neural_delta_abs_max:.6g}, approx_kl_last={ppo_last.approx_kl:.6g}, clip_fraction_last={ppo_last.clip_fraction:.6g}",
        f"- fixed_seed_eval: games={fixed.episode_count}, passed={fixed.passed_smoke_checks}, avg_rank={fixed.average_rank:.6g}, rank_pt={fixed.rank_pt:.6g}, fourth_rate={fixed.fourth_rate:.6g}, deal_in_rate={fixed.deal_in_rate:.6g}",
        f"- latency: decisions={latency.decision_count}, decisions_per_sec={latency.decisions_per_sec:.6g}, avg_ms={latency.avg_decision_latency_ms:.6g}, p95_ms={latency.p95_decision_latency_ms:.6g}",
        "",
        "## Artifacts",
        "",
        "- `zero_delta_selfplay.json`",
        "- `critic_pretrain.json`",
        "- `discard_only_ppo.json`",
        "- `fixed_seed_eval.json`",
        "- `latency.json`",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
