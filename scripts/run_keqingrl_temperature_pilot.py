#!/usr/bin/env python3
"""Train-time sampling-temperature PPO pilot for controlled KeqingRL discard-only research."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from keqingrl import DiscardOnlyMahjongEnv, run_fixed_seed_evaluation_smoke
from keqingrl.learning_signal import batch_diagnostic_rows, seed_registry_hash
from keqingrl.ppo import compute_ppo_loss, ppo_update
from keqingrl.selfplay import build_episodes_ppo_batch, collect_selfplay_episodes
from scripts.probe_keqingrl_sampling_diversity import (
    TemperaturePolicy,
    _candidate_summary,
    _load_candidates,
    _load_policy,
    _opponent_pool,
    _sampling_summary,
    _write_csv,
    _write_json,
)
from scripts.run_keqingrl_fixed_online_bridge import _file_sha256, _loss_float, _policy_delta_stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run train-time sampling-temperature PPO pilot")
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--source-config-ids", type=int, nargs="+", default=(93,))
    parser.add_argument("--rerun-config-ids", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--temperatures", type=float, nargs="+", default=(1.25, 1.5, 2.0))
    parser.add_argument("--lrs", type=float, nargs="+", default=(1e-4, 3e-4))
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--rule-kl-coef", type=float, default=0.001)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--value-coef", type=float, default=0.0)
    parser.add_argument("--rank-coef", type=float, default=0.0)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed-base", type=int, default=202604280000)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--torch-seed-base", type=int, default=202604280000)
    parser.add_argument("--eval-seed-base", type=int, default=202604290000)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--learner-seats", type=int, nargs="+", default=(0,))
    parser.add_argument("--eval-seat-rotation", type=int, nargs="+", default=(0,))
    parser.add_argument("--max-kyokus", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device is not None else torch.device("cpu")
    candidates = _load_candidates(args)
    summary_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    advantage_rows: list[dict[str, Any]] = []
    config_id = 0

    for candidate in candidates:
        source_policy = _load_policy(candidate, device)
        opponent_pool = _opponent_pool(str(candidate["opponent_mode"]))
        for temperature in args.temperatures:
            for lr in args.lrs:
                policy = copy.deepcopy(source_policy).to(device)
                rule_score_scale = float(getattr(policy, "rule_score_scale", 1.0))
                optimizer = torch.optim.Adam(policy.parameters(), lr=float(lr))
                config_rows: list[dict[str, Any]] = []
                for iteration in range(int(args.iterations)):
                    rollout_seed = _iteration_seed(args, config_id, iteration)
                    torch_seed = _iteration_torch_seed(args, config_id, iteration)
                    _seed_torch_sampling(torch_seed)
                    behavior_policy = TemperaturePolicy(policy, temperature=float(temperature)).to(device)
                    episodes = collect_selfplay_episodes(
                        DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
                        behavior_policy,
                        num_episodes=int(args.episodes),
                        opponent_pool=opponent_pool,
                        learner_seats=tuple(int(seat) for seat in args.learner_seats),
                        seed=rollout_seed,
                        seed_stride=int(args.seed_stride),
                        greedy=False,
                        max_steps=int(args.max_steps),
                        device=device,
                    )
                    _advantages, _returns, prepared_steps, batch = build_episodes_ppo_batch(
                        episodes,
                        gamma=float(args.gamma),
                        gae_lambda=float(args.gae_lambda),
                        include_rank_targets=True,
                        strict_metadata=True,
                    )
                    batch = batch.to(device)
                    diagnostic_rows, diagnostic_summary = batch_diagnostic_rows(policy, batch, prepared_steps, episodes)
                    sampling_summary = _sampling_summary(diagnostic_rows)
                    behavior_temperatures = {
                        float(row["behavior_temperature"])
                        for row in diagnostic_rows
                        if row.get("behavior_temperature") is not None
                    }
                    if behavior_temperatures != {float(temperature)}:
                        raise RuntimeError(
                            f"behavior_temperature metadata mismatch: expected {temperature}, got {sorted(behavior_temperatures)}"
                        )
                    pre_stats = _policy_delta_stats(policy, batch)
                    losses = []
                    for _epoch in range(int(args.update_epochs)):
                        losses.append(
                            ppo_update(
                                policy,
                                optimizer,
                                batch,
                                clip_eps=float(args.clip_eps),
                                value_coef=float(args.value_coef),
                                entropy_coef=float(args.entropy_coef),
                                rank_coef=float(args.rank_coef),
                                rule_kl_coef=float(args.rule_kl_coef),
                                normalize_advantages=bool(args.normalize_advantages),
                                max_grad_norm=float(args.max_grad_norm) if args.max_grad_norm is not None else None,
                            )
                        )
                    post_loss = _compute_loss(policy, batch, args)
                    post_stats = _policy_delta_stats(policy, batch)
                    iter_row = {
                        **_candidate_summary(candidate),
                        "pilot_config_id": config_id,
                        "iteration": int(iteration),
                        "rule_score_scale": rule_score_scale,
                        "rule_score_scale_version": "keqingrl_rule_score_scale_v1",
                        "behavior_temperature": float(temperature),
                        "lr": float(lr),
                        "episodes": int(args.episodes),
                        "rollout_seed": int(rollout_seed),
                        "torch_sample_seed": int(torch_seed),
                        "seed_registry_id": _iteration_seed_registry_id(args, config_id, iteration),
                        "seed_hash": seed_registry_hash(_iteration_seed_registry(args, config_id, iteration)),
                        "update_epochs": int(args.update_epochs),
                        "rule_kl_coef": float(args.rule_kl_coef),
                        "entropy_coef": float(args.entropy_coef),
                        "value_coef": float(args.value_coef),
                        "rank_coef": float(args.rank_coef),
                        "max_grad_norm": float(args.max_grad_norm) if args.max_grad_norm is not None else "",
                        **diagnostic_summary,
                        **sampling_summary,
                        **{f"pre_{key}": value for key, value in pre_stats.items()},
                        **{f"post_{key}": value for key, value in post_stats.items()},
                        "post_update_approx_kl": _loss_float(post_loss.approx_kl),
                        "post_update_clip_fraction": _loss_float(post_loss.clip_fraction),
                        "post_update_policy_loss": _loss_float(post_loss.policy_loss),
                        "post_update_rule_kl": _loss_float(post_loss.rule_kl),
                    }
                    iteration_rows.append(iter_row)
                    config_rows.append(iter_row)
                    for row in diagnostic_rows:
                        step_rows.append(
                            {
                                **_candidate_summary(candidate),
                                "pilot_config_id": config_id,
                                "iteration": int(iteration),
                                "rule_score_scale": rule_score_scale,
                                "rule_score_scale_version": "keqingrl_rule_score_scale_v1",
                                "behavior_temperature": float(temperature),
                                "lr": float(lr),
                                **row,
                            }
                        )
                    for row in _advantage_audit_rows(diagnostic_rows):
                        advantage_rows.append(
                            {
                                **_candidate_summary(candidate),
                                "pilot_config_id": config_id,
                                "iteration": int(iteration),
                                "rule_score_scale": rule_score_scale,
                                "rule_score_scale_version": "keqingrl_rule_score_scale_v1",
                                "behavior_temperature": float(temperature),
                                "lr": float(lr),
                                **row,
                            }
                        )
                    print(
                        "temp-pilot "
                        f"cfg={config_id} source={candidate['source_config_id']} "
                        f"temp={float(temperature):g} lr={float(lr):g} iter={iteration + 1}/{args.iterations} "
                        f"non_top1={sampling_summary['non_top1_selected_count']} "
                        f"non_top1_pos={sampling_summary['non_top1_positive_advantage_count']} "
                        f"top1_changed={post_stats['top1_action_changed_rate']:.6g} "
                        f"kl={_loss_float(post_loss.approx_kl):.6g} "
                        f"clip={_loss_float(post_loss.clip_fraction):.6g}",
                        flush=True,
                    )
                eval_metrics = run_fixed_seed_evaluation_smoke(
                    DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
                    policy,
                    num_games=int(args.eval_episodes),
                    seed=int(args.eval_seed_base),
                    seed_stride=int(args.seed_stride),
                    seat_rotation=tuple(int(seat) for seat in args.eval_seat_rotation),
                    opponent_pool=opponent_pool,
                    opponent_name=str(candidate["opponent_mode"]),
                    max_steps=int(args.max_steps),
                    greedy=True,
                    reuse_training_rollout=False,
                    device=device,
                )
                final_row = config_rows[-1]
                summary_rows.append(
                    {
                        **_candidate_summary(candidate),
                        "pilot_config_id": config_id,
                        "rule_score_scale": rule_score_scale,
                        "rule_score_scale_version": "keqingrl_rule_score_scale_v1",
                        "behavior_temperature": float(temperature),
                        "lr": float(lr),
                        "iterations": int(args.iterations),
                        "episodes": int(args.episodes),
                        "eval_episodes": int(args.eval_episodes),
                        "eval_seed_registry_id": _eval_seed_registry_id(args),
                        "eval_seed_hash": seed_registry_hash(_eval_seed_registry(args)),
                        "eval_scope": _eval_scope(args),
                        "eval_strength_note": "sanity check only; not duplicate strength evidence",
                        "source_checkpoint_sha256": candidate.get("checkpoint_sha256") or _file_sha256(Path(candidate["checkpoint_path"])),
                        **{f"final_{key}": value for key, value in final_row.items() if _summary_metric_key(key)},
                        "eval_rank_pt": eval_metrics.rank_pt,
                        "eval_mean_rank": eval_metrics.average_rank,
                        "eval_fourth_rate": eval_metrics.fourth_rate,
                        "eval_learner_deal_in_rate": eval_metrics.learner_deal_in_rate,
                        "eval_learner_win_rate": eval_metrics.learner_win_rate,
                        "illegal_action_rate_fail_closed": eval_metrics.illegal_action_rate_fail_closed,
                        "fallback_rate_fail_closed": eval_metrics.fallback_rate_fail_closed,
                        "forced_terminal_missed_fail_closed": eval_metrics.forced_terminal_missed_fail_closed,
                    }
                )
                config_id += 1

    payload = {
        "mode": "train_time_temperature_pilot",
        "source_type": "checkpoint",
        "candidate_summary": str(args.candidate_summary),
        "source_config_ids": [int(value) for value in args.source_config_ids],
        "temperatures": [float(value) for value in args.temperatures],
        "rule_score_scale_values": sorted({float(row["rule_score_scale"]) for row in summary_rows}),
        "rule_score_scale_version": "keqingrl_rule_score_scale_v1",
        "lrs": [float(value) for value in args.lrs],
        "iteration_count": int(args.iterations),
        "episodes": int(args.episodes),
        "eval_scope": _eval_scope(args),
        "eval_strength_note": "sanity check only; not duplicate strength evidence",
        "summaries": summary_rows,
        "iteration_rows": iteration_rows,
        "advantage_audit": advantage_rows,
    }
    _write_json(args.output_dir / "temperature_pilot.json", payload)
    _write_csv(args.output_dir / "summary.csv", summary_rows)
    _write_csv(args.output_dir / "iterations.csv", iteration_rows)
    _write_csv(args.output_dir / "batch_steps.csv", step_rows)
    _write_csv(args.output_dir / "advantage_audit.csv", advantage_rows)
    (args.output_dir / "summary.md").write_text(_summary_markdown(args, summary_rows), encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


def _compute_loss(policy, batch, args: argparse.Namespace):
    with torch.no_grad():
        return compute_ppo_loss(
            policy,
            batch,
            clip_eps=float(args.clip_eps),
            value_coef=float(args.value_coef),
            entropy_coef=float(args.entropy_coef),
            rank_coef=float(args.rank_coef),
            rule_kl_coef=float(args.rule_kl_coef),
            normalize_advantages=bool(args.normalize_advantages),
        )


def _advantage_audit_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(int(row["selected_prior_rank"]), []).append(row)
    audit_rows: list[dict[str, Any]] = []
    for rank, rank_rows in sorted(buckets.items()):
        advantages = [float(row["advantage_raw"]) for row in rank_rows]
        returns = [float(row["return"]) for row in rank_rows]
        terminal_rewards = [float(row["terminal_reward"]) for row in rank_rows]
        audit_rows.append(
            {
                "selected_prior_rank": int(rank),
                "count": len(rank_rows),
                "positive_advantage_count": sum(1 for value in advantages if value > 0.0),
                "negative_advantage_count": sum(1 for value in advantages if value < 0.0),
                "zero_advantage_count": sum(1 for value in advantages if value == 0.0),
                "advantage_mean": _mean(advantages),
                "return_mean": _mean(returns),
                "terminal_reward_mean": _mean(terminal_rewards),
            }
        )
    return audit_rows


def _summary_metric_key(key: str) -> bool:
    return key in {
        "batch_size",
        "advantage_mean",
        "advantage_std",
        "advantage_min",
        "advantage_max",
        "positive_advantage_count",
        "non_top1_selected_count",
        "non_top1_positive_advantage_count",
        "positive_advantage_top1_count",
        "positive_advantage_non_top1_count",
        "selected_prior_top1_rate",
        "post_top1_action_changed_rate",
        "post_rule_agreement",
        "post_neural_delta_abs_mean",
        "post_neural_delta_abs_max",
        "post_update_approx_kl",
        "post_update_clip_fraction",
        "post_update_policy_loss",
        "post_update_rule_kl",
    }


def _seed_torch_sampling(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _iteration_seed(args: argparse.Namespace, config_id: int, iteration: int) -> int:
    return int(args.seed_base + config_id * 100_000 + iteration * int(args.episodes) * int(args.seed_stride))


def _iteration_torch_seed(args: argparse.Namespace, config_id: int, iteration: int) -> int:
    return int(args.torch_seed_base + config_id * 100_000 + iteration)


def _iteration_seed_registry(args: argparse.Namespace, config_id: int, iteration: int) -> list[int]:
    start = _iteration_seed(args, config_id, iteration)
    return [int(start + idx * args.seed_stride) for idx in range(args.episodes)]


def _iteration_seed_registry_id(args: argparse.Namespace, config_id: int, iteration: int) -> str:
    return (
        f"config={config_id}:iter={iteration}:base={_iteration_seed(args, config_id, iteration)}:"
        f"stride={args.seed_stride}:count={args.episodes}"
    )


def _eval_seed_registry(args: argparse.Namespace) -> list[int]:
    return [int(args.eval_seed_base + idx * args.seed_stride) for idx in range(args.eval_episodes)]


def _eval_seed_registry_id(args: argparse.Namespace) -> str:
    return f"base={args.eval_seed_base}:stride={args.seed_stride}:count={args.eval_episodes}"


def _eval_scope(args: argparse.Namespace) -> str:
    seats = ",".join(str(int(seat)) for seat in args.eval_seat_rotation)
    noun = "seat" if len(tuple(args.eval_seat_rotation)) == 1 else "seats"
    return f"fixed-seed smoke; learner {noun} {seats} only"


def _summary_markdown(args: argparse.Namespace, rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# KeqingRL Train-Time Temperature Pilot",
        "",
        "source_type: `checkpoint`",
        f"candidate_summary: `{args.candidate_summary}`",
        f"source_config_ids: `{','.join(str(int(value)) for value in args.source_config_ids)}`",
        f"episodes: `{args.episodes}`",
        f"iterations: `{args.iterations}`",
        f"temperatures: `{','.join(str(float(value)) for value in args.temperatures)}`",
        f"rule_score_scale_values: `{','.join(str(value) for value in sorted({float(row['rule_score_scale']) for row in rows})) if rows else ''}`",
        "rule_score_scale_version: `keqingrl_rule_score_scale_v1`",
        f"lrs: `{','.join(str(float(value)) for value in args.lrs)}`",
        f"rule_kl_coef: `{args.rule_kl_coef}`",
        f"entropy_coef: `{args.entropy_coef}`",
        f"eval_seed_registry_id: `{_eval_seed_registry_id(args)}`",
        f"eval_seed_hash: `{seed_registry_hash(_eval_seed_registry(args))}`",
        f"eval_scope: `{_eval_scope(args)}`",
        "eval_strength_note: `sanity check only; not duplicate strength evidence`",
        "",
        "## Results",
        "",
    ]
    for row in sorted(rows, key=lambda item: (int(item["source_config_id"]), float(item["behavior_temperature"]), float(item["lr"]))):
        lines.append(
            "- "
            f"cfg={row['pilot_config_id']} "
            f"source={row['source_config_id']} "
            f"scale={row['rule_score_scale']:g} "
            f"temp={row['behavior_temperature']:g} "
            f"lr={row['lr']:g} "
            f"non_top1={row['final_non_top1_selected_count']} "
            f"non_top1_pos={row['final_non_top1_positive_advantage_count']} "
            f"top1_changed={row['final_post_top1_action_changed_rate']:.6g} "
            f"kl={row['final_post_update_approx_kl']:.6g} "
            f"clip={row['final_post_update_clip_fraction']:.6g} "
            f"delta_max={row['final_post_neural_delta_abs_max']:.6g} "
            f"eval_fourth={row['eval_fourth_rate']:.6g} "
            f"deal_in={row['eval_learner_deal_in_rate']:.6g}"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `temperature_pilot.json`",
            "- `summary.csv`",
            "- `iterations.csv`",
            "- `batch_steps.csv`",
            "- `advantage_audit.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(1, len(values))


if __name__ == "__main__":
    main()
