#!/usr/bin/env python3
"""Diagnostic-only rule_score_scale probe for KeqingRL checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from keqingrl import DiscardOnlyMahjongEnv, run_fixed_seed_evaluation_smoke
from keqingrl.distribution import MaskedCategorical
from keqingrl.learning_signal import seed_registry_hash, tensor_stats
from keqingrl.selfplay import build_episodes_ppo_batch, collect_selfplay_episodes
from scripts.probe_keqingrl_sampling_diversity import (
    _candidate_summary,
    _load_candidates,
    _load_policy,
    _opponent_pool,
    _seed_registry,
    _seed_registry_id,
    _to_csvable,
    _to_jsonable,
    _write_csv,
    _write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run diagnostic-only rule_score_scale probes")
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--source-config-ids", type=int, nargs="+", default=(93, 57, 8))
    parser.add_argument("--rerun-config-ids", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scales", type=float, nargs="+", default=(1.0, 0.5, 0.25, 0.1))
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=202604260000)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--learner-seats", type=int, nargs="+", default=(0,))
    parser.add_argument("--eval-seat-rotation", type=int, nargs="+", default=(0,))
    parser.add_argument("--max-kyokus", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device is not None else torch.device("cpu")
    candidates = _load_candidates(args)
    summary_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        policy = _load_policy(candidate, device)
        opponent_pool = _opponent_pool(str(candidate["opponent_mode"]))
        for scale in args.scales:
            policy.rule_score_scale = float(scale)
            policy.eval()
            torch.manual_seed(int(args.seed_base + int(candidate["source_config_id"]) * 1000 + int(float(scale) * 1000)))
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(int(args.seed_base + int(candidate["source_config_id"]) * 1000 + int(float(scale) * 1000)))
            episodes = collect_selfplay_episodes(
                DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
                policy,
                num_episodes=args.episodes,
                opponent_pool=opponent_pool,
                learner_seats=tuple(int(seat) for seat in args.learner_seats),
                seed=args.seed_base,
                seed_stride=args.seed_stride,
                greedy=False,
                max_steps=args.max_steps,
                device=device,
            )
            _advantages, _returns, _prepared_steps, batch = build_episodes_ppo_batch(
                episodes,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                include_rank_targets=True,
                strict_metadata=True,
            )
            batch = batch.to(device)
            diagnostics = _scale_diagnostics(policy, batch, clip_eps=args.clip_eps)
            eval_metrics = run_fixed_seed_evaluation_smoke(
                DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
                policy,
                num_games=args.eval_episodes,
                seed=args.seed_base,
                seed_stride=args.seed_stride,
                seat_rotation=tuple(int(seat) for seat in args.eval_seat_rotation),
                opponent_pool=opponent_pool,
                opponent_name=str(candidate["opponent_mode"]),
                max_steps=args.max_steps,
                greedy=True,
                reuse_training_rollout=False,
                device=device,
            )
            row = {
                **_candidate_summary(candidate),
                "source_type": "checkpoint",
                "training": False,
                "diagnostic_only": True,
                "rule_score_scale": float(scale),
                "episodes": int(args.episodes),
                "eval_episodes": int(args.eval_episodes),
                "learner_seats": list(int(seat) for seat in args.learner_seats),
                "eval_seat_rotation": list(int(seat) for seat in args.eval_seat_rotation),
                "seed_registry_id": _seed_registry_id(args),
                "seed_hash": seed_registry_hash(_seed_registry(args)),
                "eval_rank_pt": eval_metrics.rank_pt,
                "eval_mean_rank": eval_metrics.average_rank,
                "eval_fourth_rate": eval_metrics.fourth_rate,
                "eval_learner_deal_in_rate": eval_metrics.learner_deal_in_rate,
                "eval_learner_win_rate": eval_metrics.learner_win_rate,
                "illegal_action_rate_fail_closed": eval_metrics.illegal_action_rate_fail_closed,
                "fallback_rate_fail_closed": eval_metrics.fallback_rate_fail_closed,
                "forced_terminal_missed_fail_closed": eval_metrics.forced_terminal_missed_fail_closed,
                "forced_terminal_preempt_count": eval_metrics.forced_terminal_preempt_count,
                "autopilot_terminal_count": eval_metrics.autopilot_terminal_count,
                **diagnostics,
            }
            summary_rows.append(row)
            print(
                "scale "
                f"source={candidate['source_config_id']} "
                f"rerun={candidate['rerun_config_id']} "
                f"scale={float(scale):g} "
                f"top1_changed={row['top1_action_changed_rate']:.6g} "
                f"rule_agree={row['rule_agreement']:.6g} "
                f"remaining_margin={row['mean_delta_needed_to_flip_top1']:.6g} "
                f"eval_rank={row['eval_rank_pt']:.6g}",
                flush=True,
            )

    payload = {
        "mode": "rule_score_scale_probe",
        "source_type": "checkpoint",
        "training": False,
        "diagnostic_only": True,
        "candidate_summary": str(args.candidate_summary),
        "source_config_ids": list(args.source_config_ids or ()),
        "rerun_config_ids": args.rerun_config_ids,
        "scales": [float(value) for value in args.scales],
        "episodes": int(args.episodes),
        "eval_episodes": int(args.eval_episodes),
        "seed_registry_id": _seed_registry_id(args),
        "seed_hash": seed_registry_hash(_seed_registry(args)),
        "summaries": summary_rows,
    }
    _write_json(args.output_dir / "rule_score_scale_probe.json", payload)
    _write_csv(args.output_dir / "summary.csv", summary_rows)
    (args.output_dir / "summary.md").write_text(_summary_markdown(args, summary_rows), encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


def _scale_diagnostics(policy, batch, *, clip_eps: float) -> dict[str, Any]:
    with torch.no_grad():
        output = policy(batch.policy_input)
    final_logits = output.aux.get("final_logits", output.action_logits).float()
    prior_logits = batch.policy_input.prior_logits
    if prior_logits is None:
        prior_logits = output.aux.get("prior_logits")
    if prior_logits is None:
        raise ValueError("rule_score_scale diagnostics require prior_logits")
    prior_logits = prior_logits.float()
    mask = batch.policy_input.legal_action_mask.bool()
    valid_final = final_logits.masked_fill(~mask, torch.finfo(final_logits.dtype).min)
    valid_prior = prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min)
    final_top1 = valid_final.argmax(dim=-1)
    prior_top1 = valid_prior.argmax(dim=-1)
    changed = final_top1 != prior_top1
    remaining_margins: list[float] = []
    scaled_prior_margins: list[float] = []
    for row_idx in range(valid_final.shape[0]):
        row_mask = mask[row_idx]
        legal_count = int(row_mask.sum().item())
        if legal_count <= 1 or bool(changed[row_idx]):
            remaining_margins.append(0.0)
        else:
            top_idx = int(prior_top1[row_idx].item())
            competitors = valid_final[row_idx].clone()
            competitors[top_idx] = torch.finfo(competitors.dtype).min
            remaining_margins.append(float((valid_final[row_idx, top_idx] - competitors.max()).detach().cpu()))
        legal_prior = valid_prior[row_idx][row_mask]
        if legal_count <= 1:
            scaled_prior_margins.append(0.0)
        else:
            top2 = torch.topk(legal_prior, k=2).values
            scale = float(getattr(policy, "rule_score_scale", 1.0))
            scaled_prior_margins.append(float(((top2[0] - top2[1]) * scale).detach().cpu()))
    prior_dist = MaskedCategorical(valid_prior, mask)
    final_dist = MaskedCategorical(valid_final, mask)
    action_index = batch.action_index.long()
    log_prob_prior = prior_dist.log_prob(action_index)
    log_prob_final = final_dist.log_prob(action_index)
    ratio = torch.exp(log_prob_final - log_prob_prior)
    approx_kl_vs_prior = (log_prob_prior - log_prob_final).mean()
    neural_delta = output.aux.get("neural_delta")
    delta_stats = (
        tensor_stats(neural_delta.detach().abs()[mask], prefix="neural_delta_abs")
        if neural_delta is not None
        else {
            "neural_delta_abs_mean": 0.0,
            "neural_delta_abs_std": 0.0,
            "neural_delta_abs_min": 0.0,
            "neural_delta_abs_max": 0.0,
        }
    )
    margins = torch.tensor(remaining_margins, dtype=torch.float32)
    scaled_prior = torch.tensor(scaled_prior_margins, dtype=torch.float32)
    return {
        "batch_size": int(action_index.numel()),
        "rule_agreement": float(1.0 - changed.float().mean().detach().cpu()),
        "top1_action_changed_rate": float(changed.float().mean().detach().cpu()),
        "mean_delta_needed_to_flip_top1": float(margins.mean().item()) if margins.numel() else 0.0,
        "p50_delta_needed_to_flip_top1": float(margins.quantile(0.5).item()) if margins.numel() else 0.0,
        "p90_delta_needed_to_flip_top1": float(margins.quantile(0.9).item()) if margins.numel() else 0.0,
        "scaled_prior_margin_mean": float(scaled_prior.mean().item()) if scaled_prior.numel() else 0.0,
        "scaled_prior_margin_p50": float(scaled_prior.quantile(0.5).item()) if scaled_prior.numel() else 0.0,
        "approx_kl_vs_prior": float(approx_kl_vs_prior.detach().cpu()),
        "clip_fraction_vs_prior": float(((ratio - 1.0).abs() > float(clip_eps)).float().mean().detach().cpu()),
        "ratio_vs_prior_mean": float(ratio.mean().detach().cpu()),
        "ratio_vs_prior_std": float(ratio.std(unbiased=False).detach().cpu()),
        **delta_stats,
    }


def _summary_markdown(args: argparse.Namespace, rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# KeqingRL Rule Score Scale Probe",
        "",
        "source_type: `checkpoint`",
        "training: `false`",
        "diagnostic_only: `true`",
        f"candidate_summary: `{args.candidate_summary}`",
        f"episodes: `{args.episodes}`",
        f"eval_episodes: `{args.eval_episodes}`",
        f"scales: `{','.join(str(float(value)) for value in args.scales)}`",
        f"seed_registry_id: `{_seed_registry_id(args)}`",
        f"seed_hash: `{seed_registry_hash(_seed_registry(args))}`",
        "",
        "## Results",
        "",
    ]
    for row in rows:
        lines.append(
            "- "
            f"source={row['source_config_id']} "
            f"rerun={row['rerun_config_id']} "
            f"opponent={row['opponent_mode']} "
            f"scale={row['rule_score_scale']:g} "
            f"top1_changed={row['top1_action_changed_rate']:.6g} "
            f"rule_agree={row['rule_agreement']:.6g} "
            f"remaining_margin={row['mean_delta_needed_to_flip_top1']:.6g} "
            f"scaled_prior_margin={row['scaled_prior_margin_mean']:.6g} "
            f"delta_max={row['neural_delta_abs_max']:.6g} "
            f"kl_vs_prior={row['approx_kl_vs_prior']:.6g} "
            f"clip_vs_prior={row['clip_fraction_vs_prior']:.6g} "
            f"eval_rank={row['eval_rank_pt']:.6g} "
            f"fourth={row['eval_fourth_rate']:.6g} "
            f"deal_in={row['eval_learner_deal_in_rate']:.6g}"
        )
    lines.extend(["", "## Artifacts", "", "- `rule_score_scale_probe.json`", "- `summary.csv`"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
