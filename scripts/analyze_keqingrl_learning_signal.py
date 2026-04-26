#!/usr/bin/env python3
"""Analyze PPO learning signal for controlled discard-only KeqingRL checkpoints."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from keqingrl import DiscardOnlyMahjongEnv, RulePriorDeltaPolicy
from keqingrl.learning_signal import (
    PpoDiagnosticConfig,
    batch_diagnostic_rows,
    classify_learning_signal_blocker,
    gradient_norms,
    loss_gradient_decomposition,
    ppo_update_probe,
    seed_registry_hash,
    top1_margin_diagnostics,
)
from keqingrl.ppo import compute_ppo_loss
from keqingrl.metadata import resolve_rule_score_scale_metadata
from keqingrl.selfplay import build_episodes_ppo_batch, collect_selfplay_episodes, summarize_iteration
from scripts.run_keqingrl_discard_research_sweep import _build_opponent_pool, _file_sha256


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze KeqingRL discard-only PPO learning signal")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--source-summary", type=Path, default=None)
    parser.add_argument("--source-config-id", type=int, default=93)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=202604250000)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--seed-registry-id", default="learning_signal_20260425_v1")
    parser.add_argument("--output-dir", type=Path, default=Path("reports/keqingrl_learning_signal_20260425/source_93"))
    parser.add_argument("--opponent-mode", choices=("rule_prior_greedy", "rulebase"), default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max-kyokus", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--prior-kl-eps", type=float, default=1e-4)
    parser.add_argument("--skip-overfit", action="store_true")
    parser.add_argument("--overfit-lrs", type=float, nargs="+", default=(3e-4, 1e-3))
    parser.add_argument("--overfit-update-epochs", type=int, nargs="+", default=(16, 64))
    parser.add_argument("--overfit-entropy-coefs", type=float, nargs="+", default=(0.0, 0.001))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    candidate = _resolve_candidate(args)
    policy = _load_policy(candidate["config_path"], candidate["checkpoint_path"], device)
    policy.eval()
    opponent_mode = args.opponent_mode or str(candidate["config_key"].get("opponent_mode", "rule_prior_greedy"))
    opponent_pool, rulebase_policy = _build_opponent_pool(opponent_mode, strict_rulebase=True)
    seeds = [int(args.seed_base + idx * args.seed_stride) for idx in range(args.episodes)]

    episodes = collect_selfplay_episodes(
        DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
        policy,
        num_episodes=args.episodes,
        opponent_pool=opponent_pool,
        learner_seats=(0,),
        seed=args.seed_base,
        seed_stride=args.seed_stride,
        greedy=False,
        policy_version=0,
        max_steps=args.max_steps,
        device=device,
    )
    advantages, returns, prepared_steps, batch = build_episodes_ppo_batch(
        episodes,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        include_rank_targets=True,
        strict_metadata=True,
    )
    batch = batch.to(device)
    batch_rows, batch_summary = batch_diagnostic_rows(policy, batch, prepared_steps, episodes)
    loss_config = PpoDiagnosticConfig(
        clip_eps=args.clip_eps,
        value_coef=0.5,
        entropy_coef=0.0,
        rank_coef=0.05,
        rule_kl_coef=0.0,
        prior_kl_eps=args.prior_kl_eps,
        normalize_advantages=True,
    )
    loss_decomp = loss_gradient_decomposition(policy, batch, config=loss_config)
    probe_rows = ppo_update_probe(
        policy,
        batch,
        lrs=(3e-5, 3e-4, 1e-3),
        update_epochs=(1, 4, 16),
        clip_eps=args.clip_eps,
        normalize_advantages=True,
    )
    overfit_summary: dict[str, Any] | None = None
    overfit_rows: list[dict[str, Any]] = []
    if not args.skip_overfit:
        overfit_rows, overfit_summary = _run_overfit_matrix(policy, batch, batch_summary, args)

    smoke_losses = [
        compute_ppo_loss(
            policy,
            batch,
            clip_eps=args.clip_eps,
            value_coef=0.5,
            entropy_coef=0.0,
            rank_coef=0.05,
            rule_kl_coef=0.0,
            prior_kl_eps=args.prior_kl_eps,
            normalize_advantages=True,
        )
    ]
    metrics = summarize_iteration(episodes, smoke_losses, batch, learner_seats=(0,))
    blocker = classify_learning_signal_blocker(batch_summary, loss_decomp["summary"], overfit_summary)
    payload = {
        "source_config_id": candidate["source_config_id"],
        "config_key": candidate["config_key"],
        "checkpoint_path": str(candidate["checkpoint_path"]),
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "config_path": str(candidate["config_path"]),
        "opponent_mode": opponent_mode,
        "diagnostic_seed_registry_id": args.seed_registry_id,
        "diagnostic_seed_count": len(seeds),
        "diagnostic_seed_hash": seed_registry_hash(seeds),
        "diagnostic_seeds": seeds,
        "learner_deal_in_rate": metrics.learner_deal_in_rate,
        "learner_win_rate": metrics.learner_win_rate,
        "rulebase_fallback_count": 0 if rulebase_policy is None else rulebase_policy.rulebase_fallback_count,
        "illegal_action_rate_fail_closed": metrics.illegal_action_rate_fail_closed,
        "fallback_rate_fail_closed": metrics.fallback_rate_fail_closed,
        "forced_terminal_missed_fail_closed": metrics.forced_terminal_missed_fail_closed,
        "primary_blocker": blocker,
        "batch_summary": batch_summary,
        "loss_decomposition_summary": loss_decomp["summary"],
        "top1_margin_summary": top1_margin_diagnostics(policy, batch)["summary"],
        "overfit_summary": overfit_summary,
    }

    _write_json(args.output_dir / "learning_signal.json", payload)
    _write_json(args.output_dir / "batch_stats.json", batch_summary)
    _write_csv(args.output_dir / "batch_steps.csv", batch_rows)
    _write_json(args.output_dir / "loss_decomposition.json", loss_decomp)
    _write_csv(args.output_dir / "gradient_stats.csv", loss_decomp["rows"])
    _write_csv(args.output_dir / "ppo_update_probe.csv", probe_rows)
    if overfit_rows:
        _write_csv(args.output_dir / "overfit_curve.csv", overfit_rows)
    if overfit_summary is not None:
        _write_json(args.output_dir / "overfit_summary.json", overfit_summary)
        (args.output_dir / "overfit_summary.md").write_text(_overfit_markdown(overfit_summary), encoding="utf-8")
    summary = _summary_markdown(payload)
    (args.output_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


def _resolve_candidate(args: argparse.Namespace) -> dict[str, Any]:
    if args.checkpoint is not None and args.config is not None:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        return {
            "source_config_id": args.source_config_id,
            "config_key": config.get("config_key", {}),
            "checkpoint_path": args.checkpoint,
            "checkpoint_sha256": _file_sha256(args.checkpoint),
            "config_path": args.config,
        }
    if args.source_summary is None:
        raise ValueError("provide either --checkpoint/--config or --source-summary")
    rows = _read_csv(args.source_summary)
    for row in rows:
        raw_source = row.get("source_config_id") or row.get("config_id")
        if raw_source is not None and raw_source != "" and int(raw_source) == int(args.source_config_id):
            return {
                "source_config_id": int(args.source_config_id),
                "config_key": json.loads(row.get("config_key") or "{}"),
                "checkpoint_path": Path(row["checkpoint_path"]),
                "checkpoint_sha256": row.get("checkpoint_sha256") or _file_sha256(Path(row["checkpoint_path"])),
                "config_path": Path(row["config_path"]),
            }
    raise ValueError(f"source_config_id {args.source_config_id} not found in {args.source_summary}")


def _load_policy(config_path: Path, checkpoint_path: Path, device: torch.device) -> RulePriorDeltaPolicy:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    policy = RulePriorDeltaPolicy(
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_res_blocks=int(config["model"]["num_res_blocks"]),
        dropout=float(config["model"].get("dropout", 0.0)),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    policy.rule_score_scale = _checkpoint_rule_score_scale(checkpoint)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    return policy


def _checkpoint_rule_score_scale(checkpoint: dict[str, Any]) -> float:
    metadata = checkpoint.get("contract_metadata")
    if metadata is None:
        artifact_metadata = checkpoint.get("artifact_metadata")
        if isinstance(artifact_metadata, dict):
            metadata = artifact_metadata.get("contract_metadata")
    if metadata is None:
        metadata = {
            "rule_score_scale": checkpoint.get("rule_score_scale"),
            "rule_score_scale_version": checkpoint.get("rule_score_scale_version"),
        }
    if not isinstance(metadata, dict):
        metadata = {}
    return resolve_rule_score_scale_metadata(metadata, strict_metadata=False)


def _run_overfit_matrix(
    policy: RulePriorDeltaPolicy,
    batch,
    batch_summary: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    best_top1 = 0.0
    best_delta = 0.0
    final_actor_grad = 0.0
    final_mlp_grad = 0.0
    p10_margin = float(batch_summary.get("p10_delta_needed_to_flip_top1", 0.0))
    p50_margin = float(batch_summary.get("p50_delta_needed_to_flip_top1", 0.0))
    for normalize_advantages in (True, False):
        for lr in args.overfit_lrs:
            for update_epochs in args.overfit_update_epochs:
                for actor_only in (True, False):
                    for entropy_coef in args.overfit_entropy_coefs:
                        probe_policy = copy.deepcopy(policy)
                        optimizer = torch.optim.Adam(probe_policy.parameters(), lr=float(lr))
                        for epoch in range(1, int(update_epochs) + 1):
                            optimizer.zero_grad(set_to_none=True)
                            loss = compute_ppo_loss(
                                probe_policy,
                                batch,
                                clip_eps=args.clip_eps,
                                value_coef=0.0 if actor_only else 0.5,
                                entropy_coef=float(entropy_coef),
                                rank_coef=0.0 if actor_only else 0.05,
                                rule_kl_coef=0.0,
                                prior_kl_eps=args.prior_kl_eps,
                                normalize_advantages=bool(normalize_advantages),
                            )
                            loss.total_loss.backward()
                            grads = gradient_norms(probe_policy)
                            optimizer.step()
                            delta_stats = _policy_delta_stats(probe_policy, batch)
                            delta_abs_mean = delta_stats["neural_delta_abs_mean"]
                            delta_abs_max = delta_stats["neural_delta_abs_max"]
                            top1_changed = delta_stats["top1_action_changed_rate"]
                            best_top1 = max(best_top1, top1_changed)
                            best_delta = max(best_delta, delta_abs_max)
                            final_actor_grad = float(grads["actor_grad_norm_total"])
                            final_mlp_grad = float(grads["policy_mlp.final_linear.weight_grad_norm"])
                            rows.append(
                                {
                                    "normalize_advantages": bool(normalize_advantages),
                                    "lr": float(lr),
                                    "target_update_epochs": int(update_epochs),
                                    "actor_only": bool(actor_only),
                                    "entropy_coef": float(entropy_coef),
                                    "epoch": epoch,
                                    "total_loss": float(loss.total_loss.detach().cpu()),
                                    "policy_loss": float(loss.policy_loss.detach().cpu()),
                                    "actor_grad_norm_total": final_actor_grad,
                                    "policy_mlp_final_grad_norm": final_mlp_grad,
                                    "neural_delta_abs_mean": delta_abs_mean,
                                    "neural_delta_abs_max": delta_abs_max,
                                    "top1_action_changed_rate": top1_changed,
                                }
                            )
    pass_actor_can_move = best_top1 >= 0.01 or best_delta >= p50_margin
    gradient_dead = final_actor_grad == 0.0 or final_mlp_grad == 0.0
    prior_margin_too_large = (not pass_actor_can_move) and best_delta < p10_margin
    summary = {
        "best_top1_action_changed_rate": best_top1,
        "best_neural_delta_abs_max": best_delta,
        "p10_delta_needed_to_flip_top1": p10_margin,
        "p50_delta_needed_to_flip_top1": p50_margin,
        "pass_actor_can_move": pass_actor_can_move,
        "gradient_dead": gradient_dead,
        "prior_margin_too_large": prior_margin_too_large,
        "fail_signal_symmetric": False,
    }
    return rows, summary


def _policy_delta_stats(policy, batch) -> dict[str, float]:
    with torch.no_grad():
        output = policy(batch.policy_input)
    mask = batch.policy_input.legal_action_mask.bool()
    neural_delta = output.aux.get("neural_delta")
    if neural_delta is None:
        delta_abs_mean = 0.0
        delta_abs_max = 0.0
    else:
        legal_delta = neural_delta.masked_select(mask)
        delta_abs_mean = 0.0 if legal_delta.numel() == 0 else float(legal_delta.abs().mean().detach().cpu())
        delta_abs_max = 0.0 if legal_delta.numel() == 0 else float(legal_delta.abs().max().detach().cpu())
    prior_logits = output.aux.get("prior_logits")
    if prior_logits is None:
        prior_logits = batch.policy_input.prior_logits
    if prior_logits is None:
        changed = 0.0
    else:
        current_top1 = output.action_logits.masked_fill(~mask, torch.finfo(output.action_logits.dtype).min).argmax(dim=-1)
        prior_top1 = prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min).argmax(dim=-1)
        changed = float((current_top1 != prior_top1).float().mean().detach().cpu())
    return {
        "neural_delta_abs_mean": delta_abs_mean,
        "neural_delta_abs_max": delta_abs_max,
        "top1_action_changed_rate": changed,
    }


def _summary_markdown(payload: dict[str, Any]) -> str:
    batch = payload["batch_summary"]
    overfit = payload.get("overfit_summary") or {}
    lines = [
        "# KeqingRL Learning-Signal Diagnostics",
        "",
        f"source_config_id: `{payload['source_config_id']}`",
        f"checkpoint_path: `{payload['checkpoint_path']}`",
        f"checkpoint_sha256: `{payload['checkpoint_sha256']}`",
        f"config_key: `{json.dumps(payload['config_key'], sort_keys=True)}`",
        f"diagnostic_seed_registry_id: `{payload['diagnostic_seed_registry_id']}`",
        f"diagnostic_seed_hash: `{payload['diagnostic_seed_hash']}`",
        f"primary_blocker: `{payload['primary_blocker']}`",
        "",
        "## Batch Signal",
        "",
        f"- batch_size: `{batch['batch_size']}`",
        f"- advantage_std: `{batch['advantage_std']:.6g}`",
        f"- selected_non_top1_positive_advantage_count: `{batch['selected_non_top1_positive_advantage_count']}`",
        f"- mean_delta_needed_to_flip_top1: `{batch['mean_delta_needed_to_flip_top1']:.6g}`",
        f"- learner_deal_in_rate: `{payload['learner_deal_in_rate']:.6g}`",
        "",
        "## Stability",
        "",
        f"- rulebase_fallback_count: `{payload['rulebase_fallback_count']}`",
        f"- illegal_action_rate_fail_closed: `{payload['illegal_action_rate_fail_closed']:.6g}`",
        f"- fallback_rate_fail_closed: `{payload['fallback_rate_fail_closed']:.6g}`",
        f"- forced_terminal_missed_fail_closed: `{payload['forced_terminal_missed_fail_closed']}`",
    ]
    if overfit:
        lines.extend(
            [
                "",
                "## Fixed-Batch Overfit",
                "",
                f"- best_top1_action_changed_rate: `{overfit['best_top1_action_changed_rate']:.6g}`",
                f"- best_neural_delta_abs_max: `{overfit['best_neural_delta_abs_max']:.6g}`",
                f"- pass_actor_can_move: `{overfit['pass_actor_can_move']}`",
                f"- gradient_dead: `{overfit['gradient_dead']}`",
                f"- prior_margin_too_large: `{overfit['prior_margin_too_large']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _overfit_markdown(summary: dict[str, Any]) -> str:
    return "\n".join(["# Fixed-Batch Overfit Summary", "", *[f"- {key}: `{value}`" for key, value in summary.items()]]) + "\n"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows({key: _to_csvable(value) for key, value in row.items()} for row in rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return value


def _to_csvable(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


if __name__ == "__main__":
    main()
