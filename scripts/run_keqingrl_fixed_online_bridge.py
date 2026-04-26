#!/usr/bin/env python3
"""Fixed-batch overfit to online-rollout bridge diagnostics for KeqingRL."""

from __future__ import annotations

import argparse
import csv
import copy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from keqingrl import DiscardOnlyMahjongEnv
from keqingrl.learning_signal import seed_registry_hash
from keqingrl.ppo import compute_ppo_loss, ppo_update
from keqingrl.selfplay import build_episodes_ppo_batch, collect_selfplay_episodes
from scripts.probe_keqingrl_sampling_diversity import (
    _candidate_summary,
    _load_candidates,
    _load_policy,
    _opponent_pool,
    _seed_registry,
    _seed_registry_id,
    _write_csv,
    _write_json,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-batch to online PPO bridge diagnostics")
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--source-config-ids", type=int, nargs="+", default=(93, 57, 8))
    parser.add_argument("--rerun-config-ids", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--online-episodes", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=202604260000)
    parser.add_argument("--online-seed-base", type=int, default=202604270000)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--learner-seats", type=int, nargs="+", default=(0,))
    parser.add_argument("--max-kyokus", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--overfit-target-top1", type=float, default=0.01)
    parser.add_argument("--overfit-max-epochs", type=int, default=96)
    parser.add_argument("--overfit-lrs", type=float, nargs="+", default=(1e-3, 3e-3))
    parser.add_argument("--overfit-normalize-advantages", nargs="+", default=("true", "false"))
    parser.add_argument("--overfit-entropy-coef", type=float, default=0.0)
    parser.add_argument("--online-lr", type=float, default=1e-3)
    parser.add_argument("--online-update-epochs", type=int, default=4)
    parser.add_argument("--online-entropy-coef", type=float, default=0.0)
    parser.add_argument("--online-normalize-advantages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device is not None else torch.device("cpu")
    candidates = _load_candidates(args)
    summary_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        base_policy = _load_policy(candidate, device)
        opponent_pool = _opponent_pool(str(candidate["opponent_mode"]))
        fixed_batch = _collect_batch(
            base_policy,
            opponent_pool,
            args,
            episodes=args.episodes,
            seed=args.seed_base,
            device=device,
        )
        fixed_before = _policy_delta_stats(base_policy, fixed_batch)
        overfit_policy, overfit_curve, overfit_config = _overfit_until_move(
            base_policy,
            fixed_batch,
            args,
        )
        fixed_after_overfit = _policy_delta_stats(overfit_policy, fixed_batch)
        checkpoint_path = args.output_dir / "checkpoints" / f"source_{candidate['source_config_id']}_rerun_{candidate['rerun_config_id']}_fixed_batch_overfit.pt"
        torch.save(
            {
                "source_type": "fixed_batch_overfit_bridge",
                "candidate": _candidate_summary(candidate),
                "overfit_config": overfit_config,
                "policy_state_dict": overfit_policy.state_dict(),
            },
            checkpoint_path,
        )
        online_batch = _collect_batch(
            overfit_policy,
            opponent_pool,
            args,
            episodes=args.online_episodes,
            seed=args.online_seed_base,
            device=device,
        )
        online_pre = _policy_delta_stats(overfit_policy, online_batch)
        fixed_recheck_pre_online = _policy_delta_stats(overfit_policy, fixed_batch)
        online_losses = _run_online_update(overfit_policy, online_batch, args)
        online_post = _policy_delta_stats(overfit_policy, online_batch)
        fixed_recheck_post_online = _policy_delta_stats(overfit_policy, fixed_batch)
        post_online_checkpoint_path = args.output_dir / "checkpoints" / f"source_{candidate['source_config_id']}_rerun_{candidate['rerun_config_id']}_post_online.pt"
        torch.save(
            {
                "source_type": "fixed_batch_overfit_bridge_post_online",
                "candidate": _candidate_summary(candidate),
                "overfit_config": overfit_config,
                "online_config": {
                    "lr": float(args.online_lr),
                    "update_epochs": int(args.online_update_epochs),
                    "normalize_advantages": bool(args.online_normalize_advantages),
                    "entropy_coef": float(args.online_entropy_coef),
                    "value_coef": 0.0,
                    "rank_coef": 0.0,
                    "rule_kl_coef": 0.0,
                },
                "policy_state_dict": overfit_policy.state_dict(),
            },
            post_online_checkpoint_path,
        )
        row = {
            **_candidate_summary(candidate),
            "source_type": "fixed_batch_overfit_bridge",
            "training": True,
            "fixed_seed_registry_id": _seed_registry_id(args),
            "fixed_seed_hash": seed_registry_hash(_seed_registry(args)),
            "online_seed_registry_id": _online_seed_registry_id(args),
            "online_seed_hash": seed_registry_hash(_online_seed_registry(args)),
            "fixed_episodes": int(args.episodes),
            "online_episodes": int(args.online_episodes),
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": _file_sha256(checkpoint_path),
            "post_online_checkpoint_path": str(post_online_checkpoint_path),
            "post_online_checkpoint_sha256": _file_sha256(post_online_checkpoint_path),
            **{f"fixed_before_{key}": value for key, value in fixed_before.items()},
            **{f"fixed_after_overfit_{key}": value for key, value in fixed_after_overfit.items()},
            **{f"online_pre_{key}": value for key, value in online_pre.items()},
            **{f"online_post_{key}": value for key, value in online_post.items()},
            **{f"fixed_post_online_{key}": value for key, value in fixed_recheck_post_online.items()},
            "fixed_recheck_pre_online_top1_action_changed_rate": fixed_recheck_pre_online["top1_action_changed_rate"],
            "overfit_passed_target": bool(fixed_after_overfit["top1_action_changed_rate"] >= float(args.overfit_target_top1)),
            "overfit_epochs": int(overfit_config.get("epochs", 0)),
            "overfit_lr": overfit_config.get("lr"),
            "overfit_normalize_advantages": overfit_config.get("normalize_advantages"),
            "online_lr": float(args.online_lr),
            "online_update_epochs": int(args.online_update_epochs),
            "online_normalize_advantages": bool(args.online_normalize_advantages),
            "online_final_approx_kl": _loss_float(online_losses[-1].approx_kl) if online_losses else 0.0,
            "online_final_clip_fraction": _loss_float(online_losses[-1].clip_fraction) if online_losses else 0.0,
            "online_final_policy_loss": _loss_float(online_losses[-1].policy_loss) if online_losses else 0.0,
        }
        summary_rows.append(row)
        for curve in overfit_curve:
            curve_rows.append({**_candidate_summary(candidate), **curve})
        print(
            "bridge "
            f"source={candidate['source_config_id']} "
            f"fixed_after={row['fixed_after_overfit_top1_action_changed_rate']:.6g} "
            f"online_pre={row['online_pre_top1_action_changed_rate']:.6g} "
            f"online_post={row['online_post_top1_action_changed_rate']:.6g} "
            f"fixed_post_online={row['fixed_post_online_top1_action_changed_rate']:.6g}",
            flush=True,
        )

    payload = {
        "mode": "fixed_batch_online_bridge",
        "source_type": "fixed_batch_overfit_bridge",
        "candidate_summary": str(args.candidate_summary),
        "source_config_ids": list(args.source_config_ids or ()),
        "rerun_config_ids": args.rerun_config_ids,
        "fixed_seed_registry_id": _seed_registry_id(args),
        "fixed_seed_hash": seed_registry_hash(_seed_registry(args)),
        "online_seed_registry_id": _online_seed_registry_id(args),
        "online_seed_hash": seed_registry_hash(_online_seed_registry(args)),
        "summaries": summary_rows,
    }
    _write_json(args.output_dir / "fixed_online_bridge.json", payload)
    _write_csv(args.output_dir / "summary.csv", summary_rows)
    _write_csv(args.output_dir / "overfit_curve.csv", curve_rows)
    (args.output_dir / "summary.md").write_text(_summary_markdown(args, summary_rows), encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


def _collect_batch(policy, opponent_pool, args: argparse.Namespace, *, episodes: int, seed: int, device: torch.device):
    collected = collect_selfplay_episodes(
        DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
        policy,
        num_episodes=int(episodes),
        opponent_pool=opponent_pool,
        learner_seats=tuple(int(seat) for seat in args.learner_seats),
        seed=int(seed),
        seed_stride=int(args.seed_stride),
        greedy=False,
        max_steps=int(args.max_steps),
        device=device,
    )
    _advantages, _returns, _prepared_steps, batch = build_episodes_ppo_batch(
        collected,
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        include_rank_targets=True,
        strict_metadata=True,
    )
    return batch.to(device)


def _overfit_until_move(base_policy, fixed_batch, args: argparse.Namespace):
    normalize_options = [_parse_bool(value) for value in args.overfit_normalize_advantages]
    best_policy = copy.deepcopy(base_policy)
    best_curve: list[dict[str, Any]] = []
    best_config: dict[str, Any] = {"epochs": 0}
    best_top1 = -1.0
    for normalize_advantages in normalize_options:
        for lr in args.overfit_lrs:
            probe_policy = copy.deepcopy(base_policy)
            optimizer = torch.optim.Adam(probe_policy.parameters(), lr=float(lr))
            curve: list[dict[str, Any]] = []
            for epoch in range(1, int(args.overfit_max_epochs) + 1):
                optimizer.zero_grad(set_to_none=True)
                loss = compute_ppo_loss(
                    probe_policy,
                    fixed_batch,
                    clip_eps=float(args.clip_eps),
                    value_coef=0.0,
                    entropy_coef=float(args.overfit_entropy_coef),
                    rank_coef=0.0,
                    rule_kl_coef=0.0,
                    normalize_advantages=bool(normalize_advantages),
                )
                loss.total_loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(probe_policy.parameters(), float(args.max_grad_norm))
                optimizer.step()
                stats = _policy_delta_stats(probe_policy, fixed_batch)
                curve_row = {
                    "overfit_lr": float(lr),
                    "overfit_normalize_advantages": bool(normalize_advantages),
                    "epoch": epoch,
                    "loss": _loss_float(loss.total_loss),
                    "policy_loss": _loss_float(loss.policy_loss),
                    **stats,
                }
                curve.append(curve_row)
                if stats["top1_action_changed_rate"] > best_top1:
                    best_top1 = stats["top1_action_changed_rate"]
                    best_policy = copy.deepcopy(probe_policy)
                    best_curve = list(curve)
                    best_config = {
                        "lr": float(lr),
                        "normalize_advantages": bool(normalize_advantages),
                        "epochs": epoch,
                        "target_top1": float(args.overfit_target_top1),
                    }
                if stats["top1_action_changed_rate"] >= float(args.overfit_target_top1):
                    return probe_policy, curve, {
                        "lr": float(lr),
                        "normalize_advantages": bool(normalize_advantages),
                        "epochs": epoch,
                        "target_top1": float(args.overfit_target_top1),
                    }
    return best_policy, best_curve, best_config


def _run_online_update(policy, online_batch, args: argparse.Namespace):
    optimizer = torch.optim.Adam(policy.parameters(), lr=float(args.online_lr))
    losses = []
    for _ in range(int(args.online_update_epochs)):
        losses.append(
            ppo_update(
                policy,
                optimizer,
                online_batch,
                clip_eps=float(args.clip_eps),
                value_coef=0.0,
                entropy_coef=float(args.online_entropy_coef),
                rank_coef=0.0,
                rule_kl_coef=0.0,
                normalize_advantages=bool(args.online_normalize_advantages),
                max_grad_norm=float(args.max_grad_norm) if args.max_grad_norm is not None else None,
            )
        )
    return losses


def _policy_delta_stats(policy, batch) -> dict[str, float]:
    with torch.no_grad():
        output = policy(batch.policy_input)
    mask = batch.policy_input.legal_action_mask.bool()
    neural_delta = output.aux.get("neural_delta")
    legal_delta = neural_delta.masked_select(mask) if neural_delta is not None else torch.empty(0, device=mask.device)
    prior_logits = output.aux.get("prior_logits")
    if prior_logits is None:
        prior_logits = batch.policy_input.prior_logits
    current_top1 = output.action_logits.masked_fill(~mask, torch.finfo(output.action_logits.dtype).min).argmax(dim=-1)
    if prior_logits is None:
        changed = torch.zeros_like(current_top1, dtype=torch.bool)
    else:
        prior_top1 = prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min).argmax(dim=-1)
        changed = current_top1 != prior_top1
    return {
        "batch_size": int(batch.action_index.numel()),
        "top1_action_changed_rate": float(changed.float().mean().detach().cpu()),
        "rule_agreement": float(1.0 - changed.float().mean().detach().cpu()),
        "neural_delta_abs_mean": 0.0 if legal_delta.numel() == 0 else float(legal_delta.abs().mean().detach().cpu()),
        "neural_delta_abs_max": 0.0 if legal_delta.numel() == 0 else float(legal_delta.abs().max().detach().cpu()),
    }


def _summary_markdown(args: argparse.Namespace, rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# KeqingRL Fixed-Batch To Online Bridge",
        "",
        "source_type: `fixed_batch_overfit_bridge`",
        f"candidate_summary: `{args.candidate_summary}`",
        f"fixed_episodes: `{args.episodes}`",
        f"online_episodes: `{args.online_episodes}`",
        f"fixed_seed_registry_id: `{_seed_registry_id(args)}`",
        f"online_seed_registry_id: `{_online_seed_registry_id(args)}`",
        "",
        "## Results",
        "",
    ]
    for row in rows:
        lines.append(
            "- "
            f"source={row['source_config_id']} "
            f"rerun={row['rerun_config_id']} "
            f"overfit_pass={row['overfit_passed_target']} "
            f"overfit_epochs={row['overfit_epochs']} "
            f"fixed_before={row['fixed_before_top1_action_changed_rate']:.6g} "
            f"fixed_after={row['fixed_after_overfit_top1_action_changed_rate']:.6g} "
            f"online_pre={row['online_pre_top1_action_changed_rate']:.6g} "
            f"online_post={row['online_post_top1_action_changed_rate']:.6g} "
            f"fixed_post_online={row['fixed_post_online_top1_action_changed_rate']:.6g} "
            f"online_delta_max={row['online_post_neural_delta_abs_max']:.6g} "
            f"checkpoint={row['checkpoint_path']} "
            f"post_online_checkpoint={row['post_online_checkpoint_path']}"
        )
    lines.extend(["", "## Artifacts", "", "- `fixed_online_bridge.json`", "- `summary.csv`", "- `overfit_curve.csv`", "- `checkpoints/*.pt`"])
    return "\n".join(lines) + "\n"


def _online_seed_registry(args: argparse.Namespace) -> list[int]:
    return [int(args.online_seed_base + idx * args.seed_stride) for idx in range(args.online_episodes)]


def _online_seed_registry_id(args: argparse.Namespace) -> str:
    return f"base={args.online_seed_base}:stride={args.seed_stride}:count={args.online_episodes}"


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid boolean value: {value}")


def _loss_float(value: Any) -> float:
    if value is None:
        return 0.0
    if hasattr(value, "detach"):
        return float(value.detach().cpu())
    return float(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    main()
