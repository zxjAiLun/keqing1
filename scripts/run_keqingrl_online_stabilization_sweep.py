#!/usr/bin/env python3
"""Sweep mild online-update settings from a fixed-batch overfit bridge checkpoint."""

from __future__ import annotations

import argparse
import copy
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from keqingrl import DiscardOnlyMahjongEnv, RulePriorDeltaPolicy, run_fixed_seed_evaluation_smoke
from keqingrl.metadata import resolve_rule_score_scale_metadata
from keqingrl.ppo import compute_ppo_loss, ppo_update
from keqingrl.selfplay import build_episodes_ppo_batch, collect_selfplay_episodes
from scripts.probe_keqingrl_sampling_diversity import (
    _opponent_pool,
    _to_csvable,
    _to_jsonable,
    _write_csv,
    _write_json,
)
from scripts.run_keqingrl_fixed_online_bridge import _file_sha256, _loss_float, _policy_delta_stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run 93-only mild online stabilization sweep")
    parser.add_argument("--bridge-summary", type=Path, required=True)
    parser.add_argument("--source-config-id", type=int, default=93)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--online-episodes", type=int, default=16)
    parser.add_argument("--online-seed-base", type=int, default=202604270000)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--learner-seats", type=int, nargs="+", default=(0,))
    parser.add_argument("--max-kyokus", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--online-lrs", type=float, nargs="+", default=(1e-4, 3e-4, 1e-3))
    parser.add_argument("--online-update-epochs", type=int, nargs="+", default=(1, 2, 4))
    parser.add_argument("--online-rule-kl-coefs", type=float, nargs="+", default=(0.0, 0.001, 0.005))
    parser.add_argument("--online-entropy-coefs", type=float, nargs="+", default=(0.0, 0.005))
    parser.add_argument("--max-grad-norms", type=float, nargs="+", default=(0.25, 0.5, 1.0))
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--value-coef", type=float, default=0.0)
    parser.add_argument("--rank-coef", type=float, default=0.0)
    parser.add_argument("--stable-kl-threshold", type=float, default=0.03)
    parser.add_argument("--stable-clip-threshold", type=float, default=0.3)
    parser.add_argument("--target-top1-min", type=float, default=1e-9)
    parser.add_argument("--delta-max-limit", type=float, default=4.0)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--eval-top-k", type=int, default=10)
    parser.add_argument("--eval-seat-rotation", type=int, nargs="+", default=(0,))
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device is not None else torch.device("cpu")
    bridge_row = _select_bridge_row(args.bridge_summary, args.source_config_id)
    base_policy = _load_bridge_policy(bridge_row, device)
    rule_score_scale = float(getattr(base_policy, "rule_score_scale", 1.0))
    opponent_pool = _opponent_pool(str(bridge_row["opponent_mode"]))
    online_batch = _collect_online_batch(base_policy, opponent_pool, args, device)
    pre_stats = _policy_delta_stats(base_policy, online_batch)

    rows: list[dict[str, Any]] = []
    for lr in args.online_lrs:
        for update_epochs in args.online_update_epochs:
            for rule_kl_coef in args.online_rule_kl_coefs:
                for entropy_coef in args.online_entropy_coefs:
                    for max_grad_norm in args.max_grad_norms:
                        policy = copy.deepcopy(base_policy)
                        optimizer = torch.optim.Adam(policy.parameters(), lr=float(lr))
                        losses = []
                        for _ in range(int(update_epochs)):
                            losses.append(
                                ppo_update(
                                    policy,
                                    optimizer,
                                    online_batch,
                                    clip_eps=float(args.clip_eps),
                                    value_coef=float(args.value_coef),
                                    entropy_coef=float(entropy_coef),
                                    rank_coef=float(args.rank_coef),
                                    rule_kl_coef=float(rule_kl_coef),
                                    normalize_advantages=bool(args.normalize_advantages),
                                    max_grad_norm=float(max_grad_norm),
                                )
                            )
                        post_stats = _policy_delta_stats(policy, online_batch)
                        final_loss = _compute_online_loss(
                            policy,
                            online_batch,
                            args,
                            entropy_coef=float(entropy_coef),
                            rule_kl_coef=float(rule_kl_coef),
                        )
                        row = {
                            "source_config_id": int(bridge_row["source_config_id"]),
                            "rerun_config_id": int(bridge_row["rerun_config_id"]),
                            "source_type": "fixed_batch_overfit_bridge_online_stabilization",
                            "bridge_checkpoint_path": bridge_row["checkpoint_path"],
                            "bridge_checkpoint_sha256": bridge_row.get("checkpoint_sha256") or _file_sha256(Path(bridge_row["checkpoint_path"])),
                            "config_path": bridge_row["config_path"],
                            "opponent_mode": bridge_row["opponent_mode"],
                            "rule_score_scale": rule_score_scale,
                            "rule_score_scale_version": "keqingrl_rule_score_scale_v1",
                            "online_lr": float(lr),
                            "online_update_epochs": int(update_epochs),
                            "online_rule_kl_coef": float(rule_kl_coef),
                            "online_entropy_coef": float(entropy_coef),
                            "max_grad_norm": float(max_grad_norm),
                            "normalize_advantages": bool(args.normalize_advantages),
                            "value_coef": float(args.value_coef),
                            "rank_coef": float(args.rank_coef),
                            "online_update_loss_count": len(losses),
                            **{f"online_pre_{key}": value for key, value in pre_stats.items()},
                            **{f"online_post_{key}": value for key, value in post_stats.items()},
                            "online_final_approx_kl": _loss_float(final_loss.approx_kl),
                            "online_final_clip_fraction": _loss_float(final_loss.clip_fraction),
                            "online_final_policy_loss": _loss_float(final_loss.policy_loss),
                            "online_final_rule_kl": _loss_float(final_loss.rule_kl),
                        }
                        row.update(_stability_flags(row, args))
                        rows.append(row)
                        print(
                            "stabilize "
                            f"lr={float(lr):g} ep={int(update_epochs)} "
                            f"rkl={float(rule_kl_coef):g} ent={float(entropy_coef):g} "
                            f"gn={float(max_grad_norm):g} "
                            f"top1={row['online_post_top1_action_changed_rate']:.6g} "
                            f"kl={row['online_final_approx_kl']:.6g} "
                            f"clip={row['online_final_clip_fraction']:.6g} "
                            f"status={row['stabilization_status']}",
                            flush=True,
                        )

    eval_rows = _evaluate_selected(rows, bridge_row, online_batch, base_policy, opponent_pool, args, device)
    payload = {
        "mode": "online_stabilization_sweep",
        "source_config_id": int(args.source_config_id),
        "bridge_summary": str(args.bridge_summary),
        "online_seed_registry_id": _online_seed_registry_id(args),
        "rule_score_scale": rule_score_scale,
        "rule_score_scale_version": "keqingrl_rule_score_scale_v1",
        "grid_count": len(rows),
        "eval_count": len(eval_rows),
        "summaries": rows,
        "eval_rows": eval_rows,
    }
    _write_json(args.output_dir / "online_stabilization_sweep.json", payload)
    _write_csv(args.output_dir / "summary.csv", rows)
    _write_csv(args.output_dir / "eval.csv", eval_rows)
    (args.output_dir / "summary.md").write_text(_summary_markdown(args, rows, eval_rows), encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


def _select_bridge_row(path: Path, source_config_id: int) -> dict[str, str]:
    rows = _read_csv(path)
    matches = [row for row in rows if int(row["source_config_id"]) == int(source_config_id)]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one bridge row for source_config_id={source_config_id}, got {len(matches)}")
    return matches[0]


def _load_bridge_policy(row: dict[str, str], device: torch.device) -> RulePriorDeltaPolicy:
    config = json.loads(Path(row["config_path"]).read_text(encoding="utf-8"))
    policy = RulePriorDeltaPolicy(
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_res_blocks=int(config["model"]["num_res_blocks"]),
        dropout=float(config["model"].get("dropout", 0.0)),
    ).to(device)
    checkpoint = torch.load(Path(row["checkpoint_path"]), map_location=device)
    policy.rule_score_scale = _checkpoint_rule_score_scale(checkpoint)
    policy.load_state_dict(checkpoint["policy_state_dict"])
    policy.eval()
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


def _collect_online_batch(policy, opponent_pool, args: argparse.Namespace, device: torch.device):
    _seed_torch_sampling(int(args.online_seed_base))
    episodes = collect_selfplay_episodes(
        DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
        policy,
        num_episodes=int(args.online_episodes),
        opponent_pool=opponent_pool,
        learner_seats=tuple(int(seat) for seat in args.learner_seats),
        seed=int(args.online_seed_base),
        seed_stride=int(args.seed_stride),
        greedy=False,
        max_steps=int(args.max_steps),
        device=device,
    )
    _advantages, _returns, _prepared_steps, batch = build_episodes_ppo_batch(
        episodes,
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        include_rank_targets=True,
        strict_metadata=True,
    )
    return batch.to(device)


def _seed_torch_sampling(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _compute_online_loss(
    policy,
    online_batch,
    args: argparse.Namespace,
    *,
    entropy_coef: float,
    rule_kl_coef: float,
):
    with torch.no_grad():
        return compute_ppo_loss(
            policy,
            online_batch,
            clip_eps=float(args.clip_eps),
            value_coef=float(args.value_coef),
            entropy_coef=float(entropy_coef),
            rank_coef=float(args.rank_coef),
            rule_kl_coef=float(rule_kl_coef),
            normalize_advantages=bool(args.normalize_advantages),
        )


def _stability_flags(row: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    top1 = float(row["online_post_top1_action_changed_rate"])
    kl = float(row["online_final_approx_kl"])
    clip = float(row["online_final_clip_fraction"])
    delta_max = float(row["online_post_neural_delta_abs_max"])
    moves = top1 > float(args.target_top1_min)
    kl_stable = kl < float(args.stable_kl_threshold)
    clip_stable = clip < float(args.stable_clip_threshold)
    delta_non_explosive = delta_max < float(args.delta_max_limit)
    if moves and kl_stable and clip_stable and delta_non_explosive:
        status = "STABLE_MOVES"
    elif moves and (not kl_stable or not clip_stable):
        status = "MOVES_BUT_UNSTABLE"
    elif not moves and kl_stable and clip_stable:
        status = "STABLE_NO_MOVE"
    else:
        status = "UNSTABLE_NO_MOVE"
    return {
        "moves_top1": moves,
        "kl_stable": kl_stable,
        "clip_stable": clip_stable,
        "delta_non_explosive": delta_non_explosive,
        "stabilization_status": status,
    }


def _evaluate_selected(
    rows: list[dict[str, Any]],
    bridge_row: dict[str, str],
    online_batch,
    base_policy,
    opponent_pool,
    args: argparse.Namespace,
    device: torch.device,
) -> list[dict[str, Any]]:
    stable = [row for row in rows if row["stabilization_status"] == "STABLE_MOVES"]
    fallback = [row for row in rows if bool(row["moves_top1"])]
    selected = stable or sorted(
        fallback,
        key=lambda row: (
            float(row["online_final_approx_kl"]),
            float(row["online_final_clip_fraction"]),
            -float(row["online_post_top1_action_changed_rate"]),
        ),
    )
    selected = selected[: max(0, int(args.eval_top_k))]
    eval_rows: list[dict[str, Any]] = []
    for row in selected:
        policy = copy.deepcopy(base_policy)
        optimizer = torch.optim.Adam(policy.parameters(), lr=float(row["online_lr"]))
        for _ in range(int(row["online_update_epochs"])):
            ppo_update(
                policy,
                optimizer,
                online_batch,
                clip_eps=float(args.clip_eps),
                value_coef=float(args.value_coef),
                entropy_coef=float(row["online_entropy_coef"]),
                rank_coef=float(args.rank_coef),
                rule_kl_coef=float(row["online_rule_kl_coef"]),
                normalize_advantages=bool(args.normalize_advantages),
                max_grad_norm=float(row["max_grad_norm"]),
            )
        metrics = run_fixed_seed_evaluation_smoke(
            DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
            policy,
            num_games=int(args.eval_episodes),
            seed=int(args.online_seed_base),
            seed_stride=int(args.seed_stride),
            seat_rotation=tuple(int(seat) for seat in args.eval_seat_rotation),
            opponent_pool=opponent_pool,
            opponent_name=str(bridge_row["opponent_mode"]),
            max_steps=int(args.max_steps),
            greedy=True,
            reuse_training_rollout=False,
            device=device,
        )
        eval_rows.append(
            {
                **{key: row[key] for key in row if key.startswith("online_") or key in {"source_config_id", "rerun_config_id", "stabilization_status", "max_grad_norm", "moves_top1", "kl_stable", "clip_stable", "delta_non_explosive"}},
                "rule_score_scale": row["rule_score_scale"],
                "rule_score_scale_version": row["rule_score_scale_version"],
                "eval_rank_pt": metrics.rank_pt,
                "eval_mean_rank": metrics.average_rank,
                "eval_fourth_rate": metrics.fourth_rate,
                "eval_learner_deal_in_rate": metrics.learner_deal_in_rate,
                "eval_learner_win_rate": metrics.learner_win_rate,
                "eval_scope": f"fixed-seed smoke; learner seats {','.join(str(int(seat)) for seat in args.eval_seat_rotation)} only",
                "eval_strength_note": "sanity check only; not duplicate strength evidence",
                "illegal_action_rate_fail_closed": metrics.illegal_action_rate_fail_closed,
                "fallback_rate_fail_closed": metrics.fallback_rate_fail_closed,
                "forced_terminal_missed_fail_closed": metrics.forced_terminal_missed_fail_closed,
            }
        )
    return eval_rows


def _summary_markdown(args: argparse.Namespace, rows: Sequence[dict[str, Any]], eval_rows: Sequence[dict[str, Any]]) -> str:
    status_counts: dict[str, int] = {}
    for row in rows:
        status = str(row["stabilization_status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    max_top1 = max((float(row["online_post_top1_action_changed_rate"]) for row in rows), default=0.0)
    max_delta = max((float(row["online_post_neural_delta_abs_max"]) for row in rows), default=0.0)
    min_kl = min((float(row["online_final_approx_kl"]) for row in rows), default=0.0)
    max_kl = max((float(row["online_final_approx_kl"]) for row in rows), default=0.0)
    min_clip = min((float(row["online_final_clip_fraction"]) for row in rows), default=0.0)
    max_clip = max((float(row["online_final_clip_fraction"]) for row in rows), default=0.0)
    ranked = sorted(
        rows,
        key=lambda row: (
            row["stabilization_status"] != "STABLE_MOVES",
            float(row["online_final_approx_kl"]),
            float(row["online_final_clip_fraction"]),
            -float(row["online_post_top1_action_changed_rate"]),
        ),
    )
    lines = [
        "# KeqingRL Online Stabilization Sweep",
        "",
        f"source_config_id: `{args.source_config_id}`",
        f"bridge_summary: `{args.bridge_summary}`",
        f"grid_count: `{len(rows)}`",
        f"eval_count: `{len(eval_rows)}`",
        f"online_seed_registry_id: `{_online_seed_registry_id(args)}`",
        f"rule_score_scale: `{rows[0]['rule_score_scale'] if rows else 1.0}`",
        "rule_score_scale_version: `keqingrl_rule_score_scale_v1`",
        f"eval_scope: `fixed-seed smoke; learner seats {','.join(str(int(seat)) for seat in args.eval_seat_rotation)} only`",
        "eval_strength_note: `sanity check only; not duplicate strength evidence`",
        f"stable_kl_threshold: `{args.stable_kl_threshold}`",
        f"stable_clip_threshold: `{args.stable_clip_threshold}`",
        f"status_counts: `{status_counts}`",
        f"max_top1_action_changed_rate: `{max_top1:.6g}`",
        f"online_kl_range: `{min_kl:.6g}..{max_kl:.6g}`",
        f"online_clip_range: `{min_clip:.6g}..{max_clip:.6g}`",
        f"max_neural_delta_abs_max: `{max_delta:.6g}`",
        f"conclusion: `{'no stable top-1 movement under this grid' if max_top1 <= 0.0 else 'at least one config moved top-1'}`",
        "",
        "## Top Update Rows",
        "",
    ]
    for row in ranked[:20]:
        lines.append(
            "- "
            f"status={row['stabilization_status']} "
            f"lr={row['online_lr']:g} "
            f"epochs={row['online_update_epochs']} "
            f"rule_kl={row['online_rule_kl_coef']:g} "
            f"entropy={row['online_entropy_coef']:g} "
            f"grad={row['max_grad_norm']:g} "
            f"top1={row['online_post_top1_action_changed_rate']:.6g} "
            f"kl={row['online_final_approx_kl']:.6g} "
            f"clip={row['online_final_clip_fraction']:.6g} "
            f"delta_mean={row['online_post_neural_delta_abs_mean']:.6g} "
            f"delta_max={row['online_post_neural_delta_abs_max']:.6g}"
        )
    lines.extend(["", "## Eval Rows", ""])
    for row in eval_rows:
        lines.append(
            "- "
            f"status={row['stabilization_status']} "
            f"lr={row['online_lr']:g} "
            f"epochs={row['online_update_epochs']} "
            f"rule_kl={row['online_rule_kl_coef']:g} "
            f"entropy={row['online_entropy_coef']:g} "
            f"grad={row['max_grad_norm']:g} "
            f"top1={row['online_post_top1_action_changed_rate']:.6g} "
            f"kl={row['online_final_approx_kl']:.6g} "
            f"clip={row['online_final_clip_fraction']:.6g} "
            f"eval_rank={row['eval_rank_pt']:.6g} "
            f"fourth={row['eval_fourth_rate']:.6g} "
            f"deal_in={row['eval_learner_deal_in_rate']:.6g}"
        )
    lines.extend(["", "## Artifacts", "", "- `online_stabilization_sweep.json`", "- `summary.csv`", "- `eval.csv`"])
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _online_seed_registry_id(args: argparse.Namespace) -> str:
    return f"base={args.online_seed_base}:stride={args.seed_stride}:count={args.online_episodes}"


if __name__ == "__main__":
    main()
