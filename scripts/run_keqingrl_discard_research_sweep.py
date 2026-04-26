#!/usr/bin/env python3
"""Run controlled discard-only KeqingRL research sweeps."""

from __future__ import annotations

import argparse
import csv
import hashlib
import random
import subprocess
from dataclasses import asdict, is_dataclass, replace
import json
from pathlib import Path
from typing import Any

import torch
import keqing_core

from keqingrl import (
    DiscardOnlyMahjongEnv,
    InteractivePolicy,
    OpponentPool,
    OpponentPoolEntry,
    RulePriorDeltaPolicy,
    RulePriorPolicy,
    evaluate_policy,
    run_ppo_iteration,
)
from keqingrl.contracts import PolicyInput, PolicyOutput
from keqingrl.distribution import MaskedCategorical
from keqingrl.training import (
    _initialize_policy_from_env_observation,
    _loss_float,
    _ppo_delta_smoke_stats,
)
from keqingrl.learning_signal import (
    gradient_norms as _learning_signal_gradient_norms,
    ppo_update_probe,
    seed_registry_hash,
    tensor_stats,
    top1_margin_diagnostics,
)
from keqingrl.ppo import compute_ppo_loss, ppo_update
from keqingrl.selfplay import (
    DiscardOnlyIterationResult,
    build_episodes_ppo_batch,
    collect_selfplay_episodes,
    summarize_iteration,
)


DEFAULT_RULE_KL_COEFS = (0.0, 0.001, 0.01, 0.02, 0.05)
DEFAULT_ENTROPY_COEFS = (0.0, 0.001, 0.005, 0.01)
DEFAULT_LRS = (3e-5, 1e-4, 3e-4)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled discard-only KeqingRL PPO sweep")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/keqingrl_discard_research"))
    parser.add_argument("--mode", choices=("grid", "candidate-rerun", "learning-signal-ablation"), default="grid")
    parser.add_argument("--from-summary", type=Path, default=None)
    parser.add_argument("--source-config-ids", type=int, nargs="+", default=None)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--repeat-id", type=int, default=0)
    parser.add_argument("--eval-seed-base", type=int, default=202604250000)
    parser.add_argument("--eval-seed-stride", type=int, default=1)
    parser.add_argument("--rerun-config-id", type=int, default=None)
    parser.add_argument("--source-config-id", type=int, default=None)
    parser.add_argument("--source-report", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-kyokus", type=int, default=1)
    parser.add_argument("--hidden-dim", type=int, default=32)
    parser.add_argument("--num-res-blocks", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--rollout-episodes", type=int, default=2)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--eval-episodes", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--clip-eps", type=float, default=0.1)
    parser.add_argument("--value-coef", type=float, default=0.5)
    parser.add_argument("--rank-coef", type=float, default=0.05)
    parser.add_argument("--prior-kl-eps", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--diagnostic-fields", action="store_true")
    parser.add_argument("--diagnostic-seed-base", type=int, default=202604250000)
    parser.add_argument("--diagnostic-seed-stride", type=int, default=1)
    parser.add_argument(
        "--ablation-profile",
        choices=("small", "medium", "full", "actor-only-small"),
        default="small",
    )
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--random-subsample", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rule-kl-coefs", type=float, nargs="+", default=DEFAULT_RULE_KL_COEFS)
    parser.add_argument("--entropy-coefs", type=float, nargs="+", default=DEFAULT_ENTROPY_COEFS)
    parser.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS)
    parser.add_argument(
        "--opponent-modes",
        nargs="+",
        choices=("rule_prior_greedy", "rulebase"),
        default=("rule_prior_greedy",),
    )
    parser.add_argument("--rulebase-strict", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    configs = _build_run_configs(args)
    if args.dry_run:
        _write_dry_run(args, configs)
        print((args.out_dir / "dry_run.md").read_text(encoding="utf-8"))
        return

    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for config in configs:
        config_id = int(config["config_id"])
        rerun_config_id = int(config["rerun_config_id"])
        source_config_id = config.get("source_config_id")
        source_config_id = None if source_config_id is None else int(source_config_id)
        opponent_mode = str(config["opponent_mode"])
        rule_kl_coef = float(config["rule_kl_coef"])
        entropy_coef = float(config["entropy_coef"])
        lr = float(config["lr"])
        config_seed = int(args.seed + rerun_config_id * 100_000 + args.repeat_id * 10_000)
        report = _run_config(
            config_id=config_id,
            rerun_config_id=rerun_config_id,
            source_config_id=source_config_id,
            source_report=config.get("source_report"),
            opponent_mode=opponent_mode,
            rule_kl_coef=rule_kl_coef,
            entropy_coef=entropy_coef,
            lr=lr,
            seed=config_seed,
            update_epochs=int(config.get("update_epochs", args.update_epochs)),
            value_coef=float(config.get("value_coef", args.value_coef)),
            rank_coef=float(config.get("rank_coef", args.rank_coef)),
            normalize_advantages=bool(config.get("normalize_advantages", args.normalize_advantages)),
            args=args,
        )
        summaries.append(report["summary"])
        rows.extend(report["iterations"])
        print(
            "config "
            f"{rerun_config_id + 1}/{len(configs)} "
            f"source={source_config_id} "
            f"opponent={opponent_mode} "
            f"rule_kl={rule_kl_coef:g} entropy={entropy_coef:g} lr={lr:g} "
            f"rank_pt={report['summary']['eval_rank_pt']:.6g} "
            f"fourth={report['summary']['eval_fourth_rate']:.6g} "
            f"delta_max={report['summary']['neural_delta_abs_max']:.6g}",
            flush=True,
        )

    eval_seed_registry = _eval_seed_registry(args)
    run_metadata = _run_metadata(args, configs, summaries, eval_seed_registry)
    payload = {
        "scope": {
            "style_context": "neutral",
            "action_scope": "DISCARD only",
            "reward_spec": "fixed default",
            "opponents": sorted({str(row["opponent_mode"]) for row in summaries}),
            "teacher_imitation": False,
            "full_action_rl": False,
            "style_variants": False,
        },
        "mode": args.mode,
        "source_type": "retrained_config",
        "seed": args.seed,
        "train_seed_base": args.seed,
        "repeat_id": args.repeat_id,
        "eval_seed_registry_id": _eval_seed_registry_id(args),
        "eval_seed_count": len(eval_seed_registry),
        "eval_seed_hash": _eval_seed_hash(eval_seed_registry),
        "shared_eval_seeds": True,
        "eval_seed_policy": "forced_shared_across_configs",
        "config_count": len(configs),
        **_ablation_metadata(args, configs),
        "grid": {
            "rule_kl_coef": list(args.rule_kl_coefs),
            "entropy_coef": list(args.entropy_coefs),
            "lr": list(args.lrs),
            "opponent_mode": list(args.opponent_modes),
        },
        "candidate_rerun": _candidate_rerun_payload(args, configs),
        "rerun_config_mapping": _rerun_config_mapping(configs),
        "run": run_metadata,
        "summaries": summaries,
        "iterations": rows,
    }
    _write_json(args.out_dir / "sweep.json", payload)
    _write_csv(args.out_dir / "summary.csv", summaries)
    _write_csv(args.out_dir / "iterations.csv", rows)
    (args.out_dir / "summary.md").write_text(_summary_markdown(args, summaries), encoding="utf-8")

    print((args.out_dir / "summary.md").read_text(encoding="utf-8"))


def _build_run_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.mode == "grid":
        return _build_grid_configs(args)
    if args.mode == "candidate-rerun":
        return _build_candidate_rerun_configs(args)
    if args.mode == "learning-signal-ablation":
        return _build_learning_signal_ablation_configs(args)
    raise ValueError(f"unsupported mode: {args.mode}")


def _ablation_metadata(args: argparse.Namespace, configs: list[dict[str, Any]]) -> dict[str, Any]:
    if args.mode != "learning-signal-ablation":
        return {}
    profile_values = _ablation_profile_values(args.ablation_profile)
    return {
        "ablation_profile": args.ablation_profile,
        "profile_config_count": _ablation_profile_config_count(args),
        "profile_values": {key: list(value) for key, value in profile_values.items()},
        "max_configs": args.max_configs,
        "random_subsample": args.random_subsample,
    }


def _ablation_profile_config_count(args: argparse.Namespace) -> int:
    profile_values = _ablation_profile_values(args.ablation_profile)
    source_ids = list(args.source_config_ids or _default_ablation_source_ids(args))
    count = max(1, len(source_ids))
    for values in profile_values.values():
        count *= len(values)
    return count


def _run_metadata(
    args: argparse.Namespace,
    configs: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    eval_seed_registry: list[int],
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "iterations": args.iterations,
        "rollout_episodes": args.rollout_episodes,
        "update_epochs": args.update_epochs,
        "eval_episodes": args.eval_episodes,
        "eval_seed_base": args.eval_seed_base,
        "eval_seed_stride": args.eval_seed_stride,
        "eval_seed_registry_id": _eval_seed_registry_id(args),
        "eval_seed_count": len(eval_seed_registry),
        "eval_seed_hash": _eval_seed_hash(eval_seed_registry),
        "shared_eval_seeds": True,
        "eval_seed_policy": "forced_shared_across_configs",
        "learner_seats": [0],
        "opponents": sorted({str(row["opponent_mode"]) for row in summaries}),
        "config_count": len(configs),
    }
    metadata.update(_ablation_metadata(args, configs))
    return metadata


def _build_grid_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    configs: list[dict[str, Any]] = []
    raw_configs = [
        (opponent_mode, rule_kl_coef, entropy_coef, lr)
        for opponent_mode in args.opponent_modes
        for rule_kl_coef in args.rule_kl_coefs
        for entropy_coef in args.entropy_coefs
        for lr in args.lrs
    ]
    for config_id, (opponent_mode, rule_kl_coef, entropy_coef, lr) in enumerate(raw_configs):
        configs.append(
            {
                "config_id": config_id,
                "rerun_config_id": _rerun_config_id(args, config_id),
                "source_config_id": args.source_config_id,
                "source_report": args.source_report,
                "opponent_mode": str(opponent_mode),
                "rule_kl_coef": float(rule_kl_coef),
                "entropy_coef": float(entropy_coef),
                "lr": float(lr),
            }
        )
    return configs


def _build_candidate_rerun_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.from_summary is None:
        raise ValueError("--mode candidate-rerun requires --from-summary")
    if not args.source_config_ids:
        raise ValueError("--mode candidate-rerun requires --source-config-ids")
    rows = _read_summary_rows(args.from_summary)
    by_source_id: dict[int, dict[str, str]] = {}
    for row in rows:
        source_id = _row_source_config_id(row)
        by_source_id.setdefault(source_id, row)

    source_report = str(args.from_summary.parent)
    configs: list[dict[str, Any]] = []
    for rerun_config_id, source_id in enumerate(args.source_config_ids):
        if source_id not in by_source_id:
            raise ValueError(f"source_config_id {source_id} not found in {args.from_summary}")
        row = by_source_id[source_id]
        configs.append(
            {
                "config_id": rerun_config_id,
                "rerun_config_id": rerun_config_id,
                "source_config_id": source_id,
                "source_report": source_report,
                "opponent_mode": str(row["opponent_mode"]),
                "rule_kl_coef": float(row["rule_kl_coef"]),
                "entropy_coef": float(row["entropy_coef"]),
                "lr": float(row["lr"]),
            }
        )
    return configs


def _build_learning_signal_ablation_configs(args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.from_summary is None:
        raise ValueError("--mode learning-signal-ablation requires --from-summary")
    source_config_ids = list(args.source_config_ids or _default_ablation_source_ids(args))
    if not source_config_ids:
        raise ValueError("--mode learning-signal-ablation requires --source-config-ids or a usable default source set")
    rows = _read_summary_rows(args.from_summary)
    by_source_id: dict[int, dict[str, str]] = {}
    for row in rows:
        by_source_id.setdefault(_row_source_config_id(row), row)
    source_report = str(args.from_summary.parent)
    profile = _ablation_profile_values(args.ablation_profile)
    configs: list[dict[str, Any]] = []
    for source_id in source_config_ids:
        if source_id not in by_source_id:
            raise ValueError(f"source_config_id {source_id} not found in {args.from_summary}")
        source_row = by_source_id[source_id]
        opponent_mode = str(source_row["opponent_mode"])
        for normalize_advantages in profile["normalize_advantages"]:
            for update_epochs in profile["update_epochs"]:
                for rule_kl_coef in profile["rule_kl_coef"]:
                    for entropy_coef in profile["entropy_coef"]:
                        for lr in profile["lr"]:
                            for value_coef in profile["value_coef"]:
                                for rank_coef in profile["rank_coef"]:
                                    config_id = len(configs)
                                    configs.append(
                                        {
                                            "config_id": config_id,
                                            "rerun_config_id": config_id,
                                            "source_config_id": source_id,
                                            "source_report": source_report,
                                            "ablation_profile": args.ablation_profile,
                                            "opponent_mode": opponent_mode,
                                            "rule_kl_coef": float(rule_kl_coef),
                                            "entropy_coef": float(entropy_coef),
                                            "lr": float(lr),
                                            "normalize_advantages": bool(normalize_advantages),
                                            "update_epochs": int(update_epochs),
                                            "value_coef": float(value_coef),
                                            "rank_coef": float(rank_coef),
                                        }
                                    )
    return _limit_ablation_configs(configs, args)


def _default_ablation_source_ids(args: argparse.Namespace) -> list[int]:
    if args.ablation_profile in {"small", "actor-only-small"}:
        return [93, 57, 8]
    return []


def _ablation_profile_values(profile: str) -> dict[str, tuple[Any, ...]]:
    if profile == "small":
        return {
            "normalize_advantages": (True,),
            "update_epochs": (1, 4),
            "rule_kl_coef": (0.0, 0.001),
            "entropy_coef": (0.0, 0.005),
            "lr": (3e-4, 1e-3),
            "value_coef": (0.5,),
            "rank_coef": (0.05,),
        }
    if profile == "medium":
        return {
            "normalize_advantages": (True, False),
            "update_epochs": (1, 4),
            "rule_kl_coef": (0.0, 0.001),
            "entropy_coef": (0.0, 0.001, 0.005),
            "lr": (3e-4, 1e-3),
            "value_coef": (0.5,),
            "rank_coef": (0.0, 0.05),
        }
    if profile == "full":
        return {
            "normalize_advantages": (True, False),
            "update_epochs": (1, 4),
            "rule_kl_coef": (0.0, 0.001),
            "entropy_coef": (0.0, 0.001, 0.005),
            "lr": (3e-4, 1e-3),
            "value_coef": (0.0, 0.5),
            "rank_coef": (0.0, 0.05),
        }
    if profile == "actor-only-small":
        return {
            "normalize_advantages": (True, False),
            "update_epochs": (4, 8),
            "rule_kl_coef": (0.0,),
            "entropy_coef": (0.0, 0.005),
            "lr": (1e-3, 3e-3),
            "value_coef": (0.0,),
            "rank_coef": (0.0,),
        }
    raise ValueError(f"unsupported ablation profile: {profile}")


def _limit_ablation_configs(configs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    limited = list(configs)
    if args.random_subsample is not None:
        if args.random_subsample <= 0:
            raise ValueError("--random-subsample must be positive")
        rng = random.Random(args.seed)
        limited = rng.sample(limited, k=min(int(args.random_subsample), len(limited)))
    if args.max_configs is not None:
        if args.max_configs <= 0:
            raise ValueError("--max-configs must be positive")
        limited = limited[: int(args.max_configs)]
    return _renumber_configs(limited)


def _renumber_configs(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    renumbered: list[dict[str, Any]] = []
    for config_id, config in enumerate(configs):
        updated = dict(config)
        updated["config_id"] = config_id
        updated["rerun_config_id"] = config_id
        renumbered.append(updated)
    return renumbered


def _read_summary_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _row_source_config_id(row: dict[str, str]) -> int:
    raw_value = row.get("source_config_id") or row.get("config_id")
    if raw_value is None or raw_value == "":
        raise ValueError("summary row missing source_config_id/config_id")
    return int(raw_value)


def _candidate_rerun_payload(args: argparse.Namespace, configs: list[dict[str, Any]]) -> dict[str, Any] | None:
    if args.mode != "candidate-rerun":
        return None
    return {
        "source_type": "retrained_config",
        "from_summary": None if args.from_summary is None else str(args.from_summary),
        "source_config_ids": list(args.source_config_ids or []),
        "shared_eval_seeds": True,
        "eval_seed_policy": "forced_shared_across_configs",
        "configs": configs,
    }


def _rerun_config_mapping(configs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rerun_config_id": int(config["rerun_config_id"]),
            "source_config_id": config.get("source_config_id"),
            "source_report": config.get("source_report"),
            "config_key": _config_key(
                str(config["opponent_mode"]),
                float(config["rule_kl_coef"]),
                float(config["entropy_coef"]),
                float(config["lr"]),
            ),
        }
        for config in configs
    ]


def _write_dry_run(args: argparse.Namespace, configs: list[dict[str, Any]]) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode,
        "dry_run": True,
        "config_count": len(configs),
        "configs": configs,
        **_ablation_metadata(args, configs),
    }
    _write_json(args.out_dir / "dry_run.json", payload)
    _write_csv(args.out_dir / "dry_run_configs.csv", configs)
    lines = [
        "# KeqingRL Sweep Dry Run",
        "",
        f"mode: `{args.mode}`",
        f"ablation_profile: `{args.ablation_profile}`",
        f"config_count: `{len(configs)}`",
        f"max_configs: `{args.max_configs}`",
        f"random_subsample: `{args.random_subsample}`",
        "",
        "## First Configs",
        "",
    ]
    for config in configs[:20]:
        lines.append(
            "- "
            f"id={config['config_id']} "
            f"source={config.get('source_config_id')} "
            f"lr={config['lr']} "
            f"epochs={config.get('update_epochs')} "
            f"rule_kl={config['rule_kl_coef']} "
            f"entropy={config['entropy_coef']} "
            f"norm_adv={config.get('normalize_advantages')} "
            f"value={config.get('value_coef')} "
            f"rank={config.get('rank_coef')}"
        )
    (args.out_dir / "dry_run.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_config(
    *,
    config_id: int,
    rerun_config_id: int,
    source_config_id: int | None,
    source_report: str | None,
    opponent_mode: str,
    rule_kl_coef: float,
    entropy_coef: float,
    lr: float,
    seed: int,
    update_epochs: int,
    value_coef: float,
    rank_coef: float,
    normalize_advantages: bool,
    args: argparse.Namespace,
) -> dict[str, Any]:
    device = torch.device(args.device) if args.device is not None else torch.device("cpu")
    env = DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus)
    policy = RulePriorDeltaPolicy(
        hidden_dim=args.hidden_dim,
        num_res_blocks=args.num_res_blocks,
        dropout=0.0,
    ).to(device)
    _initialize_policy_from_env_observation(env, policy, seed=seed, device=device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    opponent_pool, rulebase_policy = _build_opponent_pool(opponent_mode, strict_rulebase=args.rulebase_strict)

    iteration_rows: list[dict[str, Any]] = []
    for iteration in range(args.iterations):
        iteration_seed = seed + iteration * 10_000
        if _diagnostics_enabled(args):
            result, diagnostic_fields = _run_ppo_iteration_with_diagnostics(
                env,
                policy,
                optimizer,
                num_episodes=args.rollout_episodes,
                opponent_pool=opponent_pool,
                learner_seats=(0,),
                update_epochs=update_epochs,
                seed=iteration_seed,
                policy_version=iteration,
                max_steps=args.max_steps,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                include_rank_targets=True,
                clip_eps=args.clip_eps,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                rank_coef=rank_coef,
                rule_kl_coef=rule_kl_coef,
                prior_kl_eps=args.prior_kl_eps,
                normalize_advantages=normalize_advantages,
                max_grad_norm=args.max_grad_norm,
                device=device,
                strict_metadata=True,
                lr=lr,
                args=args,
            )
        else:
            result = run_ppo_iteration(
                env,
                policy,
                optimizer,
                num_episodes=args.rollout_episodes,
                opponent_pool=opponent_pool,
                learner_seats=(0,),
                update_epochs=update_epochs,
                seed=iteration_seed,
                policy_version=iteration,
                max_steps=args.max_steps,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                include_rank_targets=True,
                clip_eps=args.clip_eps,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                rank_coef=rank_coef,
                rule_kl_coef=rule_kl_coef,
                prior_kl_eps=args.prior_kl_eps,
                normalize_advantages=normalize_advantages,
                max_grad_norm=args.max_grad_norm,
                device=device,
                strict_metadata=True,
            )
            diagnostic_fields = {}
        loss = result.losses[-1]
        metrics = result.metrics
        delta_stats = _ppo_delta_smoke_stats(policy, result.batch)
        iteration_rows.append(
            {
                "config_id": config_id,
                "source_type": "retrained_config",
                "rerun_config_id": rerun_config_id,
                "source_config_id": source_config_id,
                "source_report": source_report,
                "config_key": _config_key(opponent_mode, rule_kl_coef, entropy_coef, lr),
                "config_key_label": _config_key_label(opponent_mode, rule_kl_coef, entropy_coef, lr),
                "iteration": iteration,
                "seed": iteration_seed,
                "rule_kl_coef": rule_kl_coef,
                "entropy_coef": entropy_coef,
                "lr": lr,
                "opponent_mode": opponent_mode,
                "update_epochs": update_epochs,
                "value_coef": value_coef,
                "rank_coef": rank_coef,
                "normalize_advantages": normalize_advantages,
                "approx_kl": _loss_float(loss.approx_kl),
                "clip_fraction": _loss_float(loss.clip_fraction),
                "entropy": _loss_float(loss.entropy_bonus),
                "rule_kl": None if loss.rule_kl is None else _loss_float(loss.rule_kl),
                "rule_agreement": None if loss.rule_agreement is None else _loss_float(loss.rule_agreement),
                "neural_delta_abs_mean": delta_stats["neural_delta_abs_mean"],
                "neural_delta_abs_max": delta_stats["neural_delta_abs_max"],
                "top1_action_changed_rate": delta_stats["top1_action_changed_rate"],
                "rank_pt": metrics.mean_terminal_reward,
                "fourth_rate": metrics.fourth_rate,
                "learner_win_rate": metrics.learner_win_rate,
                "learner_deal_in_rate": metrics.learner_deal_in_rate,
                "learner_call_rate": metrics.learner_call_rate,
                "learner_riichi_rate": metrics.learner_riichi_rate,
                "deal_in_rate": metrics.learner_deal_in_rate,
                "batch_size": metrics.batch_size,
                "episode_count": metrics.episode_count,
                "illegal_action_rate_fail_closed": metrics.illegal_action_rate_fail_closed,
                "fallback_rate_fail_closed": metrics.fallback_rate_fail_closed,
                "forced_terminal_missed_fail_closed": metrics.forced_terminal_missed_fail_closed,
                "illegal_attempt_count": None,
                "fallback_policy_count": None,
                "forced_terminal_preempt_count": metrics.forced_terminal_preempt_count,
                "forced_terminal_missed_count": metrics.forced_terminal_missed_fail_closed,
                "autopilot_terminal_count": metrics.autopilot_terminal_count,
                "learner_seat_step_count": metrics.learner_seat_step_count,
                "learner_controlled_step_count": metrics.learner_controlled_step_count,
                **_rulebase_counter_fields(rulebase_policy),
                **diagnostic_fields,
                "illegal_action_rate": metrics.illegal_action_rate_fail_closed,
                "fallback_rate": metrics.fallback_rate_fail_closed,
                "forced_terminal_missed": metrics.forced_terminal_missed_fail_closed,
            }
        )

    eval_metrics = evaluate_policy(
        DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
        policy,
        num_episodes=args.eval_episodes,
        opponent_pool=opponent_pool,
        learner_seats=(0,),
        seed=args.eval_seed_base,
        seed_stride=args.eval_seed_stride,
        greedy=True,
        device=device,
    )
    final = iteration_rows[-1]
    summary = {
        "config_id": config_id,
        "source_type": "retrained_config",
        "rerun_config_id": _rerun_config_id(args, config_id),
        "source_config_id": source_config_id,
        "source_report": source_report,
        "config_key": _config_key(opponent_mode, rule_kl_coef, entropy_coef, lr),
        "config_key_label": _config_key_label(opponent_mode, rule_kl_coef, entropy_coef, lr),
        "seed": seed,
        "train_seed_base": seed,
        "repeat_id": args.repeat_id,
        "eval_seed_registry_id": _eval_seed_registry_id(args),
        "eval_seed_count": args.eval_episodes,
        "eval_seed_hash": _eval_seed_hash(_eval_seed_registry(args)),
        "shared_eval_seeds": True,
        "eval_seed_policy": "forced_shared_across_configs",
        "diagnostic_seed_registry_id": _diagnostic_seed_registry_id(args),
        "diagnostic_seed_hash": seed_registry_hash(_diagnostic_seed_registry(args)),
        "rule_kl_coef": rule_kl_coef,
        "entropy_coef": entropy_coef,
        "lr": lr,
        "opponent_mode": opponent_mode,
        "iterations": args.iterations,
        "rollout_episodes": args.rollout_episodes,
        "update_epochs": update_epochs,
        "value_coef": value_coef,
        "rank_coef": rank_coef,
        "normalize_advantages": normalize_advantages,
        "eval_episodes": args.eval_episodes,
        "approx_kl": final["approx_kl"],
        "clip_fraction": final["clip_fraction"],
        "entropy": final["entropy"],
        "rule_kl": final["rule_kl"],
        "rule_agreement": final["rule_agreement"],
        "neural_delta_abs_mean": final["neural_delta_abs_mean"],
        "neural_delta_abs_max": final["neural_delta_abs_max"],
        "top1_action_changed_rate": final["top1_action_changed_rate"],
        "advantage_mean": final.get("advantage_mean"),
        "advantage_std": final.get("advantage_std"),
        "advantage_min": final.get("advantage_min"),
        "advantage_max": final.get("advantage_max"),
        "return_mean": final.get("return_mean"),
        "return_std": final.get("return_std"),
        "return_min": final.get("return_min"),
        "return_max": final.get("return_max"),
        "selected_non_top1_positive_advantage_count": final.get("selected_non_top1_positive_advantage_count"),
        "mean_delta_needed_to_flip_top1": final.get("mean_delta_needed_to_flip_top1"),
        "actor_grad_norm_total": final.get("actor_grad_norm_total"),
        "policy_mlp_final_grad_norm": final.get("policy_mlp_final_grad_norm"),
        "post_update_kl_vs_old": final.get("post_update_kl_vs_old"),
        **_summary_learning_signal_fields(final),
        "train_rank_pt": final["rank_pt"],
        "train_fourth_rate": final["fourth_rate"],
        "train_learner_win_rate": final["learner_win_rate"],
        "train_learner_deal_in_rate": final["learner_deal_in_rate"],
        "train_learner_call_rate": final["learner_call_rate"],
        "train_learner_riichi_rate": final["learner_riichi_rate"],
        "train_deal_in_rate": final["learner_deal_in_rate"],
        "eval_rank_pt": eval_metrics.mean_terminal_reward,
        "eval_mean_rank": eval_metrics.mean_rank,
        "eval_fourth_rate": eval_metrics.fourth_place_rate,
        "eval_learner_win_rate": eval_metrics.learner_win_rate,
        "eval_learner_deal_in_rate": eval_metrics.learner_deal_in_rate,
        "eval_learner_call_rate": eval_metrics.learner_call_rate,
        "eval_learner_riichi_rate": eval_metrics.learner_riichi_rate,
        "eval_deal_in_rate": eval_metrics.learner_deal_in_rate,
        "eval_win_rate": eval_metrics.learner_win_rate,
        "illegal_action_rate_fail_closed": final["illegal_action_rate_fail_closed"],
        "fallback_rate_fail_closed": final["fallback_rate_fail_closed"],
        "forced_terminal_missed_fail_closed": final["forced_terminal_missed_fail_closed"],
        "illegal_attempt_count": final["illegal_attempt_count"],
        "fallback_policy_count": final["fallback_policy_count"],
        "forced_terminal_preempt_count": final["forced_terminal_preempt_count"],
        "forced_terminal_missed_count": final["forced_terminal_missed_count"],
        "autopilot_terminal_count": final["autopilot_terminal_count"],
        "learner_seat_step_count": final.get("learner_seat_step_count"),
        "learner_controlled_step_count": final.get("learner_controlled_step_count"),
        **_rulebase_counter_fields(rulebase_policy),
        "illegal_action_rate": final["illegal_action_rate_fail_closed"],
        "fallback_rate": final["fallback_rate_fail_closed"],
        "forced_terminal_missed": final["forced_terminal_missed_fail_closed"],
    }
    artifact_fields = _save_config_artifacts(
        args=args,
        config_id=config_id,
        rerun_config_id=rerun_config_id,
        source_config_id=source_config_id,
        opponent_mode=opponent_mode,
        rule_kl_coef=rule_kl_coef,
        entropy_coef=entropy_coef,
        lr=lr,
        seed=seed,
        policy=policy,
        optimizer=optimizer,
        iteration_rows=iteration_rows,
        eval_metrics=eval_metrics,
        summary=summary,
        update_epochs=update_epochs,
        value_coef=value_coef,
        rank_coef=rank_coef,
        normalize_advantages=normalize_advantages,
    )
    summary.update(artifact_fields)
    return {"summary": summary, "iterations": iteration_rows}


def _save_config_artifacts(
    *,
    args: argparse.Namespace,
    config_id: int,
    rerun_config_id: int,
    source_config_id: int | None,
    opponent_mode: str,
    rule_kl_coef: float,
    entropy_coef: float,
    lr: float,
    seed: int,
    policy: RulePriorDeltaPolicy,
    optimizer: torch.optim.Optimizer,
    iteration_rows: list[dict[str, Any]],
    eval_metrics: Any,
    summary: dict[str, Any],
    update_epochs: int,
    value_coef: float,
    rank_coef: float,
    normalize_advantages: bool,
) -> dict[str, Any]:
    config_dir = _config_artifact_dir(args.out_dir, rerun_config_id, source_config_id)
    config_dir.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "config_id": config_id,
        "source_type": "retrained_config",
        "rerun_config_id": rerun_config_id,
        "source_config_id": source_config_id,
        "source_report": args.source_report,
        "config_key": _config_key(opponent_mode, rule_kl_coef, entropy_coef, lr),
        "config_key_label": _config_key_label(opponent_mode, rule_kl_coef, entropy_coef, lr),
        "train_seed_base": seed,
        "repeat_id": args.repeat_id,
        "eval_seed_registry_id": _eval_seed_registry_id(args),
        "eval_seed_count": args.eval_episodes,
        "eval_seed_hash": _eval_seed_hash(_eval_seed_registry(args)),
        "shared_eval_seeds": True,
        "eval_seed_policy": "forced_shared_across_configs",
        "model": {
            "class": "RulePriorDeltaPolicy",
            "hidden_dim": args.hidden_dim,
            "num_res_blocks": args.num_res_blocks,
            "dropout": 0.0,
        },
        "optimizer": {
            "class": "torch.optim.Adam",
            "lr": lr,
        },
        "run": {
            "iterations": args.iterations,
            "rollout_episodes": args.rollout_episodes,
            "update_epochs": update_epochs,
            "eval_episodes": args.eval_episodes,
            "max_steps": args.max_steps,
            "clip_eps": args.clip_eps,
            "value_coef": value_coef,
            "rank_coef": rank_coef,
            "prior_kl_eps": args.prior_kl_eps,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "max_grad_norm": args.max_grad_norm,
            "rulebase_strict": args.rulebase_strict,
            "normalize_advantages": normalize_advantages,
        },
    }

    config_path = config_dir / "config.json"
    policy_path = config_dir / "policy_final.pt"
    optimizer_path = config_dir / "optimizer_final.pt"
    train_metrics_path = config_dir / "train_metrics.json"
    eval_metrics_path = config_dir / "eval_metrics.json"
    native_schema_path = config_dir / "native_schema.json"
    git_commit_path = config_dir / "git_commit.txt"

    _write_json(config_path, config_payload)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "config": config_payload,
            "summary": _to_jsonable(summary),
        },
        policy_path,
    )
    torch.save(
        {
            "optimizer_state_dict": optimizer.state_dict(),
            "config": config_payload,
        },
        optimizer_path,
    )
    _write_json(train_metrics_path, {"iterations": iteration_rows})
    _write_json(eval_metrics_path, eval_metrics)
    _write_json(native_schema_path, keqing_core.require_native_schema())
    git_commit_path.write_text(_git_commit_text(), encoding="utf-8")

    return {
        "artifact_dir": str(config_dir),
        "config_path": str(config_path),
        "checkpoint_path": str(policy_path),
        "checkpoint_sha256": _file_sha256(policy_path),
        "optimizer_path": str(optimizer_path),
        "optimizer_sha256": _file_sha256(optimizer_path),
        "train_metrics_path": str(train_metrics_path),
        "eval_metrics_path": str(eval_metrics_path),
        "native_schema_path": str(native_schema_path),
        "git_commit_path": str(git_commit_path),
    }


def _diagnostics_enabled(args: argparse.Namespace) -> bool:
    return bool(args.diagnostic_fields or args.mode == "learning-signal-ablation")


def _run_ppo_iteration_with_diagnostics(
    env: DiscardOnlyMahjongEnv,
    policy: RulePriorDeltaPolicy,
    optimizer: torch.optim.Optimizer,
    *,
    num_episodes: int,
    opponent_pool: OpponentPool | None,
    learner_seats: tuple[int, ...],
    update_epochs: int,
    seed: int | None,
    policy_version: int,
    max_steps: int,
    gamma: float,
    gae_lambda: float,
    include_rank_targets: bool,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    rank_coef: float,
    rule_kl_coef: float,
    prior_kl_eps: float,
    normalize_advantages: bool,
    max_grad_norm: float | None,
    device: torch.device | str | None,
    strict_metadata: bool,
    lr: float,
    args: argparse.Namespace,
) -> tuple[DiscardOnlyIterationResult, dict[str, Any]]:
    episodes = collect_selfplay_episodes(
        env,
        policy,
        num_episodes=num_episodes,
        opponent_pool=opponent_pool,
        learner_seats=learner_seats,
        seed=seed,
        greedy=False,
        policy_version=policy_version,
        max_steps=max_steps,
        device=device,
    )
    _advantages, _returns, _prepared_steps, batch = build_episodes_ppo_batch(
        episodes,
        gamma=gamma,
        gae_lambda=gae_lambda,
        include_rank_targets=include_rank_targets,
        strict_metadata=strict_metadata,
    )
    target_device = torch.device(device) if device is not None else next(policy.parameters()).device
    batch = batch.to(target_device)
    pre_fields = _phase_diagnostic_fields(
        args,
        policy,
        batch,
        prefix="pre_update",
        lr=lr,
        update_epochs=update_epochs,
        normalize_advantages=normalize_advantages,
        include_probe=True,
    )
    losses = []
    for _ in range(update_epochs):
        losses.append(
            ppo_update(
                policy,
                optimizer,
                batch,
                clip_eps=clip_eps,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                rank_coef=rank_coef,
                rule_kl_coef=rule_kl_coef,
                prior_kl_eps=prior_kl_eps,
                normalize_advantages=normalize_advantages,
                max_grad_norm=max_grad_norm,
            )
        )
    post_fields = _phase_diagnostic_fields(
        args,
        policy,
        batch,
        prefix="post_update",
        lr=lr,
        update_epochs=update_epochs,
        normalize_advantages=normalize_advantages,
        include_probe=False,
    )
    post_fields.update(_policy_vs_old_fields(policy, batch, prefix="post_update", clip_eps=clip_eps))
    metrics = summarize_iteration(episodes, losses, batch, learner_seats=learner_seats)
    result = DiscardOnlyIterationResult(
        episodes=episodes,
        batch=batch,
        losses=tuple(losses),
        metrics=metrics,
    )
    diagnostic_fields = {**pre_fields, **post_fields}
    diagnostic_fields.update(_legacy_learning_signal_fields(diagnostic_fields))
    return result, diagnostic_fields


def _phase_diagnostic_fields(
    args: argparse.Namespace,
    policy: RulePriorDeltaPolicy,
    batch: Any,
    *,
    prefix: str,
    lr: float,
    update_epochs: int,
    normalize_advantages: bool,
    include_probe: bool,
) -> dict[str, Any]:
    margins = top1_margin_diagnostics(policy, batch)["summary"]
    returns = tensor_stats(batch.returns.detach().cpu(), prefix=f"{prefix}_return")
    advantages = tensor_stats(batch.advantages.detach().cpu(), prefix=f"{prefix}_advantage")
    actor_grad_norm = None
    policy_mlp_final_grad_norm = None
    try:
        policy.zero_grad(set_to_none=True)
        loss = compute_ppo_loss(
            policy,
            batch,
            clip_eps=args.clip_eps,
            value_coef=0.0,
            entropy_coef=0.0,
            rank_coef=0.0,
            rule_kl_coef=0.0,
            normalize_advantages=normalize_advantages,
        )
        loss.total_loss.backward()
        grads = _learning_signal_gradient_norms(policy)
        actor_grad_norm = grads["actor_grad_norm_total"]
        policy_mlp_final_grad_norm = grads["policy_mlp.final_linear.weight_grad_norm"]
    finally:
        policy.zero_grad(set_to_none=True)
    fields: dict[str, Any] = {
        **advantages,
        **returns,
        f"{prefix}_selected_non_top1_positive_advantage_count": margins["selected_non_top1_positive_advantage_count"],
        f"{prefix}_mean_delta_needed_to_flip_top1": margins["mean_delta_needed_to_flip_top1"],
        f"{prefix}_actor_grad_norm": actor_grad_norm,
        f"{prefix}_policy_mlp_final_grad_norm": policy_mlp_final_grad_norm,
        "diagnostic_seed_registry_id": _diagnostic_seed_registry_id(args),
        "diagnostic_seed_hash": seed_registry_hash(_diagnostic_seed_registry(args)),
    }
    if include_probe:
        actual_probe = ppo_update_probe(
            policy,
            batch,
            lrs=(float(lr),),
            update_epochs=(int(update_epochs),),
            clip_eps=args.clip_eps,
            normalize_advantages=normalize_advantages,
        )
        standard_probe = ppo_update_probe(
            policy,
            batch,
            lrs=(3e-4,),
            update_epochs=(1,),
            clip_eps=args.clip_eps,
            normalize_advantages=normalize_advantages,
        )
        fields.update(_probe_fields(actual_probe, prefix="actual_update_probe"))
        fields.update(_probe_fields(standard_probe, prefix="standard_probe_lr3e_4_epoch1"))
    return fields


def _probe_fields(rows: list[dict[str, Any]], *, prefix: str) -> dict[str, Any]:
    if not rows:
        return {}
    row = rows[-1]
    return {
        f"{prefix}_lr": row.get("lr"),
        f"{prefix}_update_epochs": row.get("target_update_epochs"),
        f"{prefix}_post_update_kl_vs_old": row.get("post_approx_kl_vs_old"),
        f"{prefix}_top1_changed_rate": row.get("top1_action_changed_rate"),
        f"{prefix}_neural_delta_abs_max": row.get("neural_delta_abs_max"),
        f"{prefix}_ratio_std": row.get("post_ratio_std"),
        f"{prefix}_clip_fraction": row.get("post_clip_fraction"),
    }


def _policy_vs_old_fields(policy: RulePriorDeltaPolicy, batch: Any, *, prefix: str, clip_eps: float) -> dict[str, Any]:
    with torch.no_grad():
        output = policy(batch.policy_input)
        dist = MaskedCategorical(output.action_logits, batch.policy_input.legal_action_mask)
        new_log_prob = dist.log_prob(batch.action_index)
        ratio = torch.exp(new_log_prob - batch.old_log_prob)
        mask = batch.policy_input.legal_action_mask.bool()
        prior_logits = output.aux.get("prior_logits")
        if prior_logits is None:
            prior_logits = batch.policy_input.prior_logits
        if prior_logits is None:
            top1_changed = 0.0
            rule_agreement = None
        else:
            current = output.action_logits.masked_fill(~mask, torch.finfo(output.action_logits.dtype).min).argmax(dim=-1)
            prior = prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min).argmax(dim=-1)
            top1_changed = float((current != prior).float().mean().detach().cpu())
            rule_agreement = float((current == prior).float().mean().detach().cpu())
    return {
        f"{prefix}_kl_vs_old": float(0.5 * (new_log_prob - batch.old_log_prob).pow(2).mean().detach().cpu()),
        f"{prefix}_ratio_mean": float(ratio.mean().detach().cpu()),
        f"{prefix}_ratio_std": float(ratio.std(unbiased=False).detach().cpu()),
        f"{prefix}_clip_fraction": float(((ratio - 1.0).abs() > float(clip_eps)).float().mean().detach().cpu()),
        f"{prefix}_top1_action_changed_rate": top1_changed,
        f"{prefix}_rule_agreement": rule_agreement,
    }


def _legacy_learning_signal_fields(fields: dict[str, Any]) -> dict[str, Any]:
    return {
        "advantage_mean": fields.get("pre_update_advantage_mean"),
        "advantage_std": fields.get("pre_update_advantage_std"),
        "advantage_min": fields.get("pre_update_advantage_min"),
        "advantage_max": fields.get("pre_update_advantage_max"),
        "return_mean": fields.get("pre_update_return_mean"),
        "return_std": fields.get("pre_update_return_std"),
        "return_min": fields.get("pre_update_return_min"),
        "return_max": fields.get("pre_update_return_max"),
        "selected_non_top1_positive_advantage_count": fields.get("pre_update_selected_non_top1_positive_advantage_count"),
        "mean_delta_needed_to_flip_top1": fields.get("pre_update_mean_delta_needed_to_flip_top1"),
        "actor_grad_norm_total": fields.get("pre_update_actor_grad_norm"),
        "policy_mlp_final_grad_norm": fields.get("pre_update_policy_mlp_final_grad_norm"),
        "post_update_kl_vs_old": fields.get("post_update_kl_vs_old"),
        "pre_update_probe_top1_changed_rate": fields.get("actual_update_probe_top1_changed_rate"),
    }


def _summary_learning_signal_fields(final: dict[str, Any]) -> dict[str, Any]:
    names = (
        "pre_update_advantage_mean",
        "pre_update_advantage_std",
        "pre_update_advantage_min",
        "pre_update_advantage_max",
        "pre_update_return_mean",
        "pre_update_return_std",
        "pre_update_return_min",
        "pre_update_return_max",
        "pre_update_actor_grad_norm",
        "pre_update_policy_mlp_final_grad_norm",
        "pre_update_selected_non_top1_positive_advantage_count",
        "pre_update_mean_delta_needed_to_flip_top1",
        "pre_update_probe_top1_changed_rate",
        "actual_update_probe_lr",
        "actual_update_probe_update_epochs",
        "actual_update_probe_post_update_kl_vs_old",
        "actual_update_probe_top1_changed_rate",
        "actual_update_probe_neural_delta_abs_max",
        "actual_update_probe_ratio_std",
        "actual_update_probe_clip_fraction",
        "standard_probe_lr3e_4_epoch1_lr",
        "standard_probe_lr3e_4_epoch1_update_epochs",
        "standard_probe_lr3e_4_epoch1_post_update_kl_vs_old",
        "standard_probe_lr3e_4_epoch1_top1_changed_rate",
        "standard_probe_lr3e_4_epoch1_neural_delta_abs_max",
        "standard_probe_lr3e_4_epoch1_ratio_std",
        "standard_probe_lr3e_4_epoch1_clip_fraction",
        "post_update_advantage_mean",
        "post_update_advantage_std",
        "post_update_return_mean",
        "post_update_return_std",
        "post_update_actor_grad_norm",
        "post_update_policy_mlp_final_grad_norm",
        "post_update_mean_delta_needed_to_flip_top1",
        "post_update_top1_action_changed_rate",
        "post_update_rule_agreement",
        "post_update_ratio_mean",
        "post_update_ratio_std",
        "post_update_clip_fraction",
    )
    return {name: final.get(name) for name in names}


def _diagnostic_seed_registry(args: argparse.Namespace) -> list[int]:
    return [int(args.diagnostic_seed_base + idx * args.diagnostic_seed_stride) for idx in range(args.rollout_episodes)]


def _diagnostic_seed_registry_id(args: argparse.Namespace) -> str:
    return f"base={args.diagnostic_seed_base}:stride={args.diagnostic_seed_stride}:count={args.rollout_episodes}"


def _summary_ablation_lines(args: argparse.Namespace, config_count: int) -> list[str]:
    if args.mode != "learning-signal-ablation":
        return []
    lines = [
        f"ablation_profile: `{args.ablation_profile}`",
        f"profile_config_count: `{_ablation_profile_config_count(args)}`",
    ]
    if args.max_configs is not None:
        lines.append(f"max_configs: `{args.max_configs}`")
    if args.random_subsample is not None:
        lines.append(f"random_subsample: `{args.random_subsample}`")
    return lines


def _summary_markdown(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> str:
    ranked = sorted(
        summaries,
        key=lambda row: (
            -float(row["eval_rank_pt"]),
            float(row["eval_fourth_rate"]),
            float(row["eval_learner_deal_in_rate"]),
            float(row["neural_delta_abs_max"]),
        ),
    )
    lines = [
        "# KeqingRL Controlled Discard-Only Sweep",
        "",
        f"seed: `{args.seed}`",
        f"configs: `{len(summaries)}`",
        *_summary_ablation_lines(args, len(summaries)),
        f"iterations/config: `{args.iterations}`",
        f"rollout_episodes/config/iter: `{args.rollout_episodes}`",
        f"eval_episodes/config: `{args.eval_episodes}`",
        f"eval_seed_registry_id: `{_eval_seed_registry_id(args)}`",
        f"eval_seed_count: `{len(_eval_seed_registry(args))}`",
        f"eval_seed_hash: `{_eval_seed_hash(_eval_seed_registry(args))}`",
        "shared_eval_seeds: `true`",
        "eval_seed_policy: `forced_shared_across_configs`",
        "source_type: `retrained_config`",
        "comparison_note: `config repeat only; use checkpoint eval for candidate comparison`",
        "eval_scope: `single learner seat 0; use paired checkpoint eval for seat rotation validation`",
        "",
        "## Scope",
        "",
        "- style_context: neutral",
        "- action_scope: DISCARD only",
        "- reward_spec: fixed default",
        f"- opponents: {', '.join(sorted({str(row['opponent_mode']) for row in summaries}))}",
        "- no teacher imitation, no full action RL, no style variants",
        "",
        "## Stability Checks",
        "",
        f"- max illegal_action_rate_fail_closed: {_max_metric(summaries, 'illegal_action_rate_fail_closed'):.6g}",
        f"- max fallback_rate_fail_closed: {_max_metric(summaries, 'fallback_rate_fail_closed'):.6g}",
        f"- max forced_terminal_missed_fail_closed: {_max_metric(summaries, 'forced_terminal_missed_fail_closed'):.6g}",
        f"- max rulebase_fallback_count: {_max_metric(summaries, 'rulebase_fallback_count'):.6g}",
        f"- max rulebase_chosen_missing_count: {_max_metric(summaries, 'rulebase_chosen_missing_count'):.6g}",
        f"- max rulebase_chosen_not_found_count: {_max_metric(summaries, 'rulebase_chosen_not_found_count'):.6g}",
        f"- max rulebase_batch_unsupported_count: {_max_metric(summaries, 'rulebase_batch_unsupported_count'):.6g}",
        "- fail-closed note: these are hard raise gates, not observed recoverable event rates",
        "- trace counters observed: forced_terminal_preempt_count, autopilot_terminal_count",
        "- trace counters pending: illegal_attempt_count, fallback_policy_count, forced_terminal_missed_count",
        "- caveat: this is a tiny smoke-scale matrix, not strength evidence",
        "",
        "## Top Configs",
        "",
    ]
    for row in ranked[:10]:
        lines.append(
            "- "
            f"id={row['config_id']} "
            f"opponent={row['opponent_mode']} "
            f"rule_kl={row['rule_kl_coef']:g} "
            f"entropy={row['entropy_coef']:g} "
            f"lr={row['lr']:g} "
            f"eval_rank_pt={row['eval_rank_pt']:.6g} "
            f"fourth={row['eval_fourth_rate']:.6g} "
            f"learner_deal_in={row['eval_learner_deal_in_rate']:.6g} "
            f"approx_kl={row['approx_kl']:.6g} "
            f"clip={row['clip_fraction']:.6g} "
            f"rule_kl_obs={_fmt_optional(row['rule_kl'])} "
            f"rule_agree={_fmt_optional(row['rule_agreement'])} "
            f"delta_mean={row['neural_delta_abs_mean']:.6g} "
            f"delta_max={row['neural_delta_abs_max']:.6g} "
            f"top1_changed={row['top1_action_changed_rate']:.6g}"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `sweep.json`",
            "- `summary.csv`",
            "- `iterations.csv`",
            "- `configs/<rerun_config_id>_*/config.json`",
            "- `configs/<rerun_config_id>_*/policy_final.pt`",
            "- `configs/<rerun_config_id>_*/optimizer_final.pt`",
        ]
    )
    return "\n".join(lines) + "\n"


def _build_opponent_pool(
    opponent_mode: str,
    *,
    strict_rulebase: bool,
) -> tuple[OpponentPool, "RulebaseGreedyPolicy | None"]:
    rulebase_policy: RulebaseGreedyPolicy | None = None
    if opponent_mode == "rule_prior_greedy":
        policy: InteractivePolicy = RulePriorPolicy()
    elif opponent_mode == "rulebase":
        rulebase_policy = RulebaseGreedyPolicy(strict=strict_rulebase)
        policy = rulebase_policy
    else:
        raise ValueError(f"unsupported opponent_mode: {opponent_mode}")
    return (
        OpponentPool(
            (
                OpponentPoolEntry(
                    policy=policy,
                    policy_version=-1,
                    greedy=True,
                    name=opponent_mode,
                ),
            )
        ),
        rulebase_policy,
    )


class RulebaseGreedyPolicy(RulePriorPolicy):
    """Greedy policy that follows env-provided Rust rulebase choices."""

    def __init__(self, *, strict: bool = True) -> None:
        super().__init__()
        self.strict = bool(strict)
        self.rulebase_fallback_count = 0
        self.rulebase_chosen_missing_count = 0
        self.rulebase_chosen_not_found_count = 0
        self.rulebase_batch_unsupported_count = 0

    def forward(self, policy_input: PolicyInput) -> PolicyOutput:
        output = super().forward(policy_input)
        chosen_key = policy_input.metadata.get("rulebase_chosen")
        if chosen_key is None:
            self.rulebase_chosen_missing_count += 1
            return self._fallback_or_raise(output, "rulebase_chosen missing")
        if (
            policy_input.legal_actions is None
            or len(policy_input.legal_actions) != 1
            or int(output.action_logits.shape[0]) != 1
        ):
            self.rulebase_batch_unsupported_count += 1
            return self._fallback_or_raise(output, "rulebase batch shape unsupported")
        try:
            chosen_index = next(
                idx
                for idx, action in enumerate(policy_input.legal_actions[0])
                if action.canonical_key == chosen_key
            )
        except StopIteration:
            self.rulebase_chosen_not_found_count += 1
            return self._fallback_or_raise(output, f"rulebase_chosen not in legal actions: {chosen_key}")

        mask = policy_input.legal_action_mask.bool()
        logits = torch.full_like(output.action_logits, torch.finfo(output.action_logits.dtype).min)
        logits = logits.masked_fill(~mask, torch.finfo(output.action_logits.dtype).min)
        logits[0, chosen_index] = 0.0
        entropy = MaskedCategorical(logits, mask).entropy()
        aux = dict(output.aux)
        aux["final_logits"] = logits
        return replace(output, action_logits=logits, entropy=entropy, aux=aux)

    def _fallback_or_raise(self, output: PolicyOutput, reason: str) -> PolicyOutput:
        if self.strict:
            raise RuntimeError(f"strict rulebase opponent contract violation: {reason}")
        self.rulebase_fallback_count += 1
        return output


def _config_artifact_dir(out_dir: Path, rerun_config_id: int, source_config_id: int | None) -> Path:
    source_label = "none" if source_config_id is None else str(source_config_id)
    return out_dir / "configs" / f"rerun_{rerun_config_id:03d}_source_{source_label}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit_text() -> str:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        status = subprocess.check_output(["git", "status", "--short"], text=True).strip()
    except Exception as exc:
        return f"unavailable: {exc}\n"
    lines = [f"commit={commit}"]
    lines.append("dirty=0" if not status else "dirty=1")
    if status:
        lines.append(status)
    return "\n".join(lines) + "\n"


def _rulebase_counter_fields(policy: "RulebaseGreedyPolicy | None") -> dict[str, int]:
    if policy is None:
        return {
            "rulebase_fallback_count": 0,
            "rulebase_chosen_missing_count": 0,
            "rulebase_chosen_not_found_count": 0,
            "rulebase_batch_unsupported_count": 0,
        }
    return {
        "rulebase_fallback_count": int(policy.rulebase_fallback_count),
        "rulebase_chosen_missing_count": int(policy.rulebase_chosen_missing_count),
        "rulebase_chosen_not_found_count": int(policy.rulebase_chosen_not_found_count),
        "rulebase_batch_unsupported_count": int(policy.rulebase_batch_unsupported_count),
    }


def _rerun_config_id(args: argparse.Namespace, config_id: int) -> int:
    return config_id if args.rerun_config_id is None else int(args.rerun_config_id)


def _config_key(opponent_mode: str, rule_kl_coef: float, entropy_coef: float, lr: float) -> dict[str, Any]:
    return {
        "opponent_mode": opponent_mode,
        "rule_kl_coef": float(rule_kl_coef),
        "entropy_coef": float(entropy_coef),
        "lr": float(lr),
    }


def _config_key_label(opponent_mode: str, rule_kl_coef: float, entropy_coef: float, lr: float) -> str:
    return f"{opponent_mode}/rule_kl={rule_kl_coef:g}/entropy={entropy_coef:g}/lr={lr:g}"


def _eval_seed_registry(args: argparse.Namespace) -> list[int]:
    return [int(args.eval_seed_base + idx * args.eval_seed_stride) for idx in range(args.eval_episodes)]


def _eval_seed_registry_id(args: argparse.Namespace) -> str:
    return f"base={args.eval_seed_base}:stride={args.eval_seed_stride}:count={args.eval_episodes}"


def _eval_seed_hash(eval_seeds: list[int]) -> str:
    payload = json.dumps(eval_seeds, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _fmt_optional(value: object) -> str:
    if value is None:
        return "None"
    return f"{float(value):.6g}"


def _max_metric(rows: list[dict[str, Any]], field: str) -> float:
    if not rows:
        return 0.0
    return max(float(row[field]) for row in rows)


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows({key: _to_csvable(value) for key, value in row.items()} for row in rows)


def _to_csvable(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


if __name__ == "__main__":
    main()
