#!/usr/bin/env python3
"""Run controlled discard-only KeqingRL research sweeps."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass, replace
import json
from pathlib import Path
from typing import Any

import torch

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


DEFAULT_RULE_KL_COEFS = (0.0, 0.001, 0.01, 0.02, 0.05)
DEFAULT_ENTROPY_COEFS = (0.0, 0.001, 0.005, 0.01)
DEFAULT_LRS = (3e-5, 1e-4, 3e-4)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run controlled discard-only KeqingRL PPO sweep")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/keqingrl_discard_research"))
    parser.add_argument("--seed", type=int, default=20260425)
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
    parser.add_argument("--rule-kl-coefs", type=float, nargs="+", default=DEFAULT_RULE_KL_COEFS)
    parser.add_argument("--entropy-coefs", type=float, nargs="+", default=DEFAULT_ENTROPY_COEFS)
    parser.add_argument("--lrs", type=float, nargs="+", default=DEFAULT_LRS)
    parser.add_argument(
        "--opponent-modes",
        nargs="+",
        choices=("rule_prior_greedy", "rulebase"),
        default=("rule_prior_greedy",),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    configs = [
        (opponent_mode, rule_kl_coef, entropy_coef, lr)
        for opponent_mode in args.opponent_modes
        for rule_kl_coef in args.rule_kl_coefs
        for entropy_coef in args.entropy_coefs
        for lr in args.lrs
    ]
    rows: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []

    for config_id, (opponent_mode, rule_kl_coef, entropy_coef, lr) in enumerate(configs):
        config_seed = int(args.seed + config_id * 100_000)
        report = _run_config(
            config_id=config_id,
            opponent_mode=str(opponent_mode),
            rule_kl_coef=float(rule_kl_coef),
            entropy_coef=float(entropy_coef),
            lr=float(lr),
            seed=config_seed,
            args=args,
        )
        summaries.append(report["summary"])
        rows.extend(report["iterations"])
        print(
            "config "
            f"{config_id + 1}/{len(configs)} "
            f"opponent={opponent_mode} "
            f"rule_kl={rule_kl_coef:g} entropy={entropy_coef:g} lr={lr:g} "
            f"rank_pt={report['summary']['eval_rank_pt']:.6g} "
            f"fourth={report['summary']['eval_fourth_rate']:.6g} "
            f"delta_max={report['summary']['neural_delta_abs_max']:.6g}",
            flush=True,
        )

    payload = {
        "scope": {
            "style_context": "neutral",
            "action_scope": "DISCARD only",
            "reward_spec": "fixed default",
            "opponents": list(args.opponent_modes),
            "teacher_imitation": False,
            "full_action_rl": False,
            "style_variants": False,
        },
        "seed": args.seed,
        "grid": {
            "rule_kl_coef": list(args.rule_kl_coefs),
            "entropy_coef": list(args.entropy_coefs),
            "lr": list(args.lrs),
            "opponent_mode": list(args.opponent_modes),
        },
        "run": {
            "iterations": args.iterations,
            "rollout_episodes": args.rollout_episodes,
            "update_epochs": args.update_epochs,
            "eval_episodes": args.eval_episodes,
            "learner_seats": [0],
            "opponents": list(args.opponent_modes),
        },
        "summaries": summaries,
        "iterations": rows,
    }
    _write_json(args.out_dir / "sweep.json", payload)
    _write_csv(args.out_dir / "summary.csv", summaries)
    _write_csv(args.out_dir / "iterations.csv", rows)
    (args.out_dir / "summary.md").write_text(_summary_markdown(args, summaries), encoding="utf-8")

    print((args.out_dir / "summary.md").read_text(encoding="utf-8"))


def _run_config(
    *,
    config_id: int,
    opponent_mode: str,
    rule_kl_coef: float,
    entropy_coef: float,
    lr: float,
    seed: int,
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
    opponent_pool = _build_opponent_pool(opponent_mode)

    iteration_rows: list[dict[str, Any]] = []
    for iteration in range(args.iterations):
        iteration_seed = seed + iteration * 10_000
        result = run_ppo_iteration(
            env,
            policy,
            optimizer,
            num_episodes=args.rollout_episodes,
            opponent_pool=opponent_pool,
            learner_seats=(0,),
            update_epochs=args.update_epochs,
            seed=iteration_seed,
            policy_version=iteration,
            max_steps=args.max_steps,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            include_rank_targets=True,
            clip_eps=args.clip_eps,
            value_coef=args.value_coef,
            entropy_coef=entropy_coef,
            rank_coef=args.rank_coef,
            rule_kl_coef=rule_kl_coef,
            prior_kl_eps=args.prior_kl_eps,
            normalize_advantages=True,
            max_grad_norm=args.max_grad_norm,
            device=device,
            strict_metadata=True,
        )
        loss = result.losses[-1]
        metrics = result.metrics
        delta_stats = _ppo_delta_smoke_stats(policy, result.batch)
        iteration_rows.append(
            {
                "config_id": config_id,
                "iteration": iteration,
                "seed": iteration_seed,
                "rule_kl_coef": rule_kl_coef,
                "entropy_coef": entropy_coef,
                "lr": lr,
                "opponent_mode": opponent_mode,
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
                "deal_in_rate": metrics.deal_in_rate,
                "batch_size": metrics.batch_size,
                "episode_count": metrics.episode_count,
                "illegal_action_rate": metrics.illegal_action_rate,
                "fallback_rate": metrics.fallback_rate,
                "forced_terminal_missed": metrics.forced_terminal_missed,
            }
        )

    eval_metrics = evaluate_policy(
        DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
        policy,
        num_episodes=args.eval_episodes,
        opponent_pool=opponent_pool,
        learner_seats=(0,),
        seed=seed + 900_000,
        greedy=True,
        device=device,
    )
    final = iteration_rows[-1]
    summary = {
        "config_id": config_id,
        "seed": seed,
        "rule_kl_coef": rule_kl_coef,
        "entropy_coef": entropy_coef,
        "lr": lr,
        "opponent_mode": opponent_mode,
        "iterations": args.iterations,
        "rollout_episodes": args.rollout_episodes,
        "eval_episodes": args.eval_episodes,
        "approx_kl": final["approx_kl"],
        "clip_fraction": final["clip_fraction"],
        "entropy": final["entropy"],
        "rule_kl": final["rule_kl"],
        "rule_agreement": final["rule_agreement"],
        "neural_delta_abs_mean": final["neural_delta_abs_mean"],
        "neural_delta_abs_max": final["neural_delta_abs_max"],
        "top1_action_changed_rate": final["top1_action_changed_rate"],
        "train_rank_pt": final["rank_pt"],
        "train_fourth_rate": final["fourth_rate"],
        "train_deal_in_rate": final["deal_in_rate"],
        "eval_rank_pt": eval_metrics.mean_terminal_reward,
        "eval_mean_rank": eval_metrics.mean_rank,
        "eval_fourth_rate": eval_metrics.fourth_place_rate,
        "eval_deal_in_rate": eval_metrics.deal_in_rate,
        "eval_win_rate": eval_metrics.win_rate,
        "illegal_action_rate": final["illegal_action_rate"],
        "fallback_rate": final["fallback_rate"],
        "forced_terminal_missed": final["forced_terminal_missed"],
    }
    return {"summary": summary, "iterations": iteration_rows}


def _summary_markdown(args: argparse.Namespace, summaries: list[dict[str, Any]]) -> str:
    ranked = sorted(
        summaries,
        key=lambda row: (
            -float(row["eval_rank_pt"]),
            float(row["eval_fourth_rate"]),
            float(row["eval_deal_in_rate"]),
            float(row["neural_delta_abs_max"]),
        ),
    )
    lines = [
        "# KeqingRL Controlled Discard-Only Sweep",
        "",
        f"seed: `{args.seed}`",
        f"configs: `{len(summaries)}`",
        f"iterations/config: `{args.iterations}`",
        f"rollout_episodes/config/iter: `{args.rollout_episodes}`",
        f"eval_episodes/config: `{args.eval_episodes}`",
        "",
        "## Scope",
        "",
        "- style_context: neutral",
        "- action_scope: DISCARD only",
        "- reward_spec: fixed default",
        f"- opponents: {', '.join(args.opponent_modes)}",
        "- no teacher imitation, no full action RL, no style variants",
        "",
        "## Stability Checks",
        "",
        f"- max illegal_action_rate: {_max_metric(summaries, 'illegal_action_rate'):.6g}",
        f"- max fallback_rate: {_max_metric(summaries, 'fallback_rate'):.6g}",
        f"- max forced_terminal_missed: {_max_metric(summaries, 'forced_terminal_missed'):.6g}",
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
            f"deal_in={row['eval_deal_in_rate']:.6g} "
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
        ]
    )
    return "\n".join(lines) + "\n"


def _build_opponent_pool(opponent_mode: str) -> OpponentPool:
    if opponent_mode == "rule_prior_greedy":
        policy: InteractivePolicy = RulePriorPolicy()
    elif opponent_mode == "rulebase":
        policy = RulebaseGreedyPolicy()
    else:
        raise ValueError(f"unsupported opponent_mode: {opponent_mode}")
    return OpponentPool(
        (
            OpponentPoolEntry(
                policy=policy,
                policy_version=-1,
                greedy=True,
                name=opponent_mode,
            ),
        )
    )


class RulebaseGreedyPolicy(RulePriorPolicy):
    """Greedy policy that follows env-provided Rust rulebase choices when present."""

    def forward(self, policy_input: PolicyInput) -> PolicyOutput:
        output = super().forward(policy_input)
        chosen_key = policy_input.metadata.get("rulebase_chosen")
        if (
            chosen_key is None
            or policy_input.legal_actions is None
            or len(policy_input.legal_actions) != 1
            or int(output.action_logits.shape[0]) != 1
        ):
            return output
        try:
            chosen_index = next(
                idx
                for idx, action in enumerate(policy_input.legal_actions[0])
                if action.canonical_key == chosen_key
            )
        except StopIteration:
            return output

        mask = policy_input.legal_action_mask.bool()
        logits = torch.full_like(output.action_logits, torch.finfo(output.action_logits.dtype).min)
        logits = logits.masked_fill(~mask, torch.finfo(output.action_logits.dtype).min)
        logits[0, chosen_index] = 0.0
        entropy = MaskedCategorical(logits, mask).entropy()
        aux = dict(output.aux)
        aux["final_logits"] = logits
        return replace(output, action_logits=logits, entropy=entropy, aux=aux)


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
        writer.writerows(rows)


if __name__ == "__main__":
    main()
