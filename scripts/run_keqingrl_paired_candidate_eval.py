#!/usr/bin/env python3
"""Run paired checkpoint evaluation for controlled discard-only KeqingRL candidates."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from keqingrl import (
    DiscardOnlyMahjongEnv,
    OpponentPool,
    OpponentPoolEntry,
    RulePriorDeltaPolicy,
    RulePriorPolicy,
    run_fixed_seed_evaluation_smoke,
)
from keqingrl.selfplay import collect_selfplay_episodes
from keqingrl.rollout import rollout_step_policy_input
from keqingrl.training import _initialize_policy_from_env_observation
from scripts.run_keqingrl_discard_research_sweep import RulebaseGreedyPolicy


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run paired eval from saved KeqingRL candidate checkpoints")
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--source-config-ids", type=int, nargs="+", default=None)
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-seed-base", type=int, default=202604250000)
    parser.add_argument("--eval-seed-stride", type=int, default=1)
    parser.add_argument("--opponents", nargs="+", choices=("rule_prior_greedy", "rulebase"), default=("rule_prior_greedy", "rulebase"))
    parser.add_argument("--max-kyokus", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--device", default=None)
    parser.add_argument("--diagnostic-episodes", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device is not None else torch.device("cpu")
    candidates = _load_candidates(args.candidate_summary, args.source_config_ids, device)
    baselines = _build_baselines(candidates, args, device)
    policies = candidates + baselines

    rows: list[dict[str, Any]] = []
    for opponent_name in args.opponents:
        baseline_metrics: dict[str, dict[str, Any]] = {}
        for policy_record in policies:
            print(
                f"eval opponent={opponent_name} kind={policy_record['kind']} candidate={policy_record['candidate_id']}",
                flush=True,
            )
            eval_row = _evaluate_record(policy_record, opponent_name, args, device)
            if policy_record["kind"] == "baseline":
                baseline_metrics[policy_record["candidate_id"]] = eval_row
            rows.append(eval_row)
        for row in rows:
            if row["opponent_name"] != opponent_name or row["kind"] != "candidate":
                continue
            zero = baseline_metrics["zero_delta_rule_prior"]
            untrained = baseline_metrics["untrained_rule_prior_delta"]
            row["paired_rank_pt_delta_vs_zero_delta"] = row["rank_pt"] - zero["rank_pt"]
            row["paired_rank_pt_delta_vs_untrained_rule_prior_delta"] = row["rank_pt"] - untrained["rank_pt"]
            row["paired_rank_pt_delta_vs_rule_prior"] = row["paired_rank_pt_delta_vs_zero_delta"]

    _write_json(args.out_dir / "paired_eval.json", _payload(args, rows))
    _write_csv(args.out_dir / "paired_eval.csv", rows)
    summary = _summary_markdown(args, rows)
    (args.out_dir / "summary.md").write_text(summary, encoding="utf-8")
    print(summary)


def _load_candidates(summary_path: Path, source_config_ids: list[int] | None, device: torch.device) -> list[dict[str, Any]]:
    rows = _read_csv(summary_path)
    if source_config_ids is not None:
        source_set = {int(value) for value in source_config_ids}
        rows = [row for row in rows if int(row["source_config_id"]) in source_set]
    if not rows:
        raise ValueError("no candidates selected")
    selected: list[dict[str, Any]] = []
    for row in rows:
        checkpoint_path = Path(row["checkpoint_path"])
        config_path = Path(row["config_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(checkpoint_path)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        policy = RulePriorDeltaPolicy(
            hidden_dim=int(config["model"]["hidden_dim"]),
            num_res_blocks=int(config["model"]["num_res_blocks"]),
            dropout=float(config["model"].get("dropout", 0.0)),
        ).to(device)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.eval()
        selected.append(
            {
                "kind": "candidate",
                "candidate_id": str(row["source_config_id"]),
                "source_config_id": int(row["source_config_id"]),
                "rerun_config_id": int(row["rerun_config_id"]),
                "config_key": json.loads(row["config_key"]),
                "config_key_label": row["config_key_label"],
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": row["checkpoint_sha256"],
                "policy": policy,
                "train_rule_agreement": _optional_float(row.get("rule_agreement")),
                "train_top1_action_changed_rate": _optional_float(row.get("top1_action_changed_rate")),
                "train_neural_delta_abs_mean": _optional_float(row.get("neural_delta_abs_mean")),
                "train_neural_delta_abs_max": _optional_float(row.get("neural_delta_abs_max")),
                "train_approx_kl": _optional_float(row.get("approx_kl")),
                "train_clip_fraction": _optional_float(row.get("clip_fraction")),
            }
        )
    return selected


def _build_baselines(candidates: list[dict[str, Any]], args: argparse.Namespace, device: torch.device) -> list[dict[str, Any]]:
    first_config = candidates[0]["config_key"]
    first_checkpoint_config = json.loads(Path(candidates[0]["checkpoint_path"]).with_name("config.json").read_text(encoding="utf-8"))
    zero_policy = RulePriorPolicy()
    untrained = RulePriorDeltaPolicy(
        hidden_dim=int(first_checkpoint_config["model"]["hidden_dim"]),
        num_res_blocks=int(first_checkpoint_config["model"]["num_res_blocks"]),
        dropout=float(first_checkpoint_config["model"].get("dropout", 0.0)),
    ).to(device)
    _initialize_policy_from_env_observation(
        DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
        untrained,
        seed=args.eval_seed_base,
        device=device,
    )
    untrained.eval()
    return [
        {
            "kind": "baseline",
            "candidate_id": "zero_delta_rule_prior",
            "source_config_id": None,
            "rerun_config_id": None,
            "config_key": {"baseline": "zero_delta_rule_prior", "reference_config_key": first_config},
            "config_key_label": "baseline/zero_delta_rule_prior",
            "checkpoint_path": "",
            "checkpoint_sha256": "",
            "policy": zero_policy,
        },
        {
            "kind": "baseline",
            "candidate_id": "untrained_rule_prior_delta",
            "source_config_id": None,
            "rerun_config_id": None,
            "config_key": {"baseline": "untrained_rule_prior_delta", "reference_config_key": first_config},
            "config_key_label": "baseline/untrained_rule_prior_delta",
            "checkpoint_path": "",
            "checkpoint_sha256": "",
            "policy": untrained,
        },
    ]


def _evaluate_record(record: dict[str, Any], opponent_name: str, args: argparse.Namespace, device: torch.device) -> dict[str, Any]:
    opponent_pool = _opponent_pool(opponent_name)
    metrics = run_fixed_seed_evaluation_smoke(
        DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
        record["policy"],
        num_games=args.eval_episodes,
        seed=args.eval_seed_base,
        seed_stride=args.eval_seed_stride,
        seat_rotation=(0, 1, 2, 3),
        opponent_pool=opponent_pool,
        opponent_name=opponent_name,
        max_steps=args.max_steps,
        greedy=True,
        reuse_training_rollout=False,
        device=device,
    )
    diagnostics = _policy_diagnostics(record["policy"], opponent_pool, args, device)
    return {
        "kind": record["kind"],
        "candidate_id": record["candidate_id"],
        "source_config_id": record["source_config_id"],
        "rerun_config_id": record["rerun_config_id"],
        "config_key": record["config_key"],
        "config_key_label": record["config_key_label"],
        "checkpoint_path": record["checkpoint_path"],
        "checkpoint_sha256": record["checkpoint_sha256"],
        "opponent_name": opponent_name,
        "eval_seed_registry_id": _eval_seed_registry_id(args),
        "eval_seed_hash": _eval_seed_hash(_eval_seed_registry(args)),
        "eval_seed_count": args.eval_episodes,
        "policy_mode": "greedy",
        "seat_rotation": "0,1,2,3",
        "training_rollout_reuse": False,
        "rank_pt": metrics.rank_pt,
        "mean_rank": metrics.average_rank,
        "fourth_rate": metrics.fourth_rate,
        "learner_deal_in_rate": metrics.learner_deal_in_rate,
        "learner_win_rate": metrics.learner_win_rate,
        "learner_call_rate": metrics.learner_call_rate,
        "learner_riichi_rate": metrics.learner_riichi_rate,
        "rule_agreement": diagnostics["rule_agreement"],
        "top1_action_changed_rate": diagnostics["top1_action_changed_rate"],
        "neural_delta_abs_mean": diagnostics["neural_delta_abs_mean"],
        "neural_delta_abs_max": diagnostics["neural_delta_abs_max"],
        "diagnostic_episode_count": args.diagnostic_episodes,
        "diagnostic_step_count": diagnostics["diagnostic_step_count"],
        "train_rule_agreement": record.get("train_rule_agreement"),
        "train_top1_action_changed_rate": record.get("train_top1_action_changed_rate"),
        "train_neural_delta_abs_mean": record.get("train_neural_delta_abs_mean"),
        "train_neural_delta_abs_max": record.get("train_neural_delta_abs_max"),
        "approx_kl": record.get("train_approx_kl"),
        "clip_fraction": record.get("train_clip_fraction"),
        "rulebase_fallback_count": 0,
        "illegal_action_rate_fail_closed": metrics.illegal_action_rate_fail_closed,
        "fallback_rate_fail_closed": metrics.fallback_rate_fail_closed,
        "forced_terminal_missed_fail_closed": metrics.forced_terminal_missed_fail_closed,
        "paired_rank_pt_delta_vs_zero_delta": None,
        "paired_rank_pt_delta_vs_untrained_rule_prior_delta": None,
        "paired_rank_pt_delta_vs_rule_prior": None,
    }


def _policy_diagnostics(policy, opponent_pool: OpponentPool, args: argparse.Namespace, device: torch.device) -> dict[str, float | int]:
    episode_count = args.eval_episodes if args.diagnostic_episodes is None else args.diagnostic_episodes
    abs_values: list[float] = []
    changed = 0
    total = 0
    for seat in (0, 1, 2, 3):
        episodes = collect_selfplay_episodes(
            DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
            policy,
            num_episodes=episode_count,
            opponent_pool=opponent_pool,
            learner_seats=(seat,),
            seed=args.eval_seed_base,
            seed_stride=args.eval_seed_stride,
            max_steps=args.max_steps,
            greedy=True,
            device=device,
        )
        for episode in episodes:
            for step in episode.steps:
                if step.actor != seat or step.is_autopilot:
                    continue
                policy_input = rollout_step_policy_input(step, device=device)
                with torch.no_grad():
                    output = policy.forward(policy_input)
                final_logits = output.aux.get("final_logits", output.action_logits).detach().cpu()
                prior_logits = policy_input.prior_logits.detach().cpu() if policy_input.prior_logits is not None else final_logits
                mask = policy_input.legal_action_mask.detach().cpu().bool()
                valid_final = final_logits.masked_fill(~mask, torch.finfo(final_logits.dtype).min)
                valid_prior = prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min)
                changed += int(torch.argmax(valid_final, dim=1).item() != torch.argmax(valid_prior, dim=1).item())
                delta = (final_logits - prior_logits).abs()[mask]
                abs_values.extend(float(value) for value in delta.flatten())
                total += 1
    if not abs_values:
        return {
            "rule_agreement": 1.0,
            "top1_action_changed_rate": 0.0,
            "neural_delta_abs_mean": 0.0,
            "neural_delta_abs_max": 0.0,
            "diagnostic_step_count": 0,
        }
    return {
        "rule_agreement": 1.0 - changed / max(1, total),
        "top1_action_changed_rate": changed / max(1, total),
        "neural_delta_abs_mean": sum(abs_values) / len(abs_values),
        "neural_delta_abs_max": max(abs_values),
        "diagnostic_step_count": total,
    }


def _opponent_pool(name: str) -> OpponentPool:
    if name == "rule_prior_greedy":
        policy = RulePriorPolicy()
    elif name == "rulebase":
        policy = RulebaseGreedyPolicy(strict=True)
    else:
        raise ValueError(f"unsupported opponent: {name}")
    return OpponentPool((OpponentPoolEntry(policy=policy, policy_version=-1, greedy=True, name=name),))


def _payload(args: argparse.Namespace, rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "candidate_summary": str(args.candidate_summary),
        "eval_seed_registry_id": _eval_seed_registry_id(args),
        "eval_seed_hash": _eval_seed_hash(_eval_seed_registry(args)),
        "eval_seed_count": args.eval_episodes,
        "seat_rotation": [0, 1, 2, 3],
        "policy_mode": "greedy",
        "training_rollout_reuse": False,
        "opponents": list(args.opponents),
        "rows": rows,
    }


def _summary_markdown(args: argparse.Namespace, rows: list[dict[str, Any]]) -> str:
    candidates = [row for row in rows if row["kind"] == "candidate"]
    ranked = sorted(
        candidates,
        key=lambda row: (
            str(row["opponent_name"]),
            -float(row["paired_rank_pt_delta_vs_zero_delta"]),
            float(row["fourth_rate"]),
            float(row["neural_delta_abs_max"]),
        ),
    )
    lines = [
        "# KeqingRL Paired Candidate Evaluation",
        "",
        f"candidate_summary: `{args.candidate_summary}`",
        f"eval_seed_registry_id: `{_eval_seed_registry_id(args)}`",
        f"eval_seed_hash: `{_eval_seed_hash(_eval_seed_registry(args))}`",
        f"eval_seed_count: `{args.eval_episodes}`",
        f"diagnostic_episode_count: `{args.diagnostic_episodes}`",
        "seat_rotation: `0,1,2,3`",
        "policy_mode: `greedy`",
        "training_rollout_reuse: `false`",
        "",
        "## Required Fields",
        "",
        "- checkpoint_path / checkpoint_sha256",
        "- source_config_id / config_key",
        "- eval_seed_registry_id / eval_seed_hash",
        "- learner_deal_in_rate",
        "- rulebase_fallback_count",
        "- illegal_action_rate_fail_closed / fallback_rate_fail_closed / forced_terminal_missed_fail_closed",
        "",
        "## Candidate Results",
        "",
    ]
    for row in ranked:
        lines.append(
            "- "
            f"opponent={row['opponent_name']} "
            f"source_id={row['source_config_id']} "
            f"delta_vs_zero={row['paired_rank_pt_delta_vs_zero_delta']:.6g} "
            f"delta_vs_untrained={row['paired_rank_pt_delta_vs_untrained_rule_prior_delta']:.6g} "
            f"rank_pt={row['rank_pt']:.6g} "
            f"mean_rank={row['mean_rank']:.6g} "
            f"fourth={row['fourth_rate']:.6g} "
            f"learner_deal_in={row['learner_deal_in_rate']:.6g} "
            f"rule_agree={row['rule_agreement']:.6g} "
            f"top1_changed={row['top1_action_changed_rate']:.6g} "
            f"delta_mean={row['neural_delta_abs_mean']:.6g} "
            f"delta_max={row['neural_delta_abs_max']:.6g} "
            f"checkpoint={row['checkpoint_path']}"
        )
    lines.extend(["", "## Decision", "", _decision_text(candidates), ""])
    return "\n".join(lines)


def _decision_text(rows: list[dict[str, Any]]) -> str:
    if rows and all(
        float(row["top1_action_changed_rate"]) == 0.0
        and float(row["rule_agreement"]) == 1.0
        and float(row["neural_delta_abs_max"]) < 0.01
        for row in rows
    ):
        return "Current checkpoints remain rule-prior equivalent on top-1 decisions; prioritize learning-signal research before more eval scaling."
    return "At least one checkpoint changes top-1 or has non-trivial delta; continue paired strength/stability evaluation before changing learning signal."


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


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


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _to_jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(item) for item in value]
    return value


def _to_csvable(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _eval_seed_registry(args: argparse.Namespace) -> list[int]:
    return [int(args.eval_seed_base + idx * args.eval_seed_stride) for idx in range(args.eval_episodes)]


def _eval_seed_registry_id(args: argparse.Namespace) -> str:
    return f"base={args.eval_seed_base}:stride={args.eval_seed_stride}:count={args.eval_episodes}"


def _eval_seed_hash(eval_seeds: list[int]) -> str:
    payload = json.dumps(eval_seeds, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


if __name__ == "__main__":
    main()
