#!/usr/bin/env python3
"""Probe online rollout sampling diversity without training KeqingRL policies."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from keqingrl import DiscardOnlyMahjongEnv, OpponentPool, OpponentPoolEntry, RulePriorDeltaPolicy, RulePriorPolicy
from keqingrl.contracts import PolicyInput, PolicyOutput
from keqingrl.distribution import MaskedCategorical
from keqingrl.learning_signal import batch_diagnostic_rows, seed_registry_hash
from keqingrl.metadata import resolve_rule_score_scale_metadata
from keqingrl.policy import InteractivePolicy
from keqingrl.selfplay import build_episodes_ppo_batch, collect_selfplay_episodes
from scripts.run_keqingrl_discard_research_sweep import RulebaseGreedyPolicy


class TemperaturePolicy(InteractivePolicy):
    def __init__(self, base_policy: InteractivePolicy, *, temperature: float) -> None:
        super().__init__()
        if float(temperature) <= 0.0:
            raise ValueError(f"temperature must be positive, got {temperature}")
        self.base_policy = base_policy
        self.temperature = float(temperature)

    def forward(self, policy_input: PolicyInput) -> PolicyOutput:
        output = self.base_policy(policy_input)
        mask = policy_input.legal_action_mask.bool()
        logits = output.action_logits / self.temperature
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        entropy = MaskedCategorical(logits, mask).entropy()
        aux = dict(output.aux)
        if "final_logits" in aux:
            aux["untempered_final_logits"] = aux["final_logits"]
        aux["final_logits"] = logits
        aux["sampling_temperature"] = torch.tensor(
            self.temperature,
            device=logits.device,
            dtype=logits.dtype,
        )
        return PolicyOutput(
            action_logits=logits,
            value=output.value,
            rank_logits=output.rank_logits,
            entropy=entropy,
            aux=aux,
            next_recurrent_state=output.next_recurrent_state,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sampling-diversity diagnostics from saved KeqingRL checkpoints")
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--source-config-ids", type=int, nargs="+", default=(93, 57, 8))
    parser.add_argument("--rerun-config-ids", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--seed-base", type=int, default=202604260000)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--torch-seed-base", type=int, default=202604260000)
    parser.add_argument("--temperatures", type=float, nargs="+", default=(1.0, 1.5, 2.0, 3.0))
    parser.add_argument("--learner-seats", type=int, nargs="+", default=(0,))
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
    step_rows: list[dict[str, Any]] = []

    for candidate in candidates:
        policy = _load_policy(candidate, device)
        opponent_pool = _opponent_pool(str(candidate["opponent_mode"]))
        for temp_idx, temperature in enumerate(args.temperatures):
            torch_seed = int(args.torch_seed_base + int(candidate["source_config_id"]) * 1000 + temp_idx)
            torch.manual_seed(torch_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(torch_seed)
            probe_policy = TemperaturePolicy(policy, temperature=float(temperature)).to(device)
            episodes = collect_selfplay_episodes(
                DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
                probe_policy,
                num_episodes=args.episodes,
                opponent_pool=opponent_pool,
                learner_seats=tuple(int(seat) for seat in args.learner_seats),
                seed=args.seed_base,
                seed_stride=args.seed_stride,
                greedy=False,
                max_steps=args.max_steps,
                device=device,
            )
            _advantages, _returns, prepared_steps, batch = build_episodes_ppo_batch(
                episodes,
                gamma=args.gamma,
                gae_lambda=args.gae_lambda,
                include_rank_targets=True,
                strict_metadata=True,
            )
            batch = batch.to(device)
            rows, base_summary = batch_diagnostic_rows(probe_policy, batch, prepared_steps, episodes)
            summary = {
                **_candidate_summary(candidate),
                "source_type": "checkpoint",
                "training": False,
                "temperature": float(temperature),
                "rule_score_scale": float(getattr(policy, "rule_score_scale", 1.0)),
                "rule_score_scale_version": "keqingrl_rule_score_scale_v1",
                "episodes": int(args.episodes),
                "learner_seats": list(int(seat) for seat in args.learner_seats),
                "seed_registry_id": _seed_registry_id(args),
                "seed_hash": seed_registry_hash(_seed_registry(args)),
                "torch_sample_seed": torch_seed,
                **base_summary,
                **_sampling_summary(rows),
            }
            summary_rows.append(summary)
            for row in rows:
                step_rows.append(
                    {
                        **_candidate_summary(candidate),
                        "source_type": "checkpoint",
                        "training": False,
                        "temperature": float(temperature),
                        "rule_score_scale": float(getattr(policy, "rule_score_scale", 1.0)),
                        "rule_score_scale_version": "keqingrl_rule_score_scale_v1",
                        "seed_registry_id": _seed_registry_id(args),
                        "seed_hash": seed_registry_hash(_seed_registry(args)),
                        "torch_sample_seed": torch_seed,
                        **row,
                    }
                )
            print(
                "probe "
                f"source={candidate['source_config_id']} "
                f"rerun={candidate['rerun_config_id']} "
                f"temp={float(temperature):g} "
                f"selected_top1={summary['selected_prior_top1_rate']:.6g} "
                f"non_top1={summary['non_top1_selected_count']} "
                f"non_top1_pos_adv={summary['non_top1_positive_advantage_count']}",
                flush=True,
            )

    payload = {
        "mode": "sampling_diversity_probe",
        "source_type": "checkpoint",
        "training": False,
        "candidate_summary": str(args.candidate_summary),
        "source_config_ids": list(args.source_config_ids or ()),
        "rerun_config_ids": args.rerun_config_ids,
        "temperatures": [float(value) for value in args.temperatures],
        "rule_score_scale_values": sorted(
            {float(row["rule_score_scale"]) for row in summary_rows}
        ),
        "rule_score_scale_version": "keqingrl_rule_score_scale_v1",
        "episodes": int(args.episodes),
        "learner_seats": [int(seat) for seat in args.learner_seats],
        "seed_registry_id": _seed_registry_id(args),
        "seed_hash": seed_registry_hash(_seed_registry(args)),
        "summaries": summary_rows,
    }
    _write_json(args.output_dir / "sampling_diversity.json", payload)
    _write_csv(args.output_dir / "summary.csv", summary_rows)
    _write_csv(args.output_dir / "batch_steps.csv", step_rows)
    (args.output_dir / "summary.md").write_text(_summary_markdown(args, summary_rows), encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


def _load_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = _read_csv(args.candidate_summary)
    source_set = {int(value) for value in args.source_config_ids} if args.source_config_ids else None
    rerun_set = {int(value) for value in args.rerun_config_ids} if args.rerun_config_ids else None
    selected = []
    for row in rows:
        source_id = int(row["source_config_id"])
        rerun_id = int(row["rerun_config_id"])
        if source_set is not None and source_id not in source_set:
            continue
        if rerun_set is not None and rerun_id not in rerun_set:
            continue
        selected.append(row)
    if not selected:
        raise ValueError("no matching checkpoint rows selected")
    if rerun_set is None:
        source_counts: dict[int, int] = {}
        for row in selected:
            source_counts[int(row["source_config_id"])] = source_counts.get(int(row["source_config_id"]), 0) + 1
        duplicates = sorted(source_id for source_id, count in source_counts.items() if count > 1)
        if duplicates:
            raise ValueError(
                "candidate summary contains multiple checkpoints for source_config_id "
                f"{duplicates}; pass --rerun-config-ids to select exact checkpoints"
            )
    return [_normalize_candidate_row(row) for row in selected]


def _normalize_candidate_row(row: dict[str, str]) -> dict[str, Any]:
    config_key = _json_or_empty(row.get("config_key"))
    return {
        "source_config_id": int(row["source_config_id"]),
        "rerun_config_id": int(row["rerun_config_id"]),
        "opponent_mode": str(row.get("opponent_mode") or config_key.get("opponent_mode") or "rule_prior_greedy"),
        "rule_kl_coef": _optional_float(row.get("rule_kl_coef")),
        "entropy_coef": _optional_float(row.get("entropy_coef")),
        "lr": _optional_float(row.get("lr")),
        "config_key": config_key,
        "config_key_label": row.get("config_key_label") or json.dumps(config_key, sort_keys=True),
        "checkpoint_path": str(row["checkpoint_path"]),
        "checkpoint_sha256": row.get("checkpoint_sha256") or _file_sha256(Path(row["checkpoint_path"])),
        "config_path": str(row["config_path"]),
    }


def _load_policy(candidate: dict[str, Any], device: torch.device) -> RulePriorDeltaPolicy:
    config_path = Path(candidate["config_path"])
    checkpoint_path = Path(candidate["checkpoint_path"])
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(checkpoint_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    policy = RulePriorDeltaPolicy(
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_res_blocks=int(config["model"]["num_res_blocks"]),
        dropout=float(config["model"].get("dropout", 0.0)),
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
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


def _opponent_pool(opponent_mode: str) -> OpponentPool:
    if opponent_mode == "rule_prior_greedy":
        policy = RulePriorPolicy()
    elif opponent_mode == "rulebase":
        policy = RulebaseGreedyPolicy(strict=True)
    else:
        raise ValueError(f"unsupported opponent mode: {opponent_mode}")
    return OpponentPool((OpponentPoolEntry(policy=policy, policy_version=-1, greedy=True, name=opponent_mode),))


def _candidate_summary(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "source_config_id": int(candidate["source_config_id"]),
        "rerun_config_id": int(candidate["rerun_config_id"]),
        "opponent_mode": candidate["opponent_mode"],
        "rule_kl_coef": candidate["rule_kl_coef"],
        "entropy_coef": candidate["entropy_coef"],
        "lr": candidate["lr"],
        "config_key": candidate["config_key"],
        "config_key_label": candidate["config_key_label"],
        "checkpoint_path": candidate["checkpoint_path"],
        "checkpoint_sha256": candidate["checkpoint_sha256"],
        "config_path": candidate["config_path"],
    }


def _sampling_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    selected_prior_probs = [float(row["selected_prior_prob"]) for row in rows]
    selected_ranks = [int(row["selected_prior_rank"]) for row in rows]
    positive_rows = [row for row in rows if float(row["advantage_raw"]) > 0.0]
    top1_rows = [row for row in rows if bool(row["selected_is_prior_top1"])]
    non_top1_rows = [row for row in rows if not bool(row["selected_is_prior_top1"])]
    return {
        "selected_prior_rank_histogram": _histogram(selected_ranks),
        "selected_prior_prob_mean": _mean(selected_prior_probs),
        "selected_prior_prob_p90": _quantile(selected_prior_probs, 0.90),
        "non_top1_selected_count": len(non_top1_rows),
        "non_top1_selected_rate": len(non_top1_rows) / max(1, len(rows)),
        "non_top1_positive_advantage_count": sum(1 for row in non_top1_rows if float(row["advantage_raw"]) > 0.0),
        "positive_advantage_top1_count": sum(1 for row in top1_rows if float(row["advantage_raw"]) > 0.0),
        "positive_advantage_non_top1_count": sum(1 for row in non_top1_rows if float(row["advantage_raw"]) > 0.0),
        "sample_entropy_mean": _mean(float(row.get("entropy", 0.0)) for row in rows),
        "selected_prior_rank_mean": _mean(selected_ranks),
        "positive_advantage_count": len(positive_rows),
    }


def _summary_markdown(args: argparse.Namespace, rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# KeqingRL Sampling Diversity Probe",
        "",
        "source_type: `checkpoint`",
        "training: `false`",
        f"candidate_summary: `{args.candidate_summary}`",
        f"episodes: `{args.episodes}`",
        f"temperatures: `{','.join(str(float(value)) for value in args.temperatures)}`",
        f"learner_seats: `{','.join(str(int(seat)) for seat in args.learner_seats)}`",
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
            f"temp={row['temperature']:g} "
            f"batch={row['batch_size']} "
            f"selected_top1={row['selected_prior_top1_rate']:.6g} "
            f"non_top1={row['non_top1_selected_count']} "
            f"non_top1_pos_adv={row['non_top1_positive_advantage_count']} "
            f"pos_top1={row['positive_advantage_top1_count']} "
            f"pos_non_top1={row['positive_advantage_non_top1_count']} "
            f"prior_prob_mean={row['selected_prior_prob_mean']:.6g} "
            f"rank_hist={json.dumps(row['selected_prior_rank_histogram'], sort_keys=True)}"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `sampling_diversity.json`",
            "- `summary.csv`",
            "- `batch_steps.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _to_csvable(row.get(key)) for key in fieldnames})


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_jsonable(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(inner) for inner in value]
    if isinstance(value, torch.Tensor):
        return _to_jsonable(value.detach().cpu().tolist())
    if hasattr(value, "item"):
        return value.item()
    return value


def _to_csvable(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(_to_jsonable(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return value


def _json_or_empty(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _optional_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _seed_registry(args: argparse.Namespace) -> list[int]:
    return [int(args.seed_base + idx * args.seed_stride) for idx in range(args.episodes)]


def _seed_registry_id(args: argparse.Namespace) -> str:
    return f"base={args.seed_base}:stride={args.seed_stride}:count={args.episodes}"


def _histogram(values: Sequence[int]) -> dict[str, int]:
    hist: dict[str, int] = {}
    for value in values:
        key = str(int(value))
        hist[key] = hist.get(key, 0) + 1
    return dict(sorted(hist.items(), key=lambda item: int(item[0])))


def _mean(values: Sequence[float] | Any) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return float(sum(float(value) for value in materialized) / len(materialized))


def _quantile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, round((len(ordered) - 1) * float(q))))
    return float(ordered[index])


if __name__ == "__main__":
    main()
