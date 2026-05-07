#!/usr/bin/env python3
"""Evaluate KeqingRL Mortal imitation checkpoints on tensor shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.train_keqingrl_mortal_imitation_shards import (  # noqa: E402
    _adapt_policy_action_feature_dim,
    _config_mapping_to_argv,
    _index_policy_input,
    _index_teacher_batch,
    _load_json_config,
    _load_manifest,
    _manifest_action_feature_dim,
    _policy_input_to_device,
    _teacher_batch_to_device,
)
from scripts.materialize_keqingrl_mortal_imitation_shards import _sanitize_teacher_batch_with_stats  # noqa: E402
from scripts.run_keqingrl_mortal_imitation import (  # noqa: E402
    DeltaSupportProjectionPolicy,
    _load_policy,
    _teacher_support_tensors,
    _write_csv,
    _write_json,
    mortal_imitation_loss,
)
from keqingrl.actions import ActionType  # noqa: E402


_ACTION_TYPE_NAMES = tuple(action_type.name for action_type in ActionType)
_ACTION_ID_TYPE_DIVISOR = (35**4) * 5 * 256
_CALL_TYPES = {ActionType.CHI, ActionType.PON, ActionType.DAIMINKAN}
_WIN_TYPES = {ActionType.RON, ActionType.TSUMO}


def _parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, remaining_argv = config_parser.parse_known_args()
    configured_argv: list[str] = []
    if config_args.config is not None:
        configured_argv = _config_mapping_to_argv(_load_json_config(config_args.config))

    parser = argparse.ArgumentParser(description="Evaluate KeqingRL Mortal imitation tensor shards")
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size-rows", type=int, default=2048)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-support", choices=("topk", "adaptive-topk", "full-legal"), default="full-legal")
    parser.add_argument("--teacher-topk", type=int, default=3)
    parser.add_argument("--rule-score-scale", type=float, default=0.0)
    parser.add_argument("--support-policy-mode", choices=("support-only-topk", "unrestricted"), default="unrestricted")
    parser.add_argument("--delta-support-mode", choices=("topk", "all"), default="all")
    parser.add_argument("--delta-support-topk", type=int, default=3)
    parser.add_argument("--delta-support-margin-threshold", type=float, default=0.75)
    parser.add_argument("--outside-support-delta-mode", choices=("zero", "negative-clip"), default="zero")
    parser.add_argument("--high-margin-threshold", type=float, default=1.0)
    args = parser.parse_args(configured_argv + remaining_argv)
    if int(args.batch_size_rows) <= 0:
        raise ValueError("--batch-size-rows must be positive")
    return args


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(args.shard_dir)
    shard_paths = [Path(row["path"]) for row in manifest["shards"]]
    if not shard_paths:
        raise RuntimeError(f"no shards found under {args.shard_dir}")
    device = torch.device(args.device)
    candidate = _candidate_from_checkpoint(args)
    action_feature_dim = _manifest_action_feature_dim(manifest, shard_paths[0])
    base_policy = _load_policy(candidate, device)
    _adapt_policy_action_feature_dim(base_policy, action_feature_dim)
    base_policy.rule_score_scale = float(args.rule_score_scale)
    policy = DeltaSupportProjectionPolicy(
        base_policy,
        support_mode=str(args.delta_support_mode),
        topk=int(args.delta_support_topk),
        margin_threshold=float(args.delta_support_margin_threshold),
        outside_support_delta_mode=str(args.outside_support_delta_mode),
        support_policy_mode=str(args.support_policy_mode),
    ).to(device)
    policy.eval()

    started = time.perf_counter()
    shard_rows: list[dict[str, Any]] = []
    totals = _EvalTotals()
    breakdown = _ActionTypeBreakdown()
    family_breakdown = _FamilyBreakdown()
    with torch.no_grad():
        for shard_index, shard_path in enumerate(shard_paths):
            shard_start = time.perf_counter()
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            row_count = int(shard["row_count"])
            shard_totals = _EvalTotals()
            shard_breakdown = _ActionTypeBreakdown()
            for start in range(0, row_count, int(args.batch_size_rows)):
                end = min(row_count, start + int(args.batch_size_rows))
                indices = list(range(start, end))
                policy_input = _policy_input_to_device(_index_policy_input(shard["policy_input"], indices), device)
                raw_teacher_batch = _index_teacher_batch(shard["teacher_batch"], indices)
                teacher_batch, sanitize_summary = _sanitize_teacher_batch_with_stats(raw_teacher_batch)
                teacher_batch = _teacher_batch_to_device(teacher_batch, device)
                output = policy(policy_input)
                loss = mortal_imitation_loss(
                    output,
                    policy_input,
                    teacher_support=str(args.teacher_support),
                    teacher_topk=int(args.teacher_topk),
                    teacher_temperature=float(args.teacher_temperature),
                    strict_extra=True,
                    teacher_batch=teacher_batch,
                )
                rank_stats = _rank_stats(output.action_logits, teacher_batch)
                valid_rows = int(teacher_batch.row_valid_mask.sum().detach().cpu().item())
                source_rows = int(raw_teacher_batch.row_valid_mask.numel())
                row_metadata = _index_row_metadata(shard.get("row_metadata", {}), indices)
                batch_breakdown = _action_type_breakdown(output, policy_input, teacher_batch, row_metadata, float(args.teacher_temperature))
                batch_family_rows = _family_diagnostics(
                    output.action_logits,
                    policy_input,
                    teacher_batch,
                    high_margin_threshold=float(args.high_margin_threshold),
                )
                shard_totals.add(loss, rank_stats, valid_rows, source_rows, int(sanitize_summary["teacher_row_sanitized_invalid_count"]))
                totals.add(loss, rank_stats, valid_rows, source_rows, int(sanitize_summary["teacher_row_sanitized_invalid_count"]))
                shard_breakdown.add_rows(batch_breakdown)
                breakdown.add_rows(batch_breakdown)
                family_breakdown.add_rows(batch_family_rows)
            row = shard_totals.row()
            row.update(
                {
                    "shard_index": int(shard_index),
                    "shard_path": str(shard_path),
                    "shard_sec": time.perf_counter() - shard_start,
                }
            )
            shard_rows.append(row)
            print(
                f"shard-eval shard={shard_index + 1}/{len(shard_paths)} rows={row['row_count']} "
                f"ce={row['teacher_ce']:.6g} kl={row['teacher_kl']:.6g} "
                f"agree={row['teacher_policy_agreement']:.6g} rank_ge5={row['rank_ge5_rate']:.6g}",
                flush=True,
            )
            del shard

    summary = totals.row()
    summary.update(
        {
            "checkpoint": str(args.checkpoint),
            "shard_dir": str(args.shard_dir),
            "shard_count": len(shard_paths),
            "eval_sec": time.perf_counter() - started,
            "rows_per_sec": totals.row_count / max(time.perf_counter() - started, 1e-9),
        }
    )
    _write_csv(args.output_dir / "shard_eval_summary.csv", [summary])
    _write_csv(args.output_dir / "shard_eval_shards.csv", shard_rows)
    _write_csv(args.output_dir / "shard_eval_action_type_breakdown.csv", breakdown.rows())
    _write_csv(args.output_dir / "shard_eval_action_family_breakdown.csv", family_breakdown.rows())
    _write_csv(args.output_dir / "shard_eval_call_diagnostics.csv", family_breakdown.call_rows())
    _write_json(
        args.output_dir / "shard_eval.json",
        {
            "mode": "keqingrl_mortal_imitation_shard_eval_v1",
            "summary": summary,
            "shards": shard_rows,
            "action_type_breakdown": breakdown.rows(),
            "action_family_breakdown": family_breakdown.rows(),
            "call_diagnostics": family_breakdown.call_rows(),
        },
    )
    print(
        f"shard-eval done rows={summary['row_count']} ce={summary['teacher_ce']:.6g} "
        f"kl={summary['teacher_kl']:.6g} agree={summary['teacher_policy_agreement']:.6g} "
        f"rank_ge5={summary['rank_ge5_rate']:.6g} sec={summary['eval_sec']:.1f}",
        flush=True,
    )


class _EvalTotals:
    def __init__(self) -> None:
        self.row_count = 0
        self.source_row_count = 0
        self.ce_sum = 0.0
        self.kl_sum = 0.0
        self.agree_sum = 0.0
        self.rank_sum = 0.0
        self.rank_ge5_count = 0
        self.sanitized_invalid_count = 0

    def add(self, loss, rank_stats: Mapping[str, float], row_count: int, source_row_count: int, sanitized_invalid_count: int) -> None:
        rows = max(0, int(row_count))
        self.row_count += rows
        self.source_row_count += max(0, int(source_row_count))
        self.sanitized_invalid_count += max(0, int(sanitized_invalid_count))
        self.ce_sum += float(loss.teacher_ce.detach().cpu()) * rows
        self.kl_sum += float(loss.teacher_kl.detach().cpu()) * rows
        self.agree_sum += float(loss.teacher_policy_agreement.detach().cpu()) * rows
        self.rank_sum += float(rank_stats["rank_sum"])
        self.rank_ge5_count += int(rank_stats["rank_ge5_count"])

    def row(self) -> dict[str, Any]:
        denom = max(1, self.row_count)
        return {
            "row_count": int(self.row_count),
            "source_row_count": int(self.source_row_count),
            "teacher_ce": self.ce_sum / denom,
            "teacher_kl": self.kl_sum / denom,
            "teacher_policy_agreement": self.agree_sum / denom,
            "teacher_top1_rank_mean": self.rank_sum / denom,
            "rank_ge5_count": int(self.rank_ge5_count),
            "rank_ge5_rate": self.rank_ge5_count / denom,
            "teacher_row_valid_count": int(self.row_count),
            "teacher_row_invalid_count": int(max(0, self.source_row_count - self.row_count)),
            "teacher_row_sanitized_invalid_count": int(self.sanitized_invalid_count),
            "teacher_row_valid_rate": float(self.row_count / max(1, self.source_row_count)),
        }


def _candidate_from_checkpoint(args: argparse.Namespace) -> dict[str, Any]:
    config_path = args.config_path
    if config_path is None:
        loaded = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        config_payload = loaded.get("config")
        if not isinstance(config_payload, Mapping):
            raise RuntimeError("--config-path is required when checkpoint does not contain config")
        generated_config_path = args.output_dir / "checkpoint_config_from_payload.json"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        generated_config_path.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        config_path = generated_config_path
    return {
        "source_config_id": 93,
        "rerun_config_id": 0,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": "",
        "config_path": str(config_path),
    }


def _rank_stats(student_logits: torch.Tensor, teacher_batch) -> dict[str, float]:
    logits = student_logits.float()
    mask = teacher_batch.legal_action_mask
    if mask is None:
        mask = torch.ones_like(teacher_batch.teacher_scores, dtype=torch.bool)
    mask = mask.to(device=logits.device).bool()
    teacher_scores = teacher_batch.teacher_scores.to(device=logits.device).float()
    teacher_scores = teacher_scores.masked_fill(~mask, torch.finfo(torch.float32).min)
    logits = logits.masked_fill(~mask, torch.finfo(torch.float32).min)
    teacher_top1 = teacher_scores.argmax(dim=-1)
    ranks: list[int] = []
    valid = teacher_batch.row_valid_mask.to(device=logits.device).bool()
    for row_idx, teacher_idx in enumerate(teacher_top1.detach().cpu().tolist()):
        if not bool(valid[row_idx].detach().cpu().item()):
            continue
        order = torch.argsort(logits[row_idx], descending=True).detach().cpu().tolist()
        ranks.append(int(order.index(int(teacher_idx)) + 1))
    if not ranks:
        return {"rank_mean": 0.0, "rank_sum": 0.0, "rank_ge5_count": 0, "rank_ge5_rate": 0.0}
    rank_sum = sum(ranks)
    rank_ge5_count = sum(1 for rank in ranks if rank >= 5)
    return {
        "rank_mean": rank_sum / len(ranks),
        "rank_sum": float(rank_sum),
        "rank_ge5_count": int(rank_ge5_count),
        "rank_ge5_rate": rank_ge5_count / len(ranks),
    }


class _ActionTypeBreakdown:
    def __init__(self) -> None:
        self._buckets: dict[str, dict[str, float]] = {}

    def add_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            key = str(row["teacher_top1_action_type"])
            bucket = self._buckets.setdefault(
                key,
                {
                    "row_count": 0.0,
                    "teacher_ce_sum": 0.0,
                    "teacher_kl_sum": 0.0,
                    "teacher_agree_count": 0.0,
                    "rank_sum": 0.0,
                    "rank_ge5_count": 0.0,
                },
            )
            row_count = float(row["row_count"])
            bucket["row_count"] += row_count
            bucket["teacher_ce_sum"] += float(row["teacher_ce"]) * row_count
            bucket["teacher_kl_sum"] += float(row["teacher_kl"]) * row_count
            bucket["teacher_agree_count"] += float(row["teacher_agreement"]) * row_count
            bucket["rank_sum"] += float(row["teacher_top1_rank_mean"]) * row_count
            bucket["rank_ge5_count"] += float(row["rank_ge5_rate"]) * row_count

    def rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for key in sorted(self._buckets):
            bucket = self._buckets[key]
            row_count = int(bucket["row_count"])
            denom = max(1, row_count)
            rows.append(
                {
                    "teacher_top1_action_type": key,
                    "row_count": row_count,
                    "teacher_ce": bucket["teacher_ce_sum"] / denom,
                    "teacher_kl": bucket["teacher_kl_sum"] / denom,
                    "teacher_agreement": bucket["teacher_agree_count"] / denom,
                    "teacher_top1_rank_mean": bucket["rank_sum"] / denom,
                    "rank_ge5_rate": bucket["rank_ge5_count"] / denom,
                }
            )
        return rows


def _index_row_metadata(row_metadata: Mapping[str, torch.Tensor], indices: Sequence[int]) -> dict[str, torch.Tensor]:
    if not row_metadata:
        return {}
    tensor_indices = torch.tensor(indices, dtype=torch.long)
    return {str(key): value.index_select(0, tensor_indices) for key, value in row_metadata.items()}


def _action_type_breakdown(output, policy_input, teacher_batch, row_metadata: Mapping[str, torch.Tensor], teacher_temperature: float) -> list[dict[str, Any]]:
    teacher_types = row_metadata.get("teacher_top1_action_type_id")
    if teacher_types is None:
        return []
    support_logits, _, support_teacher, support_mask, _ = _teacher_support_tensors(output, policy_input, teacher_batch)
    valid = teacher_batch.row_valid_mask.to(device=support_logits.device).bool() & support_mask.any(dim=-1)
    min_value = torch.finfo(torch.float32).min
    support_logits = support_logits.masked_fill(~support_mask, min_value)
    support_teacher = support_teacher.masked_fill(~support_mask, min_value)
    teacher_probs = torch.softmax(support_teacher / float(teacher_temperature), dim=-1)
    teacher_log_probs = torch.log(teacher_probs.clamp_min(1e-12))
    policy_log_probs = torch.log_softmax(support_logits, dim=-1)
    per_row_ce = -(teacher_probs * policy_log_probs).sum(dim=-1)
    per_row_kl = (teacher_probs * (teacher_log_probs - policy_log_probs)).sum(dim=-1)
    teacher_argmax = support_teacher.argmax(dim=-1)
    policy_argmax = support_logits.argmax(dim=-1)
    rank_stats = _row_ranks(output.action_logits, teacher_batch)
    buckets: dict[str, dict[str, float]] = {}
    for row_idx, type_id in enumerate(teacher_types.detach().cpu().tolist()):
        if not bool(valid[row_idx].detach().cpu().item()) or int(type_id) < 0 or int(type_id) >= len(_ACTION_TYPE_NAMES):
            continue
        key = _ACTION_TYPE_NAMES[int(type_id)]
        bucket = buckets.setdefault(
            key,
            {
                "row_count": 0.0,
                "teacher_ce_sum": 0.0,
                "teacher_kl_sum": 0.0,
                "teacher_agree_count": 0.0,
                "rank_sum": 0.0,
                "rank_ge5_count": 0.0,
            },
        )
        bucket["row_count"] += 1.0
        bucket["teacher_ce_sum"] += float(per_row_ce[row_idx].detach().cpu().item())
        bucket["teacher_kl_sum"] += float(per_row_kl[row_idx].detach().cpu().item())
        bucket["teacher_agree_count"] += float(policy_argmax[row_idx].item() == teacher_argmax[row_idx].item())
        rank = rank_stats.get(int(row_idx), 0)
        bucket["rank_sum"] += float(rank)
        bucket["rank_ge5_count"] += float(rank >= 5)
    rows: list[dict[str, Any]] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        row_count = int(bucket["row_count"])
        denom = max(1, row_count)
        rows.append(
            {
                "teacher_top1_action_type": key,
                "row_count": row_count,
                "teacher_ce": bucket["teacher_ce_sum"] / denom,
                "teacher_kl": bucket["teacher_kl_sum"] / denom,
                "teacher_agreement": bucket["teacher_agree_count"] / denom,
                "teacher_top1_rank_mean": bucket["rank_sum"] / denom,
                "rank_ge5_rate": bucket["rank_ge5_count"] / denom,
            }
        )
    return rows


def _row_ranks(student_logits: torch.Tensor, teacher_batch) -> dict[int, int]:
    logits = student_logits.float()
    mask = teacher_batch.legal_action_mask
    if mask is None:
        mask = torch.ones_like(teacher_batch.teacher_scores, dtype=torch.bool)
    mask = mask.to(device=logits.device).bool()
    teacher_scores = teacher_batch.teacher_scores.to(device=logits.device).float().masked_fill(
        ~mask,
        torch.finfo(torch.float32).min,
    )
    logits = logits.masked_fill(~mask, torch.finfo(torch.float32).min)
    valid = teacher_batch.row_valid_mask.to(device=logits.device).bool()
    ranks: dict[int, int] = {}
    for row_idx, teacher_idx in enumerate(teacher_scores.argmax(dim=-1).detach().cpu().tolist()):
        if not bool(valid[row_idx].detach().cpu().item()):
            continue
        order = torch.argsort(logits[row_idx], descending=True).detach().cpu().tolist()
        ranks[int(row_idx)] = int(order.index(int(teacher_idx)) + 1)
    return ranks


class _FamilyBreakdown:
    def __init__(self) -> None:
        self._buckets: dict[str, dict[str, float]] = {}

    def add_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        for row in rows:
            bucket = self._buckets.setdefault(str(row["bucket"]), _empty_family_bucket())
            for key, value in row.items():
                if key == "bucket":
                    continue
                bucket[key] = float(bucket.get(key, 0.0)) + float(value)

    def rows(self) -> list[dict[str, Any]]:
        return [_family_bucket_row(name, values) for name, values in sorted(self._buckets.items())]

    def call_rows(self) -> list[dict[str, Any]]:
        return [row for row in self.rows() if row["bucket"] in {"call_opportunity", "chi_opportunity", "pon_opportunity", "teacher_call", "teacher_chi", "teacher_pon"}]


def _empty_family_bucket() -> dict[str, float]:
    return {
        "row_count": 0.0,
        "exact_agreement_count": 0.0,
        "family_agreement_count": 0.0,
        "teacher_call_count": 0.0,
        "student_call_count": 0.0,
        "rulebase_call_count": 0.0,
        "student_call_when_teacher_pass_count": 0.0,
        "student_pass_when_teacher_call_count": 0.0,
        "student_call_when_high_margin_teacher_pass_count": 0.0,
        "student_pass_when_high_margin_teacher_call_count": 0.0,
        "high_margin_disagreement_count": 0.0,
        "teacher_margin_top1_top2_sum": 0.0,
        "teacher_call_vs_pass_margin_sum": 0.0,
        "student_call_vs_pass_margin_sum": 0.0,
        "teacher_chi_best_vs_second_margin_sum": 0.0,
        "wrong_chi_shape_count": 0.0,
    }


def _family_bucket_row(name: str, bucket: Mapping[str, float]) -> dict[str, Any]:
    row_count = int(bucket.get("row_count", 0.0))
    denom = max(1, row_count)
    teacher_call_count = int(bucket.get("teacher_call_count", 0.0))
    teacher_pass_count = row_count - teacher_call_count
    return {
        "bucket": name,
        "row_count": row_count,
        "exact_agreement_rate": bucket.get("exact_agreement_count", 0.0) / denom,
        "family_agreement_rate": bucket.get("family_agreement_count", 0.0) / denom,
        "teacher_call_rate": bucket.get("teacher_call_count", 0.0) / denom,
        "student_call_rate": bucket.get("student_call_count", 0.0) / denom,
        "rulebase_call_rate": bucket.get("rulebase_call_count", 0.0) / denom,
        "student_call_when_teacher_pass_rate": bucket.get("student_call_when_teacher_pass_count", 0.0) / max(1, teacher_pass_count),
        "student_pass_when_teacher_call_rate": bucket.get("student_pass_when_teacher_call_count", 0.0) / max(1, teacher_call_count),
        "student_call_when_high_margin_teacher_pass_count": int(bucket.get("student_call_when_high_margin_teacher_pass_count", 0.0)),
        "student_pass_when_high_margin_teacher_call_count": int(bucket.get("student_pass_when_high_margin_teacher_call_count", 0.0)),
        "high_margin_disagreement_count": int(bucket.get("high_margin_disagreement_count", 0.0)),
        "teacher_margin_top1_top2_mean": bucket.get("teacher_margin_top1_top2_sum", 0.0) / denom,
        "teacher_call_vs_pass_margin_mean": bucket.get("teacher_call_vs_pass_margin_sum", 0.0) / denom,
        "student_call_vs_pass_margin_mean": bucket.get("student_call_vs_pass_margin_sum", 0.0) / denom,
        "teacher_chi_best_vs_second_margin_mean": bucket.get("teacher_chi_best_vs_second_margin_sum", 0.0) / denom,
        "wrong_chi_shape_count": int(bucket.get("wrong_chi_shape_count", 0.0)),
    }


def _family_diagnostics(
    student_logits: torch.Tensor,
    policy_input,
    teacher_batch,
    *,
    high_margin_threshold: float,
) -> list[dict[str, Any]]:
    legal_mask = policy_input.legal_action_mask.to(device=student_logits.device).bool()
    valid = teacher_batch.row_valid_mask.to(device=student_logits.device).bool()
    action_type_ids = _action_type_ids(policy_input).to(device=student_logits.device)
    teacher_scores = _legal_teacher_scores(teacher_batch, width=int(student_logits.shape[1])).to(device=student_logits.device)
    teacher_scores = teacher_scores.masked_fill(~legal_mask, torch.finfo(torch.float32).min)
    student_scores = student_logits.float().masked_fill(~legal_mask, torch.finfo(torch.float32).min)
    prior_scores = policy_input.prior_logits.to(device=student_logits.device).float().masked_fill(
        ~legal_mask,
        torch.finfo(torch.float32).min,
    )
    teacher_top1 = teacher_scores.argmax(dim=-1)
    student_top1 = student_scores.argmax(dim=-1)
    prior_top1 = prior_scores.argmax(dim=-1)
    teacher_type = action_type_ids.gather(1, teacher_top1.unsqueeze(-1)).squeeze(-1)
    student_type = action_type_ids.gather(1, student_top1.unsqueeze(-1)).squeeze(-1)
    prior_type = action_type_ids.gather(1, prior_top1.unsqueeze(-1)).squeeze(-1)
    pass_mask = legal_mask & (action_type_ids == int(ActionType.PASS))
    call_mask = legal_mask & _isin_action_types(action_type_ids, _CALL_TYPES)
    chi_mask = legal_mask & (action_type_ids == int(ActionType.CHI))
    pon_mask = legal_mask & (action_type_ids == int(ActionType.PON))
    ron_mask = legal_mask & (action_type_ids == int(ActionType.RON))
    response_mask = valid & pass_mask.any(dim=-1)
    masks = {
        "response_all": response_mask,
        "call_opportunity": response_mask & call_mask.any(dim=-1),
        "chi_opportunity": response_mask & chi_mask.any(dim=-1),
        "pon_opportunity": response_mask & pon_mask.any(dim=-1),
        "ron_opportunity": response_mask & ron_mask.any(dim=-1),
        "teacher_pass": response_mask & (teacher_type == int(ActionType.PASS)),
        "teacher_call": response_mask & _type_is_call(teacher_type),
        "teacher_chi": response_mask & (teacher_type == int(ActionType.CHI)),
        "teacher_pon": response_mask & (teacher_type == int(ActionType.PON)),
        "teacher_ron": response_mask & (teacher_type == int(ActionType.RON)),
    }
    teacher_family = _family_ids(teacher_type)
    student_family = _family_ids(student_type)
    prior_family = _family_ids(prior_type)
    teacher_top2_margin = _top1_top2_margin(teacher_scores, legal_mask)
    teacher_call_vs_pass = _best_margin(teacher_scores, call_mask, pass_mask)
    student_call_vs_pass = _best_margin(student_scores, call_mask, pass_mask)
    chi_best_second = _top1_top2_margin(teacher_scores, chi_mask)
    high_margin_teacher_pass = (teacher_type == int(ActionType.PASS)) & (-teacher_call_vs_pass >= float(high_margin_threshold))
    high_margin_teacher_call = _type_is_call(teacher_type) & (teacher_call_vs_pass >= float(high_margin_threshold))
    exact_agreement = teacher_top1 == student_top1
    family_agreement = teacher_family == student_family
    teacher_call = teacher_family == _FAMILY_CALL
    student_call = student_family == _FAMILY_CALL
    prior_call = prior_family == _FAMILY_CALL
    wrong_chi_shape = (
        (teacher_type == int(ActionType.CHI))
        & (student_type == int(ActionType.CHI))
        & ~exact_agreement
    )
    rows: list[dict[str, Any]] = []
    for bucket_name, row_mask in masks.items():
        if not bool(row_mask.any().detach().cpu().item()):
            continue
        rows.append(
            {
                "bucket": bucket_name,
                "row_count": int(row_mask.sum().detach().cpu().item()),
                "exact_agreement_count": int((exact_agreement & row_mask).sum().detach().cpu().item()),
                "family_agreement_count": int((family_agreement & row_mask).sum().detach().cpu().item()),
                "teacher_call_count": int((teacher_call & row_mask).sum().detach().cpu().item()),
                "student_call_count": int((student_call & row_mask).sum().detach().cpu().item()),
                "rulebase_call_count": int((prior_call & row_mask).sum().detach().cpu().item()),
                "student_call_when_teacher_pass_count": int((student_call & (teacher_type == int(ActionType.PASS)) & row_mask).sum().detach().cpu().item()),
                "student_pass_when_teacher_call_count": int(((student_type == int(ActionType.PASS)) & teacher_call & row_mask).sum().detach().cpu().item()),
                "student_call_when_high_margin_teacher_pass_count": int((student_call & high_margin_teacher_pass & row_mask).sum().detach().cpu().item()),
                "student_pass_when_high_margin_teacher_call_count": int(((student_type == int(ActionType.PASS)) & high_margin_teacher_call & row_mask).sum().detach().cpu().item()),
                "high_margin_disagreement_count": int((~family_agreement & (high_margin_teacher_pass | high_margin_teacher_call) & row_mask).sum().detach().cpu().item()),
                "teacher_margin_top1_top2_sum": float(teacher_top2_margin[row_mask].sum().detach().cpu().item()),
                "teacher_call_vs_pass_margin_sum": float(teacher_call_vs_pass[row_mask].sum().detach().cpu().item()),
                "student_call_vs_pass_margin_sum": float(student_call_vs_pass[row_mask].sum().detach().cpu().item()),
                "teacher_chi_best_vs_second_margin_sum": float(chi_best_second[row_mask].sum().detach().cpu().item()),
                "wrong_chi_shape_count": int((wrong_chi_shape & row_mask).sum().detach().cpu().item()),
            }
        )
    return rows


_FAMILY_PASS = 0
_FAMILY_CALL = 1
_FAMILY_WIN = 2
_FAMILY_OTHER = 3


def _action_type_ids(policy_input) -> torch.LongTensor:
    return (policy_input.legal_action_ids.long() // _ACTION_ID_TYPE_DIVISOR).clamp(min=0, max=len(_ACTION_TYPE_NAMES) - 1)


def _isin_action_types(action_type_ids: torch.Tensor, action_types: set[ActionType]) -> torch.BoolTensor:
    mask = torch.zeros_like(action_type_ids, dtype=torch.bool)
    for action_type in action_types:
        mask |= action_type_ids == int(action_type)
    return mask


def _type_is_call(action_type_ids: torch.Tensor) -> torch.BoolTensor:
    return _isin_action_types(action_type_ids, _CALL_TYPES)


def _family_ids(action_type_ids: torch.Tensor) -> torch.LongTensor:
    family = torch.full_like(action_type_ids, _FAMILY_OTHER)
    family[action_type_ids == int(ActionType.PASS)] = _FAMILY_PASS
    family[_type_is_call(action_type_ids)] = _FAMILY_CALL
    family[_isin_action_types(action_type_ids, _WIN_TYPES)] = _FAMILY_WIN
    return family


def _legal_teacher_scores(teacher_batch, *, width: int) -> torch.Tensor:
    if teacher_batch.mapped_legal_scores is not None:
        return teacher_batch.mapped_legal_scores[:, :width].float()
    scores = torch.full(
        (int(teacher_batch.teacher_scores.shape[0]), int(width)),
        torch.finfo(torch.float32).min,
        dtype=torch.float32,
        device=teacher_batch.teacher_scores.device,
    )
    teacher_scores = teacher_batch.teacher_scores.float()
    support_mask = teacher_batch.support_mask
    if support_mask is None:
        support_mask = teacher_batch.legal_action_mask
    if support_mask is not None:
        teacher_scores = teacher_scores.masked_fill(~support_mask.to(device=teacher_scores.device).bool(), torch.finfo(torch.float32).min)
    if teacher_batch.topk_indices is None:
        return teacher_scores[:, :width]
    indices = teacher_batch.topk_indices.to(device=teacher_scores.device).long()
    valid = (indices >= 0) & (indices < int(width))
    return scores.scatter(1, indices.clamp(min=0, max=max(0, int(width) - 1)), teacher_scores.masked_fill(~valid, torch.finfo(torch.float32).min))


def _top1_top2_margin(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    masked = scores.masked_fill(~mask, torch.finfo(torch.float32).min)
    valid_count = mask.sum(dim=-1)
    values = torch.topk(masked, k=min(2, int(masked.shape[1])), dim=-1).values
    if int(values.shape[1]) == 1:
        margin = torch.zeros_like(values[:, 0])
    else:
        margin = values[:, 0] - values[:, 1]
    return torch.where(valid_count >= 2, margin, torch.zeros_like(margin))


def _best_margin(scores: torch.Tensor, positive_mask: torch.Tensor, negative_mask: torch.Tensor) -> torch.Tensor:
    min_value = torch.finfo(torch.float32).min
    positive = scores.masked_fill(~positive_mask, min_value).max(dim=-1).values
    negative = scores.masked_fill(~negative_mask, min_value).max(dim=-1).values
    valid = positive_mask.any(dim=-1) & negative_mask.any(dim=-1)
    return torch.where(valid, positive - negative, torch.zeros_like(positive))


if __name__ == "__main__":
    main()
