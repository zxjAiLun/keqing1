#!/usr/bin/env python3
"""Evaluate KeqingRL Mortal imitation checkpoints on tensor shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.train_keqingrl_mortal_imitation_shards import (  # noqa: E402
    _config_mapping_to_argv,
    _index_policy_input,
    _index_teacher_batch,
    _load_json_config,
    _load_manifest,
    _policy_input_to_device,
    _teacher_batch_to_device,
)
from scripts.run_keqingrl_mortal_imitation import (  # noqa: E402
    DeltaSupportProjectionPolicy,
    _load_policy,
    _write_csv,
    _write_json,
    mortal_imitation_loss,
)


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
    base_policy = _load_policy(candidate, device)
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
    with torch.no_grad():
        for shard_index, shard_path in enumerate(shard_paths):
            shard_start = time.perf_counter()
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            row_count = int(shard["row_count"])
            shard_totals = _EvalTotals()
            for start in range(0, row_count, int(args.batch_size_rows)):
                end = min(row_count, start + int(args.batch_size_rows))
                indices = list(range(start, end))
                policy_input = _policy_input_to_device(_index_policy_input(shard["policy_input"], indices), device)
                teacher_batch = _teacher_batch_to_device(_index_teacher_batch(shard["teacher_batch"], indices), device)
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
                shard_totals.add(loss, rank_stats, valid_rows)
                totals.add(loss, rank_stats, valid_rows)
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
    _write_json(
        args.output_dir / "shard_eval.json",
        {
            "mode": "keqingrl_mortal_imitation_shard_eval_v1",
            "summary": summary,
            "shards": shard_rows,
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
        self.ce_sum = 0.0
        self.kl_sum = 0.0
        self.agree_sum = 0.0
        self.rank_sum = 0.0
        self.rank_ge5_count = 0

    def add(self, loss, rank_stats: Mapping[str, float], row_count: int) -> None:
        rows = max(0, int(row_count))
        self.row_count += rows
        self.ce_sum += float(loss.teacher_ce.detach().cpu()) * rows
        self.kl_sum += float(loss.teacher_kl.detach().cpu()) * rows
        self.agree_sum += float(loss.teacher_policy_agreement.detach().cpu()) * rows
        self.rank_sum += float(rank_stats["rank_mean"]) * rows
        self.rank_ge5_count += int(round(float(rank_stats["rank_ge5_rate"]) * rows))

    def row(self) -> dict[str, Any]:
        denom = max(1, self.row_count)
        return {
            "row_count": int(self.row_count),
            "teacher_ce": self.ce_sum / denom,
            "teacher_kl": self.kl_sum / denom,
            "teacher_policy_agreement": self.agree_sum / denom,
            "teacher_top1_rank_mean": self.rank_sum / denom,
            "rank_ge5_count": int(self.rank_ge5_count),
            "rank_ge5_rate": self.rank_ge5_count / denom,
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
    for row_idx, teacher_idx in enumerate(teacher_top1.detach().cpu().tolist()):
        order = torch.argsort(logits[row_idx], descending=True).detach().cpu().tolist()
        ranks.append(int(order.index(int(teacher_idx)) + 1))
    if not ranks:
        return {"rank_mean": 0.0, "rank_ge5_rate": 0.0}
    return {
        "rank_mean": sum(ranks) / len(ranks),
        "rank_ge5_rate": sum(1 for rank in ranks if rank >= 5) / len(ranks),
    }


if __name__ == "__main__":
    main()
