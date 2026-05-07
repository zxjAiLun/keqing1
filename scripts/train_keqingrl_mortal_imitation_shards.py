#!/usr/bin/env python3
"""Train KeqingRL Mortal imitation from pre-materialized tensor shards."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
import time
from typing import Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.materialize_keqingrl_mortal_imitation_shards import _slice_teacher_batch  # noqa: E402
from scripts.run_keqingrl_mortal_imitation import (  # noqa: E402
    DeltaSupportProjectionPolicy,
    _checkpoint_artifact_run_dir,
    _file_sha256,
    _latest_checkpoint_rows,
    _load_imitation_candidates,
    _load_policy,
    _save_imitation_checkpoint,
    _student_logit_source,
    _write_csv,
    _write_json,
    mortal_imitation_loss,
)


_BOOLEAN_OPTIONAL_CONFIG_KEYS = {"shuffle_shards", "shuffle_rows"}


def _parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, remaining_argv = config_parser.parse_known_args()
    configured_argv: list[str] = []
    if config_args.config is not None:
        configured_argv = _config_mapping_to_argv(_load_json_config(config_args.config))

    parser = argparse.ArgumentParser(description="Train KeqingRL Mortal imitation from tensor shards")
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--source-config-ids", type=int, nargs="+", default=(93,))
    parser.add_argument("--rerun-config-ids", type=int, nargs="+", default=None)
    parser.add_argument("--shard-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts/keqingrl/checkpoints"))
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size-rows", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-source", choices=("mortal-action-q",), default="mortal-action-q")
    parser.add_argument("--teacher-support", choices=("topk", "adaptive-topk", "full-legal"), default="full-legal")
    parser.add_argument("--teacher-topk", type=int, default=3)
    parser.add_argument("--rule-score-scale", type=float, default=0.0)
    parser.add_argument("--support-policy-mode", choices=("support-only-topk", "unrestricted"), default="unrestricted")
    parser.add_argument("--delta-support-mode", choices=("topk", "all"), default="all")
    parser.add_argument("--delta-support-topk", type=int, default=3)
    parser.add_argument("--delta-support-margin-threshold", type=float, default=0.75)
    parser.add_argument("--outside-support-delta-mode", choices=("zero", "negative-clip"), default="zero")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shuffle-shards", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--shuffle-rows", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=202605070000)
    parser.add_argument("--save-checkpoint-every-epoch", type=int, default=1)
    parser.add_argument("--mortal-teacher-checkpoint", type=Path, default=Path("artifacts/mortal_training/mortal.pth"))
    parser.add_argument("--mortal-teacher-strict-extra-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--episodes", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=0)
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--max-kyokus", type=int, default=0)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--self-turn-action-types", nargs="+", default=("DISCARD", "REACH_DISCARD", "TSUMO", "ANKAN", "KAKAN", "RYUKYOKU"))
    parser.add_argument("--response-action-types", nargs="*", default=("PASS", "RON", "PON", "CHI", "DAIMINKAN"))
    parser.add_argument("--forced-autopilot-action-types", nargs="*", default=("TSUMO", "RON", "RYUKYOKU"))
    args = parser.parse_args(configured_argv + remaining_argv)
    if int(args.epochs) <= 0:
        raise ValueError("--epochs must be positive")
    if int(args.batch_size_rows) <= 0:
        raise ValueError("--batch-size-rows must be positive")
    args.iterations = int(args.epochs)
    args.update_epochs = 1
    return args


def _load_json_config(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"config must be a JSON object: {path}")
    return dict(payload)


def _config_mapping_to_argv(config: Mapping[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in config.items():
        if value is None:
            continue
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool):
            if key in _BOOLEAN_OPTIONAL_CONFIG_KEYS:
                argv.append(flag if value else f"--no-{str(key).replace('_', '-')}")
                continue
            raise ValueError(f"boolean config key is not a known boolean CLI option: {key}")
        argv.append(flag)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            argv.extend(str(item) for item in value)
        else:
            argv.append(str(value))
    return argv


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = _load_manifest(args.shard_dir)
    shard_paths = [Path(row["path"]) for row in manifest["shards"]]
    if not shard_paths:
        raise RuntimeError(f"no shards found under {args.shard_dir}")
    candidates = _load_imitation_candidates(args)
    if len(candidates) != 1:
        raise RuntimeError(f"shard trainer expects exactly one candidate, got {len(candidates)}")
    candidate = candidates[0]
    device = torch.device(args.device)
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
    optimizer = torch.optim.Adam(base_policy.parameters(), lr=float(args.lr))

    iteration_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    rng = random.Random(int(args.seed))
    global_step = 0
    for epoch in range(int(args.epochs)):
        epoch_start = time.perf_counter()
        epoch_paths = list(shard_paths)
        if bool(args.shuffle_shards):
            rng.shuffle(epoch_paths)
        totals = _MetricTotals()
        for shard_path in epoch_paths:
            shard = torch.load(shard_path, map_location="cpu", weights_only=False)
            row_count = int(shard["row_count"])
            order = list(range(row_count))
            if bool(args.shuffle_rows):
                rng.shuffle(order)
            for start in range(0, row_count, int(args.batch_size_rows)):
                indices = order[start : start + int(args.batch_size_rows)]
                policy_input = _index_policy_input(shard["policy_input"], indices).to(device) if hasattr(shard["policy_input"], "to") else _policy_input_to_device(_index_policy_input(shard["policy_input"], indices), device)
                teacher_batch = _teacher_batch_to_device(_index_teacher_batch(shard["teacher_batch"], indices), device)
                optimizer.zero_grad(set_to_none=True)
                output = policy(policy_input)
                loss = mortal_imitation_loss(
                    output,
                    policy_input,
                    teacher_support=str(args.teacher_support),
                    teacher_topk=int(args.teacher_topk),
                    teacher_temperature=float(args.teacher_temperature),
                    strict_extra=bool(args.mortal_teacher_strict_extra_mask),
                    teacher_batch=teacher_batch,
                )
                loss.loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(base_policy.parameters(), float(args.max_grad_norm))
                optimizer.step()
                totals.add(loss, int(teacher_batch.row_valid_mask.sum().item()))
                global_step += 1
            del shard
        row = totals.row(epoch=epoch, global_step=global_step, epoch_sec=time.perf_counter() - epoch_start)
        row.update(
            {
                "checkpoint_path": "",
                "checkpoint_sha256": "",
                "shard_dir": str(args.shard_dir),
                "shard_count": len(shard_paths),
                "lr": float(args.lr),
                "batch_size_rows": int(args.batch_size_rows),
                "teacher_source": str(args.teacher_source),
                "teacher_support": str(args.teacher_support),
                "teacher_temperature": float(args.teacher_temperature),
                "student_logit_source": _student_logit_source(args),
            }
        )
        if _should_save(args, epoch):
            checkpoint = _save_imitation_checkpoint(
                args,
                candidate,
                base_policy,
                optimizer,
                config_id=0,
                iteration=epoch,
                summary_row=_checkpoint_summary_row(row),
            )
            row["checkpoint_path"] = checkpoint["checkpoint_path"]
            row["checkpoint_sha256"] = checkpoint["checkpoint_sha256"]
            checkpoint_rows.append(checkpoint)
        iteration_rows.append(row)
        _write_outputs(args, iteration_rows, checkpoint_rows, manifest)
        print(
            f"shard-train epoch={epoch + 1}/{int(args.epochs)} rows={row['row_count']} "
            f"teacher_ce={row['teacher_ce']:.6g} teacher_kl={row['teacher_kl']:.6g} "
            f"agree={row['teacher_policy_agreement']:.6g} sec={row['epoch_sec']:.1f}",
            flush=True,
        )


class _MetricTotals:
    def __init__(self) -> None:
        self.row_count = 0
        self.loss_sum = 0.0
        self.ce_sum = 0.0
        self.kl_sum = 0.0
        self.agree_sum = 0.0
        self.valid_sum = 0

    def add(self, loss, row_count: int) -> None:
        rows = max(0, int(row_count))
        self.row_count += rows
        self.valid_sum += rows
        self.loss_sum += float(loss.loss.detach().cpu()) * rows
        self.ce_sum += float(loss.teacher_ce.detach().cpu()) * rows
        self.kl_sum += float(loss.teacher_kl.detach().cpu()) * rows
        self.agree_sum += float(loss.teacher_policy_agreement.detach().cpu()) * rows

    def row(self, *, epoch: int, global_step: int, epoch_sec: float) -> dict[str, Any]:
        denom = max(1, self.row_count)
        return {
            "epoch": int(epoch),
            "global_step": int(global_step),
            "row_count": int(self.row_count),
            "teacher_loss": self.loss_sum / denom,
            "teacher_ce": self.ce_sum / denom,
            "teacher_kl": self.kl_sum / denom,
            "teacher_policy_agreement": self.agree_sum / denom,
            "teacher_row_valid_count": int(self.valid_sum),
            "teacher_row_valid_rate": float(self.valid_sum / denom),
            "epoch_sec": float(epoch_sec),
            "rows_per_sec": float(self.row_count / max(epoch_sec, 1e-9)),
        }


def _load_manifest(shard_dir: Path) -> dict[str, Any]:
    manifest = json.loads((Path(shard_dir) / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("mode") != "keqingrl_mortal_imitation_shards_v1":
        raise RuntimeError(f"unsupported shard manifest: {manifest.get('mode')}")
    return manifest


def _policy_input_to_device(policy_input, device: torch.device):
    return policy_input.__class__(
        obs=policy_input.obs.__class__(
            tile_obs=policy_input.obs.tile_obs.to(device),
            scalar_obs=policy_input.obs.scalar_obs.to(device),
            history_obs=None if policy_input.obs.history_obs is None else policy_input.obs.history_obs.to(device),
            extras={key: value.to(device) for key, value in policy_input.obs.extras.items()},
        ),
        legal_action_ids=policy_input.legal_action_ids.to(device),
        legal_action_features=policy_input.legal_action_features.to(device),
        legal_action_mask=policy_input.legal_action_mask.to(device),
        rule_context=policy_input.rule_context.to(device),
        raw_rule_scores=None if policy_input.raw_rule_scores is None else policy_input.raw_rule_scores.to(device),
        prior_logits=None if policy_input.prior_logits is None else policy_input.prior_logits.to(device),
        style_context=None if policy_input.style_context is None else policy_input.style_context.to(device),
        legal_actions=None,
        recurrent_state=None,
        metadata=policy_input.metadata,
    )


def _teacher_batch_to_device(teacher_batch, device: torch.device):
    from dataclasses import replace

    sanitized = _sanitize_teacher_batch(teacher_batch)
    return replace(
        sanitized,
        teacher_scores=sanitized.teacher_scores.to(device),
        prior_logits=sanitized.prior_logits.to(device),
        row_valid_mask=sanitized.row_valid_mask.to(device),
        topk_indices=None if sanitized.topk_indices is None else sanitized.topk_indices.to(device),
        support_mask=None if sanitized.support_mask is None else sanitized.support_mask.to(device),
        legal_action_mask=None if sanitized.legal_action_mask is None else sanitized.legal_action_mask.to(device),
        mapped_legal_scores=None if sanitized.mapped_legal_scores is None else sanitized.mapped_legal_scores.to(device),
    )


def _sanitize_teacher_batch(teacher_batch):
    from dataclasses import replace

    support_mask = teacher_batch.support_mask
    if support_mask is None:
        support_mask = teacher_batch.legal_action_mask
    if support_mask is None:
        support_mask = torch.ones_like(teacher_batch.teacher_scores, dtype=torch.bool)
    support_mask = support_mask.bool()
    scores = teacher_batch.teacher_scores.float()
    sentinel_floor = torch.finfo(torch.float32).min / 2
    valid_scores = torch.isfinite(scores) & (scores > sentinel_floor)
    valid_rows = teacher_batch.row_valid_mask.bool() & support_mask.any(dim=-1) & (valid_scores | ~support_mask).all(dim=-1)
    return replace(teacher_batch, row_valid_mask=valid_rows)


def _index_policy_input(policy_input, indices: Sequence[int]):
    tensor_indices = torch.tensor(indices, dtype=torch.long)
    return _slice_or_index_policy_input(policy_input, tensor_indices)


def _slice_or_index_policy_input(policy_input, indices: torch.LongTensor):
    from dataclasses import replace

    obs = policy_input.obs
    return replace(
        policy_input,
        obs=replace(
            obs,
            tile_obs=obs.tile_obs.index_select(0, indices),
            scalar_obs=obs.scalar_obs.index_select(0, indices),
            history_obs=None if obs.history_obs is None else obs.history_obs.index_select(0, indices),
            extras={key: value.index_select(0, indices) for key, value in obs.extras.items()},
        ),
        legal_action_ids=policy_input.legal_action_ids.index_select(0, indices),
        legal_action_features=policy_input.legal_action_features.index_select(0, indices),
        legal_action_mask=policy_input.legal_action_mask.index_select(0, indices),
        rule_context=policy_input.rule_context.index_select(0, indices),
        raw_rule_scores=None if policy_input.raw_rule_scores is None else policy_input.raw_rule_scores.index_select(0, indices),
        prior_logits=None if policy_input.prior_logits is None else policy_input.prior_logits.index_select(0, indices),
        style_context=None if policy_input.style_context is None else policy_input.style_context.index_select(0, indices),
        legal_actions=None,
    )


def _index_teacher_batch(teacher_batch, indices: Sequence[int]):
    tensor_indices = torch.tensor(indices, dtype=torch.long)
    # Reuse the contiguous slicing helper for small ordered microbatches when possible.
    if indices and list(indices) == list(range(indices[0], indices[0] + len(indices))):
        return _slice_teacher_batch(teacher_batch, int(indices[0]), int(indices[0]) + len(indices))
    from dataclasses import replace

    return replace(
        teacher_batch,
        teacher_scores=teacher_batch.teacher_scores.index_select(0, tensor_indices),
        prior_logits=teacher_batch.prior_logits.index_select(0, tensor_indices),
        row_valid_mask=teacher_batch.row_valid_mask.index_select(0, tensor_indices),
        topk_indices=None if teacher_batch.topk_indices is None else teacher_batch.topk_indices.index_select(0, tensor_indices),
        support_mask=None if teacher_batch.support_mask is None else teacher_batch.support_mask.index_select(0, tensor_indices),
        legal_action_mask=None if teacher_batch.legal_action_mask is None else teacher_batch.legal_action_mask.index_select(0, tensor_indices),
        mapped_legal_scores=None if teacher_batch.mapped_legal_scores is None else teacher_batch.mapped_legal_scores.index_select(0, tensor_indices),
    )


def _checkpoint_summary_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "top1_changed_vs_parent_rate": 0.0,
        "teacher_ce": float(row["teacher_ce"]),
        "teacher_kl": float(row["teacher_kl"]),
        "teacher_policy_agreement": float(row["teacher_policy_agreement"]),
        "mapping_row_count": int(row["row_count"]),
        "mapping_available_count": int(row["row_count"]),
        "teacher_row_valid_count": int(row["teacher_row_valid_count"]),
        "teacher_row_valid_rate": float(row["teacher_row_valid_rate"]),
        "teacher_row_invalid_count": 0,
        "fail_closed_count": 0,
    }


def _should_save(args: argparse.Namespace, epoch: int) -> bool:
    every = int(args.save_checkpoint_every_epoch)
    return every > 0 and ((epoch + 1) % every == 0 or (epoch + 1) == int(args.epochs))


def _write_outputs(
    args: argparse.Namespace,
    iteration_rows: Sequence[dict[str, Any]],
    checkpoint_rows: Sequence[dict[str, Any]],
    manifest: Mapping[str, Any],
) -> None:
    _write_csv(args.output_dir / "shard_training_iterations.csv", iteration_rows)
    _write_csv(args.output_dir / "checkpoint_iterations.csv", checkpoint_rows)
    _write_csv(args.output_dir / "checkpoint_summary.csv", _latest_checkpoint_rows(checkpoint_rows))
    _write_json(
        args.output_dir / "shard_training.json",
        {
            "mode": "keqingrl_mortal_imitation_shard_training_v1",
            "shard_dir": str(args.shard_dir),
            "shard_manifest_sha256": _file_sha256(Path(args.shard_dir) / "manifest.json"),
            "source_row_count": manifest.get("row_count"),
            "artifact_dir": str(_checkpoint_artifact_run_dir(args)),
            "iterations": iteration_rows,
            "checkpoints": checkpoint_rows,
        },
    )


if __name__ == "__main__":
    main()
