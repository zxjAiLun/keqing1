#!/usr/bin/env python3
"""Materialize KeqingRL Mortal imitation tensor shards from replay sidecars."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from keqingrl.contracts import ObsTensorBatch, PolicyInput  # noqa: E402
from scripts.run_keqingrl_mortal_imitation import (  # noqa: E402
    _build_replay_imitation_batch,
    _file_sha256,
    _write_json,
    prepare_mortal_imitation_teacher_data,
)


def _parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, remaining_argv = config_parser.parse_known_args()
    configured_argv: list[str] = []
    if config_args.config is not None:
        configured_argv = _config_mapping_to_argv(_load_json_config(config_args.config))

    parser = argparse.ArgumentParser(description="Materialize Mortal imitation tensor shards")
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mortal-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--actors", type=int, nargs="+", default=(0, 1, 2, 3))
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--replay-files-per-build", type=int, default=10)
    parser.add_argument("--shard-size-rows", type=int, default=8192)
    parser.add_argument("--teacher-support", choices=("topk", "adaptive-topk", "full-legal"), default="full-legal")
    parser.add_argument("--teacher-topk", type=int, default=3)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--mortal-teacher-strict-extra-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-kyokus", type=int, default=0)
    parser.add_argument("--self-turn-action-types", nargs="+", default=("DISCARD", "REACH_DISCARD", "TSUMO", "ANKAN", "KAKAN", "RYUKYOKU"))
    parser.add_argument("--response-action-types", nargs="*", default=("PASS", "RON", "PON", "CHI", "DAIMINKAN"))
    parser.add_argument("--forced-autopilot-action-types", nargs="*", default=("TSUMO", "RON", "RYUKYOKU"))
    args = parser.parse_args(configured_argv + remaining_argv)
    args.learner_seats = tuple(int(actor) for actor in args.actors)
    if int(args.replay_files_per_build) <= 0:
        raise ValueError("--replay-files-per-build must be positive")
    if int(args.shard_size_rows) <= 0:
        raise ValueError("--shard-size-rows must be positive")
    if int(args.skip) < 0:
        raise ValueError("--skip must be non-negative")
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
            if key in {"recursive", "mortal_teacher_strict_extra_mask"}:
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
    replay_paths = _replay_paths(args)
    if not replay_paths:
        raise RuntimeError(f"no .mjson files found under {args.replay_dir}")

    shard_rows: list[dict[str, Any]] = []
    pending: dict[str, Any] | None = None
    pending_rows = 0
    shard_index = 0
    total_rows = 0
    total_summary: dict[str, int] = {}
    started = time.perf_counter()

    for chunk_index, chunk_paths in enumerate(_chunks(replay_paths, int(args.replay_files_per_build))):
        chunk_start = time.perf_counter()
        steps, batch, replay_summary = _build_replay_imitation_batch(args, replay_paths=chunk_paths, rollout_seed=chunk_index)
        teacher_data = prepare_mortal_imitation_teacher_data(
            batch.policy_input,
            prepared_steps=steps,
            strict_extra=bool(args.mortal_teacher_strict_extra_mask),
            teacher_support=str(args.teacher_support),
            teacher_topk=int(args.teacher_topk),
        )
        _accumulate_summary(total_summary, replay_summary)
        _accumulate_summary(total_summary, teacher_data.summary)
        shard_payload = _payload_from_batch(batch.policy_input, teacher_data.teacher_batch)
        for start in range(0, int(shard_payload["row_count"]), int(args.shard_size_rows)):
            end = min(int(shard_payload["row_count"]), start + int(args.shard_size_rows))
            piece = _slice_payload(shard_payload, start, end)
            pending, pending_rows, shard_index, rows = _append_piece(
                args,
                pending=pending,
                pending_rows=pending_rows,
                piece=piece,
                shard_index=shard_index,
            )
            shard_rows.extend(rows)
            total_rows += end - start
        print(
            f"materialize-shards chunk={chunk_index + 1} files={len(chunk_paths)} rows={shard_payload['row_count']} "
            f"total_rows={total_rows} sec={time.perf_counter() - chunk_start:.1f}",
            flush=True,
        )

    if pending is not None and pending_rows > 0:
        shard_rows.append(_write_shard(args, pending, shard_index))

    manifest = {
        "mode": "keqingrl_mortal_imitation_shards_v1",
        "replay_dir": str(args.replay_dir),
        "replay_count": len(replay_paths),
        "row_count": total_rows,
        "shard_count": len(shard_rows),
        "teacher_support": str(args.teacher_support),
        "teacher_topk": int(args.teacher_topk),
        "teacher_temperature": float(args.teacher_temperature),
        "mortal_teacher_checkpoint": str(args.mortal_teacher_checkpoint),
        "actors": [int(actor) for actor in args.actors],
        "summary": total_summary,
        "shards": shard_rows,
        "wall_time_sec": time.perf_counter() - started,
    }
    _write_json(args.output_dir / "manifest.json", manifest)
    print(
        f"materialize-shards done replays={len(replay_paths)} rows={total_rows} shards={len(shard_rows)} "
        f"out={args.output_dir}",
        flush=True,
    )


def _replay_paths(args: argparse.Namespace) -> list[Path]:
    paths = sorted(Path(args.replay_dir).rglob("*.mjson") if bool(args.recursive) else Path(args.replay_dir).glob("*.mjson"))
    if int(args.skip) > 0:
        paths = paths[int(args.skip) :]
    if int(args.limit) > 0:
        paths = paths[: int(args.limit)]
    return paths


def _chunks(paths: Sequence[Path], size: int) -> list[list[Path]]:
    return [list(paths[index : index + size]) for index in range(0, len(paths), size)]


def _payload_from_batch(policy_input: PolicyInput, teacher_batch) -> dict[str, Any]:
    clean_obs = ObsTensorBatch(
        tile_obs=policy_input.obs.tile_obs.cpu(),
        scalar_obs=policy_input.obs.scalar_obs.cpu(),
        history_obs=None if policy_input.obs.history_obs is None else policy_input.obs.history_obs.cpu(),
        extras={},
    )
    clean_input = replace(
        policy_input,
        obs=clean_obs,
        legal_actions=None,
        recurrent_state=None,
        metadata=dict(policy_input.metadata),
    )
    return {
        "row_count": int(policy_input.legal_action_mask.shape[0]),
        "policy_input": clean_input,
        "teacher_batch": _teacher_batch_to_cpu(teacher_batch),
    }


def _teacher_batch_to_cpu(teacher_batch):
    return replace(
        teacher_batch,
        teacher_scores=teacher_batch.teacher_scores.cpu(),
        prior_logits=teacher_batch.prior_logits.cpu(),
        row_valid_mask=teacher_batch.row_valid_mask.cpu(),
        topk_indices=None if teacher_batch.topk_indices is None else teacher_batch.topk_indices.cpu(),
        support_mask=None if teacher_batch.support_mask is None else teacher_batch.support_mask.cpu(),
        legal_action_mask=None if teacher_batch.legal_action_mask is None else teacher_batch.legal_action_mask.cpu(),
        mapped_legal_scores=None if teacher_batch.mapped_legal_scores is None else teacher_batch.mapped_legal_scores.cpu(),
    )


def _slice_payload(payload: Mapping[str, Any], start: int, end: int) -> dict[str, Any]:
    return {
        "row_count": int(end - start),
        "policy_input": _slice_policy_input(payload["policy_input"], start, end),
        "teacher_batch": _slice_teacher_batch(payload["teacher_batch"], start, end),
    }


def _slice_policy_input(policy_input: PolicyInput, start: int, end: int) -> PolicyInput:
    obs = policy_input.obs
    return replace(
        policy_input,
        obs=replace(
            obs,
            tile_obs=obs.tile_obs[start:end],
            scalar_obs=obs.scalar_obs[start:end],
            history_obs=None if obs.history_obs is None else obs.history_obs[start:end],
            extras={key: value[start:end] for key, value in obs.extras.items()},
        ),
        legal_action_ids=policy_input.legal_action_ids[start:end],
        legal_action_features=policy_input.legal_action_features[start:end],
        legal_action_mask=policy_input.legal_action_mask[start:end],
        rule_context=policy_input.rule_context[start:end],
        raw_rule_scores=None if policy_input.raw_rule_scores is None else policy_input.raw_rule_scores[start:end],
        prior_logits=None if policy_input.prior_logits is None else policy_input.prior_logits[start:end],
        style_context=None if policy_input.style_context is None else policy_input.style_context[start:end],
        legal_actions=None,
    )


def _slice_teacher_batch(teacher_batch, start: int, end: int):
    return replace(
        teacher_batch,
        teacher_scores=teacher_batch.teacher_scores[start:end],
        prior_logits=teacher_batch.prior_logits[start:end],
        row_valid_mask=teacher_batch.row_valid_mask[start:end],
        topk_indices=None if teacher_batch.topk_indices is None else teacher_batch.topk_indices[start:end],
        support_mask=None if teacher_batch.support_mask is None else teacher_batch.support_mask[start:end],
        legal_action_mask=None if teacher_batch.legal_action_mask is None else teacher_batch.legal_action_mask[start:end],
        mapped_legal_scores=None if teacher_batch.mapped_legal_scores is None else teacher_batch.mapped_legal_scores[start:end],
    )


def _append_piece(
    args: argparse.Namespace,
    *,
    pending: dict[str, Any] | None,
    pending_rows: int,
    piece: Mapping[str, Any],
    shard_index: int,
) -> tuple[dict[str, Any] | None, int, int, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    current = pending
    current_rows = int(pending_rows)
    rest = dict(piece)
    while int(rest["row_count"]) > 0:
        if current is None:
            current = rest
            current_rows = int(rest["row_count"])
            break
        capacity = int(args.shard_size_rows) - current_rows
        if capacity <= 0:
            rows.append(_write_shard(args, current, shard_index))
            shard_index += 1
            current = None
            current_rows = 0
            continue
        take = min(capacity, int(rest["row_count"]))
        current = _concat_payloads(current, _slice_payload(rest, 0, take))
        current_rows += take
        rest = _slice_payload(rest, take, int(rest["row_count"]))
        if current_rows >= int(args.shard_size_rows):
            rows.append(_write_shard(args, current, shard_index))
            shard_index += 1
            current = None
            current_rows = 0
    return current, current_rows, shard_index, rows


def _concat_payloads(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row_count": int(left["row_count"]) + int(right["row_count"]),
        "policy_input": _concat_policy_inputs(left["policy_input"], right["policy_input"]),
        "teacher_batch": _concat_teacher_batches(left["teacher_batch"], right["teacher_batch"]),
    }


def _concat_policy_inputs(left: PolicyInput, right: PolicyInput) -> PolicyInput:
    obs = left.obs
    return replace(
        left,
        obs=replace(
            obs,
            tile_obs=torch.cat([left.obs.tile_obs, right.obs.tile_obs], dim=0),
            scalar_obs=torch.cat([left.obs.scalar_obs, right.obs.scalar_obs], dim=0),
            history_obs=None if left.obs.history_obs is None else torch.cat([left.obs.history_obs, right.obs.history_obs], dim=0),
            extras={},
        ),
        legal_action_ids=_cat_padded(left.legal_action_ids, right.legal_action_ids, pad_value=0),
        legal_action_features=_cat_padded(left.legal_action_features, right.legal_action_features, pad_value=0.0),
        legal_action_mask=_cat_padded(left.legal_action_mask, right.legal_action_mask, pad_value=False),
        rule_context=torch.cat([left.rule_context, right.rule_context], dim=0),
        raw_rule_scores=None if left.raw_rule_scores is None else _cat_padded(left.raw_rule_scores, right.raw_rule_scores, pad_value=0.0),
        prior_logits=None if left.prior_logits is None else _cat_padded(left.prior_logits, right.prior_logits, pad_value=0.0),
        style_context=None if left.style_context is None else torch.cat([left.style_context, right.style_context], dim=0),
    )


def _concat_teacher_batches(left, right):
    return replace(
        left,
        teacher_scores=_cat_padded(left.teacher_scores, right.teacher_scores, pad_value=0.0),
        prior_logits=_cat_padded(left.prior_logits, right.prior_logits, pad_value=0.0),
        row_valid_mask=torch.cat([left.row_valid_mask, right.row_valid_mask], dim=0),
        topk_indices=None if left.topk_indices is None else _cat_padded(left.topk_indices, right.topk_indices, pad_value=0),
        support_mask=None if left.support_mask is None else _cat_padded(left.support_mask, right.support_mask, pad_value=False),
        legal_action_mask=None if left.legal_action_mask is None else _cat_padded(left.legal_action_mask, right.legal_action_mask, pad_value=False),
        mapped_legal_scores=None if left.mapped_legal_scores is None else _cat_padded(left.mapped_legal_scores, right.mapped_legal_scores, pad_value=0.0),
    )


def _cat_padded(left: torch.Tensor, right: torch.Tensor, *, pad_value: float | bool | int) -> torch.Tensor:
    if left.ndim < 2 or right.ndim < 2 or int(left.shape[1]) == int(right.shape[1]):
        return torch.cat([left, right], dim=0)
    width = max(int(left.shape[1]), int(right.shape[1]))
    return torch.cat([_pad_width(left, width, pad_value=pad_value), _pad_width(right, width, pad_value=pad_value)], dim=0)


def _pad_width(tensor: torch.Tensor, width: int, *, pad_value: float | bool | int) -> torch.Tensor:
    if int(tensor.shape[1]) == int(width):
        return tensor
    shape = list(tensor.shape)
    shape[1] = int(width) - int(tensor.shape[1])
    pad = torch.full(tuple(shape), pad_value, dtype=tensor.dtype, device=tensor.device)
    return torch.cat([tensor, pad], dim=1)


def _write_shard(args: argparse.Namespace, payload: Mapping[str, Any], shard_index: int) -> dict[str, Any]:
    path = args.output_dir / f"shard_{shard_index:06d}.pt"
    torch.save(
        {
            "format": "keqingrl_mortal_imitation_shard_v1",
            "row_count": int(payload["row_count"]),
            "policy_input": payload["policy_input"],
            "teacher_batch": payload["teacher_batch"],
        },
        path,
    )
    return {"path": str(path), "sha256": _file_sha256(path), "row_count": int(payload["row_count"])}


def _accumulate_summary(total: dict[str, int], row: Mapping[str, Any]) -> None:
    for key, value in row.items():
        if key.endswith("_count") or key in {"mapping_row_count", "mapping_available_count"}:
            try:
                total[key] = int(total.get(key, 0)) + int(value)
            except (TypeError, ValueError):
                continue


if __name__ == "__main__":
    main()
