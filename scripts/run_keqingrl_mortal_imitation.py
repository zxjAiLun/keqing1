#!/usr/bin/env python3
"""Train KeqingRL by imitating Mortal action-Q over KeqingCore legal actions."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass, replace
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from keqingrl.actions import ActionSpec, ActionType
from keqingrl.contracts import PolicyInput, PolicyOutput
from keqingrl.metadata import (
    RULE_SCORE_SCALE_VERSION,
    default_checkpoint_metadata,
    validate_checkpoint_metadata,
)
from keqingrl.mortal_runtime import load_mortal_teacher_runtime
from keqingrl.mortal_observation import MortalObservationBridge
from keqingrl.mortal_teacher import (
    MORTAL_ACTION_MASK_EXTRA_KEY,
    MORTAL_ACTION_SPACE,
    MORTAL_ACTION_TEACHER_CONTRACT_VERSION,
    MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY,
    MORTAL_ENCODED_OBS_EXTRA_KEY,
    MORTAL_Q_VALUES_EXTRA_KEY,
    MortalTeacherMappingError,
    mortal_action_mapping_audit_row,
    mortal_discard_teacher_tensors_from_extras,
    mortal_scores_for_legal_actions,
)
from keqingrl.selfplay import build_episodes_ppo_batch, collect_selfplay_episodes
from scripts.probe_keqingrl_sampling_diversity import (
    TemperaturePolicy,
    _candidate_summary,
    _load_candidates as _load_probe_candidates,
    _load_policy,
    _opponent_pool,
)
from scripts.run_keqingrl_discard_research_sweep import _stable_json_hash
from scripts.run_keqingrl_fixed_online_bridge import _file_sha256
from scripts.run_keqingrl_tempered_ratio_pilot import (
    DeltaSupportProjectionPolicy,
    _action_type_tuple,
    _single_delta_support_projection_config,
)


ALLOWED_IMITATION_TEACHER_SOURCES = ("mortal-action-q",)
TEACHER_SUPPORT_MODES = ("topk", "full-legal")
_ACTION_TYPE_CHOICES = tuple(action_type.name for action_type in ActionType)


@dataclass(frozen=True)
class MortalImitationLossResult:
    loss: torch.Tensor
    teacher_ce: torch.Tensor
    teacher_kl: torch.Tensor
    teacher_entropy: torch.Tensor
    teacher_margin: torch.Tensor
    teacher_policy_agreement: torch.Tensor
    teacher_prior_agreement: torch.Tensor
    policy_top1_vs_teacher_top1_rate: torch.Tensor
    policy_top1_vs_rule_top1_rate: torch.Tensor
    teacher_row_valid_count: torch.Tensor


@dataclass(frozen=True)
class MortalMappingAuditResult:
    summary: dict[str, Any]
    audit_rows: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class MortalImitationTeacherBatch:
    teacher_support: str
    teacher_topk: int
    teacher_scores: torch.Tensor
    prior_logits: torch.Tensor
    row_valid_mask: torch.BoolTensor
    topk_indices: torch.LongTensor | None = None
    legal_action_mask: torch.BoolTensor | None = None


@dataclass(frozen=True)
class MortalImitationTeacherData:
    summary: dict[str, Any]
    audit_rows: tuple[dict[str, Any], ...]
    teacher_batch: MortalImitationTeacherBatch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KeqingRL Mortal action-Q imitation training")
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--source-config-ids", type=int, nargs="+", default=(93,))
    parser.add_argument("--rerun-config-ids", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.004)
    parser.add_argument("--update-epochs", type=int, default=3)
    parser.add_argument("--teacher-source", choices=ALLOWED_IMITATION_TEACHER_SOURCES, default="mortal-action-q")
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--teacher-support", choices=TEACHER_SUPPORT_MODES, default="topk")
    parser.add_argument("--teacher-topk", type=int, default=3)
    parser.add_argument("--mortal-teacher-checkpoint", type=Path, required=True)
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--mortal-teacher-device", default=None)
    parser.add_argument("--defer-mortal-teacher-runtime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--defer-mortal-observation-bridge", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mortal-teacher-eval-batch-size", type=int, default=256)
    parser.add_argument("--mortal-teacher-strict-extra-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--mortal-action-audit-max-examples", type=int, default=20)
    parser.add_argument("--rule-score-scale", type=float, default=0.25)
    parser.add_argument("--behavior-temperature", type=float, default=1.25)
    parser.add_argument("--support-policy-mode", choices=("support-only-topk", "unrestricted"), default="support-only-topk")
    parser.add_argument("--delta-support-mode", choices=("topk", "all"), default="topk")
    parser.add_argument("--delta-support-topk", type=int, default=3)
    parser.add_argument("--delta-support-margin-threshold", type=float, default=0.75)
    parser.add_argument("--outside-support-delta-mode", choices=("zero", "negative-clip"), default="zero")
    parser.add_argument("--save-checkpoint-every", type=int, default=8)
    parser.add_argument("--seed-base", type=int, default=202604330000)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--torch-seed-base", type=int, default=202604330000)
    parser.add_argument("--learner-seats", type=int, nargs="+", default=(0,))
    parser.add_argument("--max-kyokus", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--forced-autopilot-action-types",
        choices=_ACTION_TYPE_CHOICES,
        nargs="*",
        default=("TSUMO", "RON", "RYUKYOKU"),
    )
    parser.add_argument(
        "--self-turn-action-types",
        choices=_ACTION_TYPE_CHOICES,
        nargs="+",
        default=("DISCARD", "REACH_DISCARD", "TSUMO", "RYUKYOKU"),
    )
    parser.add_argument(
        "--response-action-types",
        choices=_ACTION_TYPE_CHOICES,
        nargs="*",
        default=("PASS", "RON", "PON", "CHI"),
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _validate_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device is not None else torch.device("cpu")
    candidates = _load_imitation_candidates(args)
    teacher_runtime = _load_teacher_runtime(args, device=device)

    summary_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []

    config_id = 0
    for candidate in candidates:
        base_policy = _load_policy(candidate, device)
        base_policy.rule_score_scale = float(args.rule_score_scale)
        source_reference = copy.deepcopy(base_policy).to(device).eval()
        policy = DeltaSupportProjectionPolicy(
            base_policy,
            support_mode=str(args.delta_support_mode),
            topk=int(args.delta_support_topk),
            margin_threshold=float(args.delta_support_margin_threshold),
            outside_support_delta_mode=str(args.outside_support_delta_mode),
            support_policy_mode=str(args.support_policy_mode),
        ).to(device)
        optimizer = torch.optim.Adam(base_policy.parameters(), lr=float(args.lr))
        opponent_pool = _opponent_pool(str(candidate["opponent_mode"]))

        final_row: dict[str, Any] | None = None
        for iteration in range(int(args.iterations)):
            iteration_start = time.perf_counter()
            rollout_seed = _iteration_seed(args, config_id, iteration)
            torch_seed = _iteration_torch_seed(args, config_id, iteration)
            _seed_torch_sampling(torch_seed)
            parent_policy = copy.deepcopy(policy).to(device).eval()
            behavior_policy = TemperaturePolicy(policy, temperature=float(args.behavior_temperature)).to(device)
            rollout_start = time.perf_counter()
            episodes = collect_selfplay_episodes(
                _rollout_env(args, teacher_runtime),
                behavior_policy,
                num_episodes=int(args.episodes),
                opponent_pool=opponent_pool,
                learner_seats=tuple(int(seat) for seat in args.learner_seats),
                seed=rollout_seed,
                seed_stride=int(args.seed_stride),
                greedy=False,
                max_steps=int(args.max_steps),
                device=device,
                include_mortal_teacher_extras=not bool(args.defer_mortal_observation_bridge),
                collect_mortal_teacher_events=bool(args.defer_mortal_observation_bridge),
            )
            rollout_sec = time.perf_counter() - rollout_start
            build_batch_start = time.perf_counter()
            _advantages, _returns, prepared_steps, batch = build_episodes_ppo_batch(
                episodes,
                gamma=float(args.gamma),
                gae_lambda=float(args.gae_lambda),
                include_rank_targets=True,
                strict_metadata=True,
            )
            build_batch_sec = time.perf_counter() - build_batch_start
            mortal_obs_bridge_sec = 0.0
            if bool(args.defer_mortal_observation_bridge):
                mortal_obs_bridge_start = time.perf_counter()
                policy_input = _ensure_mortal_encoded_observation_extras(
                    batch.policy_input,
                    prepared_steps=prepared_steps,
                    bridge=MortalObservationBridge(mortal_root=Path(args.mortal_root)),
                )
                if policy_input is not batch.policy_input:
                    batch = replace(batch, policy_input=policy_input)
                mortal_obs_bridge_sec = time.perf_counter() - mortal_obs_bridge_start
            device_transfer_start = time.perf_counter()
            batch = batch.to(device)
            device_transfer_sec = time.perf_counter() - device_transfer_start
            teacher_eval_start = time.perf_counter()
            policy_input = _ensure_mortal_teacher_q_extras(
                batch.policy_input,
                teacher_runtime=teacher_runtime,
                eval_batch_size=int(args.mortal_teacher_eval_batch_size),
            )
            if policy_input is not batch.policy_input:
                batch = replace(batch, policy_input=policy_input)
            teacher_eval_sec = time.perf_counter() - teacher_eval_start
            teacher_cache_start = time.perf_counter()
            teacher_data = prepare_mortal_imitation_teacher_data(
                batch.policy_input,
                prepared_steps=prepared_steps,
                strict_extra=bool(args.mortal_teacher_strict_extra_mask),
                teacher_support=str(args.teacher_support),
                teacher_topk=int(args.teacher_topk),
            )
            teacher_cache_sec = time.perf_counter() - teacher_cache_start
            for row in teacher_data.audit_rows:
                _append_limited_audit_row(
                    audit_rows,
                    {
                        "pilot_config_id": int(config_id),
                        "iteration": int(iteration),
                        "rollout_seed": int(rollout_seed),
                        "torch_seed": int(torch_seed),
                        **row,
                    },
                    max_examples=int(args.mortal_action_audit_max_examples),
                )
            if int(teacher_data.summary["fail_closed_count"]) > 0:
                raise MortalTeacherMappingError(
                    "Mortal imitation mapping failed closed: "
                    f"missing={teacher_data.summary['missing_legal_count']} "
                    f"extra={teacher_data.summary['extra_mortal_count']}"
                )

            update_start = time.perf_counter()
            for _epoch in range(int(args.update_epochs)):
                optimizer.zero_grad(set_to_none=True)
                output = policy(batch.policy_input)
                loss_result = mortal_imitation_loss(
                    output,
                    batch.policy_input,
                    teacher_support=str(args.teacher_support),
                    teacher_topk=int(args.teacher_topk),
                    teacher_temperature=float(args.teacher_temperature),
                    strict_extra=bool(args.mortal_teacher_strict_extra_mask),
                    teacher_batch=teacher_data.teacher_batch,
                )
                loss_result.loss.backward()
                if args.max_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(base_policy.parameters(), float(args.max_grad_norm))
                optimizer.step()
            update_sec = time.perf_counter() - update_start

            metrics_start = time.perf_counter()
            with torch.no_grad():
                output = policy(batch.policy_input)
                parent_output = parent_policy(batch.policy_input)
                source_output = source_reference(batch.policy_input)
                final_loss = mortal_imitation_loss(
                    output,
                    batch.policy_input,
                    teacher_support=str(args.teacher_support),
                    teacher_topk=int(args.teacher_topk),
                    teacher_temperature=float(args.teacher_temperature),
                    strict_extra=bool(args.mortal_teacher_strict_extra_mask),
                    teacher_batch=teacher_data.teacher_batch,
                )
                metrics, new_changed_rows = imitation_metrics(
                    output,
                    batch.policy_input,
                    parent_output=parent_output,
                    source_output=source_output,
                    prepared_steps=prepared_steps,
                    teacher_support=str(args.teacher_support),
                    teacher_topk=int(args.teacher_topk),
                    teacher_temperature=float(args.teacher_temperature),
                )
            metrics_sec = time.perf_counter() - metrics_start
            changed_rows.extend(
                {
                    "pilot_config_id": int(config_id),
                    "iteration": int(iteration),
                    **row,
                }
                for row in new_changed_rows
            )
            batch_row_count = int(batch.policy_input.legal_action_mask.shape[0])
            checkpoint_sec = 0.0
            final_row = {
                **_candidate_summary(candidate),
                "pilot_config_id": int(config_id),
                "iteration": int(iteration),
                "episode_count": int(args.episodes),
                "batch_row_count": batch_row_count,
                "rollout_seed": int(rollout_seed),
                "torch_seed": int(torch_seed),
                "teacher_source": str(args.teacher_source),
                "teacher_contract_version": MORTAL_ACTION_TEACHER_CONTRACT_VERSION,
                "teacher_temperature": float(args.teacher_temperature),
                "teacher_support": str(args.teacher_support),
                "teacher_topk": int(args.teacher_topk),
                "lr": float(args.lr),
                "update_epochs": int(args.update_epochs),
                "rule_score_scale": float(args.rule_score_scale),
                "rule_score_scale_version": RULE_SCORE_SCALE_VERSION,
                "support_policy_mode": str(args.support_policy_mode),
                "delta_support_mode": str(args.delta_support_mode),
                "delta_support_topk": int(args.delta_support_topk),
                "mortal_teacher_checkpoint": str(args.mortal_teacher_checkpoint),
                "defer_mortal_observation_bridge": bool(args.defer_mortal_observation_bridge),
                "mortal_teacher_eval_batch_size": int(args.mortal_teacher_eval_batch_size),
                "mortal_teacher_strict_extra_mask": bool(args.mortal_teacher_strict_extra_mask),
                **_action_scope_fields(args),
                **teacher_data.summary,
                **{key: _scalar(value) for key, value in final_loss.__dict__.items() if key != "loss"},
                "teacher_loss": _scalar(final_loss.loss),
                **metrics,
                "rollout_sec": rollout_sec,
                "build_batch_sec": build_batch_sec,
                "mortal_obs_bridge_sec": mortal_obs_bridge_sec,
                "device_transfer_sec": device_transfer_sec,
                "teacher_eval_sec": teacher_eval_sec,
                "teacher_cache_sec": teacher_cache_sec,
                "update_sec": update_sec,
                "metrics_sec": metrics_sec,
                "checkpoint_sec": checkpoint_sec,
            }

            if _should_save_checkpoint(args, iteration):
                checkpoint_start = time.perf_counter()
                checkpoint_rows.append(
                    _save_imitation_checkpoint(
                        args,
                        candidate,
                        base_policy,
                        optimizer,
                        config_id=config_id,
                        iteration=iteration,
                        summary_row=final_row,
                    )
                )
                checkpoint_sec = time.perf_counter() - checkpoint_start
                final_row["checkpoint_sec"] = checkpoint_sec
            iteration_sec = time.perf_counter() - iteration_start
            final_row["iteration_sec"] = iteration_sec
            final_row["rows_per_sec"] = batch_row_count / max(iteration_sec, 1e-9)
            iteration_rows.append(final_row)
            print(
                "mortal-imitation "
                f"cfg={config_id} iter={iteration + 1}/{int(args.iterations)} "
                f"rows={final_row['batch_row_count']} "
                f"mapping={final_row['mapping_available_count']}/{final_row['mapping_row_count']} "
                f"fail_closed={final_row['fail_closed_count']} "
                f"teacher_ce={final_row['teacher_ce']:.6g} "
                f"teacher_kl={final_row['teacher_kl']:.6g} "
                f"teacher_agree={final_row['teacher_policy_agreement']:.6g} "
                f"top1_parent={final_row['top1_changed_vs_parent_rate']:.6g} "
                f"rank_ge5={final_row['rank_ge5_rate']:.6g} "
                f"sec={iteration_sec:.1f} "
                f"rollout={rollout_sec:.1f} "
                f"mortal_obs={mortal_obs_bridge_sec:.1f} "
                f"to_device={device_transfer_sec:.1f} "
                f"teacher_eval={teacher_eval_sec:.1f} "
                f"teacher_cache={teacher_cache_sec:.1f} "
                f"update={update_sec:.1f} "
                f"metrics={metrics_sec:.1f}",
                flush=True,
            )
            _write_incremental_outputs(args, iteration_rows, audit_rows, changed_rows, checkpoint_rows)

        if final_row is not None:
            final_checkpoint = checkpoint_rows[-1] if checkpoint_rows else {}
            summary_rows.append(
                {
                    **final_row,
                    "checkpoint_path": final_checkpoint.get("checkpoint_path", ""),
                    "checkpoint_sha256": final_checkpoint.get("checkpoint_sha256", ""),
                    "config_path": final_checkpoint.get("config_path", ""),
                }
            )
        config_id += 1

    _write_outputs(args, summary_rows, iteration_rows, audit_rows, changed_rows, checkpoint_rows)


def _validate_args(args: argparse.Namespace) -> None:
    if str(args.teacher_source) not in ALLOWED_IMITATION_TEACHER_SOURCES:
        raise ValueError(f"unsupported imitation teacher source: {args.teacher_source}")
    if args.mortal_teacher_checkpoint is None:
        raise ValueError("--mortal-teacher-checkpoint is required for mortal-action-q imitation")
    if float(args.teacher_temperature) <= 0.0:
        raise ValueError("--teacher-temperature must be positive")
    if int(args.teacher_topk) <= 0:
        raise ValueError("--teacher-topk must be positive")
    if int(args.episodes) <= 0 or int(args.iterations) <= 0:
        raise ValueError("--episodes and --iterations must be positive")
    if int(args.update_epochs) <= 0:
        raise ValueError("--update-epochs must be positive")


def _load_teacher_runtime(args: argparse.Namespace, *, device: torch.device):
    teacher_device = args.mortal_teacher_device if args.mortal_teacher_device is not None else str(device)
    return load_mortal_teacher_runtime(
        Path(args.mortal_teacher_checkpoint),
        mortal_root=Path(args.mortal_root),
        device=teacher_device,
    )


def _load_imitation_candidates(args: argparse.Namespace) -> list[dict[str, Any]]:
    candidates = _load_probe_candidates(args)
    latest_by_identity: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    order: list[tuple[int, int, str, str]] = []
    for candidate in candidates:
        key = (
            int(candidate["source_config_id"]),
            int(candidate["rerun_config_id"]),
            str(candidate["checkpoint_path"]),
            str(candidate["config_path"]),
        )
        if key not in latest_by_identity:
            order.append(key)
        latest_by_identity[key] = candidate
    deduped = [latest_by_identity[key] for key in order]
    dropped = len(candidates) - len(deduped)
    if dropped:
        print(f"mortal-imitation deduped {dropped} stale duplicate candidate row(s)", flush=True)
    return deduped


def _rollout_env(args: argparse.Namespace, teacher_runtime):
    from keqingrl import DiscardOnlyMahjongEnv

    defer_runtime = bool(args.defer_mortal_teacher_runtime)
    bridge = None if bool(args.defer_mortal_observation_bridge) else MortalObservationBridge(mortal_root=Path(args.mortal_root))
    return DiscardOnlyMahjongEnv(
        max_kyokus=int(args.max_kyokus),
        self_turn_action_types=_action_type_tuple(args.self_turn_action_types),
        response_action_types=_action_type_tuple(args.response_action_types),
        forced_autopilot_action_types=_action_type_tuple(args.forced_autopilot_action_types),
        mortal_teacher_runtime=None if defer_runtime else teacher_runtime,
        mortal_observation_bridge=bridge,
        mortal_teacher_strict_extra_mask=bool(args.mortal_teacher_strict_extra_mask),
        mortal_teacher_defer_runtime=defer_runtime,
    )


def _ensure_mortal_encoded_observation_extras(
    policy_input: PolicyInput,
    *,
    prepared_steps: Sequence[Any],
    bridge: MortalObservationBridge,
) -> PolicyInput:
    extras = policy_input.obs.extras
    if MORTAL_Q_VALUES_EXTRA_KEY in extras and MORTAL_ACTION_MASK_EXTRA_KEY in extras:
        return policy_input
    if MORTAL_ENCODED_OBS_EXTRA_KEY in extras and MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY in extras:
        return policy_input
    row_count = int(policy_input.legal_action_mask.shape[0])
    if len(prepared_steps) != row_count:
        raise MortalTeacherMappingError(
            "Mortal observation bridge step count mismatch: "
            f"prepared_steps={len(prepared_steps)}, policy_rows={row_count}"
        )
    encoded_obs_rows: list[torch.Tensor] = []
    encoded_mask_rows: list[torch.Tensor] = []
    previous_episode_id: object = object()
    for row_idx, step in enumerate(prepared_steps):
        events = tuple(getattr(step, "mortal_teacher_events", ()))
        if not events:
            raise MortalTeacherMappingError(
                f"Mortal observation events missing for deferred bridge row {row_idx}"
            )
        episode_id = getattr(step, "episode_id", None)
        if episode_id != previous_episode_id:
            bridge.reset_cache()
            previous_episode_id = episode_id
        encoded = bridge.encode_from_events(events, int(step.actor))
        encoded_obs_rows.append(encoded.obs)
        encoded_mask_rows.append(encoded.action_mask)
    new_extras = dict(extras)
    new_extras[MORTAL_ENCODED_OBS_EXTRA_KEY] = torch.stack(encoded_obs_rows, dim=0)
    new_extras[MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY] = torch.stack(encoded_mask_rows, dim=0)
    return replace(policy_input, obs=replace(policy_input.obs, extras=new_extras))


def _ensure_mortal_teacher_q_extras(
    policy_input: PolicyInput,
    *,
    teacher_runtime,
    eval_batch_size: int | None = 256,
) -> PolicyInput:
    extras = policy_input.obs.extras
    has_q_values = MORTAL_Q_VALUES_EXTRA_KEY in extras
    has_action_mask = MORTAL_ACTION_MASK_EXTRA_KEY in extras
    if has_q_values and has_action_mask:
        return policy_input
    if has_q_values or has_action_mask:
        present = [key for key in (MORTAL_Q_VALUES_EXTRA_KEY, MORTAL_ACTION_MASK_EXTRA_KEY) if key in extras]
        raise MortalTeacherMappingError(f"Mortal teacher q/mask extras are incomplete: present={present}")
    missing_encoded = [
        key
        for key in (MORTAL_ENCODED_OBS_EXTRA_KEY, MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY)
        if key not in extras
    ]
    if missing_encoded:
        raise MortalTeacherMappingError(f"Mortal encoded teacher extras missing required keys: {missing_encoded}")
    if teacher_runtime is None:
        raise MortalTeacherMappingError("Mortal teacher runtime is required to materialize deferred q/mask extras")

    encoded_obs = extras[MORTAL_ENCODED_OBS_EXTRA_KEY]
    encoded_action_mask = extras[MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY].bool()
    if encoded_obs.ndim < 3:
        raise MortalTeacherMappingError(
            "Deferred Mortal encoded obs must be batched, "
            f"got shape={tuple(encoded_obs.shape)}"
        )
    if encoded_action_mask.ndim != 2 or int(encoded_action_mask.shape[-1]) != MORTAL_ACTION_SPACE:
        raise MortalTeacherMappingError(
            f"Deferred Mortal encoded action mask must have shape [B, {MORTAL_ACTION_SPACE}], "
            f"got shape={tuple(encoded_action_mask.shape)}"
        )
    batch_size = int(policy_input.legal_action_mask.shape[0])
    if int(encoded_obs.shape[0]) != batch_size or int(encoded_action_mask.shape[0]) != batch_size:
        raise MortalTeacherMappingError(
            "Deferred Mortal encoded extras batch size mismatch: "
            f"policy={batch_size}, obs={tuple(encoded_obs.shape)}, mask={tuple(encoded_action_mask.shape)}"
        )

    output_q_values: list[torch.Tensor] = []
    output_action_masks: list[torch.Tensor] = []
    chunk_size = int(eval_batch_size or 0)
    if chunk_size <= 0:
        chunk_size = batch_size
    for start in range(0, batch_size, chunk_size):
        end = min(batch_size, start + chunk_size)
        output = teacher_runtime.evaluate(encoded_obs[start:end], encoded_action_mask[start:end])
        output_q_values.append(output.q_values)
        output_action_masks.append(output.action_mask)
    target_device = policy_input.legal_action_mask.device
    new_extras = {
        key: value
        for key, value in extras.items()
        if key not in {MORTAL_ENCODED_OBS_EXTRA_KEY, MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY}
    }
    new_extras[MORTAL_Q_VALUES_EXTRA_KEY] = torch.cat(output_q_values, dim=0).to(device=target_device)
    new_extras[MORTAL_ACTION_MASK_EXTRA_KEY] = torch.cat(output_action_masks, dim=0).to(device=target_device)
    return replace(policy_input, obs=replace(policy_input.obs, extras=new_extras))


def audit_mortal_action_mapping(
    policy_input: PolicyInput,
    *,
    prepared_steps: Sequence[Any] | None = None,
    strict_extra: bool,
) -> MortalMappingAuditResult:
    if policy_input.legal_actions is None:
        raise MortalTeacherMappingError("Mortal imitation requires policy_input.legal_actions")
    q_values, mortal_masks = mortal_discard_teacher_tensors_from_extras(policy_input.obs.extras)
    mapping_row_count = int(q_values.shape[0])
    available_count = 0
    missing_count = 0
    extra_count = 0
    fail_closed_count = 0
    audit_rows: list[dict[str, Any]] = []
    for row_idx, legal_actions in enumerate(policy_input.legal_actions):
        try:
            mapped = mortal_scores_for_legal_actions(
                q_values[row_idx],
                mortal_masks[row_idx],
                legal_actions,
                strict_mask=False,
            )
            row_missing = len(mapped.missing_legal_keys)
            row_extra = len(mapped.extra_mortal_action_ids)
            fatal_extra = bool(strict_extra and row_extra)
            if row_missing or fatal_extra:
                fail_closed_count += 1
            else:
                available_count += 1
            missing_count += row_missing
            extra_count += row_extra
            row = mortal_action_mapping_audit_row(
                q_values[row_idx],
                mortal_masks[row_idx],
                legal_actions,
                context=_mapping_context(prepared_steps, row_idx),
            )
            if row is not None:
                audit_rows.append(row)
        except MortalTeacherMappingError as exc:
            fail_closed_count += 1
            missing_count += len(exc.missing_legal_keys)
            extra_count += len(exc.extra_mortal_action_ids)
            audit_rows.append(
                {
                    **_mapping_context(prepared_steps, row_idx),
                    "mismatch_kind": exc.mismatch_kind or "unsupported_action",
                    "missing_legal_keys_json": _json_dumps(list(exc.missing_legal_keys)),
                    "missing_legal_types_json": _json_dumps(()),
                    "extra_mortal_action_ids_json": _json_dumps(list(exc.extra_mortal_action_ids)),
                    "error": str(exc),
                }
            )
    return MortalMappingAuditResult(
        summary={
            "mapping_row_count": mapping_row_count,
            "mapping_available_count": available_count,
            "mapping_available_rate": available_count / max(1, mapping_row_count),
            "missing_legal_count": missing_count,
            "extra_mortal_count": extra_count,
            "fail_closed_count": fail_closed_count,
        },
        audit_rows=tuple(audit_rows),
    )


def prepare_mortal_imitation_teacher_data(
    policy_input: PolicyInput,
    *,
    prepared_steps: Sequence[Any] | None = None,
    strict_extra: bool,
    teacher_support: str,
    teacher_topk: int,
) -> MortalImitationTeacherData:
    if policy_input.legal_actions is None:
        raise MortalTeacherMappingError("Mortal imitation requires policy_input.legal_actions")
    if policy_input.prior_logits is None:
        raise MortalTeacherMappingError("Mortal imitation requires policy_input.prior_logits")
    q_values, mortal_masks = mortal_discard_teacher_tensors_from_extras(policy_input.obs.extras)
    mapping_row_count = int(q_values.shape[0])
    if str(teacher_support) == "topk":
        teacher_batch, summary, audit_rows = _prepare_mortal_topk_teacher_batch(
            policy_input,
            q_values=q_values,
            mortal_masks=mortal_masks,
            prepared_steps=prepared_steps,
            strict_extra=bool(strict_extra),
            teacher_topk=int(teacher_topk),
        )
    elif str(teacher_support) == "full-legal":
        teacher_batch, summary, audit_rows = _prepare_mortal_full_legal_teacher_batch(
            policy_input,
            q_values=q_values,
            mortal_masks=mortal_masks,
            prepared_steps=prepared_steps,
            strict_extra=bool(strict_extra),
        )
    else:
        raise ValueError(f"unsupported teacher support: {teacher_support}")
    if int(summary["mapping_row_count"]) != mapping_row_count:
        raise MortalTeacherMappingError("Mortal teacher cache row count changed while preparing batch")
    return MortalImitationTeacherData(
        summary=summary,
        audit_rows=audit_rows,
        teacher_batch=teacher_batch,
    )


def _prepare_mortal_topk_teacher_batch(
    policy_input: PolicyInput,
    *,
    q_values: torch.Tensor,
    mortal_masks: torch.Tensor,
    prepared_steps: Sequence[Any] | None,
    strict_extra: bool,
    teacher_topk: int,
) -> tuple[MortalImitationTeacherBatch, dict[str, Any], tuple[dict[str, Any], ...]]:
    assert policy_input.legal_actions is not None
    assert policy_input.prior_logits is not None
    prior_logits = policy_input.prior_logits.float()
    legal_mask = policy_input.legal_action_mask.bool()
    topk_indices, valid_rows = _unique_source_topk_indices(
        prior_logits,
        legal_mask,
        policy_input.legal_actions,
        k=int(teacher_topk),
    )
    masked_prior = prior_logits.masked_fill(~legal_mask, torch.finfo(torch.float32).min)
    topk_prior = masked_prior.gather(1, topk_indices.to(device=masked_prior.device))

    teacher_scores_rows: list[torch.Tensor] = []
    summary, audit_rows = _collect_mortal_mapping_summary_and_audit(
        q_values=q_values,
        mortal_masks=mortal_masks,
        legal_actions=policy_input.legal_actions,
        prepared_steps=prepared_steps,
        strict_extra=bool(strict_extra),
        mapped_scores_callback=lambda row_idx, mapped: teacher_scores_rows.append(
            mapped.scores.gather(0, topk_indices[row_idx].to(mapped.scores.device)).to(
                device=prior_logits.device,
                dtype=prior_logits.dtype,
            )
        ),
    )
    if len(teacher_scores_rows) != int(q_values.shape[0]):
        raise MortalTeacherMappingError("Mortal topK teacher cache row count mismatch")
    teacher_scores = torch.stack(teacher_scores_rows, dim=0)
    if not torch.isfinite(teacher_scores[valid_rows.to(device=teacher_scores.device)]).all():
        raise MortalTeacherMappingError("Mortal imitation topK scores contain non-finite values")
    return (
        MortalImitationTeacherBatch(
            teacher_support="topk",
            teacher_topk=int(teacher_topk),
            teacher_scores=teacher_scores,
            prior_logits=topk_prior.to(device=prior_logits.device),
            row_valid_mask=valid_rows.to(device=prior_logits.device),
            topk_indices=topk_indices.to(device=prior_logits.device),
        ),
        summary,
        audit_rows,
    )


def _prepare_mortal_full_legal_teacher_batch(
    policy_input: PolicyInput,
    *,
    q_values: torch.Tensor,
    mortal_masks: torch.Tensor,
    prepared_steps: Sequence[Any] | None,
    strict_extra: bool,
) -> tuple[MortalImitationTeacherBatch, dict[str, Any], tuple[dict[str, Any], ...]]:
    assert policy_input.legal_actions is not None
    assert policy_input.prior_logits is not None
    mask = policy_input.legal_action_mask.bool()
    teacher_scores = torch.full_like(policy_input.prior_logits.float(), float("-inf"))

    def fill_row(row_idx: int, mapped: Any) -> None:
        width = int(mapped.scores.shape[0])
        teacher_scores[row_idx, :width] = mapped.scores.to(device=teacher_scores.device)

    summary, audit_rows = _collect_mortal_mapping_summary_and_audit(
        q_values=q_values,
        mortal_masks=mortal_masks,
        legal_actions=policy_input.legal_actions,
        prepared_steps=prepared_steps,
        strict_extra=bool(strict_extra),
        mapped_scores_callback=fill_row,
    )
    teacher_scores = teacher_scores.masked_fill(~mask, torch.finfo(torch.float32).min)
    prior_logits = policy_input.prior_logits.float().masked_fill(~mask, torch.finfo(torch.float32).min)
    return (
        MortalImitationTeacherBatch(
            teacher_support="full-legal",
            teacher_topk=0,
            teacher_scores=teacher_scores,
            prior_logits=prior_logits,
            row_valid_mask=mask.any(dim=-1),
            topk_indices=None,
            legal_action_mask=mask,
        ),
        summary,
        audit_rows,
    )


def _collect_mortal_mapping_summary_and_audit(
    *,
    q_values: torch.Tensor,
    mortal_masks: torch.Tensor,
    legal_actions: Sequence[Sequence[ActionSpec]],
    prepared_steps: Sequence[Any] | None,
    strict_extra: bool,
    mapped_scores_callback,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    mapping_row_count = int(q_values.shape[0])
    available_count = 0
    missing_count = 0
    extra_count = 0
    fail_closed_count = 0
    audit_rows: list[dict[str, Any]] = []
    for row_idx, row_legal_actions in enumerate(legal_actions):
        try:
            mapped = mortal_scores_for_legal_actions(
                q_values[row_idx],
                mortal_masks[row_idx],
                row_legal_actions,
                strict_mask=False,
            )
            row_missing = len(mapped.missing_legal_keys)
            row_extra = len(mapped.extra_mortal_action_ids)
            fatal_extra = bool(strict_extra and row_extra)
            if row_missing or fatal_extra:
                fail_closed_count += 1
            else:
                available_count += 1
            missing_count += row_missing
            extra_count += row_extra
            mapped_scores_callback(row_idx, mapped)
            row = mortal_action_mapping_audit_row(
                q_values[row_idx],
                mortal_masks[row_idx],
                row_legal_actions,
                context=_mapping_context(prepared_steps, row_idx),
            )
            if row is not None:
                audit_rows.append(row)
        except MortalTeacherMappingError as exc:
            fail_closed_count += 1
            missing_count += len(exc.missing_legal_keys)
            extra_count += len(exc.extra_mortal_action_ids)
            mapped_scores_callback(
                row_idx,
                type(
                    "MissingMappedScores",
                    (),
                    {
                        "scores": torch.zeros(
                            (len(row_legal_actions),),
                            dtype=q_values.dtype,
                            device=q_values.device,
                        )
                    },
                )(),
            )
            audit_rows.append(
                {
                    **_mapping_context(prepared_steps, row_idx),
                    "mismatch_kind": exc.mismatch_kind or "unsupported_action",
                    "missing_legal_keys_json": _json_dumps(list(exc.missing_legal_keys)),
                    "missing_legal_types_json": _json_dumps(()),
                    "extra_mortal_action_ids_json": _json_dumps(list(exc.extra_mortal_action_ids)),
                    "error": str(exc),
                }
            )
    return (
        {
            "mapping_row_count": mapping_row_count,
            "mapping_available_count": available_count,
            "mapping_available_rate": available_count / max(1, mapping_row_count),
            "missing_legal_count": missing_count,
            "extra_mortal_count": extra_count,
            "fail_closed_count": fail_closed_count,
        },
        tuple(audit_rows),
    )


def mortal_imitation_loss(
    output: PolicyOutput,
    policy_input: PolicyInput,
    *,
    teacher_support: str,
    teacher_topk: int,
    teacher_temperature: float,
    strict_extra: bool,
    teacher_batch: MortalImitationTeacherBatch | None = None,
) -> MortalImitationLossResult:
    _ = strict_extra  # Full-row strictness is enforced before loss computation.
    if teacher_batch is not None:
        return _mortal_cached_imitation_loss(
            output,
            teacher_batch,
            teacher_temperature=float(teacher_temperature),
        )
    if policy_input.legal_actions is None:
        raise MortalTeacherMappingError("Mortal imitation requires policy_input.legal_actions")
    if policy_input.prior_logits is None:
        raise MortalTeacherMappingError("Mortal imitation requires policy_input.prior_logits")
    if str(teacher_support) == "topk":
        return _mortal_topk_imitation_loss(
            output,
            policy_input,
            teacher_topk=int(teacher_topk),
            teacher_temperature=float(teacher_temperature),
        )
    if str(teacher_support) == "full-legal":
        return _mortal_full_legal_imitation_loss(
            output,
            policy_input,
            teacher_temperature=float(teacher_temperature),
        )
    raise ValueError(f"unsupported teacher support: {teacher_support}")


def _mortal_cached_imitation_loss(
    output: PolicyOutput,
    teacher_batch: MortalImitationTeacherBatch,
    *,
    teacher_temperature: float,
) -> MortalImitationLossResult:
    if teacher_batch.teacher_support == "topk":
        if teacher_batch.topk_indices is None:
            raise MortalTeacherMappingError("cached topK teacher batch is missing topk_indices")
        policy_logits = output.action_logits.gather(
            1,
            teacher_batch.topk_indices.to(device=output.action_logits.device),
        )
        return _imitation_loss_from_scores(
            policy_logits=policy_logits,
            prior_logits=teacher_batch.prior_logits.to(device=policy_logits.device),
            teacher_scores=teacher_batch.teacher_scores.to(device=policy_logits.device),
            row_valid_mask=teacher_batch.row_valid_mask.to(device=policy_logits.device),
            teacher_temperature=float(teacher_temperature),
        )
    if teacher_batch.teacher_support == "full-legal":
        if teacher_batch.legal_action_mask is None:
            raise MortalTeacherMappingError("cached full-legal teacher batch is missing legal_action_mask")
        mask = teacher_batch.legal_action_mask.to(device=output.action_logits.device)
        policy_logits = output.action_logits.float().masked_fill(~mask, torch.finfo(torch.float32).min)
        return _imitation_loss_from_scores(
            policy_logits=policy_logits,
            prior_logits=teacher_batch.prior_logits.to(device=output.action_logits.device),
            teacher_scores=teacher_batch.teacher_scores.to(device=output.action_logits.device),
            row_valid_mask=teacher_batch.row_valid_mask.to(device=output.action_logits.device),
            teacher_temperature=float(teacher_temperature),
        )
    raise ValueError(f"unsupported cached teacher support: {teacher_batch.teacher_support}")


def _mortal_topk_imitation_loss(
    output: PolicyOutput,
    policy_input: PolicyInput,
    *,
    teacher_topk: int,
    teacher_temperature: float,
) -> MortalImitationLossResult:
    q_values, mortal_masks = mortal_discard_teacher_tensors_from_extras(policy_input.obs.extras)
    prior_logits = policy_input.prior_logits.float()
    assert policy_input.legal_actions is not None
    topk_indices, valid_rows = _unique_source_topk_indices(
        prior_logits,
        policy_input.legal_action_mask.bool(),
        policy_input.legal_actions,
        k=int(teacher_topk),
    )
    teacher_scores_rows: list[torch.Tensor] = []
    for row_idx, legal_actions in enumerate(policy_input.legal_actions):
        mapped = mortal_scores_for_legal_actions(
            q_values[row_idx],
            mortal_masks[row_idx],
            legal_actions,
            strict_mask=False,
        )
        if mapped.missing_legal_keys:
            raise MortalTeacherMappingError(
                "Mortal imitation topK row has missing legal actions",
                missing_legal_keys=mapped.missing_legal_keys,
                mismatch_kind="missing_legal",
            )
        row_scores = mapped.scores.gather(0, topk_indices[row_idx].to(mapped.scores.device))
        if not torch.isfinite(row_scores).all():
            raise MortalTeacherMappingError("Mortal imitation topK scores contain non-finite values")
        teacher_scores_rows.append(row_scores.to(device=output.action_logits.device, dtype=output.action_logits.dtype))
    teacher_scores = torch.stack(teacher_scores_rows, dim=0)
    policy_logits = output.action_logits.gather(1, topk_indices.to(device=output.action_logits.device))
    prior_topk = prior_logits.gather(1, topk_indices.to(device=prior_logits.device))
    return _imitation_loss_from_scores(
        policy_logits=policy_logits,
        prior_logits=prior_topk.to(device=policy_logits.device),
        teacher_scores=teacher_scores,
        row_valid_mask=valid_rows.to(device=policy_logits.device),
        teacher_temperature=float(teacher_temperature),
    )


def _mortal_full_legal_imitation_loss(
    output: PolicyOutput,
    policy_input: PolicyInput,
    *,
    teacher_temperature: float,
) -> MortalImitationLossResult:
    q_values, mortal_masks = mortal_discard_teacher_tensors_from_extras(policy_input.obs.extras)
    assert policy_input.legal_actions is not None
    teacher_scores = torch.full_like(output.action_logits.float(), float("-inf"))
    for row_idx, legal_actions in enumerate(policy_input.legal_actions):
        mapped = mortal_scores_for_legal_actions(
            q_values[row_idx],
            mortal_masks[row_idx],
            legal_actions,
            strict_mask=False,
        )
        if mapped.missing_legal_keys:
            raise MortalTeacherMappingError(
                "Mortal imitation full-legal row has missing legal actions",
                missing_legal_keys=mapped.missing_legal_keys,
                mismatch_kind="missing_legal",
            )
        width = int(mapped.scores.shape[0])
        teacher_scores[row_idx, :width] = mapped.scores.to(device=teacher_scores.device)
    mask = policy_input.legal_action_mask.bool().to(device=output.action_logits.device)
    teacher_scores = teacher_scores.masked_fill(~mask, torch.finfo(teacher_scores.dtype).min)
    policy_logits = output.action_logits.float().masked_fill(~mask, torch.finfo(output.action_logits.dtype).min)
    prior_logits = policy_input.prior_logits.float().to(device=policy_logits.device).masked_fill(
        ~mask,
        torch.finfo(policy_logits.dtype).min,
    )
    return _imitation_loss_from_scores(
        policy_logits=policy_logits,
        prior_logits=prior_logits,
        teacher_scores=teacher_scores,
        row_valid_mask=mask.any(dim=-1),
        teacher_temperature=float(teacher_temperature),
    )


def _imitation_loss_from_scores(
    *,
    policy_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    teacher_scores: torch.Tensor,
    row_valid_mask: torch.Tensor,
    teacher_temperature: float,
) -> MortalImitationLossResult:
    teacher_probs = torch.softmax(teacher_scores / float(teacher_temperature), dim=-1)
    teacher_log_probs = torch.log(teacher_probs.clamp_min(1e-12))
    policy_log_probs = torch.log_softmax(policy_logits.float(), dim=-1)
    per_row_ce = -(teacher_probs * policy_log_probs).sum(dim=-1)
    per_row_kl = (teacher_probs * (teacher_log_probs - policy_log_probs)).sum(dim=-1)
    valid = row_valid_mask.bool()
    weights = valid.to(dtype=policy_logits.dtype)
    teacher_argmax = teacher_scores.argmax(dim=-1)
    policy_argmax = policy_logits.argmax(dim=-1)
    prior_argmax = prior_logits.argmax(dim=-1)
    if teacher_scores.shape[-1] > 1:
        teacher_top2 = torch.topk(teacher_scores, k=2, dim=-1).values
        teacher_margin = teacher_top2[:, 0] - teacher_top2[:, 1]
    else:
        teacher_margin = torch.zeros_like(per_row_ce)
    entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
    return MortalImitationLossResult(
        loss=_weighted_mean(per_row_ce, weights),
        teacher_ce=_weighted_mean(per_row_ce, weights),
        teacher_kl=_weighted_mean(per_row_kl, weights),
        teacher_entropy=_weighted_mean(entropy, weights),
        teacher_margin=_weighted_mean(teacher_margin, weights),
        teacher_policy_agreement=_weighted_mean((policy_argmax == teacher_argmax).float(), weights),
        teacher_prior_agreement=_weighted_mean((prior_argmax == teacher_argmax).float(), weights),
        policy_top1_vs_teacher_top1_rate=_weighted_mean((policy_argmax == teacher_argmax).float(), weights),
        policy_top1_vs_rule_top1_rate=_weighted_mean((policy_argmax == prior_argmax).float(), weights),
        teacher_row_valid_count=weights.sum(),
    )


def imitation_metrics(
    output: PolicyOutput,
    policy_input: PolicyInput,
    *,
    parent_output: PolicyOutput,
    source_output: PolicyOutput,
    prepared_steps: Sequence[Any],
    teacher_support: str,
    teacher_topk: int,
    teacher_temperature: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    del teacher_support, teacher_topk, teacher_temperature
    mask = policy_input.legal_action_mask.bool().to(device=output.action_logits.device)
    final_logits = output.action_logits.float().masked_fill(~mask, torch.finfo(torch.float32).min)
    parent_logits = parent_output.action_logits.float().masked_fill(~mask, torch.finfo(torch.float32).min)
    source_logits = source_output.action_logits.float().masked_fill(~mask, torch.finfo(torch.float32).min)
    prior_logits = policy_input.prior_logits.float().to(device=final_logits.device).masked_fill(
        ~mask,
        torch.finfo(torch.float32).min,
    )
    final_top1 = final_logits.argmax(dim=-1)
    parent_top1 = parent_logits.argmax(dim=-1)
    source_top1 = source_logits.argmax(dim=-1)
    prior_top1 = prior_logits.argmax(dim=-1)
    changed_vs_parent = final_top1 != parent_top1
    changed_vs_source = final_top1 != source_top1
    changed_vs_prior = final_top1 != prior_top1
    ranks: list[float] = []
    rank_ge5 = 0
    changed_rows: list[dict[str, Any]] = []
    for row_idx in range(int(final_logits.shape[0])):
        if not bool(changed_vs_prior[row_idx].item()):
            continue
        row_mask = mask[row_idx]
        chosen = int(final_top1[row_idx].item())
        row_prior = prior_logits[row_idx]
        legal_prior = row_prior[row_mask]
        rank = 1 + int((legal_prior > row_prior[chosen]).sum().item())
        ranks.append(float(rank))
        rank_ge5 += int(rank >= 5)
        changed_rows.append(_changed_decision_row(policy_input, prepared_steps, row_idx, final_top1, prior_top1, rank))
    legal_delta = (final_logits - parent_logits).abs().masked_fill(~mask, 0.0)
    return (
        {
            "top1_changed_vs_parent_rate": _mean_bool(changed_vs_parent),
            "top1_changed_vs_source_rate": _mean_bool(changed_vs_source),
            "policy_top1_vs_rule_top1_rate": 1.0 - _mean_bool(changed_vs_prior),
            "changed_rank_mean": _mean_float(ranks),
            "rank_ge5_rate": rank_ge5 / max(1, len(ranks)),
            "clip_like_logit_delta_max": float(legal_delta.max().detach().cpu().item()),
        },
        changed_rows,
    )


def _changed_decision_row(
    policy_input: PolicyInput,
    prepared_steps: Sequence[Any],
    row_idx: int,
    final_top1: torch.Tensor,
    prior_top1: torch.Tensor,
    rank: int,
) -> dict[str, Any]:
    step = prepared_steps[row_idx] if row_idx < len(prepared_steps) else None
    legal_actions = () if policy_input.legal_actions is None else policy_input.legal_actions[row_idx]
    student_idx = int(final_top1[row_idx].item())
    prior_idx = int(prior_top1[row_idx].item())
    return {
        "episode_id": "" if step is None else step.episode_id,
        "step_id": "" if step is None else int(step.step_id),
        "actor": "" if step is None else int(step.actor),
        "student_top1": _action_label(legal_actions, student_idx),
        "rule_prior_top1": _action_label(legal_actions, prior_idx),
        "changed_action_prior_rank": int(rank),
        "legal_actions": " | ".join(action.canonical_key for action in legal_actions),
        "hand": "unavailable_in_rollout_step",
        "last_discard": "unavailable_in_rollout_step",
        "events_tail": "unavailable_in_rollout_step",
    }


def _unique_source_topk_indices(
    prior_logits: torch.Tensor,
    legal_action_mask: torch.Tensor,
    legal_actions: Sequence[Sequence[ActionSpec]],
    *,
    k: int,
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    from keqingrl.mortal_teacher import mortal_action_ids_for_action_spec

    rows: list[torch.Tensor] = []
    valid_rows: list[bool] = []
    masked_prior = prior_logits.float().masked_fill(~legal_action_mask.bool(), torch.finfo(torch.float32).min)
    for row_idx, row_actions in enumerate(legal_actions):
        legal_indices = [
            index
            for index in range(len(row_actions))
            if index < int(legal_action_mask.shape[1]) and bool(legal_action_mask[row_idx, index].item())
        ]
        sorted_indices = sorted(
            legal_indices,
            key=lambda index: float(masked_prior[row_idx, index].detach().cpu()),
            reverse=True,
        )
        selected: list[int] = []
        seen: set[tuple[int, ...]] = set()
        for index in sorted_indices:
            source = mortal_action_ids_for_action_spec(row_actions[index])
            if source in seen:
                continue
            selected.append(index)
            seen.add(source)
            if len(selected) == int(k):
                break
        valid = len(selected) == int(k)
        if not selected:
            selected = [0]
        while len(selected) < int(k):
            selected.append(selected[-1])
        rows.append(torch.tensor(selected[: int(k)], dtype=torch.long, device=prior_logits.device))
        valid_rows.append(valid)
    return torch.stack(rows, dim=0), torch.tensor(valid_rows, dtype=torch.bool, device=prior_logits.device)


def _save_imitation_checkpoint(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    policy,
    optimizer,
    *,
    config_id: int,
    iteration: int,
    summary_row: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint_dir = args.output_dir / f"checkpoint_config_{config_id:03d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source_config = json.loads(Path(candidate["config_path"]).read_text(encoding="utf-8"))
    config_hash = _stable_json_hash(
        {
            "mode": "mortal_action_q_imitation",
            "source_config_id": int(candidate["source_config_id"]),
            "rerun_config_id": int(candidate["rerun_config_id"]),
            "iteration": int(iteration),
            "episodes": int(args.episodes),
            "lr": float(args.lr),
            "update_epochs": int(args.update_epochs),
            "teacher_source": str(args.teacher_source),
            "teacher_support": str(args.teacher_support),
            "teacher_topk": int(args.teacher_topk),
            "teacher_temperature": float(args.teacher_temperature),
            "rule_score_scale": float(args.rule_score_scale),
        }
    )
    contract_metadata = default_checkpoint_metadata(
        controlled_action_types=tuple(args.self_turn_action_types) + tuple(args.response_action_types),
        rule_score_scale=float(args.rule_score_scale),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        ppo_config_hash=config_hash,
    )
    contract_metadata.update(
        {
            "teacher_source": str(args.teacher_source),
            "teacher_version": MORTAL_ACTION_TEACHER_CONTRACT_VERSION,
            "teacher_contract_version": "keqingrl_mortal_imitation_v1",
            "teacher_target_type": "topk_distribution" if args.teacher_support == "topk" else "full_legal_distribution",
            "teacher_topk": int(args.teacher_topk),
            "teacher_temperature": float(args.teacher_temperature),
            "support_policy_mode": str(args.support_policy_mode),
            "support_topk": int(args.delta_support_topk),
            "delta_support_mode": str(args.delta_support_mode),
            "delta_support_topk": int(args.delta_support_topk),
        }
    )
    validate_checkpoint_metadata(contract_metadata, expected_rule_score_scale=float(args.rule_score_scale))
    config_payload = {
        **source_config,
        "source_type": "mortal_action_q_imitation",
        "source_config_id": int(candidate["source_config_id"]),
        "rerun_config_id": int(candidate["rerun_config_id"]),
        "parent_checkpoint_path": candidate.get("checkpoint_path"),
        "parent_checkpoint_sha256": candidate.get("checkpoint_sha256") or _file_sha256(Path(candidate["checkpoint_path"])),
        "rule_score_scale": float(args.rule_score_scale),
        "rule_score_scale_version": RULE_SCORE_SCALE_VERSION,
        "mortal_imitation_config": {
            "episodes": int(args.episodes),
            "iterations": int(args.iterations),
            "selected_iteration": int(iteration),
            "lr": float(args.lr),
            "update_epochs": int(args.update_epochs),
            "teacher_source": str(args.teacher_source),
            "teacher_support": str(args.teacher_support),
            "teacher_topk": int(args.teacher_topk),
            "teacher_temperature": float(args.teacher_temperature),
            "mortal_teacher_checkpoint": str(args.mortal_teacher_checkpoint),
            "mortal_teacher_strict_extra_mask": bool(args.mortal_teacher_strict_extra_mask),
            "action_scope": _action_scope_fields(args),
        },
        "tempered_ratio_config": {
            "support_policy_mode": str(args.support_policy_mode),
            "delta_support_mode": str(args.delta_support_mode),
            "delta_support_topk": int(args.delta_support_topk),
            "delta_support_margin_threshold": float(args.delta_support_margin_threshold),
            "outside_support_delta_mode": str(args.outside_support_delta_mode),
            "delta_support_projection": _single_delta_support_projection_config(
                str(args.delta_support_mode),
                int(args.delta_support_topk),
                float(args.delta_support_margin_threshold),
                str(args.outside_support_delta_mode),
                support_policy_mode=str(args.support_policy_mode),
            ),
        },
    }
    suffix = f"iter_{int(iteration) + 1:04d}"
    config_path = checkpoint_dir / f"config_{suffix}.json"
    policy_path = checkpoint_dir / f"policy_{suffix}.pt"
    optimizer_path = checkpoint_dir / f"optimizer_{suffix}.pt"
    _write_json(config_path, config_payload)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "config": config_payload,
            "contract_metadata": contract_metadata,
            "rule_score_scale": contract_metadata["rule_score_scale"],
            "rule_score_scale_version": contract_metadata["rule_score_scale_version"],
            "summary": _to_jsonable(dict(summary_row)),
        },
        policy_path,
    )
    torch.save({"optimizer_state_dict": optimizer.state_dict(), "config": config_payload}, optimizer_path)
    return {
        "config_id": int(config_id),
        "rerun_config_id": int(candidate["rerun_config_id"]),
        "source_config_id": int(candidate["source_config_id"]),
        "checkpoint_path": str(policy_path),
        "checkpoint_sha256": _file_sha256(policy_path),
        "config_path": str(config_path),
        "optimizer_path": str(optimizer_path),
        "optimizer_sha256": _file_sha256(optimizer_path),
        "teacher_source": str(args.teacher_source),
        "teacher_support": str(args.teacher_support),
        "teacher_topk": int(args.teacher_topk),
        "teacher_temperature": float(args.teacher_temperature),
        "top1_changed_vs_parent_rate": float(summary_row["top1_changed_vs_parent_rate"]),
        "teacher_ce": float(summary_row["teacher_ce"]),
        "teacher_kl": float(summary_row["teacher_kl"]),
        "teacher_policy_agreement": float(summary_row["teacher_policy_agreement"]),
        "mapping_row_count": int(summary_row["mapping_row_count"]),
        "mapping_available_count": int(summary_row["mapping_available_count"]),
        "fail_closed_count": int(summary_row["fail_closed_count"]),
        "rule_score_scale": float(args.rule_score_scale),
        "rule_score_scale_version": RULE_SCORE_SCALE_VERSION,
    }


def _write_outputs(
    args: argparse.Namespace,
    summary_rows: Sequence[dict[str, Any]],
    iteration_rows: Sequence[dict[str, Any]],
    audit_rows: Sequence[dict[str, Any]],
    changed_rows: Sequence[dict[str, Any]],
    checkpoint_rows: Sequence[dict[str, Any]],
) -> None:
    _write_csv(args.output_dir / "imitation_summary.csv", summary_rows)
    _write_csv(args.output_dir / "imitation_iterations.csv", iteration_rows)
    _write_csv(args.output_dir / "mortal_action_mapping_audit.csv", audit_rows)
    _write_jsonl(args.output_dir / "mortal_action_mapping_examples.jsonl", audit_rows)
    _write_csv(args.output_dir / "changed_decisions.csv", changed_rows)
    _write_csv(args.output_dir / "checkpoint_iterations.csv", checkpoint_rows)
    _write_csv(args.output_dir / "checkpoint_summary.csv", _latest_checkpoint_rows(checkpoint_rows))
    _write_json(
        args.output_dir / "mortal_imitation.json",
        {
            "mode": "mortal_action_q_imitation",
            "candidate_summary": str(args.candidate_summary),
            "source_config_ids": [int(value) for value in args.source_config_ids],
            "teacher_source": str(args.teacher_source),
            "teacher_checkpoint": str(args.mortal_teacher_checkpoint),
            "teacher_support": str(args.teacher_support),
            "teacher_topk": int(args.teacher_topk),
            "summaries": summary_rows,
            "iterations": iteration_rows,
            "checkpoints": checkpoint_rows,
        },
    )
    (args.output_dir / "imitation_summary.md").write_text(
        _summary_markdown(args, summary_rows, checkpoint_rows),
        encoding="utf-8",
    )
    (args.output_dir / "changed_decisions.readable.md").write_text(
        _changed_decisions_markdown(changed_rows),
        encoding="utf-8",
    )
    print((args.output_dir / "imitation_summary.md").read_text(encoding="utf-8"))


def _write_incremental_outputs(
    args: argparse.Namespace,
    iteration_rows: Sequence[dict[str, Any]],
    audit_rows: Sequence[dict[str, Any]],
    changed_rows: Sequence[dict[str, Any]],
    checkpoint_rows: Sequence[dict[str, Any]],
) -> None:
    """Persist per-iteration artifacts so interrupted runs remain inspectable."""
    _write_csv(args.output_dir / "imitation_iterations.csv", iteration_rows)
    _write_csv(args.output_dir / "mortal_action_mapping_audit.csv", audit_rows)
    _write_jsonl(args.output_dir / "mortal_action_mapping_examples.jsonl", audit_rows)
    _write_csv(args.output_dir / "changed_decisions.csv", changed_rows)
    _write_csv(args.output_dir / "checkpoint_iterations.csv", checkpoint_rows)
    _write_csv(args.output_dir / "checkpoint_summary.csv", _latest_checkpoint_rows(checkpoint_rows))
    _write_json(
        args.output_dir / "mortal_imitation_progress.json",
        {
            "mode": "mortal_action_q_imitation_progress",
            "candidate_summary": str(args.candidate_summary),
            "source_config_ids": [int(value) for value in args.source_config_ids],
            "teacher_source": str(args.teacher_source),
            "teacher_checkpoint": str(args.mortal_teacher_checkpoint),
            "teacher_support": str(args.teacher_support),
            "teacher_topk": int(args.teacher_topk),
            "iterations": iteration_rows,
            "checkpoints": checkpoint_rows,
        },
    )


def _latest_checkpoint_rows(checkpoint_rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    latest_by_config: dict[int, dict[str, Any]] = {}
    for row in checkpoint_rows:
        latest_by_config[int(row["config_id"])] = dict(row)
    return [latest_by_config[key] for key in sorted(latest_by_config)]


def _summary_markdown(
    args: argparse.Namespace,
    summary_rows: Sequence[dict[str, Any]],
    checkpoint_rows: Sequence[dict[str, Any]],
) -> str:
    lines = [
        "# KeqingRL Mortal Action-Q Imitation",
        "",
        f"candidate_summary: `{args.candidate_summary}`",
        f"teacher_source: `{args.teacher_source}`",
        f"teacher_checkpoint: `{args.mortal_teacher_checkpoint}`",
        f"teacher_support: `{args.teacher_support}`",
        f"teacher_topk: `{args.teacher_topk}`",
        f"episodes: `{args.episodes}`",
        f"iterations: `{args.iterations}`",
        "",
        "## Results",
        "",
    ]
    for row in summary_rows:
        lines.append(
            "- "
            f"cfg={row['pilot_config_id']} source={row['source_config_id']} "
            f"mapping={row['mapping_available_count']}/{row['mapping_row_count']} "
            f"fail_closed={row['fail_closed_count']} "
            f"teacher_ce={row['teacher_ce']:.6g} "
            f"teacher_kl={row['teacher_kl']:.6g} "
            f"teacher_agree={row['teacher_policy_agreement']:.6g} "
            f"top1_parent={row['top1_changed_vs_parent_rate']:.6g} "
            f"top1_source={row['top1_changed_vs_source_rate']:.6g} "
            f"changed_rank={row['changed_rank_mean']:.6g} "
            f"rank_ge5={row['rank_ge5_rate']:.6g} "
            f"checkpoint={row.get('checkpoint_path', '')}"
        )
    lines.extend(["", "## Checkpoints", ""])
    for row in checkpoint_rows:
        lines.append(f"- `{row['checkpoint_path']}` sha256=`{row['checkpoint_sha256']}`")
    return "\n".join(lines) + "\n"


def _changed_decisions_markdown(rows: Sequence[dict[str, Any]]) -> str:
    lines = ["# Mortal Imitation Changed Decisions", ""]
    if not rows:
        lines.append("_No changed decisions recorded._")
        return "\n".join(lines) + "\n"
    for row in rows[:200]:
        lines.append(
            "- "
            f"iter={row.get('iteration')} episode={row.get('episode_id')} step={row.get('step_id')} "
            f"actor={row.get('actor')} rule=`{row.get('rule_prior_top1')}` "
            f"student=`{row.get('student_top1')}` rank={row.get('changed_action_prior_rank')}"
        )
    return "\n".join(lines) + "\n"


def _should_save_checkpoint(args: argparse.Namespace, iteration: int) -> bool:
    every = int(args.save_checkpoint_every)
    return every > 0 and ((int(iteration) + 1) % every == 0 or int(iteration) + 1 == int(args.iterations))


def _mapping_context(prepared_steps: Sequence[Any] | None, row_idx: int) -> dict[str, Any]:
    if prepared_steps is None or row_idx >= len(prepared_steps):
        return {}
    step = prepared_steps[row_idx]
    return {
        "episode_id": step.episode_id,
        "step_index": int(step.step_id),
        "actor": int(step.actor),
        "control_action_types": tuple(step.control_action_types),
    }


def _append_limited_audit_row(rows: list[dict[str, Any]], row: dict[str, Any], *, max_examples: int) -> None:
    kind = str(row.get("mismatch_kind", "unknown"))
    if sum(1 for existing in rows if str(existing.get("mismatch_kind", "unknown")) == kind) < int(max_examples):
        rows.append(row)


def _iteration_seed(args: argparse.Namespace, config_id: int, iteration: int) -> int:
    return int(args.seed_base + config_id * 100_000 + iteration * int(args.episodes) * int(args.seed_stride))


def _iteration_torch_seed(args: argparse.Namespace, config_id: int, iteration: int) -> int:
    return int(args.torch_seed_base + config_id * 100_000 + iteration)


def _seed_torch_sampling(seed: int) -> None:
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def _action_scope_fields(args: argparse.Namespace) -> dict[str, str]:
    return {
        "self_turn_action_types": ",".join(str(value) for value in args.self_turn_action_types),
        "response_action_types": ",".join(str(value) for value in args.response_action_types),
        "forced_autopilot_action_types": ",".join(str(value) for value in args.forced_autopilot_action_types),
    }


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def _mean_bool(values: torch.Tensor) -> float:
    return float(values.float().mean().detach().cpu().item()) if values.numel() else 0.0


def _mean_float(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / len(values)) if values else 0.0


def _scalar(value: torch.Tensor | float | int) -> float:
    if isinstance(value, torch.Tensor):
        return float(value.detach().cpu().item())
    return float(value)


def _action_label(legal_actions: Sequence[ActionSpec], index: int) -> str:
    if int(index) < 0 or int(index) >= len(legal_actions):
        return f"index:{int(index)}"
    return legal_actions[int(index)].canonical_key


def _json_dumps(value: object) -> str:
    return json.dumps(_to_jsonable(value), ensure_ascii=True, separators=(",", ":"), default=str)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(_to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(_to_jsonable(row), ensure_ascii=True, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


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


if __name__ == "__main__":
    main()
