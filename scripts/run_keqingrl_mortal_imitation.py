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

import keqing_core
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
    MORTAL_KAN_SELECT_ACTION_MASK_EXTRA_KEY,
    MORTAL_KAN_SELECT_ENCODED_ACTION_MASK_EXTRA_KEY,
    MORTAL_KAN_SELECT_ENCODED_OBS_EXTRA_KEY,
    MORTAL_KAN_SELECT_Q_VALUES_EXTRA_KEY,
    MORTAL_Q_VALUES_EXTRA_KEY,
    MortalTeacherMappingError,
    mortal_action_mapping_audit_row,
    mortal_discard_teacher_tensors_from_extras,
    mortal_scores_for_legal_actions,
)
from keqingrl.selfplay import build_episodes_ppo_batch, collect_selfplay_episodes
from mahjong_env.action_space import IDX_TO_TILE_NAME
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
TEACHER_SUPPORT_MODES = ("topk", "adaptive-topk", "full-legal")
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
    support_mask: torch.BoolTensor | None = None
    legal_action_mask: torch.BoolTensor | None = None
    mapped_legal_scores: torch.Tensor | None = None


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
    parser.add_argument("--support-policy-mode", choices=("support-only-topk", "unrestricted"), default="unrestricted")
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
        default=("DISCARD", "REACH_DISCARD", "TSUMO", "ANKAN", "KAKAN", "RYUKYOKU"),
    )
    parser.add_argument(
        "--response-action-types",
        choices=_ACTION_TYPE_CHOICES,
        nargs="*",
        default=("PASS", "RON", "PON", "CHI", "DAIMINKAN"),
    )
    parser.add_argument("--export-decision-review-cases", action="store_true")
    parser.add_argument("--decision-review-case-limit", type=int, default=500)
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
    action_type_breakdown_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    changed_rows: list[dict[str, Any]] = []
    teacher_disagreement_rows: list[dict[str, Any]] = []
    decision_review_rows: list[dict[str, Any]] = []
    decision_review_case_rows: list[dict[str, Any]] = []
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
                _write_incremental_outputs(
                    args,
                    iteration_rows,
                    action_type_breakdown_rows,
                    audit_rows,
                    changed_rows,
                    teacher_disagreement_rows,
                    decision_review_rows,
                    decision_review_case_rows,
                    checkpoint_rows,
                )
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
                (
                    metrics,
                    new_changed_rows,
                    new_teacher_disagreement_rows,
                    new_decision_review_rows,
                    new_decision_review_case_rows,
                    new_action_type_breakdown_rows,
                ) = imitation_metrics(
                    output,
                    batch.policy_input,
                    parent_output=parent_output,
                    source_output=source_output,
                    prepared_steps=prepared_steps,
                    teacher_support=str(args.teacher_support),
                    teacher_topk=int(args.teacher_topk),
                    teacher_temperature=float(args.teacher_temperature),
                    teacher_batch=teacher_data.teacher_batch,
                    mapping_summary=teacher_data.summary,
                    export_decision_review_cases=bool(args.export_decision_review_cases),
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
            teacher_disagreement_rows.extend(
                {
                    "pilot_config_id": int(config_id),
                    "iteration": int(iteration),
                    **row,
                }
                for row in new_teacher_disagreement_rows
            )
            decision_review_rows.extend(
                {
                    "pilot_config_id": int(config_id),
                    "iteration": int(iteration),
                    **row,
                }
                for row in new_decision_review_rows
            )
            if bool(args.export_decision_review_cases):
                remaining_case_slots = max(0, int(args.decision_review_case_limit) - len(decision_review_case_rows))
                decision_review_case_rows.extend(
                    {
                        **row,
                        "pilot_config_id": int(config_id),
                        "iteration": int(iteration),
                        "case_id": f"cfg{int(config_id)}_iter{int(iteration)}_{row.get('case_id', '')}",
                        "run_id": args.output_dir.name,
                        "rollout_seed": int(rollout_seed),
                        "torch_seed": int(torch_seed),
                    }
                    for row in new_decision_review_case_rows[:remaining_case_slots]
                )
            action_type_breakdown_rows.extend(
                {
                    "pilot_config_id": int(config_id),
                    "iteration": int(iteration),
                    "rollout_seed": int(rollout_seed),
                    "torch_seed": int(torch_seed),
                    "teacher_source": str(args.teacher_source),
                    "teacher_support": str(args.teacher_support),
                    "teacher_topk": int(args.teacher_topk),
                    **row,
                }
                for row in new_action_type_breakdown_rows
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
            _write_incremental_outputs(
                args,
                iteration_rows,
                action_type_breakdown_rows,
                audit_rows,
                changed_rows,
                teacher_disagreement_rows,
                decision_review_rows,
                decision_review_case_rows,
                checkpoint_rows,
            )

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

    _write_outputs(
        args,
        summary_rows,
        iteration_rows,
        action_type_breakdown_rows,
        audit_rows,
        changed_rows,
        teacher_disagreement_rows,
        decision_review_rows,
        decision_review_case_rows,
        checkpoint_rows,
    )


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
    kan_encoded_obs_rows: list[torch.Tensor] = []
    kan_encoded_mask_rows: list[torch.Tensor] = []
    has_kan_rows: list[bool] = []
    kan_bridge: MortalObservationBridge | None = None
    for row_idx, step in enumerate(prepared_steps):
        events = tuple(getattr(step, "mortal_teacher_events", ()))
        if not events:
            raise MortalTeacherMappingError(
                f"Mortal observation events missing for deferred bridge row {row_idx}"
            )
        # Prepared PPO rows are not guaranteed to be chronological per actor.
        # Reusing Mortal PlayerState across out-of-order rows can replay an
        # invalid event suffix into libriichi, so post-rollout materialization
        # rebuilds each row independently.
        bridge.reset_cache()
        encoded = bridge.encode_from_events(events, int(step.actor))
        encoded_obs_rows.append(encoded.obs)
        encoded_mask_rows.append(encoded.action_mask)
        row_actions = policy_input.legal_actions[row_idx] if policy_input.legal_actions is not None else ()
        has_kan = any(action.action_type in (ActionType.DAIMINKAN, ActionType.ANKAN, ActionType.KAKAN) for action in row_actions)
        has_kan_rows.append(has_kan)
        if has_kan:
            if kan_bridge is None:
                kan_bridge = MortalObservationBridge(
                    mortal_root=Path(getattr(bridge, "mortal_root", "third_party/Mortal")),
                    at_kan_select=True,
                    enable_incremental_cache=bool(getattr(bridge, "enable_incremental_cache", True)),
                )
            kan_bridge.reset_cache()
            kan_encoded = kan_bridge.encode_from_events(events, int(step.actor))
            kan_encoded_obs_rows.append(kan_encoded.obs)
            kan_encoded_mask_rows.append(kan_encoded.action_mask)
        else:
            kan_encoded_obs_rows.append(torch.zeros_like(encoded.obs))
            kan_encoded_mask_rows.append(torch.zeros_like(encoded.action_mask))
    new_extras = dict(extras)
    new_extras[MORTAL_ENCODED_OBS_EXTRA_KEY] = torch.stack(encoded_obs_rows, dim=0)
    new_extras[MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY] = torch.stack(encoded_mask_rows, dim=0)
    if any(has_kan_rows):
        new_extras[MORTAL_KAN_SELECT_ENCODED_OBS_EXTRA_KEY] = torch.stack(kan_encoded_obs_rows, dim=0)
        new_extras[MORTAL_KAN_SELECT_ENCODED_ACTION_MASK_EXTRA_KEY] = torch.stack(kan_encoded_mask_rows, dim=0)
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
    has_kan_encoded = (
        MORTAL_KAN_SELECT_ENCODED_OBS_EXTRA_KEY in extras
        or MORTAL_KAN_SELECT_ENCODED_ACTION_MASK_EXTRA_KEY in extras
    )
    has_kan_q_values = MORTAL_KAN_SELECT_Q_VALUES_EXTRA_KEY in extras
    has_kan_action_mask = MORTAL_KAN_SELECT_ACTION_MASK_EXTRA_KEY in extras
    if has_q_values and has_action_mask and (not has_kan_encoded or (has_kan_q_values and has_kan_action_mask)):
        return policy_input
    if has_q_values != has_action_mask:
        present = [key for key in (MORTAL_Q_VALUES_EXTRA_KEY, MORTAL_ACTION_MASK_EXTRA_KEY) if key in extras]
        raise MortalTeacherMappingError(f"Mortal teacher q/mask extras are incomplete: present={present}")
    if teacher_runtime is None:
        raise MortalTeacherMappingError("Mortal teacher runtime is required to materialize deferred q/mask extras")

    batch_size = int(policy_input.legal_action_mask.shape[0])
    chunk_size = int(eval_batch_size or 0)
    if chunk_size <= 0:
        chunk_size = batch_size
    target_device = policy_input.legal_action_mask.device
    new_extras = dict(extras)
    if not (has_q_values and has_action_mask):
        missing_encoded = [
            key
            for key in (MORTAL_ENCODED_OBS_EXTRA_KEY, MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY)
            if key not in extras
        ]
        if missing_encoded:
            raise MortalTeacherMappingError(f"Mortal encoded teacher extras missing required keys: {missing_encoded}")
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
        if int(encoded_obs.shape[0]) != batch_size or int(encoded_action_mask.shape[0]) != batch_size:
            raise MortalTeacherMappingError(
                "Deferred Mortal encoded extras batch size mismatch: "
                f"policy={batch_size}, obs={tuple(encoded_obs.shape)}, mask={tuple(encoded_action_mask.shape)}"
            )
        output_q_values: list[torch.Tensor] = []
        output_action_masks: list[torch.Tensor] = []
        for start in range(0, batch_size, chunk_size):
            end = min(batch_size, start + chunk_size)
            output = teacher_runtime.evaluate(encoded_obs[start:end], encoded_action_mask[start:end])
            output_q_values.append(output.q_values)
            output_action_masks.append(output.action_mask)
        new_extras.pop(MORTAL_ENCODED_OBS_EXTRA_KEY, None)
        new_extras.pop(MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY, None)
        new_extras[MORTAL_Q_VALUES_EXTRA_KEY] = torch.cat(output_q_values, dim=0).to(device=target_device)
        new_extras[MORTAL_ACTION_MASK_EXTRA_KEY] = torch.cat(output_action_masks, dim=0).to(device=target_device)
    if has_kan_encoded:
        missing_kan = [
            key
            for key in (MORTAL_KAN_SELECT_ENCODED_OBS_EXTRA_KEY, MORTAL_KAN_SELECT_ENCODED_ACTION_MASK_EXTRA_KEY)
            if key not in extras
        ]
        if missing_kan:
            raise MortalTeacherMappingError(f"Mortal kan-select encoded extras are incomplete: missing={missing_kan}")
        kan_encoded_obs = extras[MORTAL_KAN_SELECT_ENCODED_OBS_EXTRA_KEY]
        kan_encoded_mask = extras[MORTAL_KAN_SELECT_ENCODED_ACTION_MASK_EXTRA_KEY].bool()
        if int(kan_encoded_obs.shape[0]) != batch_size or int(kan_encoded_mask.shape[0]) != batch_size:
            raise MortalTeacherMappingError(
                "Deferred Mortal kan-select encoded extras batch size mismatch: "
                f"policy={batch_size}, obs={tuple(kan_encoded_obs.shape)}, mask={tuple(kan_encoded_mask.shape)}"
            )
        kan_output_q_values: list[torch.Tensor] = []
        kan_output_action_masks: list[torch.Tensor] = []
        for start in range(0, batch_size, chunk_size):
            end = min(batch_size, start + chunk_size)
            output = teacher_runtime.evaluate(kan_encoded_obs[start:end], kan_encoded_mask[start:end])
            kan_output_q_values.append(output.q_values)
            kan_output_action_masks.append(output.action_mask)
        new_extras.pop(MORTAL_KAN_SELECT_ENCODED_OBS_EXTRA_KEY, None)
        new_extras.pop(MORTAL_KAN_SELECT_ENCODED_ACTION_MASK_EXTRA_KEY, None)
        new_extras[MORTAL_KAN_SELECT_Q_VALUES_EXTRA_KEY] = torch.cat(kan_output_q_values, dim=0).to(device=target_device)
        new_extras[MORTAL_KAN_SELECT_ACTION_MASK_EXTRA_KEY] = torch.cat(kan_output_action_masks, dim=0).to(device=target_device)
    return replace(policy_input, obs=replace(policy_input.obs, extras=new_extras))


def _optional_kan_select_teacher_tensors_from_extras(
    extras: Mapping[str, torch.Tensor],
) -> tuple[torch.Tensor | None, torch.BoolTensor | None]:
    has_q = MORTAL_KAN_SELECT_Q_VALUES_EXTRA_KEY in extras
    has_mask = MORTAL_KAN_SELECT_ACTION_MASK_EXTRA_KEY in extras
    if not has_q and not has_mask:
        return None, None
    if has_q != has_mask:
        present = [
            key
            for key in (MORTAL_KAN_SELECT_Q_VALUES_EXTRA_KEY, MORTAL_KAN_SELECT_ACTION_MASK_EXTRA_KEY)
            if key in extras
        ]
        raise MortalTeacherMappingError(f"Mortal kan-select q/mask extras are incomplete: present={present}")
    q_values = extras[MORTAL_KAN_SELECT_Q_VALUES_EXTRA_KEY].float()
    action_mask = extras[MORTAL_KAN_SELECT_ACTION_MASK_EXTRA_KEY].bool()
    if q_values.ndim != 2 or int(q_values.shape[-1]) != MORTAL_ACTION_SPACE:
        raise MortalTeacherMappingError(
            f"Mortal kan-select q values must have shape [B, {MORTAL_ACTION_SPACE}], "
            f"got {tuple(q_values.shape)}"
        )
    if action_mask.shape != q_values.shape:
        raise MortalTeacherMappingError(
            "Mortal kan-select mask shape must match q values: "
            f"{tuple(action_mask.shape)} != {tuple(q_values.shape)}"
        )
    return q_values, action_mask


def audit_mortal_action_mapping(
    policy_input: PolicyInput,
    *,
    prepared_steps: Sequence[Any] | None = None,
    strict_extra: bool,
) -> MortalMappingAuditResult:
    if policy_input.legal_actions is None:
        raise MortalTeacherMappingError("Mortal imitation requires policy_input.legal_actions")
    q_values, mortal_masks = mortal_discard_teacher_tensors_from_extras(policy_input.obs.extras)
    kan_q_values, kan_masks = _optional_kan_select_teacher_tensors_from_extras(policy_input.obs.extras)
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
                kan_select_q_values=None if kan_q_values is None else kan_q_values[row_idx],
                kan_select_mask=None if kan_masks is None else kan_masks[row_idx],
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
                kan_select_q_values=None if kan_q_values is None else kan_q_values[row_idx],
                kan_select_mask=None if kan_masks is None else kan_masks[row_idx],
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
    kan_q_values, kan_masks = _optional_kan_select_teacher_tensors_from_extras(policy_input.obs.extras)
    mapping_row_count = int(q_values.shape[0])
    if str(teacher_support) in {"topk", "adaptive-topk"}:
        teacher_batch, summary, audit_rows = _prepare_mortal_topk_teacher_batch(
            policy_input,
            q_values=q_values,
            mortal_masks=mortal_masks,
            kan_q_values=kan_q_values,
            kan_masks=kan_masks,
            prepared_steps=prepared_steps,
            strict_extra=bool(strict_extra),
            teacher_topk=int(teacher_topk),
            adaptive=str(teacher_support) == "adaptive-topk",
        )
    elif str(teacher_support) == "full-legal":
        teacher_batch, summary, audit_rows = _prepare_mortal_full_legal_teacher_batch(
            policy_input,
            q_values=q_values,
            mortal_masks=mortal_masks,
            kan_q_values=kan_q_values,
            kan_masks=kan_masks,
            prepared_steps=prepared_steps,
            strict_extra=bool(strict_extra),
        )
    else:
        raise ValueError(f"unsupported teacher support: {teacher_support}")
    if int(summary["mapping_row_count"]) != mapping_row_count:
        raise MortalTeacherMappingError("Mortal teacher cache row count changed while preparing batch")
    summary.update(_teacher_validity_summary(policy_input, teacher_batch))
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
    kan_q_values: torch.Tensor | None,
    kan_masks: torch.Tensor | None,
    prepared_steps: Sequence[Any] | None,
    strict_extra: bool,
    teacher_topk: int,
    adaptive: bool = False,
) -> tuple[MortalImitationTeacherBatch, dict[str, Any], tuple[dict[str, Any], ...]]:
    assert policy_input.legal_actions is not None
    assert policy_input.prior_logits is not None
    prior_logits = policy_input.prior_logits.float()
    legal_mask = policy_input.legal_action_mask.bool()
    mapped_score_rows: list[torch.Tensor] = []
    summary, audit_rows = _collect_mortal_mapping_summary_and_audit(
        q_values=q_values,
        mortal_masks=mortal_masks,
        kan_q_values=kan_q_values,
        kan_masks=kan_masks,
        legal_actions=policy_input.legal_actions,
        prepared_steps=prepared_steps,
        strict_extra=bool(strict_extra),
        mapped_scores_callback=lambda _row_idx, mapped: mapped_score_rows.append(
            mapped.scores.to(device=prior_logits.device, dtype=prior_logits.dtype)
        ),
    )
    if len(mapped_score_rows) != int(q_values.shape[0]):
        raise MortalTeacherMappingError("Mortal topK teacher cache row count mismatch")
    mapped_scores = torch.full_like(prior_logits, float("-inf"))
    for row_idx, row_scores in enumerate(mapped_score_rows):
        width = min(int(row_scores.shape[0]), int(mapped_scores.shape[1]))
        mapped_scores[row_idx, :width] = row_scores[:width]
    topk_indices, valid_rows, support_mask, distinct_counts = _source_topk_indices(
        mapped_scores,
        legal_mask,
        policy_input.legal_actions,
        k=int(teacher_topk),
        adaptive=bool(adaptive),
    )
    masked_prior = prior_logits.masked_fill(~legal_mask, torch.finfo(torch.float32).min)
    topk_prior = masked_prior.gather(1, topk_indices.to(device=masked_prior.device))
    teacher_scores = mapped_scores.gather(1, topk_indices.to(device=mapped_scores.device))
    support_mask = support_mask.to(device=teacher_scores.device)
    finite_supported = torch.isfinite(teacher_scores) | ~support_mask
    finite_rows = finite_supported.all(dim=-1) & support_mask.any(dim=-1)
    valid_rows = valid_rows.to(device=finite_rows.device) & finite_rows
    teacher_scores = torch.where(
        torch.isfinite(teacher_scores),
        teacher_scores,
        torch.zeros_like(teacher_scores),
    )
    return (
        MortalImitationTeacherBatch(
            teacher_support="adaptive-topk" if bool(adaptive) else "topk",
            teacher_topk=int(teacher_topk),
            teacher_scores=teacher_scores,
            prior_logits=topk_prior.to(device=prior_logits.device),
            row_valid_mask=valid_rows.to(device=prior_logits.device),
            topk_indices=topk_indices.to(device=prior_logits.device),
            support_mask=support_mask.to(device=prior_logits.device),
            mapped_legal_scores=mapped_scores.to(device=prior_logits.device),
        ),
        {
            **summary,
            "teacher_topk_distinct_source_count_min": int(distinct_counts.min().detach().cpu().item()) if distinct_counts.numel() else 0,
            "teacher_topk_distinct_source_count_mean": float(distinct_counts.float().mean().detach().cpu().item()) if distinct_counts.numel() else 0.0,
        },
        audit_rows,
    )


def _prepare_mortal_full_legal_teacher_batch(
    policy_input: PolicyInput,
    *,
    q_values: torch.Tensor,
    mortal_masks: torch.Tensor,
    kan_q_values: torch.Tensor | None,
    kan_masks: torch.Tensor | None,
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
        kan_q_values=kan_q_values,
        kan_masks=kan_masks,
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
            mapped_legal_scores=teacher_scores,
        ),
        summary,
        audit_rows,
    )


def _collect_mortal_mapping_summary_and_audit(
    *,
    q_values: torch.Tensor,
    mortal_masks: torch.Tensor,
    kan_q_values: torch.Tensor | None,
    kan_masks: torch.Tensor | None,
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
                kan_select_q_values=None if kan_q_values is None else kan_q_values[row_idx],
                kan_select_mask=None if kan_masks is None else kan_masks[row_idx],
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
                kan_select_q_values=None if kan_q_values is None else kan_q_values[row_idx],
                kan_select_mask=None if kan_masks is None else kan_masks[row_idx],
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


def _teacher_validity_summary(
    policy_input: PolicyInput,
    teacher_batch: MortalImitationTeacherBatch,
) -> dict[str, Any]:
    if policy_input.legal_actions is None:
        return {}
    valid = teacher_batch.row_valid_mask.detach().cpu().bool()
    row_count = int(valid.numel())
    invalid_by_scope: dict[str, int] = {}
    invalid_by_types: dict[str, int] = {}
    invalid_by_flags: dict[str, int] = {}
    for row_idx, is_valid in enumerate(valid.tolist()):
        if bool(is_valid):
            continue
        legal_actions = policy_input.legal_actions[row_idx]
        scope = _action_scope_for_legal_actions(legal_actions)
        type_key = ",".join(sorted({action.action_type.name for action in legal_actions}))
        invalid_by_scope[scope] = invalid_by_scope.get(scope, 0) + 1
        invalid_by_types[type_key] = invalid_by_types.get(type_key, 0) + 1
        for flag in _legal_contains_flags(legal_actions):
            if flag.startswith("contains_") and flag.endswith("=true"):
                invalid_by_flags[flag] = invalid_by_flags.get(flag, 0) + 1
    valid_count = int(valid.sum().item())
    return {
        "teacher_row_valid_count": valid_count,
        "teacher_row_valid_rate": valid_count / max(1, row_count),
        "teacher_row_invalid_count": row_count - valid_count,
        "teacher_row_invalid_by_scope_json": _json_dumps(invalid_by_scope),
        "teacher_row_invalid_by_legal_action_types_json": _json_dumps(invalid_by_types),
        "teacher_row_invalid_by_legal_flags_json": _json_dumps(invalid_by_flags),
    }


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
    if str(teacher_support) in {"topk", "adaptive-topk"}:
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
    student_logits = _student_imitation_logits(output)
    if teacher_batch.teacher_support in {"topk", "adaptive-topk"}:
        if teacher_batch.topk_indices is None:
            raise MortalTeacherMappingError("cached topK teacher batch is missing topk_indices")
        policy_logits = student_logits.gather(
            1,
            teacher_batch.topk_indices.to(device=student_logits.device),
        )
        return _imitation_loss_from_scores(
            policy_logits=policy_logits,
            prior_logits=teacher_batch.prior_logits.to(device=policy_logits.device),
            teacher_scores=teacher_batch.teacher_scores.to(device=policy_logits.device),
            row_valid_mask=teacher_batch.row_valid_mask.to(device=policy_logits.device),
            teacher_temperature=float(teacher_temperature),
            support_mask=None
            if teacher_batch.support_mask is None
            else teacher_batch.support_mask.to(device=policy_logits.device),
        )
    if teacher_batch.teacher_support == "full-legal":
        if teacher_batch.legal_action_mask is None:
            raise MortalTeacherMappingError("cached full-legal teacher batch is missing legal_action_mask")
        mask = teacher_batch.legal_action_mask.to(device=student_logits.device)
        policy_logits = student_logits.masked_fill(~mask, torch.finfo(torch.float32).min)
        return _imitation_loss_from_scores(
            policy_logits=policy_logits,
            prior_logits=teacher_batch.prior_logits.to(device=student_logits.device),
            teacher_scores=teacher_batch.teacher_scores.to(device=student_logits.device),
            row_valid_mask=teacher_batch.row_valid_mask.to(device=student_logits.device),
            teacher_temperature=float(teacher_temperature),
            support_mask=mask,
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
    legal_mask = policy_input.legal_action_mask.bool()
    mapped_scores = torch.full_like(prior_logits, float("-inf"))
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
        width = min(int(mapped.scores.shape[0]), int(mapped_scores.shape[1]))
        mapped_scores[row_idx, :width] = mapped.scores[:width].to(device=mapped_scores.device)
    topk_indices, valid_rows = _unique_source_topk_indices(
        mapped_scores,
        legal_mask,
        policy_input.legal_actions,
        k=int(teacher_topk),
    )
    teacher_scores = mapped_scores.gather(1, topk_indices.to(device=mapped_scores.device)).to(
        device=output.action_logits.device,
        dtype=output.action_logits.dtype,
    )
    finite_rows = torch.isfinite(teacher_scores).all(dim=-1)
    valid_rows = valid_rows.to(device=finite_rows.device) & finite_rows
    teacher_scores = torch.where(torch.isfinite(teacher_scores), teacher_scores, torch.zeros_like(teacher_scores))
    student_logits = _student_imitation_logits(output)
    policy_logits = student_logits.gather(1, topk_indices.to(device=student_logits.device))
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
    student_logits = _student_imitation_logits(output)
    mask = policy_input.legal_action_mask.bool().to(device=student_logits.device)
    teacher_scores = teacher_scores.masked_fill(~mask, torch.finfo(teacher_scores.dtype).min)
    policy_logits = student_logits.masked_fill(~mask, torch.finfo(student_logits.dtype).min)
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


def _student_imitation_logits(output: PolicyOutput) -> torch.Tensor:
    logits = output.aux.get("unprojected_final_logits", output.action_logits)
    return logits.float()


def _imitation_loss_from_scores(
    *,
    policy_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    teacher_scores: torch.Tensor,
    row_valid_mask: torch.Tensor,
    teacher_temperature: float,
    support_mask: torch.Tensor | None = None,
) -> MortalImitationLossResult:
    if support_mask is not None:
        support_mask = support_mask.bool().to(device=teacher_scores.device)
        min_value = torch.finfo(torch.float32).min
        teacher_scores = teacher_scores.float().masked_fill(~support_mask, min_value)
        policy_logits = policy_logits.float().masked_fill(~support_mask, min_value)
        prior_logits = prior_logits.float().masked_fill(~support_mask, min_value)
        row_valid_mask = row_valid_mask.bool().to(device=teacher_scores.device) & support_mask.any(dim=-1)
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
    teacher_margin = _teacher_margin_from_scores(teacher_scores, support_mask=support_mask)
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


def _teacher_margin_from_scores(
    teacher_scores: torch.Tensor,
    *,
    support_mask: torch.Tensor | None,
) -> torch.Tensor:
    if teacher_scores.shape[-1] <= 1:
        return torch.zeros(teacher_scores.shape[0], dtype=teacher_scores.dtype, device=teacher_scores.device)
    scores = teacher_scores.float()
    if support_mask is None:
        valid_count = torch.full(
            (scores.shape[0],),
            int(scores.shape[-1]),
            dtype=torch.long,
            device=scores.device,
        )
    else:
        support_mask = support_mask.bool().to(device=scores.device)
        valid_count = support_mask.sum(dim=-1)
        scores = scores.masked_fill(~support_mask, torch.finfo(torch.float32).min)
    top2 = torch.topk(scores, k=2, dim=-1).values
    margin = top2[:, 0] - top2[:, 1]
    return torch.where(valid_count >= 2, margin, torch.zeros_like(margin))


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
    teacher_batch: MortalImitationTeacherBatch,
    mapping_summary: Mapping[str, Any],
    export_decision_review_cases: bool = False,
) -> tuple[
    dict[str, Any],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    del teacher_support, teacher_topk
    student_logits = _student_imitation_logits(output)
    parent_student_logits = _student_imitation_logits(parent_output)
    source_student_logits = _student_imitation_logits(source_output)
    mask = policy_input.legal_action_mask.bool().to(device=student_logits.device)
    final_logits = student_logits.masked_fill(~mask, torch.finfo(torch.float32).min)
    parent_logits = parent_student_logits.to(device=student_logits.device).masked_fill(~mask, torch.finfo(torch.float32).min)
    source_logits = source_student_logits.to(device=student_logits.device).masked_fill(~mask, torch.finfo(torch.float32).min)
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
    teacher_disagreement_rows: list[dict[str, Any]] = []
    review_rows: list[dict[str, Any]] = []
    review_case_rows: list[dict[str, Any]] = []
    for row_idx in range(int(final_logits.shape[0])):
        selected_changed = bool(changed_vs_parent[row_idx].item())
        teacher_idx = _teacher_top1_legal_index(teacher_batch, row_idx)
        teacher_disagreed = int(final_top1[row_idx].item()) != int(teacher_idx)
        if not selected_changed and not teacher_disagreed:
            continue
        row_mask = mask[row_idx]
        chosen = int(final_top1[row_idx].item())
        row_prior = prior_logits[row_idx]
        legal_prior = row_prior[row_mask]
        rank = 1 + int((legal_prior > row_prior[chosen]).sum().item())
        if bool(changed_vs_prior[row_idx].item()):
            ranks.append(float(rank))
            rank_ge5 += int(rank >= 5)
        review_reason = _review_reason(
            policy_input,
            row_idx,
            final_index=chosen,
            parent_index=int(parent_top1[row_idx].item()),
            teacher_index=teacher_idx,
            selected_changed=selected_changed,
            teacher_disagreed=teacher_disagreed,
        )
        row = _changed_decision_row(
            policy_input,
            prepared_steps,
            row_idx,
            final_top1=final_top1,
            parent_top1=parent_top1,
            source_top1=source_top1,
            prior_top1=prior_top1,
            rank=rank,
            final_logits=final_logits,
            parent_logits=parent_logits,
            prior_logits=prior_logits,
            teacher_batch=teacher_batch,
            selected_changed=selected_changed,
            teacher_disagreed=teacher_disagreed,
            review_reason=review_reason,
        )
        review_rows.append(row)
        if selected_changed:
            changed_rows.append(row)
        if teacher_disagreed:
            teacher_disagreement_rows.append(row)
        if bool(export_decision_review_cases):
            review_case_rows.append(
                _decision_review_case_row(
                    policy_input,
                    prepared_steps,
                    row_idx,
                    final_top1=final_top1,
                    parent_top1=parent_top1,
                    prior_top1=prior_top1,
                    final_logits=final_logits,
                    parent_logits=parent_logits,
                    prior_logits=prior_logits,
                    teacher_batch=teacher_batch,
                    selected_changed=selected_changed,
                    teacher_disagreed=teacher_disagreed,
                    review_reason=review_reason,
                )
            )
    legal_delta = (final_logits - parent_logits).abs().masked_fill(~mask, 0.0)
    action_type_breakdown = _imitation_action_type_breakdown(
        output,
        policy_input,
        parent_output=parent_output,
        prior_logits=prior_logits,
        teacher_batch=teacher_batch,
        teacher_temperature=float(teacher_temperature),
        mapping_summary=mapping_summary,
    )
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
        teacher_disagreement_rows,
        review_rows,
        review_case_rows,
        action_type_breakdown,
    )


def _changed_decision_row(
    policy_input: PolicyInput,
    prepared_steps: Sequence[Any],
    row_idx: int,
    *,
    final_top1: torch.Tensor,
    parent_top1: torch.Tensor,
    source_top1: torch.Tensor,
    prior_top1: torch.Tensor,
    rank: int,
    final_logits: torch.Tensor,
    parent_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    teacher_batch: MortalImitationTeacherBatch,
    selected_changed: bool,
    teacher_disagreed: bool,
    review_reason: str,
) -> dict[str, Any]:
    step = prepared_steps[row_idx] if row_idx < len(prepared_steps) else None
    legal_actions = () if policy_input.legal_actions is None else policy_input.legal_actions[row_idx]
    student_idx = int(final_top1[row_idx].item())
    parent_idx = int(parent_top1[row_idx].item())
    source_idx = int(source_top1[row_idx].item())
    prior_idx = int(prior_top1[row_idx].item())
    teacher_idx = _teacher_top1_legal_index(teacher_batch, row_idx)
    state = _review_state_context(step)
    before_type = _action_type_name(legal_actions, parent_idx)
    after_type = _action_type_name(legal_actions, student_idx)
    teacher_type = _action_type_name(legal_actions, teacher_idx)
    return {
        "row_id": int(row_idx),
        "episode_id": "" if step is None else step.episode_id,
        "step_id": "" if step is None else int(step.step_id),
        "kyoku": state.get("kyoku", ""),
        "honba": state.get("honba", ""),
        "actor": "" if step is None else int(step.actor),
        "action_scope": _action_scope_for_legal_actions(legal_actions),
        "hand": state.get("hand_json", ""),
        "draw": state.get("draw", ""),
        "dora": state.get("dora_json", ""),
        "scores": state.get("scores_json", ""),
        "riichi_sticks": state.get("riichi_sticks", ""),
        "kyotaku": state.get("riichi_sticks", ""),
        "last_discard": state.get("last_discard_json", ""),
        "legal_actions": _json_dumps([_action_display(action) for action in legal_actions]),
        "rule_prior_topk": _topk_action_labels(legal_actions, prior_logits[row_idx], k=5),
        "mortal_q_topk": _teacher_topk_action_labels(legal_actions, teacher_batch, row_idx, k=5),
        "student_before_topk": _topk_action_labels(legal_actions, parent_logits[row_idx], k=5),
        "student_after_topk": _topk_action_labels(legal_actions, final_logits[row_idx], k=5),
        "selected_before": _action_label(legal_actions, parent_idx),
        "selected_after": _action_label(legal_actions, student_idx),
        "selected_source": _action_label(legal_actions, source_idx),
        "teacher_top1": _action_label(legal_actions, teacher_idx),
        "teacher_top1_action_type": teacher_type,
        "student_top1": _action_label(legal_actions, student_idx),
        "rule_prior_top1": _action_label(legal_actions, prior_idx),
        "student_before_top1": _action_label(legal_actions, parent_idx),
        "student_after_top1": _action_label(legal_actions, student_idx),
        "changed_action_prior_rank": int(rank),
        "selected_changed": bool(selected_changed),
        "teacher_disagreed": bool(teacher_disagreed),
        "review_reason": str(review_reason),
        "changed_kind": _changed_kind(before_type, after_type),
        "changed_to_reach": bool(after_type == "REACH_DISCARD" and before_type != "REACH_DISCARD"),
        "changed_from_reach": bool(before_type == "REACH_DISCARD" and after_type != "REACH_DISCARD"),
        "changed_to_pass": bool(after_type == "PASS" and before_type != "PASS"),
        "changed_from_pass": bool(before_type == "PASS" and after_type != "PASS"),
        "changed_to_ron": bool(after_type == "RON" and before_type != "RON"),
        "changed_to_chi_or_pon": bool(after_type in {"CHI", "PON"} and before_type not in {"CHI", "PON"}),
        "changed_to_kan": bool(after_type in {"ANKAN", "KAKAN", "DAIMINKAN"} and before_type not in {"ANKAN", "KAKAN", "DAIMINKAN"}),
        "changed_from_kan": bool(before_type in {"ANKAN", "KAKAN", "DAIMINKAN"} and after_type not in {"ANKAN", "KAKAN", "DAIMINKAN"}),
        "events_tail": state.get("events_tail_json", ""),
    }


def _decision_review_case_row(
    policy_input: PolicyInput,
    prepared_steps: Sequence[Any],
    row_idx: int,
    *,
    final_top1: torch.Tensor,
    parent_top1: torch.Tensor,
    prior_top1: torch.Tensor,
    final_logits: torch.Tensor,
    parent_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    teacher_batch: MortalImitationTeacherBatch,
    selected_changed: bool,
    teacher_disagreed: bool,
    review_reason: str,
) -> dict[str, Any]:
    step = prepared_steps[row_idx] if row_idx < len(prepared_steps) else None
    legal_actions = () if policy_input.legal_actions is None else policy_input.legal_actions[row_idx]
    actor = int(getattr(step, "actor", 0)) if step is not None else 0
    episode_id = "" if step is None else str(getattr(step, "episode_id", ""))
    step_id = "" if step is None else int(getattr(step, "step_id", row_idx))
    snapshot, events = _review_snapshot_and_events(step, actor)
    teacher_idx = _teacher_top1_legal_index(teacher_batch, row_idx)
    student_idx = int(final_top1[row_idx].item())
    parent_idx = int(parent_top1[row_idx].item())
    prior_idx = int(prior_top1[row_idx].item())
    return {
        "case_id": f"iter_row{int(row_idx)}_ep{episode_id}_step{step_id}_actor{actor}",
        "episode_id": episode_id,
        "step_id": step_id,
        "row_id": int(row_idx),
        "actor": actor,
        "kyoku": snapshot.get("kyoku", ""),
        "honba": snapshot.get("honba", ""),
        "row_scope": _action_scope_for_legal_actions(legal_actions),
        "review_reason": str(review_reason),
        "selected_changed": bool(selected_changed),
        "teacher_disagreed": bool(teacher_disagreed),
        "selected_before": _action_label(legal_actions, parent_idx),
        "selected_after": _action_label(legal_actions, student_idx),
        "teacher_top1": _action_label(legal_actions, teacher_idx),
        "rulebase_top1": _action_label(legal_actions, prior_idx),
        "round_state": _review_round_state(snapshot, events),
        "players": _review_players(snapshot, actor),
        "legal_actions": _review_legal_action_table(
            legal_actions,
            row_idx=row_idx,
            actor=actor,
            teacher_idx=teacher_idx,
            parent_idx=parent_idx,
            student_idx=student_idx,
            prior_idx=prior_idx,
            final_logits=final_logits,
            parent_logits=parent_logits,
            prior_logits=prior_logits,
            teacher_batch=teacher_batch,
            legal_mask=policy_input.legal_action_mask[row_idx].bool(),
        ),
        "events_prefix": list(events),
        "events_tail": list(events[-12:]),
        "replay_hint": {
            "episode_id": episode_id,
            "step_id": step_id,
            "actor": actor,
        },
    }


def _imitation_action_type_breakdown(
    output: PolicyOutput,
    policy_input: PolicyInput,
    *,
    parent_output: PolicyOutput,
    prior_logits: torch.Tensor,
    teacher_batch: MortalImitationTeacherBatch,
    teacher_temperature: float,
    mapping_summary: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if policy_input.legal_actions is None:
        return []
    support_logits, support_prior, support_teacher, support_mask, support_indices = _teacher_support_tensors(
        output,
        policy_input,
        teacher_batch,
    )
    parent_support_logits = _student_imitation_logits(parent_output).to(device=support_logits.device).gather(
        1,
        support_indices.to(device=support_logits.device),
    )
    parent_support_logits = parent_support_logits.masked_fill(~support_mask, torch.finfo(torch.float32).min)
    teacher_probs = torch.softmax(support_teacher / float(teacher_temperature), dim=-1)
    teacher_log_probs = torch.log(teacher_probs.clamp_min(1e-12))
    policy_log_probs = torch.log_softmax(support_logits, dim=-1)
    per_row_ce = -(teacher_probs * policy_log_probs).sum(dim=-1)
    per_row_kl = (teacher_probs * (teacher_log_probs - policy_log_probs)).sum(dim=-1)
    per_row_entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
    per_row_margin = _teacher_margin_from_scores(support_teacher, support_mask=support_mask)
    teacher_argmax = support_teacher.argmax(dim=-1)
    policy_argmax = support_logits.argmax(dim=-1)
    parent_argmax = parent_support_logits.argmax(dim=-1)
    prior_argmax = support_prior.argmax(dim=-1)
    row_valid = teacher_batch.row_valid_mask.to(device=support_logits.device).bool() & support_mask.any(dim=-1)
    buckets: dict[str, dict[str, Any]] = {}
    for row_idx, row_actions in enumerate(policy_input.legal_actions):
        if not bool(row_valid[row_idx].item()):
            continue
        support_pos = int(teacher_argmax[row_idx].item())
        legal_idx = int(support_indices[row_idx, support_pos].item())
        action_type = _action_type_name(row_actions, legal_idx)
        scope = _action_scope_for_legal_actions(row_actions)
        legal_types = ",".join(sorted({action.action_type.name for action in row_actions}))
        contains_flags = _legal_contains_flags(row_actions)
        bucket_key = (action_type, scope, legal_types, tuple(contains_flags))
        bucket = buckets.setdefault(
            str(bucket_key),
            {
                "action_type": action_type,
                "teacher_top1_action_type": action_type,
                "row_scope": scope,
                "legal_action_types": legal_types,
                **dict(flag.split("=", 1) for flag in contains_flags),
                "row_count": 0,
                "teacher_ce_values": [],
                "teacher_kl_values": [],
                "teacher_margin_values": [],
                "teacher_entropy_values": [],
                "teacher_policy_agree_count": 0,
                "teacher_prior_agree_count": 0,
                "policy_top1_vs_teacher_top1_count": 0,
                "top1_changed_vs_parent_count": 0,
                "rank_ge5_count": 0,
            },
        )
        bucket["row_count"] += 1
        bucket["teacher_ce_values"].append(float(per_row_ce[row_idx].detach().cpu().item()))
        bucket["teacher_kl_values"].append(float(per_row_kl[row_idx].detach().cpu().item()))
        bucket["teacher_margin_values"].append(float(per_row_margin[row_idx].detach().cpu().item()))
        bucket["teacher_entropy_values"].append(float(per_row_entropy[row_idx].detach().cpu().item()))
        policy_matches_teacher = bool(policy_argmax[row_idx].item() == teacher_argmax[row_idx].item())
        prior_matches_teacher = bool(prior_argmax[row_idx].item() == teacher_argmax[row_idx].item())
        changed_vs_parent = bool(policy_argmax[row_idx].item() != parent_argmax[row_idx].item())
        bucket["teacher_policy_agree_count"] += int(policy_matches_teacher)
        bucket["teacher_prior_agree_count"] += int(prior_matches_teacher)
        bucket["policy_top1_vs_teacher_top1_count"] += int(policy_matches_teacher)
        bucket["top1_changed_vs_parent_count"] += int(changed_vs_parent)
        final_legal_idx = int(support_indices[row_idx, int(policy_argmax[row_idx].item())].item())
        rank = _prior_rank_for_index(policy_input, prior_logits, row_idx, final_legal_idx)
        bucket["rank_ge5_count"] += int(rank >= 5)

    rows: list[dict[str, Any]] = []
    for bucket_key in sorted(buckets):
        bucket = buckets[bucket_key]
        row_count = int(bucket["row_count"])
        rows.append(
            {
                "action_type": bucket["action_type"],
                "teacher_top1_action_type": bucket["teacher_top1_action_type"],
                "row_scope": bucket["row_scope"],
                "legal_action_types": bucket["legal_action_types"],
                "contains_reach": bucket["contains_reach"],
                "contains_kan": bucket["contains_kan"],
                "contains_call": bucket["contains_call"],
                "contains_terminal": bucket["contains_terminal"],
                "row_count": row_count,
                "teacher_ce": _mean_float(bucket["teacher_ce_values"]),
                "teacher_kl": _mean_float(bucket["teacher_kl_values"]),
                "teacher_agreement": bucket["teacher_policy_agree_count"] / max(1, row_count),
                "teacher_prior_agreement": bucket["teacher_prior_agree_count"] / max(1, row_count),
                "teacher_margin_mean": _mean_float(bucket["teacher_margin_values"]),
                "teacher_entropy_mean": _mean_float(bucket["teacher_entropy_values"]),
                "policy_top1_vs_teacher_top1_rate": bucket["policy_top1_vs_teacher_top1_count"] / max(1, row_count),
                "top1_changed_vs_parent_rate": bucket["top1_changed_vs_parent_count"] / max(1, row_count),
                "rank_ge5_rate": bucket["rank_ge5_count"] / max(1, row_count),
                "mapping_available_rate": float(mapping_summary.get("mapping_available_rate", 0.0)),
                "mapping_row_count": int(mapping_summary.get("mapping_row_count", 0)),
                "mapping_available_count": int(mapping_summary.get("mapping_available_count", 0)),
                "fail_closed_count": int(mapping_summary.get("fail_closed_count", 0)),
                "teacher_row_valid_count": int(mapping_summary.get("teacher_row_valid_count", 0)),
                "teacher_row_valid_rate": float(mapping_summary.get("teacher_row_valid_rate", 0.0)),
                "teacher_row_invalid_count": int(mapping_summary.get("teacher_row_invalid_count", 0)),
            }
        )
    return rows


def _teacher_support_tensors(
    output: PolicyOutput,
    policy_input: PolicyInput,
    teacher_batch: MortalImitationTeacherBatch,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.BoolTensor, torch.LongTensor]:
    student_logits = _student_imitation_logits(output)
    device = student_logits.device
    if teacher_batch.teacher_support in {"topk", "adaptive-topk"}:
        if teacher_batch.topk_indices is None:
            raise MortalTeacherMappingError("cached topK teacher batch is missing topk_indices")
        indices = teacher_batch.topk_indices.to(device=device)
        support_logits = student_logits.gather(1, indices)
        support_prior = teacher_batch.prior_logits.to(device=device).float()
        support_teacher = teacher_batch.teacher_scores.to(device=device).float()
        support_mask = (
            torch.ones_like(support_teacher, dtype=torch.bool, device=device)
            if teacher_batch.support_mask is None
            else teacher_batch.support_mask.to(device=device).bool()
        )
        min_value = torch.finfo(torch.float32).min
        support_logits = support_logits.masked_fill(~support_mask, min_value)
        support_prior = support_prior.masked_fill(~support_mask, min_value)
        support_teacher = support_teacher.masked_fill(~support_mask, min_value)
        return support_logits, support_prior, support_teacher, support_mask, indices
    if teacher_batch.teacher_support == "full-legal":
        if teacher_batch.legal_action_mask is None:
            raise MortalTeacherMappingError("cached full-legal teacher batch is missing legal_action_mask")
        width = int(student_logits.shape[1])
        indices = torch.arange(width, dtype=torch.long, device=device).unsqueeze(0).repeat(student_logits.shape[0], 1)
        support_mask = teacher_batch.legal_action_mask.to(device=device).bool()
        support_logits = student_logits.masked_fill(~support_mask, torch.finfo(torch.float32).min)
        support_prior = teacher_batch.prior_logits.to(device=device).float()
        support_teacher = teacher_batch.teacher_scores.to(device=device).float()
        return support_logits, support_prior, support_teacher, support_mask, indices
    raise ValueError(f"unsupported teacher support: {teacher_batch.teacher_support}")


def _teacher_top1_legal_index(teacher_batch: MortalImitationTeacherBatch, row_idx: int) -> int:
    teacher_scores = teacher_batch.teacher_scores[row_idx].float()
    if teacher_batch.support_mask is not None:
        mask = teacher_batch.support_mask[row_idx].bool().to(device=teacher_scores.device)
        teacher_scores = teacher_scores.masked_fill(~mask, torch.finfo(torch.float32).min)
    support_pos = int(teacher_scores.argmax().detach().cpu().item())
    if teacher_batch.topk_indices is not None:
        return int(teacher_batch.topk_indices[row_idx, support_pos].detach().cpu().item())
    return support_pos


def _teacher_topk_action_labels(
    legal_actions: Sequence[ActionSpec],
    teacher_batch: MortalImitationTeacherBatch,
    row_idx: int,
    *,
    k: int,
) -> str:
    scores = teacher_batch.teacher_scores[row_idx].detach().cpu()
    if teacher_batch.support_mask is not None:
        mask = teacher_batch.support_mask[row_idx].detach().cpu().bool()
        scores = scores.float().masked_fill(~mask, torch.finfo(torch.float32).min)
    order = torch.argsort(scores, descending=True).tolist()
    labels: list[dict[str, Any]] = []
    for support_pos in order[: int(k)]:
        if teacher_batch.support_mask is not None and not bool(teacher_batch.support_mask[row_idx, support_pos].detach().cpu().item()):
            continue
        legal_idx = (
            int(teacher_batch.topk_indices[row_idx, support_pos].detach().cpu().item())
            if teacher_batch.topk_indices is not None
            else int(support_pos)
        )
        labels.append(
            {
                "index": legal_idx,
                "score": float(scores[support_pos].item()),
                "action": _action_display(legal_actions[legal_idx]) if 0 <= legal_idx < len(legal_actions) else f"index:{legal_idx}",
            }
        )
    return _json_dumps(labels)


def _topk_action_labels(legal_actions: Sequence[ActionSpec], logits: torch.Tensor, *, k: int) -> str:
    width = min(int(logits.shape[0]), len(legal_actions))
    if width <= 0:
        return "[]"
    values = logits[:width].detach().cpu()
    order = torch.argsort(values, descending=True).tolist()
    labels = [
        {
            "index": int(index),
            "score": float(values[index].item()),
            "action": _action_display(legal_actions[int(index)]),
        }
        for index in order[: int(k)]
    ]
    return _json_dumps(labels)


def _review_legal_action_table(
    legal_actions: Sequence[ActionSpec],
    *,
    row_idx: int,
    actor: int,
    teacher_idx: int,
    parent_idx: int,
    student_idx: int,
    prior_idx: int,
    final_logits: torch.Tensor,
    parent_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    teacher_batch: MortalImitationTeacherBatch,
    legal_mask: torch.Tensor,
) -> list[dict[str, Any]]:
    width = min(len(legal_actions), int(final_logits.shape[1]))
    teacher_scores = _teacher_legal_score_row(teacher_batch, row_idx, width)
    teacher_probs = _teacher_legal_prob_map(teacher_batch, row_idx)
    rule_ranks = _rank_map(prior_logits[row_idx], legal_mask, width)
    parent_ranks = _rank_map(parent_logits[row_idx], legal_mask, width)
    student_ranks = _rank_map(final_logits[row_idx], legal_mask, width)
    teacher_ranks = _rank_map(teacher_scores, legal_mask, width)
    rows: list[dict[str, Any]] = []
    for index, action in enumerate(legal_actions[:width]):
        rows.append(
            {
                "index": index,
                "type": action.action_type.name,
                "pai": _action_tile_label(action),
                "canonical_key": action.canonical_key,
                "mjai_events": _action_mjai_events(action, actor),
                "rulebase_score": _finite_float(prior_logits[row_idx, index]),
                "mortal_q": _finite_float(teacher_scores[index]) if index < int(teacher_scores.shape[0]) else None,
                "student_before_score": _finite_float(parent_logits[row_idx, index]),
                "student_after_score": _finite_float(final_logits[row_idx, index]),
                "teacher_prob": teacher_probs.get(index),
                "teacher_rank": teacher_ranks.get(index),
                "rulebase_rank": rule_ranks.get(index),
                "student_before_rank": parent_ranks.get(index),
                "student_after_rank": student_ranks.get(index),
                "is_rulebase_top1": index == int(prior_idx),
                "is_mortal_top1": index == int(teacher_idx),
                "is_student_before_top1": index == int(parent_idx),
                "is_student_after_top1": index == int(student_idx),
            }
        )
    return rows


def _teacher_legal_score_row(teacher_batch: MortalImitationTeacherBatch, row_idx: int, width: int) -> torch.Tensor:
    if teacher_batch.mapped_legal_scores is not None:
        return teacher_batch.mapped_legal_scores[row_idx, :width].detach().cpu().float()
    scores = torch.full((int(width),), float("-inf"), dtype=torch.float32)
    teacher_scores = teacher_batch.teacher_scores[row_idx].detach().cpu().float()
    if teacher_batch.support_mask is not None:
        support_mask = teacher_batch.support_mask[row_idx].detach().cpu().bool()
        teacher_scores = teacher_scores.masked_fill(~support_mask, torch.finfo(torch.float32).min)
    for support_pos, score in enumerate(teacher_scores.tolist()):
        legal_idx = (
            int(teacher_batch.topk_indices[row_idx, support_pos].detach().cpu().item())
            if teacher_batch.topk_indices is not None
            else int(support_pos)
        )
        if 0 <= legal_idx < int(width):
            scores[legal_idx] = float(score)
    return scores


def _teacher_legal_prob_map(teacher_batch: MortalImitationTeacherBatch, row_idx: int) -> dict[int, float]:
    scores = teacher_batch.teacher_scores[row_idx].detach().cpu().float()
    if teacher_batch.support_mask is not None:
        mask = teacher_batch.support_mask[row_idx].detach().cpu().bool()
    elif teacher_batch.legal_action_mask is not None:
        mask = teacher_batch.legal_action_mask[row_idx, : int(scores.shape[0])].detach().cpu().bool()
    else:
        mask = torch.ones_like(scores, dtype=torch.bool)
    if not bool(mask.any().item()):
        return {}
    probs = torch.softmax(scores.masked_fill(~mask, torch.finfo(torch.float32).min), dim=-1)
    result: dict[int, float] = {}
    for support_pos in torch.nonzero(mask, as_tuple=False).flatten().tolist():
        legal_idx = (
            int(teacher_batch.topk_indices[row_idx, support_pos].detach().cpu().item())
            if teacher_batch.topk_indices is not None
            else int(support_pos)
        )
        result[legal_idx] = float(probs[int(support_pos)].item())
    return result


def _rank_map(values: torch.Tensor, legal_mask: torch.Tensor, width: int) -> dict[int, int]:
    width = min(int(width), int(values.shape[0]), int(legal_mask.shape[0]))
    scores = values[:width].detach().cpu().float()
    mask = legal_mask[:width].detach().cpu().bool() & torch.isfinite(scores)
    order = [
        int(index)
        for index in torch.argsort(scores.masked_fill(~mask, torch.finfo(torch.float32).min), descending=True).tolist()
        if bool(mask[int(index)].item())
    ]
    return {index: rank for rank, index in enumerate(order, start=1)}


def _finite_float(value: torch.Tensor | float | int) -> float | None:
    result = _scalar(value)
    if result == float("inf") or result == float("-inf") or result != result:
        return None
    return result


def _prior_rank_for_index(
    policy_input: PolicyInput,
    prior_logits: torch.Tensor,
    row_idx: int,
    action_index: int,
) -> int:
    mask = policy_input.legal_action_mask[row_idx].bool().to(device=prior_logits.device)
    if int(action_index) < 0 or int(action_index) >= int(prior_logits.shape[1]):
        return 999
    row_prior = prior_logits[row_idx]
    return 1 + int((row_prior[mask] > row_prior[int(action_index)]).sum().detach().cpu().item())


def _review_state_context(step: Any | None) -> dict[str, Any]:
    if step is None:
        return {}
    events = tuple(getattr(step, "mortal_teacher_events", ()))
    actor = int(getattr(step, "actor", 0))
    context: dict[str, Any] = {
        "events_tail_json": _json_dumps(events[-8:]),
    }
    if not events:
        return context
    try:
        snapshot = keqing_core.replay_state_snapshot(events, actor)
    except Exception as exc:  # pragma: no cover - diagnostic best effort.
        context["state_error"] = str(exc)
        return context
    context.update(
        {
            "kyoku": snapshot.get("kyoku", ""),
            "honba": snapshot.get("honba", ""),
            "riichi_sticks": snapshot.get("kyotaku", ""),
            "hand_json": _json_dumps(snapshot.get("hand", ())),
            "draw": snapshot.get("tsumo_pai") or snapshot.get("drawn_tile") or "",
            "dora_json": _json_dumps(snapshot.get("dora_markers", ())),
            "scores_json": _json_dumps(snapshot.get("scores", ())),
            "last_discard_json": _json_dumps(snapshot.get("last_discard")),
        }
    )
    return context


def _review_snapshot_and_events(step: Any | None, actor: int) -> tuple[dict[str, Any], tuple[Any, ...]]:
    if step is None:
        return {}, ()
    events = tuple(getattr(step, "mortal_teacher_events", ()))
    if not events:
        return {}, events
    try:
        snapshot = keqing_core.replay_state_snapshot(events, int(actor))
    except Exception:
        return {}, events
    return dict(snapshot), events


def _review_round_state(snapshot: Mapping[str, Any], events: Sequence[Any]) -> dict[str, Any]:
    return {
        "bakaze": snapshot.get("bakaze", snapshot.get("bakaze_pai", "")),
        "kyoku": snapshot.get("kyoku", ""),
        "honba": snapshot.get("honba", ""),
        "riichi_sticks": snapshot.get("kyotaku", snapshot.get("riichi_sticks", "")),
        "dora_indicators": _tile_list_labels(snapshot.get("dora_markers", snapshot.get("dora_indicators", ()))),
        "scores": snapshot.get("scores", ()),
        "wall_remaining": snapshot.get("wall_remaining", snapshot.get("remaining_wall", "")),
        "turn_index": max(0, len(events) - 1),
        "last_discard": _discard_entry(snapshot.get("last_discard")),
    }


def _review_players(snapshot: Mapping[str, Any], actor: int) -> list[dict[str, Any]]:
    players: list[dict[str, Any]] = []
    hands = snapshot.get("hands", ())
    discards = snapshot.get("discards", ())
    melds = snapshot.get("melds", ())
    riichi = snapshot.get("riichi", snapshot.get("reached", ()))
    winds = snapshot.get("winds", snapshot.get("jikaze", ()))
    scores = snapshot.get("scores", ())
    actor_hand = snapshot.get("hand", ())
    for seat in range(4):
        players.append(
            {
                "seat": seat,
                "is_actor": seat == int(actor),
                "hand": _tile_list_labels(_indexed_or_empty(hands, seat) or (actor_hand if seat == int(actor) else ())),
                "draw": snapshot.get("tsumo_pai") or snapshot.get("drawn_tile") or "",
                "discards": [_discard_entry(discard) for discard in _list_or_empty(_indexed_or_empty(discards, seat))],
                "melds": [_meld_entry(meld) for meld in _list_or_empty(_indexed_or_empty(melds, seat))],
                "riichi": bool(_indexed_or_empty(riichi, seat)),
                "wind": _indexed_or_empty(winds, seat),
                "score": _indexed_or_empty(scores, seat),
            }
        )
    return players


def _indexed_or_empty(values: Any, index: int) -> Any:
    if isinstance(values, Mapping):
        return values.get(index, values.get(str(index), ()))
    if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) and 0 <= int(index) < len(values):
        return values[int(index)]
    return ()


def _list_or_empty(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _tile_list_labels(values: Any) -> list[str]:
    return [_tile_label_value(value) for value in _list_or_empty(values)]


def _discard_entry(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if isinstance(value, Mapping):
        return {
            "pai": _tile_label_value(value.get("pai", value.get("tile", ""))),
            "tsumogiri": bool(value.get("tsumogiri", False)),
            "reach_declared": bool(value.get("reach_declared", value.get("reach", False))),
        }
    return {"pai": _tile_label_value(value), "tsumogiri": False, "reach_declared": False}


def _meld_entry(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return {
            "type": value.get("type", value.get("kind", "")),
            "pai": _tile_label_value(value.get("pai", value.get("tile", ""))),
            "consumed": _tile_list_labels(value.get("consumed", ())),
            "from": value.get("from", value.get("target", "")),
        }
    return {"type": "", "pai": "", "consumed": _tile_list_labels(value), "from": ""}


def _tile_label_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, Mapping):
        return _tile_label_value(value.get("pai", value.get("tile", value.get("name", ""))))
    if isinstance(value, int):
        return str(IDX_TO_TILE_NAME.get(int(value), value))
    text = str(value)
    return text if text != "None" else ""


def _action_scope_for_legal_actions(legal_actions: Sequence[ActionSpec]) -> str:
    response_types = {ActionType.PASS, ActionType.RON, ActionType.PON, ActionType.CHI, ActionType.DAIMINKAN}
    if any(action.action_type in response_types for action in legal_actions):
        return "response"
    return "self-turn"


def _legal_contains_flags(legal_actions: Sequence[ActionSpec]) -> tuple[str, ...]:
    action_types = {action.action_type for action in legal_actions}
    contains_reach = ActionType.REACH_DISCARD in action_types
    contains_kan = bool({ActionType.ANKAN, ActionType.KAKAN, ActionType.DAIMINKAN} & action_types)
    contains_call = bool({ActionType.CHI, ActionType.PON, ActionType.DAIMINKAN} & action_types)
    contains_terminal = bool({ActionType.TSUMO, ActionType.RON, ActionType.RYUKYOKU} & action_types)
    return (
        f"contains_reach={str(contains_reach).lower()}",
        f"contains_kan={str(contains_kan).lower()}",
        f"contains_call={str(contains_call).lower()}",
        f"contains_terminal={str(contains_terminal).lower()}",
    )


def _action_type_name(legal_actions: Sequence[ActionSpec], index: int) -> str:
    if int(index) < 0 or int(index) >= len(legal_actions):
        return "UNKNOWN"
    return legal_actions[int(index)].action_type.name


def _changed_kind(before_type: str, after_type: str) -> str:
    if before_type == after_type:
        return f"same_type:{after_type}"
    return f"{before_type.lower()}_to_{after_type.lower()}"


def _review_reason(
    policy_input: PolicyInput,
    row_idx: int,
    *,
    final_index: int,
    parent_index: int,
    teacher_index: int,
    selected_changed: bool,
    teacher_disagreed: bool,
) -> str:
    legal_actions = () if policy_input.legal_actions is None else policy_input.legal_actions[row_idx]
    reasons: list[str] = []
    if selected_changed:
        reasons.append("selected_changed")
    if teacher_disagreed:
        reasons.append("teacher_disagreement")
    involved = {
        _action_type_name(legal_actions, final_index),
        _action_type_name(legal_actions, parent_index),
        _action_type_name(legal_actions, teacher_index),
    }
    if "REACH_DISCARD" in involved:
        reasons.append("reach_related")
    if involved & {"ANKAN", "KAKAN", "DAIMINKAN"}:
        reasons.append("kan_related")
    if involved & {"CHI", "PON", "DAIMINKAN"}:
        reasons.append("call_related")
    if not reasons:
        reasons.append("diagnostic")
    return ",".join(dict.fromkeys(reasons))


def _action_display(action: ActionSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "type": action.action_type.name,
        "canonical_key": action.canonical_key,
    }
    try:
        payload["mjai_events"] = action.to_mjai_events(actor=0)
    except Exception:
        try:
            payload["mjai"] = action.to_mjai_action(actor=0)
        except Exception as exc:  # pragma: no cover - diagnostic best effort.
            payload["error"] = str(exc)
    return payload


def _action_mjai_events(action: ActionSpec, actor: int) -> list[dict[str, Any]]:
    try:
        return list(action.to_mjai_events(actor=int(actor)))
    except Exception:
        try:
            return [action.to_mjai_action(actor=int(actor))]
        except Exception:
            return []


def _action_tile_label(action: ActionSpec) -> str:
    tile = getattr(action, "tile", None)
    if tile is None:
        consumed = getattr(action, "consumed", None)
        if consumed:
            tile = consumed[0]
    if tile is None:
        return ""
    return str(IDX_TO_TILE_NAME.get(int(tile), tile))


def _unique_source_topk_indices(
    prior_logits: torch.Tensor,
    legal_action_mask: torch.Tensor,
    legal_actions: Sequence[Sequence[ActionSpec]],
    *,
    k: int,
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    indices, valid_rows, _support_mask, _distinct_counts = _source_topk_indices(
        prior_logits,
        legal_action_mask,
        legal_actions,
        k=int(k),
        adaptive=False,
    )
    return indices, valid_rows


def _source_topk_indices(
    prior_logits: torch.Tensor,
    legal_action_mask: torch.Tensor,
    legal_actions: Sequence[Sequence[ActionSpec]],
    *,
    k: int,
    adaptive: bool,
) -> tuple[torch.LongTensor, torch.BoolTensor, torch.BoolTensor, torch.LongTensor]:
    from keqingrl.mortal_teacher import mortal_action_ids_for_action_spec

    rows: list[torch.Tensor] = []
    valid_rows: list[bool] = []
    support_masks: list[torch.Tensor] = []
    distinct_counts: list[int] = []
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
        distinct_count = len(selected)
        valid = distinct_count > 0 if bool(adaptive) else distinct_count == int(k)
        if not selected:
            selected = [0]
        support = [True] * len(selected)
        while len(selected) < int(k):
            selected.append(selected[-1])
            support.append(False)
        rows.append(torch.tensor(selected[: int(k)], dtype=torch.long, device=prior_logits.device))
        support_masks.append(torch.tensor(support[: int(k)], dtype=torch.bool, device=prior_logits.device))
        valid_rows.append(valid)
        distinct_counts.append(distinct_count)
    return (
        torch.stack(rows, dim=0),
        torch.tensor(valid_rows, dtype=torch.bool, device=prior_logits.device),
        torch.stack(support_masks, dim=0),
        torch.tensor(distinct_counts, dtype=torch.long, device=prior_logits.device),
    )


def _finite_teacher_topk_indices(
    prior_logits: torch.Tensor,
    legal_action_mask: torch.Tensor,
    teacher_scores: torch.Tensor,
    legal_actions: Sequence[Sequence[ActionSpec]],
    *,
    k: int,
) -> tuple[torch.LongTensor, torch.BoolTensor]:
    rows: list[torch.Tensor] = []
    valid_rows: list[bool] = []
    masked_prior = prior_logits.float().masked_fill(~legal_action_mask.bool(), torch.finfo(torch.float32).min)
    finite_teacher = torch.isfinite(teacher_scores.float())
    for row_idx, row_actions in enumerate(legal_actions):
        legal_indices = [
            index
            for index in range(len(row_actions))
            if index < int(legal_action_mask.shape[1])
            and bool(legal_action_mask[row_idx, index].item())
            and bool(finite_teacher[row_idx, index].item())
        ]
        sorted_indices = sorted(
            legal_indices,
            key=lambda index: float(masked_prior[row_idx, index].detach().cpu()),
            reverse=True,
        )
        selected: list[int] = []
        seen: set[str] = set()
        for index in sorted_indices:
            source = row_actions[index].canonical_key
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
            "teacher_target_type": (
                "full_legal_distribution"
                if args.teacher_support == "full-legal"
                else f"{args.teacher_support}_distribution"
            ),
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
        "teacher_row_valid_count": int(summary_row.get("teacher_row_valid_count", 0)),
        "teacher_row_valid_rate": float(summary_row.get("teacher_row_valid_rate", 0.0)),
        "teacher_row_invalid_count": int(summary_row.get("teacher_row_invalid_count", 0)),
        "fail_closed_count": int(summary_row["fail_closed_count"]),
        "rule_score_scale": float(args.rule_score_scale),
        "rule_score_scale_version": RULE_SCORE_SCALE_VERSION,
    }


def _write_outputs(
    args: argparse.Namespace,
    summary_rows: Sequence[dict[str, Any]],
    iteration_rows: Sequence[dict[str, Any]],
    action_type_breakdown_rows: Sequence[dict[str, Any]],
    audit_rows: Sequence[dict[str, Any]],
    changed_rows: Sequence[dict[str, Any]],
    teacher_disagreement_rows: Sequence[dict[str, Any]],
    decision_review_rows: Sequence[dict[str, Any]],
    decision_review_case_rows: Sequence[dict[str, Any]],
    checkpoint_rows: Sequence[dict[str, Any]],
) -> None:
    _write_csv(args.output_dir / "imitation_summary.csv", summary_rows)
    _write_csv(args.output_dir / "imitation_iterations.csv", iteration_rows)
    _write_csv(args.output_dir / "imitation_action_type_breakdown.csv", action_type_breakdown_rows)
    _write_csv(args.output_dir / "mortal_action_mapping_audit.csv", audit_rows)
    _write_jsonl(args.output_dir / "mortal_action_mapping_examples.jsonl", audit_rows)
    _write_csv(args.output_dir / "changed_decisions.csv", changed_rows)
    _write_csv(args.output_dir / "teacher_disagreements.csv", teacher_disagreement_rows)
    _write_csv(args.output_dir / "decision_review_candidates.csv", decision_review_rows)
    if bool(getattr(args, "export_decision_review_cases", False)):
        _write_jsonl(args.output_dir / "decision_review_cases.jsonl", decision_review_case_rows)
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
            "action_type_breakdown": action_type_breakdown_rows,
            "changed_decisions": changed_rows,
            "teacher_disagreements": teacher_disagreement_rows,
            "decision_review_candidates": decision_review_rows,
            "decision_review_cases": decision_review_case_rows,
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
    action_type_breakdown_rows: Sequence[dict[str, Any]],
    audit_rows: Sequence[dict[str, Any]],
    changed_rows: Sequence[dict[str, Any]],
    teacher_disagreement_rows: Sequence[dict[str, Any]],
    decision_review_rows: Sequence[dict[str, Any]],
    decision_review_case_rows: Sequence[dict[str, Any]],
    checkpoint_rows: Sequence[dict[str, Any]],
) -> None:
    """Persist per-iteration artifacts so interrupted runs remain inspectable."""
    _write_csv(args.output_dir / "imitation_iterations.csv", iteration_rows)
    _write_csv(args.output_dir / "imitation_action_type_breakdown.csv", action_type_breakdown_rows)
    _write_csv(args.output_dir / "mortal_action_mapping_audit.csv", audit_rows)
    _write_jsonl(args.output_dir / "mortal_action_mapping_examples.jsonl", audit_rows)
    _write_csv(args.output_dir / "changed_decisions.csv", changed_rows)
    _write_csv(args.output_dir / "teacher_disagreements.csv", teacher_disagreement_rows)
    _write_csv(args.output_dir / "decision_review_candidates.csv", decision_review_rows)
    if bool(getattr(args, "export_decision_review_cases", False)):
        _write_jsonl(args.output_dir / "decision_review_cases.jsonl", decision_review_case_rows)
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
            "action_type_breakdown": action_type_breakdown_rows,
            "changed_decisions": changed_rows,
            "teacher_disagreements": teacher_disagreement_rows,
            "decision_review_candidates": decision_review_rows,
            "decision_review_cases": decision_review_case_rows,
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
            f"teacher_valid={row.get('teacher_row_valid_count', 0)}/{row['mapping_row_count']} "
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
    for index, row in enumerate(rows[:200], start=1):
        lines.extend(
            [
                f"## Row {index}",
                "",
                (
                    f"- iter={row.get('iteration')} episode={row.get('episode_id')} "
                    f"step={row.get('step_id')} actor={row.get('actor')} "
                    f"kyoku={row.get('kyoku')} honba={row.get('honba')} "
                    f"scope={row.get('action_scope')} kind={row.get('changed_kind')}"
                ),
                (
                    f"- selected_before=`{row.get('selected_before')}` "
                    f"selected_after=`{row.get('selected_after')}` "
                    f"teacher_top1=`{row.get('teacher_top1')}` "
                    f"rank={row.get('changed_action_prior_rank')}"
                ),
                (
                    f"- flags: reach +{row.get('changed_to_reach')} -{row.get('changed_from_reach')} "
                    f"pass +{row.get('changed_to_pass')} -{row.get('changed_from_pass')} "
                    f"ron +{row.get('changed_to_ron')} "
                    f"call +{row.get('changed_to_chi_or_pon')} "
                    f"kan +{row.get('changed_to_kan')} -{row.get('changed_from_kan')}"
                ),
                f"- hand: `{row.get('hand')}` draw: `{row.get('draw')}` dora: `{row.get('dora')}`",
                f"- scores: `{row.get('scores')}` riichi_sticks={row.get('riichi_sticks')} last_discard: `{row.get('last_discard')}`",
                f"- rule_prior_topk: `{row.get('rule_prior_topk')}`",
                f"- mortal_q_topk: `{row.get('mortal_q_topk')}`",
                f"- student_before_topk: `{row.get('student_before_topk')}`",
                f"- student_after_topk: `{row.get('student_after_topk')}`",
                f"- legal_actions: `{row.get('legal_actions')}`",
                "",
            ]
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
        "mortal_teacher_events": tuple(getattr(step, "mortal_teacher_events", ())),
        "events_tail": tuple(getattr(step, "mortal_teacher_events", ()))[-8:],
        "mortal_events_tail": tuple(getattr(step, "mortal_teacher_events", ()))[-8:],
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
