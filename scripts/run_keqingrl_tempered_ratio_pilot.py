#!/usr/bin/env python3
"""Tempered-ratio PPO diagnostic for KeqingRL discard-only research."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch
import torch.nn.functional as F

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from keqingrl import DiscardOnlyMahjongEnv, run_fixed_seed_evaluation_smoke
from keqingrl.contracts import PolicyInput, PolicyOutput
from keqingrl.distribution import MaskedCategorical
from keqingrl.learning_signal import batch_diagnostic_rows, seed_registry_hash
from keqingrl.metadata import (
    RULE_SCORE_SCALE_VERSION,
    default_checkpoint_metadata,
    validate_checkpoint_metadata,
)
from keqingrl.policy import InteractivePolicy
from keqingrl.ppo import PPOLossBreakdown, compute_ppo_loss, validate_ppo_batch_rule_score_scale
from keqingrl.rule_score import smoothed_prior_probs
from keqingrl.selfplay import build_episodes_ppo_batch, collect_selfplay_episodes
from scripts.probe_keqingrl_sampling_diversity import (
    TemperaturePolicy,
    _candidate_summary,
    _load_candidates,
    _load_policy,
    _opponent_pool,
    _sampling_summary,
    _write_csv,
    _write_json,
)
from scripts.run_keqingrl_fixed_online_bridge import _file_sha256, _loss_float, _policy_delta_stats
from scripts.run_keqingrl_discard_research_sweep import _stable_json_hash, _to_jsonable
from scripts.run_keqingrl_temperature_pilot import (
    _advantage_audit_rows,
    _eval_scope,
    _eval_seed_registry,
    _eval_seed_registry_id,
    _iteration_seed,
    _iteration_seed_registry,
    _iteration_seed_registry_id,
    _iteration_torch_seed,
    _mean,
    _seed_torch_sampling,
)


_ACTOR_UPDATE_SUPPORT_MODES = (
    "all",
    "topk",
    "weak-margin",
    "topk-or-weak-margin",
    "topk-and-weak-margin",
)

_DELTA_SUPPORT_MODES = _ACTOR_UPDATE_SUPPORT_MODES
_OUTSIDE_SUPPORT_DELTA_MODES = ("zero", "negative-clip")
_SUPPORT_POLICY_MODES = ("unrestricted", "delta-topk-zero", "support-only-topk")


class DeltaSupportProjectionPolicy(InteractivePolicy):
    """Diagnostic wrapper that constrains neural_delta support before action selection."""

    def __init__(
        self,
        base_policy: InteractivePolicy,
        *,
        support_mode: str,
        topk: int,
        margin_threshold: float,
        outside_support_delta_mode: str,
        support_policy_mode: str = "delta-projection",
    ) -> None:
        super().__init__()
        if support_mode not in _DELTA_SUPPORT_MODES:
            raise ValueError(f"unsupported delta support mode: {support_mode}")
        if outside_support_delta_mode not in _OUTSIDE_SUPPORT_DELTA_MODES:
            raise ValueError(f"unsupported outside-support delta mode: {outside_support_delta_mode}")
        if int(topk) <= 0:
            raise ValueError(f"delta_support_topk must be positive, got {topk}")
        if float(margin_threshold) < 0.0:
            raise ValueError(f"delta_support_margin_threshold must be non-negative, got {margin_threshold}")
        self.base_policy = base_policy
        self.support_mode = str(support_mode)
        self.topk = int(topk)
        self.margin_threshold = float(margin_threshold)
        self.outside_support_delta_mode = str(outside_support_delta_mode)
        self.support_policy_mode = str(support_policy_mode)
        self.rule_score_scale = float(getattr(base_policy, "rule_score_scale", 1.0))

    def forward(self, policy_input: PolicyInput) -> PolicyOutput:
        output = self.base_policy(policy_input)
        prior_logits = output.aux.get("prior_logits")
        if prior_logits is None:
            prior_logits = policy_input.prior_logits
        if prior_logits is None:
            if self.support_mode != "all":
                raise ValueError("delta support projection requires prior_logits")
            prior_logits = torch.zeros_like(output.action_logits)
        prior_logits = prior_logits.float()
        mask = policy_input.legal_action_mask.bool()
        unprojected_delta = output.aux.get("neural_delta")
        if unprojected_delta is None:
            unprojected_delta = output.action_logits.float() - self.rule_score_scale * prior_logits
        else:
            unprojected_delta = unprojected_delta.float()

        support_mask = _delta_support_mask(
            policy_input,
            prior_logits=prior_logits,
            support_mode=self.support_mode,
            topk=self.topk,
            margin_threshold=self.margin_threshold,
        )
        if self.outside_support_delta_mode == "zero":
            projected_delta = torch.where(support_mask, unprojected_delta, torch.zeros_like(unprojected_delta))
        elif self.outside_support_delta_mode == "negative-clip":
            projected_delta = torch.where(support_mask, unprojected_delta, unprojected_delta.clamp_max(0.0))
        else:
            raise ValueError(f"unsupported outside-support delta mode: {self.outside_support_delta_mode}")

        final_logits = self.rule_score_scale * prior_logits + projected_delta
        action_support_mask = mask
        if self.support_policy_mode == "support-only-topk":
            action_support_mask = support_mask
        final_logits = final_logits.masked_fill(~action_support_mask, torch.finfo(final_logits.dtype).min)
        entropy = MaskedCategorical(final_logits, mask).entropy()
        aux = dict(output.aux)
        aux["prior_logits"] = prior_logits
        aux["unprojected_neural_delta"] = unprojected_delta
        aux["unprojected_final_logits"] = output.aux.get("final_logits", output.action_logits)
        aux["neural_delta"] = projected_delta
        aux["final_logits"] = final_logits
        aux["delta_support_mask"] = support_mask
        aux["action_support_mask"] = action_support_mask
        aux["delta_support_mode"] = torch.tensor(0, device=final_logits.device, dtype=torch.int64)
        aux["delta_support_action_rate"] = support_mask.masked_select(mask).float().mean()
        aux["action_support_rate"] = action_support_mask.masked_select(mask).float().mean()
        aux["outside_support_delta_mode"] = torch.tensor(0, device=final_logits.device, dtype=torch.int64)
        return PolicyOutput(
            action_logits=final_logits,
            value=output.value,
            rank_logits=output.rank_logits,
            entropy=entropy,
            aux=aux,
            next_recurrent_state=output.next_recurrent_state,
        )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run tempered-ratio PPO diagnostic")
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--source-config-ids", type=int, nargs="+", default=(93,))
    parser.add_argument("--rerun-config-ids", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--temperatures", type=float, nargs="+", default=(1.25,))
    parser.add_argument("--rule-score-scales", type=float, nargs="+", default=(1.0,))
    parser.add_argument("--lrs", type=float, nargs="+", default=(1e-4, 3e-4))
    parser.add_argument("--clip-eps-values", type=float, nargs="+", default=(0.1, 0.2))
    parser.add_argument("--update-epochs", type=int, default=1)
    parser.add_argument("--update-epochs-values", type=int, nargs="+", default=None)
    parser.add_argument("--rule-kl-coef", type=float, default=0.001)
    parser.add_argument("--rule-kl-coef-values", type=float, nargs="+", default=None)
    parser.add_argument("--entropy-coef", type=float, default=0.005)
    parser.add_argument("--value-coef", type=float, default=0.0)
    parser.add_argument("--rank-coef", type=float, default=0.0)
    parser.add_argument("--delta-l2-coef-values", type=float, nargs="+", default=(0.0,))
    parser.add_argument("--delta-clip-values", type=float, nargs="+", default=(0.0,))
    parser.add_argument("--delta-clip-coef-values", type=float, nargs="+", default=(0.0,))
    parser.add_argument("--low-rank-flip-topk-values", type=int, nargs="+", default=(3,))
    parser.add_argument("--low-rank-flip-penalty-coef-values", type=float, nargs="+", default=(0.0,))
    parser.add_argument("--weak-margin-threshold-values", type=float, nargs="+", default=(0.75,))
    parser.add_argument("--weak-margin-flip-penalty-coef-values", type=float, nargs="+", default=(0.0,))
    parser.add_argument(
        "--support-policy-mode",
        "--support-policy-modes",
        dest="support_policy_modes",
        choices=_SUPPORT_POLICY_MODES,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--delta-support-mode",
        "--delta-support-modes",
        dest="delta_support_modes",
        choices=_DELTA_SUPPORT_MODES,
        nargs="+",
        default=("all",),
    )
    parser.add_argument(
        "--delta-support-topk",
        "--delta-support-topk-values",
        dest="delta_support_topk_values",
        type=int,
        nargs="+",
        default=(3,),
    )
    parser.add_argument(
        "--delta-support-margin-threshold",
        "--delta-support-margin-threshold-values",
        dest="delta_support_margin_threshold_values",
        type=float,
        nargs="+",
        default=(0.75,),
    )
    parser.add_argument(
        "--outside-support-delta-mode",
        "--outside-support-delta-modes",
        dest="outside_support_delta_modes",
        choices=_OUTSIDE_SUPPORT_DELTA_MODES,
        nargs="+",
        default=("zero",),
    )
    parser.add_argument(
        "--actor-update-support-mode",
        "--actor-update-support-modes",
        dest="actor_update_support_modes",
        choices=_ACTOR_UPDATE_SUPPORT_MODES,
        nargs="+",
        default=("all",),
    )
    parser.add_argument(
        "--actor-update-topk",
        "--actor-update-topk-values",
        dest="actor_update_topk_values",
        type=int,
        nargs="+",
        default=(3,),
    )
    parser.add_argument(
        "--actor-update-margin-threshold",
        "--actor-update-margin-threshold-values",
        dest="actor_update_margin_threshold_values",
        type=float,
        nargs="+",
        default=(0.75,),
    )
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--normalize-advantages", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed-base", type=int, default=202604300000)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--torch-seed-base", type=int, default=202604300000)
    parser.add_argument("--eval-seed-base", type=int, default=202604310000)
    parser.add_argument("--eval-episodes", type=int, default=16)
    parser.add_argument("--learner-seats", type=int, nargs="+", default=(0,))
    parser.add_argument("--eval-seat-rotation", type=int, nargs="+", default=(0,))
    parser.add_argument("--max-kyokus", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--pass-min-top1-changed", type=float, default=0.02)
    parser.add_argument("--pass-max-top1-changed", type=float, default=0.25)
    parser.add_argument("--pass-max-tempered-kl", type=float, default=0.03)
    parser.add_argument("--pass-max-tempered-clip", type=float, default=0.3)
    parser.add_argument("--pass-max-untempered-clip", type=float, default=0.8)
    parser.add_argument("--pass-max-eval-fourth", type=float, default=0.5)
    parser.add_argument("--pass-max-eval-deal-in", type=float, default=0.25)
    parser.add_argument("--adaptive-recovery-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--recovery-max-extra-epochs", type=int, default=0)
    parser.add_argument("--recovery-min-top1-changed", type=float, default=None)
    parser.add_argument("--recovery-max-top1-changed", type=float, default=None)
    parser.add_argument("--recovery-max-tempered-kl", type=float, default=None)
    parser.add_argument("--recovery-max-tempered-clip", type=float, default=None)
    parser.add_argument("--recovery-max-untempered-clip", type=float, default=None)
    parser.add_argument("--movement-quality-gate", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quality-train-min-top1-changed", type=float, default=None)
    parser.add_argument("--quality-train-max-top1-changed", type=float, default=0.15)
    parser.add_argument("--quality-max-changed-prior-rank-mean", type=float, default=3.0)
    parser.add_argument("--quality-max-rank-ge5-rate", type=float, default=0.10)
    parser.add_argument("--quality-max-prior-margin-p50", type=float, default=1.0)
    parser.add_argument("--fresh-validation-episodes", type=int, default=0)
    parser.add_argument("--fresh-validation-seed-base", type=int, default=None)
    parser.add_argument("--fresh-validation-min-top1-changed", type=float, default=0.01)
    parser.add_argument("--fresh-validation-max-top1-changed", type=float, default=0.10)
    parser.add_argument("--save-final-checkpoint", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device) if args.device is not None else torch.device("cpu")
    candidates = _load_candidates(args)
    summary_rows: list[dict[str, Any]] = []
    iteration_rows: list[dict[str, Any]] = []
    step_rows: list[dict[str, Any]] = []
    advantage_rows: list[dict[str, Any]] = []
    checkpoint_rows: list[dict[str, Any]] = []
    config_id = 0
    update_epochs_values = _update_epochs_values(args)
    rule_score_scales = _rule_score_scales(args)
    rule_kl_coef_values = _rule_kl_coef_values(args)
    delta_l2_coef_values = _nonnegative_float_values(
        args.delta_l2_coef_values,
        name="delta_l2_coef",
    )
    delta_clip_values = _nonnegative_float_values(
        args.delta_clip_values,
        name="delta_clip",
    )
    delta_clip_coef_values = _nonnegative_float_values(
        args.delta_clip_coef_values,
        name="delta_clip_coef",
    )
    low_rank_flip_topk_values = _positive_int_values(
        args.low_rank_flip_topk_values,
        name="low_rank_flip_topk",
    )
    low_rank_flip_penalty_coef_values = _nonnegative_float_values(
        args.low_rank_flip_penalty_coef_values,
        name="low_rank_flip_penalty_coef",
    )
    weak_margin_threshold_values = _nonnegative_float_values(
        args.weak_margin_threshold_values,
        name="weak_margin_threshold",
    )
    weak_margin_flip_penalty_coef_values = _nonnegative_float_values(
        args.weak_margin_flip_penalty_coef_values,
        name="weak_margin_flip_penalty_coef",
    )
    delta_support_configs = _support_policy_configs(args)
    actor_update_support_configs = _actor_update_support_configs(args)

    for candidate in candidates:
        source_policy = _load_policy(candidate, device)
        opponent_pool = _opponent_pool(str(candidate["opponent_mode"]))
        for rule_score_scale in rule_score_scales:
            for temperature in args.temperatures:
                for lr in args.lrs:
                    for update_epochs in update_epochs_values:
                        for clip_eps in args.clip_eps_values:
                            for rule_kl_coef in rule_kl_coef_values:
                                for delta_l2_coef in delta_l2_coef_values:
                                    for delta_clip in delta_clip_values:
                                        for delta_clip_coef in delta_clip_coef_values:
                                            for low_rank_flip_topk in low_rank_flip_topk_values:
                                                for low_rank_flip_penalty_coef in low_rank_flip_penalty_coef_values:
                                                    for weak_margin_threshold in weak_margin_threshold_values:
                                                        for weak_margin_flip_penalty_coef in weak_margin_flip_penalty_coef_values:
                                                            for delta_support in delta_support_configs:
                                                                for actor_support in actor_update_support_configs:
                                                                    config_id = _run_tempered_ratio_config(
                                                                        args,
                                                                        candidate,
                                                                        source_policy,
                                                                        opponent_pool,
                                                                        device,
                                                                        config_id,
                                                                        rule_score_scale=float(rule_score_scale),
                                                                        temperature=float(temperature),
                                                                        lr=float(lr),
                                                                        update_epochs=int(update_epochs),
                                                                        clip_eps=float(clip_eps),
                                                                        rule_kl_coef=float(rule_kl_coef),
                                                                        delta_l2_coef=float(delta_l2_coef),
                                                                        delta_clip=float(delta_clip),
                                                                        delta_clip_coef=float(delta_clip_coef),
                                                                        low_rank_flip_topk=int(low_rank_flip_topk),
                                                                        low_rank_flip_penalty_coef=float(
                                                                            low_rank_flip_penalty_coef
                                                                        ),
                                                                        weak_margin_threshold=float(weak_margin_threshold),
                                                                        weak_margin_flip_penalty_coef=float(
                                                                            weak_margin_flip_penalty_coef
                                                                        ),
                                                                        support_policy_mode=str(
                                                                            delta_support["support_policy_mode"]
                                                                        ),
                                                                        delta_support_mode=str(
                                                                            delta_support["mode"]
                                                                        ),
                                                                        delta_support_topk=int(delta_support["topk"]),
                                                                        delta_support_margin_threshold=float(
                                                                            delta_support["margin_threshold"]
                                                                        ),
                                                                        outside_support_delta_mode=str(
                                                                            delta_support["outside_support_delta_mode"]
                                                                        ),
                                                                        actor_update_support_mode=str(
                                                                            actor_support["mode"]
                                                                        ),
                                                                        actor_update_topk=int(actor_support["topk"]),
                                                                        actor_update_margin_threshold=float(
                                                                            actor_support["margin_threshold"]
                                                                        ),
                                                                        summary_rows=summary_rows,
                                                                        iteration_rows=iteration_rows,
                                                                        step_rows=step_rows,
                                                                        advantage_rows=advantage_rows,
                                                                        checkpoint_rows=checkpoint_rows,
                                                                    )

    payload = {
        "mode": _run_mode_label(args),
        "source_type": "checkpoint",
        "candidate_summary": str(args.candidate_summary),
        "source_config_ids": [int(value) for value in args.source_config_ids],
        "rule_score_scales": [float(value) for value in rule_score_scales],
        "rule_score_scale_version": RULE_SCORE_SCALE_VERSION,
        "temperatures": [float(value) for value in args.temperatures],
        "lrs": [float(value) for value in args.lrs],
        "clip_eps_values": [float(value) for value in args.clip_eps_values],
        "update_epochs_values": [int(value) for value in update_epochs_values],
        "rule_kl_coef_values": [float(value) for value in rule_kl_coef_values],
        "delta_l2_coef_values": [float(value) for value in delta_l2_coef_values],
        "delta_clip_values": [float(value) for value in delta_clip_values],
        "delta_clip_coef_values": [float(value) for value in delta_clip_coef_values],
        "low_rank_flip_topk_values": [int(value) for value in low_rank_flip_topk_values],
        "low_rank_flip_penalty_coef_values": [float(value) for value in low_rank_flip_penalty_coef_values],
        "weak_margin_threshold_values": [float(value) for value in weak_margin_threshold_values],
        "weak_margin_flip_penalty_coef_values": [float(value) for value in weak_margin_flip_penalty_coef_values],
        "pass_criteria": _pass_criteria(args),
        "support_policy": _support_policy_config(args),
        "delta_support_projection": _delta_support_projection_config(args),
        "actor_update_support": _actor_update_support_config(args),
        "adaptive_recovery": _adaptive_recovery_config(args),
        "movement_regularization": _movement_regularization_config(args),
        "movement_quality_gate": _movement_quality_gate_config(args),
        "fresh_validation": _fresh_validation_config(args),
        "iteration_count": int(args.iterations),
        "episodes": int(args.episodes),
        "eval_scope": _eval_scope(args),
        "eval_strength_note": "sanity check only; not duplicate strength evidence",
        "summaries": summary_rows,
        "checkpoints": checkpoint_rows,
        "iteration_rows": iteration_rows,
        "advantage_audit": advantage_rows,
    }
    _write_json(args.output_dir / "tempered_ratio_pilot.json", payload)
    _write_csv(args.output_dir / "summary.csv", summary_rows)
    _write_csv(args.output_dir / "iterations.csv", iteration_rows)
    _write_csv(args.output_dir / "batch_steps.csv", step_rows)
    _write_csv(args.output_dir / "advantage_audit.csv", advantage_rows)
    _write_csv(args.output_dir / "checkpoint_summary.csv", checkpoint_rows)
    (args.output_dir / "summary.md").write_text(_summary_markdown(args, summary_rows), encoding="utf-8")
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))


def _run_tempered_ratio_config(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    source_policy,
    opponent_pool,
    device: torch.device,
    config_id: int,
    *,
    rule_score_scale: float,
    temperature: float,
    lr: float,
    update_epochs: int,
    clip_eps: float,
    rule_kl_coef: float,
    delta_l2_coef: float,
    delta_clip: float,
    delta_clip_coef: float,
    low_rank_flip_topk: int,
    low_rank_flip_penalty_coef: float,
    weak_margin_threshold: float,
    weak_margin_flip_penalty_coef: float,
    support_policy_mode: str,
    delta_support_mode: str,
    delta_support_topk: int,
    delta_support_margin_threshold: float,
    outside_support_delta_mode: str,
    actor_update_support_mode: str,
    actor_update_topk: int,
    actor_update_margin_threshold: float,
    summary_rows: list[dict[str, Any]],
    iteration_rows: list[dict[str, Any]],
    step_rows: list[dict[str, Any]],
    advantage_rows: list[dict[str, Any]],
    checkpoint_rows: list[dict[str, Any]],
) -> int:
    base_policy = copy.deepcopy(source_policy).to(device)
    base_policy.rule_score_scale = float(rule_score_scale)
    policy = DeltaSupportProjectionPolicy(
        base_policy,
        support_mode=str(delta_support_mode),
        topk=int(delta_support_topk),
        margin_threshold=float(delta_support_margin_threshold),
        outside_support_delta_mode=str(outside_support_delta_mode),
        support_policy_mode=str(support_policy_mode),
    ).to(device)
    optimizer = torch.optim.Adam(base_policy.parameters(), lr=float(lr))
    config_rows: list[dict[str, Any]] = []
    for iteration in range(int(args.iterations)):
        rollout_seed = _iteration_seed(args, config_id, iteration)
        torch_seed = _iteration_torch_seed(args, config_id, iteration)
        _seed_torch_sampling(torch_seed)
        behavior_policy = TemperaturePolicy(policy, temperature=float(temperature)).to(device)
        episodes = collect_selfplay_episodes(
            DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
            behavior_policy,
            num_episodes=int(args.episodes),
            opponent_pool=opponent_pool,
            learner_seats=tuple(int(seat) for seat in args.learner_seats),
            seed=rollout_seed,
            seed_stride=int(args.seed_stride),
            greedy=False,
            max_steps=int(args.max_steps),
            device=device,
        )
        _advantages, _returns, prepared_steps, batch = build_episodes_ppo_batch(
            episodes,
            gamma=float(args.gamma),
            gae_lambda=float(args.gae_lambda),
            include_rank_targets=True,
            strict_metadata=True,
        )
        batch = batch.to(device)
        diagnostic_rows, diagnostic_summary = batch_diagnostic_rows(policy, batch, prepared_steps, episodes)
        diagnostic_rows = _annotate_actor_update_support_rows(
            diagnostic_rows,
            support_mode=str(actor_update_support_mode),
            topk=int(actor_update_topk),
            margin_threshold=float(actor_update_margin_threshold),
        )
        sampling_summary = _sampling_summary(diagnostic_rows)
        actor_update_stats = _actor_update_support_stats(
            batch,
            support_mode=str(actor_update_support_mode),
            topk=int(actor_update_topk),
            margin_threshold=float(actor_update_margin_threshold),
        )
        _assert_behavior_temperature(diagnostic_rows, float(temperature))
        pre_stats = _policy_delta_stats(policy, batch)
        pre_margin_stats = _effective_margin_stats(policy, batch)
        pre_quality_stats = _movement_quality_stats(policy, batch)
        iteration_pre_state = copy.deepcopy(policy.state_dict())
        iteration_pre_optimizer_state = copy.deepcopy(optimizer.state_dict())
        for _epoch in range(int(update_epochs)):
            _tempered_ppo_update(
                policy,
                optimizer,
                batch,
                temperature=float(temperature),
                clip_eps=float(clip_eps),
                value_coef=float(args.value_coef),
                entropy_coef=float(args.entropy_coef),
                rank_coef=float(args.rank_coef),
                rule_kl_coef=float(rule_kl_coef),
                delta_l2_coef=float(delta_l2_coef),
                delta_clip=float(delta_clip),
                delta_clip_coef=float(delta_clip_coef),
                low_rank_flip_topk=int(low_rank_flip_topk),
                low_rank_flip_penalty_coef=float(low_rank_flip_penalty_coef),
                weak_margin_threshold=float(weak_margin_threshold),
                weak_margin_flip_penalty_coef=float(weak_margin_flip_penalty_coef),
                actor_update_support_mode=str(actor_update_support_mode),
                actor_update_topk=int(actor_update_topk),
                actor_update_margin_threshold=float(actor_update_margin_threshold),
                normalize_advantages=bool(args.normalize_advantages),
                max_grad_norm=float(args.max_grad_norm) if args.max_grad_norm is not None else None,
            )
        tempered_post_loss, untempered_post_loss, post_stats, post_margin_stats = _post_update_metrics(
            policy,
            batch,
            args,
            temperature=float(temperature),
            clip_eps=float(clip_eps),
            rule_kl_coef=float(rule_kl_coef),
            delta_l2_coef=float(delta_l2_coef),
            delta_clip=float(delta_clip),
            delta_clip_coef=float(delta_clip_coef),
            low_rank_flip_topk=int(low_rank_flip_topk),
            low_rank_flip_penalty_coef=float(low_rank_flip_penalty_coef),
            weak_margin_threshold=float(weak_margin_threshold),
            weak_margin_flip_penalty_coef=float(weak_margin_flip_penalty_coef),
            actor_update_support_mode=str(actor_update_support_mode),
            actor_update_topk=int(actor_update_topk),
            actor_update_margin_threshold=float(actor_update_margin_threshold),
        )
        recovery_result = _apply_adaptive_recovery_gate(
            policy,
            batch,
            optimizer,
            args,
            temperature=float(temperature),
            clip_eps=float(clip_eps),
            rule_kl_coef=float(rule_kl_coef),
            delta_l2_coef=float(delta_l2_coef),
            delta_clip=float(delta_clip),
            delta_clip_coef=float(delta_clip_coef),
            low_rank_flip_topk=int(low_rank_flip_topk),
            low_rank_flip_penalty_coef=float(low_rank_flip_penalty_coef),
            weak_margin_threshold=float(weak_margin_threshold),
            weak_margin_flip_penalty_coef=float(weak_margin_flip_penalty_coef),
            actor_update_support_mode=str(actor_update_support_mode),
            actor_update_topk=int(actor_update_topk),
            actor_update_margin_threshold=float(actor_update_margin_threshold),
            tempered_post_loss=tempered_post_loss,
            untempered_post_loss=untempered_post_loss,
            post_stats=post_stats,
            post_margin_stats=post_margin_stats,
            iteration_pre_state=iteration_pre_state,
            iteration_pre_optimizer_state=iteration_pre_optimizer_state,
            base_update_epochs=int(update_epochs),
        )
        tempered_post_loss = recovery_result["tempered_post_loss"]
        untempered_post_loss = recovery_result["untempered_post_loss"]
        post_stats = recovery_result["post_stats"]
        post_margin_stats = recovery_result["post_margin_stats"]
        post_quality_stats = recovery_result["post_quality_stats"]
        iter_row = {
            **_candidate_summary(candidate),
            "pilot_config_id": int(config_id),
            "iteration": int(iteration),
            "ratio_mode": "tempered_current_logits",
            "rule_score_scale": float(rule_score_scale),
            "rule_score_scale_version": RULE_SCORE_SCALE_VERSION,
            "behavior_temperature": float(temperature),
            "lr": float(lr),
            "clip_eps": float(clip_eps),
            "episodes": int(args.episodes),
            "rollout_seed": int(rollout_seed),
            "torch_sample_seed": int(torch_seed),
            "seed_registry_id": _iteration_seed_registry_id(args, config_id, iteration),
            "seed_hash": seed_registry_hash(_iteration_seed_registry(args, config_id, iteration)),
            "update_epochs": int(update_epochs),
            "rule_kl_coef": float(rule_kl_coef),
            "delta_l2_coef": float(delta_l2_coef),
            "delta_clip": float(delta_clip),
            "delta_clip_coef": float(delta_clip_coef),
            "low_rank_flip_topk": int(low_rank_flip_topk),
            "low_rank_flip_penalty_coef": float(low_rank_flip_penalty_coef),
            "weak_margin_threshold": float(weak_margin_threshold),
            "weak_margin_flip_penalty_coef": float(weak_margin_flip_penalty_coef),
            "support_policy_mode": str(support_policy_mode),
            "delta_support_mode": str(delta_support_mode),
            "delta_support_topk": int(delta_support_topk),
            "delta_support_margin_threshold": float(delta_support_margin_threshold),
            "outside_support_delta_mode": str(outside_support_delta_mode),
            "actor_update_support_mode": str(actor_update_support_mode),
            "actor_update_topk": int(actor_update_topk),
            "actor_update_margin_threshold": float(actor_update_margin_threshold),
            "entropy_coef": float(args.entropy_coef),
            "value_coef": float(args.value_coef),
            "rank_coef": float(args.rank_coef),
            "max_grad_norm": float(args.max_grad_norm) if args.max_grad_norm is not None else "",
            "adaptive_recovery_enabled": bool(args.adaptive_recovery_gate),
            "recovery_extra_epochs": int(recovery_result["extra_epochs"]),
            "recovery_attempted_epochs": int(recovery_result["attempted_epochs"]),
            "recovery_stop_reason": str(recovery_result["stop_reason"]),
            "recovery_rejected_epochs": int(recovery_result["rejected_epochs"]),
            "recovery_pre_top1_changed": float(recovery_result["pre_top1_changed"]),
            "recovery_pre_tempered_kl": float(recovery_result["pre_tempered_kl"]),
            "recovery_pre_tempered_clip": float(recovery_result["pre_tempered_clip"]),
            "recovery_pre_untempered_clip": float(recovery_result["pre_untempered_clip"]),
            "recovery_min_top1_changed": _recovery_min_top1_changed(args),
            "recovery_max_top1_changed": _recovery_max_top1_changed(args),
            "recovery_max_tempered_kl": _recovery_max_tempered_kl(args),
            "recovery_max_tempered_clip": _recovery_max_tempered_clip(args),
            "recovery_max_untempered_clip": _recovery_max_untempered_clip(args),
            "movement_quality_gate_enabled": bool(args.movement_quality_gate),
            "quality_max_changed_prior_rank_mean": float(args.quality_max_changed_prior_rank_mean),
            "quality_max_rank_ge5_rate": float(args.quality_max_rank_ge5_rate),
            "quality_max_prior_margin_p50": float(args.quality_max_prior_margin_p50),
            "train_movement_quality_pass": _train_movement_quality_gate_pass(post_stats, post_quality_stats, args),
            **diagnostic_summary,
            **sampling_summary,
            **actor_update_stats,
            **{f"untempered_pre_{key}": value for key, value in pre_stats.items()},
            **{f"untempered_pre_{key}": value for key, value in pre_margin_stats.items()},
            **{f"untempered_pre_{key}": value for key, value in pre_quality_stats.items()},
            **{f"untempered_post_{key}": value for key, value in post_stats.items()},
            **{f"untempered_post_{key}": value for key, value in post_margin_stats.items()},
            **{f"untempered_post_{key}": value for key, value in post_quality_stats.items()},
            "tempered_post_update_approx_kl": _loss_float(tempered_post_loss.approx_kl),
            "tempered_post_update_clip_fraction": _loss_float(tempered_post_loss.clip_fraction),
            "tempered_post_update_policy_loss": _loss_float(tempered_post_loss.policy_loss),
            "tempered_post_update_rule_kl": _loss_float(tempered_post_loss.rule_kl),
            "tempered_post_update_low_rank_flip_penalty": _loss_float(
                tempered_post_loss.low_rank_flip_penalty
            ),
            "tempered_post_update_weak_margin_flip_penalty": _loss_float(
                tempered_post_loss.weak_margin_flip_penalty
            ),
            "untempered_post_update_approx_kl": _loss_float(untempered_post_loss.approx_kl),
            "untempered_post_update_clip_fraction": _loss_float(untempered_post_loss.clip_fraction),
        }
        iteration_rows.append(iter_row)
        config_rows.append(iter_row)
        for row in diagnostic_rows:
            step_rows.append(
                {
                    **_candidate_summary(candidate),
                    "pilot_config_id": int(config_id),
                    "iteration": int(iteration),
                    "ratio_mode": "tempered_current_logits",
                    "rule_score_scale": float(rule_score_scale),
                    "rule_score_scale_version": RULE_SCORE_SCALE_VERSION,
                    "behavior_temperature": float(temperature),
                    "lr": float(lr),
                    "clip_eps": float(clip_eps),
                    "update_epochs": int(update_epochs),
                    "rule_kl_coef": float(rule_kl_coef),
                    "delta_l2_coef": float(delta_l2_coef),
                    "delta_clip": float(delta_clip),
                    "delta_clip_coef": float(delta_clip_coef),
                    "low_rank_flip_topk": int(low_rank_flip_topk),
                    "low_rank_flip_penalty_coef": float(low_rank_flip_penalty_coef),
                    "weak_margin_threshold": float(weak_margin_threshold),
                    "weak_margin_flip_penalty_coef": float(weak_margin_flip_penalty_coef),
                    "support_policy_mode": str(support_policy_mode),
                    "delta_support_mode": str(delta_support_mode),
                    "delta_support_topk": int(delta_support_topk),
                    "delta_support_margin_threshold": float(delta_support_margin_threshold),
                    "outside_support_delta_mode": str(outside_support_delta_mode),
                    "actor_update_support_mode": str(actor_update_support_mode),
                    "actor_update_topk": int(actor_update_topk),
                    "actor_update_margin_threshold": float(actor_update_margin_threshold),
                    **row,
                }
            )
        for row in _advantage_audit_rows(diagnostic_rows):
            advantage_rows.append(
                {
                    **_candidate_summary(candidate),
                    "pilot_config_id": int(config_id),
                    "iteration": int(iteration),
                    "ratio_mode": "tempered_current_logits",
                    "rule_score_scale": float(rule_score_scale),
                    "rule_score_scale_version": RULE_SCORE_SCALE_VERSION,
                    "behavior_temperature": float(temperature),
                    "lr": float(lr),
                    "clip_eps": float(clip_eps),
                    "update_epochs": int(update_epochs),
                    "rule_kl_coef": float(rule_kl_coef),
                    "delta_l2_coef": float(delta_l2_coef),
                    "delta_clip": float(delta_clip),
                    "delta_clip_coef": float(delta_clip_coef),
                    "low_rank_flip_topk": int(low_rank_flip_topk),
                    "low_rank_flip_penalty_coef": float(low_rank_flip_penalty_coef),
                    "weak_margin_threshold": float(weak_margin_threshold),
                    "weak_margin_flip_penalty_coef": float(weak_margin_flip_penalty_coef),
                    "support_policy_mode": str(support_policy_mode),
                    "delta_support_mode": str(delta_support_mode),
                    "delta_support_topk": int(delta_support_topk),
                    "delta_support_margin_threshold": float(delta_support_margin_threshold),
                    "outside_support_delta_mode": str(outside_support_delta_mode),
                    "actor_update_support_mode": str(actor_update_support_mode),
                    "actor_update_topk": int(actor_update_topk),
                    "actor_update_margin_threshold": float(actor_update_margin_threshold),
                    **row,
                }
            )
        print(
            "tempered-ratio "
            f"cfg={config_id} source={candidate['source_config_id']} "
            f"scale={float(rule_score_scale):g} "
            f"temp={float(temperature):g} lr={float(lr):g} "
            f"epochs={int(update_epochs)} clip={float(clip_eps):g} "
            f"rule_kl={float(rule_kl_coef):g} "
            f"delta_l2={float(delta_l2_coef):g} "
            f"delta_clip={float(delta_clip):g}/{float(delta_clip_coef):g} "
            f"low_rank_k={int(low_rank_flip_topk)} "
            f"low_rank_coef={float(low_rank_flip_penalty_coef):g} "
            f"weak_margin={float(weak_margin_threshold):g}/{float(weak_margin_flip_penalty_coef):g} "
            f"support_policy={support_policy_mode} "
            f"delta_support={delta_support_mode}/{int(delta_support_topk)}/{float(delta_support_margin_threshold):g}/{outside_support_delta_mode} "
            f"actor_support={actor_update_support_mode}/{int(actor_update_topk)}/{float(actor_update_margin_threshold):g} "
            f"actor_kept={actor_update_stats['actor_update_kept_rate']:.6g} "
            f"iter={iteration + 1}/{args.iterations} "
            f"non_top1={sampling_summary['non_top1_selected_count']} "
            f"non_top1_pos={sampling_summary['non_top1_positive_advantage_count']} "
            f"top1_changed={post_stats['top1_action_changed_rate']:.6g} "
            f"changed_rank={post_quality_stats['changed_action_prior_rank_mean']:.6g} "
            f"rank_ge5={post_quality_stats['changed_to_rank_ge5_rate']:.6g} "
            f"low_rank_pen={_loss_float(tempered_post_loss.low_rank_flip_penalty):.6g} "
            f"weak_margin_pen={_loss_float(tempered_post_loss.weak_margin_flip_penalty):.6g} "
            f"t_kl={_loss_float(tempered_post_loss.approx_kl):.6g} "
            f"t_clip={_loss_float(tempered_post_loss.clip_fraction):.6g} "
            f"delta_max={post_stats['neural_delta_abs_max']:.6g} "
            f"recovery_extra={int(recovery_result['extra_epochs'])} "
            f"recovery_stop={recovery_result['stop_reason']}",
            flush=True,
        )

    final_row = config_rows[-1]
    fresh_validation = _fresh_validation_metrics(
        args,
        policy,
        opponent_pool,
        device,
        config_id=int(config_id),
    )
    qualified_for_eval = _qualified_for_eval(final_row, fresh_validation, args)
    eval_skipped_reason = "" if qualified_for_eval else _eval_skip_reason(final_row, fresh_validation, args)
    if qualified_for_eval:
        eval_metrics = run_fixed_seed_evaluation_smoke(
            DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
            policy,
            num_games=int(args.eval_episodes),
            seed=int(args.eval_seed_base),
            seed_stride=int(args.seed_stride),
            seat_rotation=tuple(int(seat) for seat in args.eval_seat_rotation),
            opponent_pool=opponent_pool,
            opponent_name=str(candidate["opponent_mode"]),
            max_steps=int(args.max_steps),
            greedy=True,
            reuse_training_rollout=False,
            device=device,
        )
        eval_fields = _eval_fields(eval_metrics)
    else:
        eval_fields = _skipped_eval_fields()
    summary_rows.append(
        {
            **_candidate_summary(candidate),
            "pilot_config_id": int(config_id),
            "ratio_mode": "tempered_current_logits",
            "rule_score_scale": float(rule_score_scale),
            "rule_score_scale_version": RULE_SCORE_SCALE_VERSION,
            "behavior_temperature": float(temperature),
            "lr": float(lr),
            "clip_eps": float(clip_eps),
            "rule_kl_coef": float(rule_kl_coef),
            "delta_l2_coef": float(delta_l2_coef),
            "delta_clip": float(delta_clip),
            "delta_clip_coef": float(delta_clip_coef),
            "low_rank_flip_topk": int(low_rank_flip_topk),
            "low_rank_flip_penalty_coef": float(low_rank_flip_penalty_coef),
            "weak_margin_threshold": float(weak_margin_threshold),
            "weak_margin_flip_penalty_coef": float(weak_margin_flip_penalty_coef),
            "support_policy_mode": str(support_policy_mode),
            "delta_support_mode": str(delta_support_mode),
            "delta_support_topk": int(delta_support_topk),
            "delta_support_margin_threshold": float(delta_support_margin_threshold),
            "outside_support_delta_mode": str(outside_support_delta_mode),
            "actor_update_support_mode": str(actor_update_support_mode),
            "actor_update_topk": int(actor_update_topk),
            "actor_update_margin_threshold": float(actor_update_margin_threshold),
            "update_epochs": int(update_epochs),
            "iterations": int(args.iterations),
            "episodes": int(args.episodes),
            "eval_episodes": int(args.eval_episodes),
            "eval_seed_registry_id": _eval_seed_registry_id(args),
            "eval_seed_hash": seed_registry_hash(_eval_seed_registry(args)),
            "eval_scope": _eval_scope(args),
            "eval_strength_note": "sanity check only; not duplicate strength evidence",
            "fresh_validation_episodes": int(args.fresh_validation_episodes),
            "fresh_validation_seed_registry_id": _fresh_validation_seed_registry_id(args, config_id),
            "train_movement_quality_gate_pass": bool(final_row.get("train_movement_quality_pass", True)),
            "qualified_for_eval": bool(qualified_for_eval),
            "eval_skipped_reason": eval_skipped_reason,
            **fresh_validation,
            "source_checkpoint_sha256": candidate.get("checkpoint_sha256")
            or _file_sha256(Path(candidate["checkpoint_path"])),
            **{f"final_{key}": value for key, value in final_row.items() if _summary_metric_key(key)},
            "final_recovery_stop_reason": final_row.get("recovery_stop_reason", ""),
            "adaptive_recovery_enabled": bool(args.adaptive_recovery_gate),
            **eval_fields,
        }
    )
    _annotate_step38a_status(summary_rows[-1], args)
    if bool(args.save_final_checkpoint) and bool(summary_rows[-1]["qualified_for_eval"]):
        checkpoint_rows.append(
            _save_tempered_ratio_checkpoint(
                args,
                candidate,
                base_policy,
                optimizer,
                config_id=int(config_id),
                rule_score_scale=float(rule_score_scale),
                temperature=float(temperature),
                lr=float(lr),
                update_epochs=int(update_epochs),
                clip_eps=float(clip_eps),
                rule_kl_coef=float(rule_kl_coef),
                delta_l2_coef=float(delta_l2_coef),
                delta_clip=float(delta_clip),
                delta_clip_coef=float(delta_clip_coef),
                low_rank_flip_topk=int(low_rank_flip_topk),
                low_rank_flip_penalty_coef=float(low_rank_flip_penalty_coef),
                weak_margin_threshold=float(weak_margin_threshold),
                weak_margin_flip_penalty_coef=float(weak_margin_flip_penalty_coef),
                support_policy_mode=str(support_policy_mode),
                delta_support_mode=str(delta_support_mode),
                delta_support_topk=int(delta_support_topk),
                delta_support_margin_threshold=float(delta_support_margin_threshold),
                outside_support_delta_mode=str(outside_support_delta_mode),
                actor_update_support_mode=str(actor_update_support_mode),
                actor_update_topk=int(actor_update_topk),
                actor_update_margin_threshold=float(actor_update_margin_threshold),
                summary_row=summary_rows[-1],
            )
        )
    return int(config_id) + 1


def _save_tempered_ratio_checkpoint(
    args: argparse.Namespace,
    candidate: dict[str, Any],
    policy,
    optimizer,
    *,
    config_id: int,
    rule_score_scale: float,
    temperature: float,
    lr: float,
    update_epochs: int,
    clip_eps: float,
    rule_kl_coef: float,
    delta_l2_coef: float,
    delta_clip: float,
    delta_clip_coef: float,
    low_rank_flip_topk: int,
    low_rank_flip_penalty_coef: float,
    weak_margin_threshold: float,
    weak_margin_flip_penalty_coef: float,
    support_policy_mode: str,
    delta_support_mode: str,
    delta_support_topk: int,
    delta_support_margin_threshold: float,
    outside_support_delta_mode: str,
    actor_update_support_mode: str,
    actor_update_topk: int,
    actor_update_margin_threshold: float,
    summary_row: dict[str, Any],
) -> dict[str, Any]:
    checkpoint_dir = args.output_dir / f"checkpoint_config_{config_id:03d}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    source_config = json.loads(Path(candidate["config_path"]).read_text(encoding="utf-8"))
    ppo_config_hash = _stable_json_hash(
        {
            "mode": _run_mode_label(args),
            "source_config_id": int(candidate["source_config_id"]),
            "rerun_config_id": int(candidate["rerun_config_id"]),
            "rule_score_scale": float(rule_score_scale),
            "behavior_temperature": float(temperature),
            "lr": float(lr),
            "update_epochs": int(update_epochs),
            "clip_eps": float(clip_eps),
            "rule_kl_coef": float(rule_kl_coef),
            "delta_l2_coef": float(delta_l2_coef),
            "delta_clip": float(delta_clip),
            "delta_clip_coef": float(delta_clip_coef),
            "low_rank_flip_topk": int(low_rank_flip_topk),
            "low_rank_flip_penalty_coef": float(low_rank_flip_penalty_coef),
            "weak_margin_threshold": float(weak_margin_threshold),
            "weak_margin_flip_penalty_coef": float(weak_margin_flip_penalty_coef),
            "support_policy_mode": str(support_policy_mode),
            "delta_support_mode": str(delta_support_mode),
            "delta_support_topk": int(delta_support_topk),
            "delta_support_margin_threshold": float(delta_support_margin_threshold),
            "outside_support_delta_mode": str(outside_support_delta_mode),
            "delta_support_projection": _single_delta_support_projection_config(
                str(delta_support_mode),
                int(delta_support_topk),
                float(delta_support_margin_threshold),
                str(outside_support_delta_mode),
                support_policy_mode=str(support_policy_mode),
            ),
            "actor_update_support_mode": str(actor_update_support_mode),
            "actor_update_topk": int(actor_update_topk),
            "actor_update_margin_threshold": float(actor_update_margin_threshold),
            "actor_update_support": _single_actor_update_support_config(
                str(actor_update_support_mode),
                int(actor_update_topk),
                float(actor_update_margin_threshold),
            ),
            "adaptive_recovery": _adaptive_recovery_config(args),
            "movement_regularization": _movement_regularization_config(args),
            "movement_quality_gate": _movement_quality_gate_config(args),
            "fresh_validation": _fresh_validation_config(args),
            "iterations": int(args.iterations),
            "episodes": int(args.episodes),
            "seed_base": int(args.seed_base),
            "torch_seed_base": int(args.torch_seed_base),
        }
    )
    contract_metadata = default_checkpoint_metadata(
        rule_score_scale=float(rule_score_scale),
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        ppo_config_hash=ppo_config_hash,
    )
    validate_checkpoint_metadata(contract_metadata, expected_rule_score_scale=float(rule_score_scale))
    config_payload = {
        **source_config,
        "source_type": _run_mode_label(args),
        "source_config_id": int(candidate["source_config_id"]),
        "rerun_config_id": int(candidate["rerun_config_id"]),
        "parent_checkpoint_path": candidate.get("checkpoint_path"),
        "parent_checkpoint_sha256": candidate.get("checkpoint_sha256")
        or _file_sha256(Path(candidate["checkpoint_path"])),
        "rule_score_scale": float(rule_score_scale),
        "rule_score_scale_version": RULE_SCORE_SCALE_VERSION,
        "tempered_ratio_config": {
            "behavior_temperature": float(temperature),
            "lr": float(lr),
            "update_epochs": int(update_epochs),
            "clip_eps": float(clip_eps),
            "rule_kl_coef": float(rule_kl_coef),
            "delta_l2_coef": float(delta_l2_coef),
            "delta_clip": float(delta_clip),
            "delta_clip_coef": float(delta_clip_coef),
            "low_rank_flip_topk": int(low_rank_flip_topk),
            "low_rank_flip_penalty_coef": float(low_rank_flip_penalty_coef),
            "weak_margin_threshold": float(weak_margin_threshold),
            "weak_margin_flip_penalty_coef": float(weak_margin_flip_penalty_coef),
            "support_policy_mode": str(support_policy_mode),
            "delta_support_mode": str(delta_support_mode),
            "delta_support_topk": int(delta_support_topk),
            "delta_support_margin_threshold": float(delta_support_margin_threshold),
            "outside_support_delta_mode": str(outside_support_delta_mode),
            "delta_support_projection": _single_delta_support_projection_config(
                str(delta_support_mode),
                int(delta_support_topk),
                float(delta_support_margin_threshold),
                str(outside_support_delta_mode),
                support_policy_mode=str(support_policy_mode),
            ),
            "actor_update_support_mode": str(actor_update_support_mode),
            "actor_update_topk": int(actor_update_topk),
            "actor_update_margin_threshold": float(actor_update_margin_threshold),
            "actor_update_support": _single_actor_update_support_config(
                str(actor_update_support_mode),
                int(actor_update_topk),
                float(actor_update_margin_threshold),
            ),
            "entropy_coef": float(args.entropy_coef),
            "value_coef": float(args.value_coef),
            "rank_coef": float(args.rank_coef),
            "adaptive_recovery": _adaptive_recovery_config(args),
            "movement_regularization": _movement_regularization_config(args),
            "movement_quality_gate": _movement_quality_gate_config(args),
            "fresh_validation": _fresh_validation_config(args),
        },
        "run": {
            **dict(source_config.get("run", {})),
            "iterations": int(args.iterations),
            "rollout_episodes": int(args.episodes),
            "update_epochs": int(update_epochs),
            "eval_episodes": int(args.eval_episodes),
            "max_steps": int(args.max_steps),
            "clip_eps": float(clip_eps),
            "value_coef": float(args.value_coef),
            "rank_coef": float(args.rank_coef),
            "gamma": float(args.gamma),
            "gae_lambda": float(args.gae_lambda),
            "max_grad_norm": float(args.max_grad_norm) if args.max_grad_norm is not None else None,
        },
    }
    config_path = checkpoint_dir / "config.json"
    policy_path = checkpoint_dir / "policy_final.pt"
    optimizer_path = checkpoint_dir / "optimizer_final.pt"
    _write_json(config_path, config_payload)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "config": config_payload,
            "contract_metadata": contract_metadata,
            "rule_score_scale": contract_metadata["rule_score_scale"],
            "rule_score_scale_version": contract_metadata["rule_score_scale_version"],
            "summary": _to_jsonable(summary_row),
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
    return {
        "config_id": int(config_id),
        "rerun_config_id": int(candidate["rerun_config_id"]),
        "source_config_id": int(candidate["source_config_id"]),
        "source_report": candidate.get("source_report"),
        "config_key": json.dumps(config_payload.get("config_key", {}), sort_keys=True, separators=(",", ":")),
        "config_key_label": f"{_run_mode_label(args)}/{candidate.get('config_key_label', '')}",
        "checkpoint_path": str(policy_path),
        "checkpoint_sha256": _file_sha256(policy_path),
        "config_path": str(config_path),
        "optimizer_path": str(optimizer_path),
        "optimizer_sha256": _file_sha256(optimizer_path),
        "rule_score_scale": float(rule_score_scale),
        "rule_score_scale_version": RULE_SCORE_SCALE_VERSION,
        "top1_action_changed_rate": float(summary_row["final_untempered_post_top1_action_changed_rate"]),
        "rule_agreement": float(summary_row["final_untempered_post_rule_agreement"]),
        "neural_delta_abs_mean": float(summary_row["final_untempered_post_neural_delta_abs_mean"]),
        "neural_delta_abs_max": float(summary_row["final_untempered_post_neural_delta_abs_max"]),
        "low_rank_flip_topk": int(low_rank_flip_topk),
        "low_rank_flip_penalty_coef": float(low_rank_flip_penalty_coef),
        "weak_margin_threshold": float(weak_margin_threshold),
        "weak_margin_flip_penalty_coef": float(weak_margin_flip_penalty_coef),
        "low_rank_flip_penalty": float(summary_row["final_tempered_post_update_low_rank_flip_penalty"]),
        "weak_margin_flip_penalty": float(summary_row["final_tempered_post_update_weak_margin_flip_penalty"]),
        "support_policy_mode": str(support_policy_mode),
        "delta_support_mode": str(delta_support_mode),
        "delta_support_topk": int(delta_support_topk),
        "delta_support_margin_threshold": float(delta_support_margin_threshold),
        "outside_support_delta_mode": str(outside_support_delta_mode),
        "actor_update_support_mode": str(actor_update_support_mode),
        "actor_update_topk": int(actor_update_topk),
        "actor_update_margin_threshold": float(actor_update_margin_threshold),
        "actor_update_kept_rate": float(summary_row["final_actor_update_kept_rate"]),
        "actor_update_dropped_positive_advantage_count": int(
            summary_row["final_actor_update_dropped_positive_advantage_count"]
        ),
        "changed_action_prior_rank_mean": float(summary_row["final_untempered_post_changed_action_prior_rank_mean"]),
        "changed_to_rank_ge5_rate": float(summary_row["final_untempered_post_changed_to_rank_ge5_rate"]),
        "changed_state_prior_margin_p50": float(summary_row["final_untempered_post_changed_state_prior_margin_p50"]),
        "fresh_validation_top1_action_changed_rate": float(summary_row["fresh_validation_top1_action_changed_rate"]),
        "fresh_validation_gate_pass": bool(summary_row["fresh_validation_gate_pass"]),
        "approx_kl": float(summary_row["final_tempered_post_update_approx_kl"]),
        "clip_fraction": float(summary_row["final_tempered_post_update_clip_fraction"]),
        "eval_rank_pt": float(summary_row["eval_rank_pt"]),
        "eval_mean_rank": float(summary_row["eval_mean_rank"]),
        "eval_fourth_rate": float(summary_row["eval_fourth_rate"]),
        "eval_learner_deal_in_rate": float(summary_row["eval_learner_deal_in_rate"]),
    }


def _post_update_metrics(
    policy,
    batch,
    args: argparse.Namespace,
    *,
    temperature: float,
    clip_eps: float,
    rule_kl_coef: float,
    delta_l2_coef: float,
    delta_clip: float,
    delta_clip_coef: float,
    low_rank_flip_topk: int,
    low_rank_flip_penalty_coef: float,
    weak_margin_threshold: float,
    weak_margin_flip_penalty_coef: float,
    actor_update_support_mode: str,
    actor_update_topk: int,
    actor_update_margin_threshold: float,
) -> tuple[PPOLossBreakdown, PPOLossBreakdown, dict[str, float], dict[str, float]]:
    tempered_post_loss = _compute_tempered_ppo_loss(
        policy,
        batch,
        temperature=float(temperature),
        clip_eps=float(clip_eps),
        value_coef=float(args.value_coef),
        entropy_coef=float(args.entropy_coef),
        rank_coef=float(args.rank_coef),
        rule_kl_coef=float(rule_kl_coef),
        delta_l2_coef=float(delta_l2_coef),
        delta_clip=float(delta_clip),
        delta_clip_coef=float(delta_clip_coef),
        low_rank_flip_topk=int(low_rank_flip_topk),
        low_rank_flip_penalty_coef=float(low_rank_flip_penalty_coef),
        weak_margin_threshold=float(weak_margin_threshold),
        weak_margin_flip_penalty_coef=float(weak_margin_flip_penalty_coef),
        actor_update_support_mode=str(actor_update_support_mode),
        actor_update_topk=int(actor_update_topk),
        actor_update_margin_threshold=float(actor_update_margin_threshold),
        normalize_advantages=bool(args.normalize_advantages),
    )
    untempered_post_loss = _compute_untempered_loss(
        policy,
        batch,
        args,
        clip_eps=float(clip_eps),
        rule_kl_coef=float(rule_kl_coef),
    )
    post_stats = _policy_delta_stats(policy, batch)
    post_margin_stats = _effective_margin_stats(policy, batch)
    return tempered_post_loss, untempered_post_loss, post_stats, post_margin_stats


def _apply_adaptive_recovery_gate(
    policy,
    batch,
    optimizer,
    args: argparse.Namespace,
    *,
    temperature: float,
    clip_eps: float,
    rule_kl_coef: float,
    delta_l2_coef: float,
    delta_clip: float,
    delta_clip_coef: float,
    low_rank_flip_topk: int,
    low_rank_flip_penalty_coef: float,
    weak_margin_threshold: float,
    weak_margin_flip_penalty_coef: float,
    actor_update_support_mode: str,
    actor_update_topk: int,
    actor_update_margin_threshold: float,
    tempered_post_loss: PPOLossBreakdown,
    untempered_post_loss: PPOLossBreakdown,
    post_stats: dict[str, float],
    post_margin_stats: dict[str, float],
    iteration_pre_state: dict[str, Any],
    iteration_pre_optimizer_state: dict[str, Any],
    base_update_epochs: int,
) -> dict[str, Any]:
    post_quality_stats = _movement_quality_stats(policy, batch)
    result: dict[str, Any] = {
        "tempered_post_loss": tempered_post_loss,
        "untempered_post_loss": untempered_post_loss,
        "post_stats": post_stats,
        "post_margin_stats": post_margin_stats,
        "post_quality_stats": post_quality_stats,
        "extra_epochs": 0,
        "attempted_epochs": 0,
        "rejected_epochs": 0,
        "stop_reason": "disabled" if not bool(args.adaptive_recovery_gate) else "not_needed",
        "pre_top1_changed": float(post_stats["top1_action_changed_rate"]),
        "pre_tempered_kl": _loss_float(tempered_post_loss.approx_kl),
        "pre_tempered_clip": _loss_float(tempered_post_loss.clip_fraction),
        "pre_untempered_clip": _loss_float(untempered_post_loss.clip_fraction),
    }
    if not bool(args.adaptive_recovery_gate):
        return result
    max_extra_epochs = int(args.recovery_max_extra_epochs)
    if float(post_stats["top1_action_changed_rate"]) > _recovery_max_top1_changed(
        args
    ) or not _recovery_state_is_stable(
        tempered_post_loss,
        untempered_post_loss,
        args,
    ) or not _movement_quality_is_acceptable(post_quality_stats, args):
        policy.load_state_dict(iteration_pre_state)
        optimizer.load_state_dict(iteration_pre_optimizer_state)
        rollback_metrics = _post_update_metrics(
            policy,
            batch,
            args,
            temperature=float(temperature),
            clip_eps=float(clip_eps),
            rule_kl_coef=float(rule_kl_coef),
            delta_l2_coef=float(delta_l2_coef),
            delta_clip=float(delta_clip),
            delta_clip_coef=float(delta_clip_coef),
            low_rank_flip_topk=int(low_rank_flip_topk),
            low_rank_flip_penalty_coef=float(low_rank_flip_penalty_coef),
            weak_margin_threshold=float(weak_margin_threshold),
            weak_margin_flip_penalty_coef=float(weak_margin_flip_penalty_coef),
            actor_update_support_mode=str(actor_update_support_mode),
            actor_update_topk=int(actor_update_topk),
            actor_update_margin_threshold=float(actor_update_margin_threshold),
        )
        (
            result["tempered_post_loss"],
            result["untempered_post_loss"],
            result["post_stats"],
            result["post_margin_stats"],
        ) = rollback_metrics
        result["post_quality_stats"] = _movement_quality_stats(policy, batch)
        result["rejected_epochs"] = int(result["rejected_epochs"]) + int(base_update_epochs)
        result["stop_reason"] = "base_rejected_unstable_overmove_or_quality"
        return result
    if max_extra_epochs <= 0:
        result["stop_reason"] = "no_budget"
        return result
    if float(post_stats["top1_action_changed_rate"]) >= _recovery_min_top1_changed(args):
        return result

    for _extra_epoch in range(max_extra_epochs):
        previous_state = copy.deepcopy(policy.state_dict())
        previous_optimizer_state = copy.deepcopy(optimizer.state_dict())
        previous_metrics = (
            result["tempered_post_loss"],
            result["untempered_post_loss"],
            result["post_stats"],
            result["post_margin_stats"],
        )
        result["attempted_epochs"] = int(result["attempted_epochs"]) + 1
        _tempered_ppo_update(
            policy,
            optimizer,
            batch,
            temperature=float(temperature),
            clip_eps=float(clip_eps),
            value_coef=float(args.value_coef),
            entropy_coef=float(args.entropy_coef),
            rank_coef=float(args.rank_coef),
            rule_kl_coef=float(rule_kl_coef),
            delta_l2_coef=float(delta_l2_coef),
            delta_clip=float(delta_clip),
            delta_clip_coef=float(delta_clip_coef),
            low_rank_flip_topk=int(low_rank_flip_topk),
            low_rank_flip_penalty_coef=float(low_rank_flip_penalty_coef),
            weak_margin_threshold=float(weak_margin_threshold),
            weak_margin_flip_penalty_coef=float(weak_margin_flip_penalty_coef),
            actor_update_support_mode=str(actor_update_support_mode),
            actor_update_topk=int(actor_update_topk),
            actor_update_margin_threshold=float(actor_update_margin_threshold),
            normalize_advantages=bool(args.normalize_advantages),
            max_grad_norm=float(args.max_grad_norm) if args.max_grad_norm is not None else None,
        )
        candidate_metrics = _post_update_metrics(
            policy,
            batch,
            args,
            temperature=float(temperature),
            clip_eps=float(clip_eps),
            rule_kl_coef=float(rule_kl_coef),
            delta_l2_coef=float(delta_l2_coef),
            delta_clip=float(delta_clip),
            delta_clip_coef=float(delta_clip_coef),
            low_rank_flip_topk=int(low_rank_flip_topk),
            low_rank_flip_penalty_coef=float(low_rank_flip_penalty_coef),
            weak_margin_threshold=float(weak_margin_threshold),
            weak_margin_flip_penalty_coef=float(weak_margin_flip_penalty_coef),
            actor_update_support_mode=str(actor_update_support_mode),
            actor_update_topk=int(actor_update_topk),
            actor_update_margin_threshold=float(actor_update_margin_threshold),
        )
        candidate_top1 = float(candidate_metrics[2]["top1_action_changed_rate"])
        candidate_quality_stats = _movement_quality_stats(policy, batch)
        if candidate_top1 > _recovery_max_top1_changed(args) or not _recovery_state_is_stable(
            candidate_metrics[0],
            candidate_metrics[1],
            args,
        ) or not _movement_quality_is_acceptable(candidate_quality_stats, args):
            policy.load_state_dict(previous_state)
            optimizer.load_state_dict(previous_optimizer_state)
            (
                result["tempered_post_loss"],
                result["untempered_post_loss"],
                result["post_stats"],
                result["post_margin_stats"],
            ) = previous_metrics
            result["post_quality_stats"] = _movement_quality_stats(policy, batch)
            result["rejected_epochs"] = int(result["rejected_epochs"]) + 1
            result["stop_reason"] = "rejected_unstable_overmove_or_quality"
            break

        (
            result["tempered_post_loss"],
            result["untempered_post_loss"],
            result["post_stats"],
            result["post_margin_stats"],
        ) = candidate_metrics
        result["post_quality_stats"] = candidate_quality_stats
        result["extra_epochs"] = int(result["extra_epochs"]) + 1
        if candidate_top1 >= _recovery_min_top1_changed(args):
            result["stop_reason"] = "target_reached"
            break
    else:
        result["stop_reason"] = "budget_exhausted"

    return result


def _tempered_ppo_update(
    policy,
    optimizer,
    batch,
    *,
    temperature: float,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    rank_coef: float,
    rule_kl_coef: float,
    delta_l2_coef: float,
    delta_clip: float,
    delta_clip_coef: float,
    low_rank_flip_topk: int,
    low_rank_flip_penalty_coef: float,
    weak_margin_threshold: float,
    weak_margin_flip_penalty_coef: float,
    actor_update_support_mode: str,
    actor_update_topk: int,
    actor_update_margin_threshold: float,
    normalize_advantages: bool,
    max_grad_norm: float | None,
) -> PPOLossBreakdown:
    optimizer.zero_grad(set_to_none=True)
    losses = _compute_tempered_ppo_loss(
        policy,
        batch,
        temperature=temperature,
        clip_eps=clip_eps,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        rank_coef=rank_coef,
        rule_kl_coef=rule_kl_coef,
        delta_l2_coef=delta_l2_coef,
        delta_clip=delta_clip,
        delta_clip_coef=delta_clip_coef,
        low_rank_flip_topk=low_rank_flip_topk,
        low_rank_flip_penalty_coef=low_rank_flip_penalty_coef,
        weak_margin_threshold=weak_margin_threshold,
        weak_margin_flip_penalty_coef=weak_margin_flip_penalty_coef,
        actor_update_support_mode=actor_update_support_mode,
        actor_update_topk=actor_update_topk,
        actor_update_margin_threshold=actor_update_margin_threshold,
        normalize_advantages=normalize_advantages,
    )
    losses.total_loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return losses


def _compute_tempered_ppo_loss(
    policy,
    batch,
    *,
    temperature: float,
    clip_eps: float,
    value_coef: float,
    entropy_coef: float,
    rank_coef: float,
    rule_kl_coef: float,
    delta_l2_coef: float,
    delta_clip: float,
    delta_clip_coef: float,
    low_rank_flip_topk: int,
    low_rank_flip_penalty_coef: float,
    weak_margin_threshold: float,
    weak_margin_flip_penalty_coef: float,
    actor_update_support_mode: str,
    actor_update_topk: int,
    actor_update_margin_threshold: float,
    normalize_advantages: bool,
) -> PPOLossBreakdown:
    if float(temperature) <= 0.0:
        raise ValueError(f"temperature must be positive, got {temperature}")
    validate_ppo_batch_rule_score_scale(policy, batch, strict_metadata=True)
    output = policy(batch.policy_input)
    mask = batch.policy_input.legal_action_mask.bool()
    tempered_logits = (output.action_logits / float(temperature)).masked_fill(
        ~mask,
        torch.finfo(output.action_logits.dtype).min,
    )
    dist = MaskedCategorical(tempered_logits, batch.policy_input.legal_action_mask)
    new_log_prob = dist.log_prob(batch.action_index)
    entropy = dist.entropy()

    raw_advantages = batch.advantages.float()
    advantage_mean = raw_advantages.mean()
    advantage_std = raw_advantages.std(unbiased=False)
    return_mean = batch.returns.float().mean()

    advantages = batch.advantages
    if normalize_advantages and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

    ratio = torch.exp(new_log_prob - batch.old_log_prob)
    ratio_mean = ratio.mean()
    ratio_std = ratio.std(unbiased=False)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    per_sample_policy_loss = -torch.min(unclipped, clipped)
    actor_weights = _actor_update_support_weights(
        batch,
        support_mode=str(actor_update_support_mode),
        topk=int(actor_update_topk),
        margin_threshold=float(actor_update_margin_threshold),
    ).to(device=per_sample_policy_loss.device, dtype=per_sample_policy_loss.dtype)
    if int(actor_weights.numel()) != int(per_sample_policy_loss.numel()):
        raise ValueError(
            "actor update support weights must match policy loss samples: "
            f"{actor_weights.numel()} != {per_sample_policy_loss.numel()}"
        )
    policy_loss = (per_sample_policy_loss * actor_weights).sum() / actor_weights.sum().clamp_min(1.0)
    value_loss = F.smooth_l1_loss(output.value, batch.returns)
    entropy_bonus = entropy.mean()
    approx_kl = 0.5 * (new_log_prob - batch.old_log_prob).pow(2).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()

    total_loss = policy_loss + float(value_coef) * value_loss - float(entropy_coef) * entropy_bonus
    rank_loss = None
    if rank_coef > 0.0 and batch.final_rank_target is not None:
        rank_loss = F.cross_entropy(output.rank_logits, batch.final_rank_target)
        total_loss = total_loss + float(rank_coef) * rank_loss

    rule_kl = _tempered_rule_kl(output, dist, batch)
    if rule_kl is not None and rule_kl_coef > 0.0:
        total_loss = total_loss + float(rule_kl_coef) * rule_kl

    delta_l2, delta_clip_penalty = _delta_regularization_terms(
        output,
        batch,
        delta_clip=float(delta_clip),
    )
    if delta_l2 is not None and delta_l2_coef > 0.0:
        total_loss = total_loss + float(delta_l2_coef) * delta_l2
    if delta_clip_penalty is not None and delta_clip_coef > 0.0:
        total_loss = total_loss + float(delta_clip_coef) * delta_clip_penalty

    low_rank_flip_penalty, weak_margin_flip_penalty = _movement_regularization_terms(
        output,
        dist,
        batch,
        low_rank_flip_topk=int(low_rank_flip_topk),
        weak_margin_threshold=float(weak_margin_threshold),
    )
    if low_rank_flip_penalty is not None and low_rank_flip_penalty_coef > 0.0:
        total_loss = total_loss + float(low_rank_flip_penalty_coef) * low_rank_flip_penalty
    if weak_margin_flip_penalty is not None and weak_margin_flip_penalty_coef > 0.0:
        total_loss = total_loss + float(weak_margin_flip_penalty_coef) * weak_margin_flip_penalty

    avg_abs_neural_delta, delta_norm = _delta_metrics(output, batch)
    rule_agreement = _untempered_rule_agreement(output, batch)
    return PPOLossBreakdown(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_bonus=entropy_bonus,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        ratio_mean=ratio_mean,
        ratio_std=ratio_std,
        advantage_mean=advantage_mean,
        advantage_std=advantage_std,
        return_mean=return_mean,
        rank_loss=rank_loss,
        rule_kl=rule_kl,
        rule_agreement=rule_agreement,
        avg_abs_neural_delta=avg_abs_neural_delta,
        delta_norm=delta_norm,
        low_rank_flip_penalty=low_rank_flip_penalty,
        weak_margin_flip_penalty=weak_margin_flip_penalty,
    )


def _compute_untempered_loss(
    policy,
    batch,
    args: argparse.Namespace,
    *,
    clip_eps: float,
    rule_kl_coef: float,
):
    with torch.no_grad():
        return compute_ppo_loss(
            policy,
            batch,
            clip_eps=float(clip_eps),
            value_coef=float(args.value_coef),
            entropy_coef=float(args.entropy_coef),
            rank_coef=float(args.rank_coef),
            rule_kl_coef=float(rule_kl_coef),
            normalize_advantages=bool(args.normalize_advantages),
        )


def _effective_margin_stats(policy, batch) -> dict[str, float]:
    with torch.no_grad():
        output = policy(batch.policy_input)
    mask = batch.policy_input.legal_action_mask.bool()
    final_logits = output.aux.get("final_logits", output.action_logits).float()
    prior_logits = output.aux.get("prior_logits")
    if prior_logits is None:
        prior_logits = batch.policy_input.prior_logits
    if prior_logits is None:
        return {
            "effective_margin_to_flip_mean": 0.0,
            "effective_margin_to_flip_p50": 0.0,
            "effective_margin_to_flip_p90": 0.0,
            "scaled_prior_margin_mean": 0.0,
            "scaled_prior_margin_p50": 0.0,
        }
    prior_logits = prior_logits.float()
    valid_final = final_logits.masked_fill(~mask, torch.finfo(final_logits.dtype).min)
    valid_prior = prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min)
    final_top1 = valid_final.argmax(dim=-1)
    prior_top1 = valid_prior.argmax(dim=-1)
    changed = final_top1 != prior_top1
    effective_margins: list[float] = []
    scaled_prior_margins: list[float] = []
    scale = float(getattr(policy, "rule_score_scale", 1.0))
    for row_idx in range(valid_final.shape[0]):
        row_mask = mask[row_idx]
        legal_count = int(row_mask.sum().item())
        if legal_count <= 1 or bool(changed[row_idx]):
            effective_margins.append(0.0)
        else:
            top_idx = int(prior_top1[row_idx].item())
            competitors = valid_final[row_idx].clone()
            competitors[top_idx] = torch.finfo(competitors.dtype).min
            effective_margins.append(float((valid_final[row_idx, top_idx] - competitors.max()).detach().cpu()))
        legal_prior = valid_prior[row_idx][row_mask]
        if legal_count <= 1:
            scaled_prior_margins.append(0.0)
        else:
            top2 = torch.topk(legal_prior, k=2).values
            scaled_prior_margins.append(float(((top2[0] - top2[1]) * scale).detach().cpu()))
    margins = torch.tensor(effective_margins, dtype=torch.float32)
    scaled_prior = torch.tensor(scaled_prior_margins, dtype=torch.float32)
    return {
        "effective_margin_to_flip_mean": float(margins.mean().item()) if margins.numel() else 0.0,
        "effective_margin_to_flip_p50": float(margins.quantile(0.5).item()) if margins.numel() else 0.0,
        "effective_margin_to_flip_p90": float(margins.quantile(0.9).item()) if margins.numel() else 0.0,
        "scaled_prior_margin_mean": float(scaled_prior.mean().item()) if scaled_prior.numel() else 0.0,
        "scaled_prior_margin_p50": float(scaled_prior.quantile(0.5).item()) if scaled_prior.numel() else 0.0,
    }


def _movement_quality_stats(policy, batch) -> dict[str, float]:
    with torch.no_grad():
        output = policy(batch.policy_input)
    mask = batch.policy_input.legal_action_mask.bool()
    final_logits = output.aux.get("final_logits", output.action_logits).float()
    prior_logits = output.aux.get("prior_logits")
    if prior_logits is None:
        prior_logits = batch.policy_input.prior_logits
    if prior_logits is None:
        return _empty_movement_quality_stats()
    prior_logits = prior_logits.float()
    valid_final = final_logits.masked_fill(~mask, torch.finfo(final_logits.dtype).min)
    valid_prior = prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min)
    final_top1 = valid_final.argmax(dim=-1)
    prior_top1 = valid_prior.argmax(dim=-1)
    prior_probs = torch.softmax(valid_prior, dim=-1).masked_fill(~mask, 0.0)

    ranks: list[float] = []
    margins_to_selected: list[float] = []
    margins_to_top2: list[float] = []
    selected_probs: list[float] = []
    top1_probs: list[float] = []
    for row_idx in range(valid_final.shape[0]):
        if int(final_top1[row_idx].item()) == int(prior_top1[row_idx].item()):
            continue
        row_mask = mask[row_idx]
        legal_count = int(row_mask.sum().item())
        if legal_count <= 0:
            continue
        chosen_idx = int(final_top1[row_idx].item())
        prior_idx = int(prior_top1[row_idx].item())
        legal_prior = prior_logits[row_idx][row_mask]
        rank = 1 + int((legal_prior > prior_logits[row_idx, chosen_idx]).sum().item())
        ranks.append(float(rank))
        margins_to_selected.append(float((prior_logits[row_idx, prior_idx] - prior_logits[row_idx, chosen_idx]).detach().cpu()))
        if legal_count <= 1:
            margins_to_top2.append(0.0)
        else:
            top2 = torch.topk(legal_prior, k=2).values
            margins_to_top2.append(float((top2[0] - top2[1]).detach().cpu()))
        selected_probs.append(float(prior_probs[row_idx, chosen_idx].detach().cpu()))
        top1_probs.append(float(prior_probs[row_idx, prior_idx].detach().cpu()))
    return _movement_quality_fields(
        ranks=ranks,
        margins_to_selected=margins_to_selected,
        margins_to_top2=margins_to_top2,
        selected_probs=selected_probs,
        top1_probs=top1_probs,
    )


def _empty_movement_quality_stats() -> dict[str, float]:
    return _movement_quality_fields(
        ranks=[],
        margins_to_selected=[],
        margins_to_top2=[],
        selected_probs=[],
        top1_probs=[],
    )


def _movement_quality_fields(
    *,
    ranks: Sequence[float],
    margins_to_selected: Sequence[float],
    margins_to_top2: Sequence[float],
    selected_probs: Sequence[float],
    top1_probs: Sequence[float],
) -> dict[str, float]:
    return {
        "changed_action_prior_rank_mean": _mean_float(ranks),
        "changed_action_prior_rank_p50": _quantile_float(ranks, 0.50),
        "changed_action_prior_rank_p90": _quantile_float(ranks, 0.90),
        "changed_to_top2_rate": _rank_rate(ranks, max_rank=2),
        "changed_to_top3_rate": _rank_rate(ranks, max_rank=3),
        "changed_to_top5_rate": _rank_rate(ranks, max_rank=5),
        "changed_to_rank_ge5_rate": (
            sum(1 for rank in ranks if float(rank) >= 5.0) / len(ranks)
            if ranks
            else 0.0
        ),
        "changed_state_prior_margin_mean": _mean_float(margins_to_selected),
        "changed_state_prior_margin_p50": _quantile_float(margins_to_selected, 0.50),
        "changed_state_prior_margin_p90": _quantile_float(margins_to_selected, 0.90),
        "changed_state_prior_margin_top1_to_top2_mean": _mean_float(margins_to_top2),
        "changed_state_prior_margin_top1_to_top2_p50": _quantile_float(margins_to_top2, 0.50),
        "changed_state_prior_margin_top1_to_top2_p90": _quantile_float(margins_to_top2, 0.90),
        "changed_state_selected_prior_prob_mean": _mean_float(selected_probs),
        "changed_state_prior_top1_prob_mean": _mean_float(top1_probs),
    }


def _mean_float(values: Sequence[float]) -> float:
    return sum(float(value) for value in values) / len(values) if values else 0.0


def _quantile_float(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    tensor = torch.tensor([float(value) for value in values], dtype=torch.float32)
    return float(tensor.quantile(float(q)).item())


def _rank_rate(ranks: Sequence[float], *, max_rank: int) -> float:
    return (
        sum(1 for rank in ranks if float(rank) <= float(max_rank)) / len(ranks)
        if ranks
        else 0.0
    )


def _tempered_rule_kl(output, dist: MaskedCategorical, batch) -> torch.Tensor | None:
    prior_logits = output.aux.get("prior_logits")
    if prior_logits is None:
        prior_logits = batch.policy_input.prior_logits
    if prior_logits is None:
        return None
    mask = batch.policy_input.legal_action_mask.bool()
    current_probs = dist.probs().masked_fill(~mask, 0.0)
    current_log_probs = torch.log(current_probs.clamp_min(1e-12))
    prior_probs = smoothed_prior_probs(prior_logits.float(), mask, eps=1e-4)
    prior_log_probs = torch.log(prior_probs.clamp_min(1e-12))
    kl = (current_probs * (current_log_probs - prior_log_probs)).masked_fill(~mask, 0.0)
    return kl.sum(dim=-1).mean()


def _untempered_rule_agreement(output, batch) -> torch.Tensor | None:
    prior_logits = output.aux.get("prior_logits")
    if prior_logits is None:
        prior_logits = batch.policy_input.prior_logits
    if prior_logits is None:
        return None
    mask = batch.policy_input.legal_action_mask.bool()
    current = output.action_logits.masked_fill(~mask, torch.finfo(output.action_logits.dtype).min).argmax(dim=-1)
    prior = prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min).argmax(dim=-1)
    return (current == prior).float().mean()


def _delta_metrics(output, batch) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    neural_delta = output.aux.get("neural_delta")
    if neural_delta is None:
        return None, None
    mask = batch.policy_input.legal_action_mask.bool()
    legal_delta = neural_delta.masked_select(mask)
    if legal_delta.numel() == 0:
        return None, None
    return legal_delta.abs().mean(), legal_delta.pow(2).mean().sqrt()


def _delta_regularization_terms(
    output,
    batch,
    *,
    delta_clip: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    neural_delta = output.aux.get("neural_delta")
    if neural_delta is None:
        return None, None
    mask = batch.policy_input.legal_action_mask.bool()
    legal_delta = neural_delta.masked_select(mask)
    if legal_delta.numel() == 0:
        return None, None
    delta_l2 = legal_delta.pow(2).mean()
    if float(delta_clip) <= 0.0:
        delta_clip_penalty = torch.zeros((), device=legal_delta.device, dtype=legal_delta.dtype)
    else:
        delta_clip_penalty = (legal_delta.abs() - float(delta_clip)).clamp_min(0.0).pow(2).mean()
    return delta_l2, delta_clip_penalty


def _movement_regularization_terms(
    output,
    dist: MaskedCategorical,
    batch,
    *,
    low_rank_flip_topk: int,
    weak_margin_threshold: float,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    prior_logits = output.aux.get("prior_logits")
    if prior_logits is None:
        prior_logits = batch.policy_input.prior_logits
    if prior_logits is None:
        return None, None

    mask = batch.policy_input.legal_action_mask.bool()
    prior_logits = prior_logits.float()
    masked_prior = prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min)
    probs = dist.probs().masked_fill(~mask, 0.0)

    ranks = 1.0 + (masked_prior.unsqueeze(2) > masked_prior.unsqueeze(1)).sum(dim=1).float()
    rank_excess = (ranks - float(low_rank_flip_topk)).clamp_min(0.0).masked_fill(~mask, 0.0)
    low_rank_flip_penalty = (probs * rank_excess).sum(dim=-1).mean()

    legal_count = mask.sum(dim=-1)
    top_values = torch.topk(masked_prior, k=min(2, masked_prior.shape[-1]), dim=-1).values
    top1 = top_values[:, 0]
    if top_values.shape[-1] > 1:
        second = torch.where(legal_count > 1, top_values[:, 1], top1)
    else:
        second = top1
    prior_top1_margin = top1 - second
    prior_top1 = masked_prior.argmax(dim=-1)
    prior_top1_prob = probs.gather(1, prior_top1.unsqueeze(1)).squeeze(1)
    strong_margin = (legal_count > 1) & (prior_top1_margin > float(weak_margin_threshold))
    weak_margin_flip_penalty = ((1.0 - prior_top1_prob) * strong_margin.float()).mean()
    return low_rank_flip_penalty, weak_margin_flip_penalty


def _delta_support_mask(
    policy_input: PolicyInput,
    *,
    prior_logits: torch.Tensor,
    support_mode: str,
    topk: int,
    margin_threshold: float,
) -> torch.Tensor:
    mode = str(support_mode)
    if mode not in _DELTA_SUPPORT_MODES:
        raise ValueError(f"unsupported delta support mode: {mode}")
    mask = policy_input.legal_action_mask.bool()
    if mode == "all":
        return mask

    prior_logits = prior_logits.float()
    masked_prior = prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min)
    ranks = 1.0 + (masked_prior.unsqueeze(2) > masked_prior.unsqueeze(1)).sum(dim=1).float()
    topk_keep = ranks <= float(topk)

    legal_count = mask.sum(dim=-1)
    top_values = torch.topk(masked_prior, k=min(2, masked_prior.shape[-1]), dim=-1).values
    top1 = top_values[:, 0]
    if top_values.shape[-1] > 1:
        second = torch.where(legal_count > 1, top_values[:, 1], top1)
    else:
        second = top1
    prior_margin = torch.where(legal_count > 1, top1 - second, torch.zeros_like(top1))
    weak_margin_keep = (prior_margin <= float(margin_threshold)).unsqueeze(1).expand_as(mask)

    if mode == "topk":
        support = topk_keep
    elif mode == "weak-margin":
        support = weak_margin_keep
    elif mode == "topk-or-weak-margin":
        support = topk_keep | weak_margin_keep
    elif mode == "topk-and-weak-margin":
        support = topk_keep & weak_margin_keep
    else:
        raise ValueError(f"unsupported delta support mode: {mode}")
    return support & mask


def _annotate_actor_update_support_rows(
    rows: Sequence[dict[str, Any]],
    *,
    support_mode: str,
    topk: int,
    margin_threshold: float,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        selected_rank = float(row.get("selected_prior_rank", 0.0))
        prior_margin = float(row.get("prior_top1_margin_to_second", 0.0))
        keep = _actor_update_support_keep(
            selected_prior_rank=selected_rank,
            prior_top1_margin_to_second=prior_margin,
            support_mode=str(support_mode),
            topk=int(topk),
            margin_threshold=float(margin_threshold),
        )
        annotated.append(
            {
                **row,
                "actor_update_support_mode": str(support_mode),
                "actor_update_topk": int(topk),
                "actor_update_margin_threshold": float(margin_threshold),
                "actor_update_keep": bool(keep),
                "actor_update_dropped": not bool(keep),
            }
        )
    return annotated


def _actor_update_support_stats(
    batch,
    *,
    support_mode: str,
    topk: int,
    margin_threshold: float,
) -> dict[str, float | int]:
    selected_rank, prior_margin = _actor_update_prior_rank_and_margin(
        batch,
        require_prior=str(support_mode) != "all",
    )
    weights = _actor_update_support_weights(
        batch,
        support_mode=str(support_mode),
        topk=int(topk),
        margin_threshold=float(margin_threshold),
        selected_rank=selected_rank,
        prior_margin=prior_margin,
    )
    keep = weights > 0.0
    dropped = ~keep
    advantages = batch.advantages.detach().float().to(device=weights.device)
    positive = advantages > 0.0
    non_top1 = selected_rank.to(device=weights.device) > 1.0
    batch_size = int(weights.numel())
    kept_count = int(keep.sum().detach().cpu())
    dropped_count = int(dropped.sum().detach().cpu())
    kept_rank = selected_rank[keep.to(device=selected_rank.device)]
    dropped_rank = selected_rank[dropped.to(device=selected_rank.device)]
    kept_margin = prior_margin[keep.to(device=prior_margin.device)]
    dropped_margin = prior_margin[dropped.to(device=prior_margin.device)]
    return {
        "actor_update_kept_count": kept_count,
        "actor_update_dropped_count": dropped_count,
        "actor_update_kept_rate": kept_count / batch_size if batch_size else 0.0,
        "actor_update_dropped_rate": dropped_count / batch_size if batch_size else 0.0,
        "actor_update_kept_positive_advantage_count": int((keep & positive).sum().detach().cpu()),
        "actor_update_dropped_positive_advantage_count": int((dropped & positive).sum().detach().cpu()),
        "actor_update_kept_non_top1_positive_advantage_count": int(
            (keep & non_top1 & positive).sum().detach().cpu()
        ),
        "actor_update_dropped_non_top1_positive_advantage_count": int(
            (dropped & non_top1 & positive).sum().detach().cpu()
        ),
        "actor_update_kept_selected_prior_rank_mean": _tensor_mean_float(kept_rank),
        "actor_update_dropped_selected_prior_rank_mean": _tensor_mean_float(dropped_rank),
        "actor_update_kept_prior_margin_mean": _tensor_mean_float(kept_margin),
        "actor_update_dropped_prior_margin_mean": _tensor_mean_float(dropped_margin),
    }


def _actor_update_support_weights(
    batch,
    *,
    support_mode: str,
    topk: int,
    margin_threshold: float,
    selected_rank: torch.Tensor | None = None,
    prior_margin: torch.Tensor | None = None,
) -> torch.Tensor:
    mode = str(support_mode)
    if mode not in _ACTOR_UPDATE_SUPPORT_MODES:
        raise ValueError(f"unsupported actor update support mode: {mode}")
    if selected_rank is None or prior_margin is None:
        selected_rank, prior_margin = _actor_update_prior_rank_and_margin(
            batch,
            require_prior=mode != "all",
        )
    if mode == "all":
        return torch.ones_like(batch.advantages.detach().float())
    keep = _actor_update_support_tensor(
        selected_rank=selected_rank,
        prior_top1_margin_to_second=prior_margin,
        support_mode=mode,
        topk=int(topk),
        margin_threshold=float(margin_threshold),
    )
    return keep.to(device=batch.advantages.device, dtype=batch.advantages.dtype)


def _actor_update_prior_rank_and_margin(
    batch,
    *,
    require_prior: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    prior_logits = batch.policy_input.prior_logits
    if prior_logits is None:
        if require_prior:
            raise ValueError("actor-update support mask requires prior_logits")
        defaults = torch.ones_like(batch.advantages.detach().float())
        return defaults, torch.zeros_like(defaults)

    mask = batch.policy_input.legal_action_mask.bool()
    prior_logits = prior_logits.float()
    masked_prior = prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min)
    ranks = 1.0 + (masked_prior.unsqueeze(2) > masked_prior.unsqueeze(1)).sum(dim=1).float()
    action_index = batch.action_index.long().view(-1, 1)
    selected_rank = ranks.gather(1, action_index).squeeze(1)

    legal_count = mask.sum(dim=-1)
    top_values = torch.topk(masked_prior, k=min(2, masked_prior.shape[-1]), dim=-1).values
    top1 = top_values[:, 0]
    if top_values.shape[-1] > 1:
        second = torch.where(legal_count > 1, top_values[:, 1], top1)
    else:
        second = top1
    prior_margin = torch.where(legal_count > 1, top1 - second, torch.zeros_like(top1))
    return selected_rank.detach(), prior_margin.detach()


def _actor_update_support_tensor(
    *,
    selected_rank: torch.Tensor,
    prior_top1_margin_to_second: torch.Tensor,
    support_mode: str,
    topk: int,
    margin_threshold: float,
) -> torch.Tensor:
    topk_keep = selected_rank <= float(topk)
    weak_margin_keep = prior_top1_margin_to_second <= float(margin_threshold)
    if support_mode == "topk":
        return topk_keep
    if support_mode == "weak-margin":
        return weak_margin_keep
    if support_mode == "topk-or-weak-margin":
        return topk_keep | weak_margin_keep
    if support_mode == "topk-and-weak-margin":
        return topk_keep & weak_margin_keep
    if support_mode == "all":
        return torch.ones_like(topk_keep, dtype=torch.bool)
    raise ValueError(f"unsupported actor update support mode: {support_mode}")


def _actor_update_support_keep(
    *,
    selected_prior_rank: float,
    prior_top1_margin_to_second: float,
    support_mode: str,
    topk: int,
    margin_threshold: float,
) -> bool:
    topk_keep = float(selected_prior_rank) <= float(topk)
    weak_margin_keep = float(prior_top1_margin_to_second) <= float(margin_threshold)
    if support_mode == "all":
        return True
    if support_mode == "topk":
        return topk_keep
    if support_mode == "weak-margin":
        return weak_margin_keep
    if support_mode == "topk-or-weak-margin":
        return topk_keep or weak_margin_keep
    if support_mode == "topk-and-weak-margin":
        return topk_keep and weak_margin_keep
    raise ValueError(f"unsupported actor update support mode: {support_mode}")


def _tensor_mean_float(values: torch.Tensor) -> float:
    return 0.0 if int(values.numel()) == 0 else float(values.float().mean().detach().cpu())


def _assert_behavior_temperature(rows: Sequence[dict[str, Any]], expected: float) -> None:
    temperatures = {
        float(row["behavior_temperature"])
        for row in rows
        if row.get("behavior_temperature") is not None
    }
    if temperatures != {float(expected)}:
        raise RuntimeError(f"behavior_temperature metadata mismatch: expected {expected}, got {sorted(temperatures)}")


def _update_epochs_values(args: argparse.Namespace) -> tuple[int, ...]:
    values = args.update_epochs_values if args.update_epochs_values is not None else (args.update_epochs,)
    epoch_values = tuple(int(value) for value in values)
    if not epoch_values:
        raise ValueError("update epoch matrix must not be empty")
    if any(value <= 0 for value in epoch_values):
        raise ValueError(f"update epochs must be positive, got {epoch_values}")
    return epoch_values


def _rule_score_scales(args: argparse.Namespace) -> tuple[float, ...]:
    values = tuple(float(value) for value in args.rule_score_scales)
    if not values:
        raise ValueError("rule score scale matrix must not be empty")
    if any(value < 0.0 for value in values):
        raise ValueError(f"rule score scales must be non-negative, got {values}")
    return values


def _rule_kl_coef_values(args: argparse.Namespace) -> tuple[float, ...]:
    raw_values = args.rule_kl_coef_values
    values = (
        (float(args.rule_kl_coef),)
        if raw_values is None
        else tuple(float(value) for value in raw_values)
    )
    if not values:
        raise ValueError("rule_kl_coef matrix must not be empty")
    if any(value < 0.0 for value in values):
        raise ValueError(f"rule_kl_coef values must be non-negative, got {values}")
    return values


def _nonnegative_float_values(values: Sequence[float], *, name: str) -> tuple[float, ...]:
    clean = tuple(float(value) for value in values)
    if not clean:
        raise ValueError(f"{name} matrix must not be empty")
    if any(value < 0.0 for value in clean):
        raise ValueError(f"{name} values must be non-negative, got {clean}")
    return clean


def _positive_int_values(values: Sequence[int], *, name: str) -> tuple[int, ...]:
    clean = tuple(int(value) for value in values)
    if not clean:
        raise ValueError(f"{name} matrix must not be empty")
    if any(value <= 0 for value in clean):
        raise ValueError(f"{name} values must be positive, got {clean}")
    return clean


def _support_policy_configs(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    if args.support_policy_modes is None:
        return _delta_support_projection_configs(args)
    modes = tuple(str(value) for value in args.support_policy_modes)
    if not modes:
        raise ValueError("support policy mode matrix must not be empty")
    invalid = [mode for mode in modes if mode not in _SUPPORT_POLICY_MODES]
    if invalid:
        raise ValueError(f"unsupported support policy modes: {invalid}")
    topk_values = _positive_int_values(args.delta_support_topk_values, name="delta_support_topk")
    margin_thresholds = _nonnegative_float_values(
        args.delta_support_margin_threshold_values,
        name="delta_support_margin_threshold",
    )
    configs: list[dict[str, Any]] = []
    seen: set[tuple[str, int, float, str]] = set()
    for mode in modes:
        if mode == "unrestricted":
            raw_configs = (
                _single_delta_support_projection_config(
                    "all",
                    int(topk_values[0]),
                    float(margin_thresholds[0]),
                    "zero",
                    support_policy_mode=mode,
                ),
            )
        elif mode == "delta-topk-zero":
            raw_configs = tuple(
                _single_delta_support_projection_config(
                    "topk",
                    int(topk),
                    float(margin_thresholds[0]),
                    "zero",
                    support_policy_mode=mode,
                )
                for topk in topk_values
            )
        elif mode == "support-only-topk":
            raw_configs = tuple(
                _single_delta_support_projection_config(
                    "topk",
                    int(topk),
                    float(margin_thresholds[0]),
                    "zero",
                    support_policy_mode=mode,
                )
                for topk in topk_values
            )
        else:
            raise ValueError(f"unsupported support policy mode: {mode}")
        for config in raw_configs:
            key = (
                str(config["support_policy_mode"]),
                int(config["topk"]),
                round(float(config["margin_threshold"]), 12),
                str(config["outside_support_delta_mode"]),
            )
            if key in seen:
                continue
            seen.add(key)
            configs.append(config)
    return tuple(configs)


def _support_policy_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "modes": None if args.support_policy_modes is None else tuple(str(value) for value in args.support_policy_modes),
        "expanded_configs": _support_policy_configs(args),
        "legacy_delta_support_args_used": args.support_policy_modes is None,
        "scope": "policy_forward_rollout_loss_fresh_eval",
    }


def _delta_support_modes(args: argparse.Namespace) -> tuple[str, ...]:
    modes = tuple(str(value) for value in args.delta_support_modes)
    if not modes:
        raise ValueError("delta support mode matrix must not be empty")
    invalid = [mode for mode in modes if mode not in _DELTA_SUPPORT_MODES]
    if invalid:
        raise ValueError(f"unsupported delta support modes: {invalid}")
    return modes


def _outside_support_delta_modes(args: argparse.Namespace) -> tuple[str, ...]:
    modes = tuple(str(value) for value in args.outside_support_delta_modes)
    if not modes:
        raise ValueError("outside-support delta mode matrix must not be empty")
    invalid = [mode for mode in modes if mode not in _OUTSIDE_SUPPORT_DELTA_MODES]
    if invalid:
        raise ValueError(f"unsupported outside-support delta modes: {invalid}")
    return modes


def _delta_support_projection_configs(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    modes = _delta_support_modes(args)
    topk_values = _positive_int_values(args.delta_support_topk_values, name="delta_support_topk")
    margin_thresholds = _nonnegative_float_values(
        args.delta_support_margin_threshold_values,
        name="delta_support_margin_threshold",
    )
    outside_modes = _outside_support_delta_modes(args)
    configs: list[dict[str, Any]] = []
    seen: set[tuple[str, int, float, str]] = set()
    for mode in modes:
        if mode == "all":
            pairs = ((topk_values[0], margin_thresholds[0]),)
        elif mode == "topk":
            pairs = tuple((topk, margin_thresholds[0]) for topk in topk_values)
        elif mode == "weak-margin":
            pairs = tuple((topk_values[0], threshold) for threshold in margin_thresholds)
        else:
            pairs = tuple(
                (topk, threshold)
                for topk in topk_values
                for threshold in margin_thresholds
            )
        for outside_mode in outside_modes:
            for topk, threshold in pairs:
                key = (str(mode), int(topk), round(float(threshold), 12), str(outside_mode))
                if key in seen:
                    continue
                seen.add(key)
                configs.append(
                    _single_delta_support_projection_config(
                        str(mode),
                        int(topk),
                        float(threshold),
                        str(outside_mode),
                    )
                )
    return tuple(configs)


def _delta_support_projection_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "modes": _delta_support_modes(args),
        "topk_values": _positive_int_values(args.delta_support_topk_values, name="delta_support_topk"),
        "margin_threshold_values": _nonnegative_float_values(
            args.delta_support_margin_threshold_values,
            name="delta_support_margin_threshold",
        ),
        "outside_support_delta_modes": _outside_support_delta_modes(args),
        "expanded_configs": _delta_support_projection_configs(args),
        "scope": "projected_policy_forward_and_loss",
        "margin_threshold_units": "unscaled_prior_logits",
    }


def _single_delta_support_projection_config(
    mode: str,
    topk: int,
    margin_threshold: float,
    outside_support_delta_mode: str,
    *,
    support_policy_mode: str | None = None,
) -> dict[str, Any]:
    if mode not in _DELTA_SUPPORT_MODES:
        raise ValueError(f"unsupported delta support mode: {mode}")
    if outside_support_delta_mode not in _OUTSIDE_SUPPORT_DELTA_MODES:
        raise ValueError(f"unsupported outside-support delta mode: {outside_support_delta_mode}")
    if int(topk) <= 0:
        raise ValueError(f"delta_support_topk must be positive, got {topk}")
    if float(margin_threshold) < 0.0:
        raise ValueError(f"delta_support_margin_threshold must be non-negative, got {margin_threshold}")
    policy_mode = _support_policy_mode_from_delta_config(
        mode=str(mode),
        outside_support_delta_mode=str(outside_support_delta_mode),
        explicit=support_policy_mode,
    )
    return {
        "support_policy_mode": policy_mode,
        "mode": str(mode),
        "topk": int(topk),
        "margin_threshold": float(margin_threshold),
        "outside_support_delta_mode": str(outside_support_delta_mode),
        "margin_threshold_units": "unscaled_prior_logits",
    }


def _support_policy_mode_from_delta_config(
    *,
    mode: str,
    outside_support_delta_mode: str,
    explicit: str | None,
) -> str:
    if explicit is not None:
        if explicit not in _SUPPORT_POLICY_MODES and explicit != "delta-projection":
            raise ValueError(f"unsupported support policy mode: {explicit}")
        return str(explicit)
    if mode == "all":
        return "unrestricted"
    if mode == "topk" and outside_support_delta_mode == "zero":
        return "delta-topk-zero"
    return "delta-projection"


def _actor_update_support_modes(args: argparse.Namespace) -> tuple[str, ...]:
    modes = tuple(str(value) for value in args.actor_update_support_modes)
    if not modes:
        raise ValueError("actor update support mode matrix must not be empty")
    invalid = [mode for mode in modes if mode not in _ACTOR_UPDATE_SUPPORT_MODES]
    if invalid:
        raise ValueError(f"unsupported actor update support modes: {invalid}")
    return modes


def _actor_update_support_configs(args: argparse.Namespace) -> tuple[dict[str, Any], ...]:
    modes = _actor_update_support_modes(args)
    topk_values = _positive_int_values(args.actor_update_topk_values, name="actor_update_topk")
    margin_thresholds = _nonnegative_float_values(
        args.actor_update_margin_threshold_values,
        name="actor_update_margin_threshold",
    )
    configs: list[dict[str, Any]] = []
    seen: set[tuple[str, int, float]] = set()
    for mode in modes:
        if mode == "all":
            pairs = ((topk_values[0], margin_thresholds[0]),)
        elif mode == "topk":
            pairs = tuple((topk, margin_thresholds[0]) for topk in topk_values)
        elif mode == "weak-margin":
            pairs = tuple((topk_values[0], threshold) for threshold in margin_thresholds)
        else:
            pairs = tuple(
                (topk, threshold)
                for topk in topk_values
                for threshold in margin_thresholds
            )
        for topk, threshold in pairs:
            key = (str(mode), int(topk), round(float(threshold), 12))
            if key in seen:
                continue
            seen.add(key)
            configs.append(
                _single_actor_update_support_config(
                    str(mode),
                    int(topk),
                    float(threshold),
                )
            )
    return tuple(configs)


def _actor_update_support_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "modes": _actor_update_support_modes(args),
        "topk_values": _positive_int_values(args.actor_update_topk_values, name="actor_update_topk"),
        "margin_threshold_values": _nonnegative_float_values(
            args.actor_update_margin_threshold_values,
            name="actor_update_margin_threshold",
        ),
        "expanded_configs": _actor_update_support_configs(args),
        "scope": "actor_policy_loss_only",
        "value_rank_losses_weighted": False,
        "margin_threshold_units": "unscaled_prior_logits",
    }


def _single_actor_update_support_config(
    mode: str,
    topk: int,
    margin_threshold: float,
) -> dict[str, Any]:
    if mode not in _ACTOR_UPDATE_SUPPORT_MODES:
        raise ValueError(f"unsupported actor update support mode: {mode}")
    if int(topk) <= 0:
        raise ValueError(f"actor_update_topk must be positive, got {topk}")
    if float(margin_threshold) < 0.0:
        raise ValueError(f"actor_update_margin_threshold must be non-negative, got {margin_threshold}")
    return {
        "mode": str(mode),
        "topk": int(topk),
        "margin_threshold": float(margin_threshold),
        "margin_threshold_units": "unscaled_prior_logits",
    }


def _run_mode_label(args: argparse.Namespace) -> str:
    if bool(args.movement_quality_gate):
        return "tempered_ratio_movement_quality_gate"
    if bool(args.adaptive_recovery_gate):
        return "tempered_ratio_adaptive_movement_diagnostic"
    return "tempered_ratio_ppo_diagnostic"


def _pass_criteria(args: argparse.Namespace) -> dict[str, float]:
    return {
        "min_top1_changed": float(args.pass_min_top1_changed),
        "max_top1_changed": float(args.pass_max_top1_changed),
        "max_tempered_kl": float(args.pass_max_tempered_kl),
        "max_tempered_clip": float(args.pass_max_tempered_clip),
        "max_untempered_clip": float(args.pass_max_untempered_clip),
        "max_eval_fourth": float(args.pass_max_eval_fourth),
        "max_eval_deal_in": float(args.pass_max_eval_deal_in),
    }


def _movement_regularization_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "low_rank_flip_topk_values": tuple(
            int(value) for value in _positive_int_values(args.low_rank_flip_topk_values, name="low_rank_flip_topk")
        ),
        "low_rank_flip_penalty_coef_values": tuple(
            float(value)
            for value in _nonnegative_float_values(
                args.low_rank_flip_penalty_coef_values,
                name="low_rank_flip_penalty_coef",
            )
        ),
        "weak_margin_threshold_values": tuple(
            float(value)
            for value in _nonnegative_float_values(
                args.weak_margin_threshold_values,
                name="weak_margin_threshold",
            )
        ),
        "weak_margin_flip_penalty_coef_values": tuple(
            float(value)
            for value in _nonnegative_float_values(
                args.weak_margin_flip_penalty_coef_values,
                name="weak_margin_flip_penalty_coef",
            )
        ),
        "weak_margin_threshold_units": "unscaled_prior_logits",
    }


def _movement_quality_gate_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enabled": bool(args.movement_quality_gate),
        "role": "candidate_selector",
        "train_min_top1_changed": _quality_train_min_top1_changed(args),
        "train_max_top1_changed": float(args.quality_train_max_top1_changed),
        "max_changed_prior_rank_mean": float(args.quality_max_changed_prior_rank_mean),
        "max_rank_ge5_rate": float(args.quality_max_rank_ge5_rate),
        "max_prior_margin_p50": float(args.quality_max_prior_margin_p50),
    }


def _fresh_validation_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "episodes": int(args.fresh_validation_episodes),
        "seed_base": _fresh_validation_seed_base(args),
        "seed_stride": int(args.seed_stride),
        "seat_rotation": tuple(int(seat) for seat in args.eval_seat_rotation),
        "min_top1_changed": float(args.fresh_validation_min_top1_changed),
        "max_top1_changed": float(args.fresh_validation_max_top1_changed),
        "policy_mode": "greedy",
    }


def _adaptive_recovery_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "enabled": bool(args.adaptive_recovery_gate),
        "role": "adaptive_movement_diagnostic_not_candidate_selector",
        "max_extra_epochs": int(args.recovery_max_extra_epochs),
        "min_top1_changed": _recovery_min_top1_changed(args),
        "max_top1_changed": _recovery_max_top1_changed(args),
        "max_tempered_kl": _recovery_max_tempered_kl(args),
        "max_tempered_clip": _recovery_max_tempered_clip(args),
        "max_untempered_clip": _recovery_max_untempered_clip(args),
        "rollback_on_unstable_overmove_or_quality": True,
    }


def _recovery_min_top1_changed(args: argparse.Namespace) -> float:
    value = args.recovery_min_top1_changed
    return float(args.pass_min_top1_changed if value is None else value)


def _recovery_max_top1_changed(args: argparse.Namespace) -> float:
    value = args.recovery_max_top1_changed
    return float(args.pass_max_top1_changed if value is None else value)


def _recovery_max_tempered_kl(args: argparse.Namespace) -> float:
    value = args.recovery_max_tempered_kl
    return float(args.pass_max_tempered_kl if value is None else value)


def _recovery_max_tempered_clip(args: argparse.Namespace) -> float:
    value = args.recovery_max_tempered_clip
    return float(args.pass_max_tempered_clip if value is None else value)


def _recovery_max_untempered_clip(args: argparse.Namespace) -> float:
    value = args.recovery_max_untempered_clip
    return float(args.pass_max_untempered_clip if value is None else value)


def _recovery_state_is_stable(
    tempered_loss: PPOLossBreakdown,
    untempered_loss: PPOLossBreakdown,
    args: argparse.Namespace,
) -> bool:
    return (
        _loss_float(tempered_loss.approx_kl) < _recovery_max_tempered_kl(args)
        and _loss_float(tempered_loss.clip_fraction) < _recovery_max_tempered_clip(args)
        and _loss_float(untempered_loss.clip_fraction) < _recovery_max_untempered_clip(args)
    )


def _quality_train_min_top1_changed(args: argparse.Namespace) -> float:
    value = args.quality_train_min_top1_changed
    return _recovery_min_top1_changed(args) if value is None else float(value)


def _movement_quality_is_acceptable(quality_stats: dict[str, float], args: argparse.Namespace) -> bool:
    if not bool(args.movement_quality_gate):
        return True
    return (
        float(quality_stats["changed_action_prior_rank_mean"]) <= float(args.quality_max_changed_prior_rank_mean)
        and float(quality_stats["changed_to_rank_ge5_rate"]) <= float(args.quality_max_rank_ge5_rate)
        and float(quality_stats["changed_state_prior_margin_p50"]) <= float(args.quality_max_prior_margin_p50)
    )


def _train_movement_quality_gate_pass(
    delta_stats: dict[str, float],
    quality_stats: dict[str, float],
    args: argparse.Namespace,
) -> bool:
    if not bool(args.movement_quality_gate):
        return True
    top1_changed = float(delta_stats["top1_action_changed_rate"])
    return (
        _quality_train_min_top1_changed(args)
        <= top1_changed
        <= float(args.quality_train_max_top1_changed)
        and _movement_quality_is_acceptable(quality_stats, args)
    )


def _fresh_validation_metrics(
    args: argparse.Namespace,
    policy,
    opponent_pool,
    device: torch.device,
    *,
    config_id: int,
) -> dict[str, Any]:
    if int(args.fresh_validation_episodes) <= 0:
        return {
            "fresh_validation_enabled": False,
            "fresh_validation_gate_pass": True,
            "fresh_validation_top1_action_changed_rate": 0.0,
            **{f"fresh_validation_{key}": value for key, value in _empty_movement_quality_stats().items()},
        }
    episodes = collect_selfplay_episodes(
        DiscardOnlyMahjongEnv(max_kyokus=args.max_kyokus),
        policy,
        num_episodes=int(args.fresh_validation_episodes),
        opponent_pool=opponent_pool,
        learner_seats=tuple(int(seat) for seat in args.eval_seat_rotation),
        seed=_fresh_validation_seed_base(args),
        seed_stride=int(args.seed_stride),
        greedy=True,
        max_steps=int(args.max_steps),
        device=device,
    )
    _advantages, _returns, _prepared_steps, batch = build_episodes_ppo_batch(
        episodes,
        gamma=float(args.gamma),
        gae_lambda=float(args.gae_lambda),
        include_rank_targets=True,
        strict_metadata=True,
    )
    batch = batch.to(device)
    delta_stats = _policy_delta_stats(policy, batch)
    quality_stats = _movement_quality_stats(policy, batch)
    gate_pass = _fresh_validation_gate_pass(delta_stats, quality_stats, args)
    return {
        "fresh_validation_enabled": True,
        "fresh_validation_gate_pass": bool(gate_pass),
        "fresh_validation_seed_hash": seed_registry_hash(_fresh_validation_seed_registry(args, config_id)),
        "fresh_validation_episode_count": int(args.fresh_validation_episodes),
        "fresh_validation_top1_action_changed_rate": float(delta_stats["top1_action_changed_rate"]),
        "fresh_validation_rule_agreement": float(delta_stats["rule_agreement"]),
        "fresh_validation_neural_delta_abs_mean": float(delta_stats["neural_delta_abs_mean"]),
        "fresh_validation_neural_delta_abs_max": float(delta_stats["neural_delta_abs_max"]),
        **{f"fresh_validation_{key}": value for key, value in quality_stats.items()},
    }


def _fresh_validation_gate_pass(
    delta_stats: dict[str, float],
    quality_stats: dict[str, float],
    args: argparse.Namespace,
) -> bool:
    if not bool(args.movement_quality_gate):
        return True
    if int(args.fresh_validation_episodes) <= 0:
        return True
    top1_changed = float(delta_stats["top1_action_changed_rate"])
    return (
        float(args.fresh_validation_min_top1_changed)
        <= top1_changed
        <= float(args.fresh_validation_max_top1_changed)
        and _movement_quality_is_acceptable(quality_stats, args)
    )


def _qualified_for_eval(final_row: dict[str, Any], fresh_validation: dict[str, Any], args: argparse.Namespace) -> bool:
    if not bool(args.movement_quality_gate):
        return True
    return bool(final_row.get("train_movement_quality_pass", False)) and bool(
        fresh_validation.get("fresh_validation_gate_pass", False)
    )


def _eval_skip_reason(final_row: dict[str, Any], fresh_validation: dict[str, Any], args: argparse.Namespace) -> str:
    if not bool(args.movement_quality_gate):
        return ""
    reasons: list[str] = []
    if not bool(final_row.get("train_movement_quality_pass", False)):
        reasons.append("train_movement_quality_gate_failed")
    if not bool(fresh_validation.get("fresh_validation_gate_pass", False)):
        reasons.append("fresh_validation_gate_failed")
    return ",".join(reasons)


def _fresh_validation_seed_base(args: argparse.Namespace) -> int:
    value = args.fresh_validation_seed_base
    return int(args.eval_seed_base + 10000 if value is None else value)


def _fresh_validation_seed_registry(args: argparse.Namespace, config_id: int) -> tuple[int, ...]:
    base = _fresh_validation_seed_base(args)
    return tuple(
        int(base + index * int(args.seed_stride))
        for index in range(int(args.fresh_validation_episodes))
    )


def _fresh_validation_seed_registry_id(args: argparse.Namespace, config_id: int) -> str:
    if int(args.fresh_validation_episodes) <= 0:
        return "disabled"
    return (
        f"base={_fresh_validation_seed_base(args)}:"
        f"stride={int(args.seed_stride)}:"
        f"count={int(args.fresh_validation_episodes)}:"
        f"seats={','.join(str(int(seat)) for seat in args.eval_seat_rotation)}"
    )


def _eval_fields(eval_metrics) -> dict[str, Any]:
    return {
        "eval_rank_pt": eval_metrics.rank_pt,
        "eval_mean_rank": eval_metrics.average_rank,
        "eval_fourth_rate": eval_metrics.fourth_rate,
        "eval_learner_deal_in_rate": eval_metrics.learner_deal_in_rate,
        "eval_learner_win_rate": eval_metrics.learner_win_rate,
        "illegal_action_rate_fail_closed": eval_metrics.illegal_action_rate_fail_closed,
        "fallback_rate_fail_closed": eval_metrics.fallback_rate_fail_closed,
        "forced_terminal_missed_fail_closed": eval_metrics.forced_terminal_missed_fail_closed,
    }


def _skipped_eval_fields() -> dict[str, Any]:
    nan = float("nan")
    return {
        "eval_rank_pt": nan,
        "eval_mean_rank": nan,
        "eval_fourth_rate": nan,
        "eval_learner_deal_in_rate": nan,
        "eval_learner_win_rate": nan,
        "illegal_action_rate_fail_closed": nan,
        "fallback_rate_fail_closed": nan,
        "forced_terminal_missed_fail_closed": nan,
    }


def _annotate_step38a_status(row: dict[str, Any], args: argparse.Namespace) -> None:
    top1_changed = float(row["final_untempered_post_top1_action_changed_rate"])
    tempered_kl = float(row["final_tempered_post_update_approx_kl"])
    tempered_clip = float(row["final_tempered_post_update_clip_fraction"])
    untempered_clip = float(row["final_untempered_post_update_clip_fraction"])
    eval_fourth = float(row["eval_fourth_rate"])
    eval_deal_in = float(row["eval_learner_deal_in_rate"])

    movement_pass = (
        float(args.pass_min_top1_changed)
        < top1_changed
        < float(args.pass_max_top1_changed)
    )
    stability_pass = (
        tempered_kl < float(args.pass_max_tempered_kl)
        and tempered_clip < float(args.pass_max_tempered_clip)
        and untempered_clip < float(args.pass_max_untempered_clip)
    )
    eval_sanity_pass = (
        eval_fourth <= float(args.pass_max_eval_fourth)
        and eval_deal_in <= float(args.pass_max_eval_deal_in)
    )
    quality_pass = (
        not bool(args.movement_quality_gate)
        or (
            bool(row.get("train_movement_quality_gate_pass", False))
            and bool(row.get("fresh_validation_gate_pass", False))
        )
    )

    row["step38a_movement_pass"] = movement_pass
    row["step38a_stability_pass"] = stability_pass
    row["step38a_eval_sanity_pass"] = eval_sanity_pass
    row["step39_movement_quality_pass"] = quality_pass
    row["step38a_pass"] = movement_pass and stability_pass and eval_sanity_pass and quality_pass


def _summary_metric_key(key: str) -> bool:
    return key in {
        "batch_size",
        "advantage_mean",
        "advantage_std",
        "advantage_min",
        "advantage_max",
        "mean_delta_needed_to_flip_top1",
        "p10_delta_needed_to_flip_top1",
        "p50_delta_needed_to_flip_top1",
        "p90_delta_needed_to_flip_top1",
        "positive_advantage_count",
        "non_top1_selected_count",
        "non_top1_positive_advantage_count",
        "positive_advantage_top1_count",
        "positive_advantage_non_top1_count",
        "selected_prior_top1_rate",
        "actor_update_kept_count",
        "actor_update_dropped_count",
        "actor_update_kept_rate",
        "actor_update_dropped_rate",
        "actor_update_kept_positive_advantage_count",
        "actor_update_dropped_positive_advantage_count",
        "actor_update_kept_non_top1_positive_advantage_count",
        "actor_update_dropped_non_top1_positive_advantage_count",
        "actor_update_kept_selected_prior_rank_mean",
        "actor_update_dropped_selected_prior_rank_mean",
        "actor_update_kept_prior_margin_mean",
        "actor_update_dropped_prior_margin_mean",
        "untempered_post_top1_action_changed_rate",
        "untempered_post_rule_agreement",
        "untempered_post_neural_delta_abs_mean",
        "untempered_post_neural_delta_abs_max",
        "untempered_post_changed_action_prior_rank_mean",
        "untempered_post_changed_action_prior_rank_p50",
        "untempered_post_changed_action_prior_rank_p90",
        "untempered_post_changed_to_top2_rate",
        "untempered_post_changed_to_top3_rate",
        "untempered_post_changed_to_top5_rate",
        "untempered_post_changed_to_rank_ge5_rate",
        "untempered_post_changed_state_prior_margin_mean",
        "untempered_post_changed_state_prior_margin_p50",
        "untempered_post_changed_state_prior_margin_p90",
        "untempered_post_changed_state_selected_prior_prob_mean",
        "untempered_post_changed_state_prior_top1_prob_mean",
        "untempered_post_effective_margin_to_flip_mean",
        "untempered_post_effective_margin_to_flip_p50",
        "untempered_post_effective_margin_to_flip_p90",
        "untempered_post_scaled_prior_margin_mean",
        "untempered_post_scaled_prior_margin_p50",
        "tempered_post_update_approx_kl",
        "tempered_post_update_clip_fraction",
        "tempered_post_update_policy_loss",
        "tempered_post_update_rule_kl",
        "tempered_post_update_low_rank_flip_penalty",
        "tempered_post_update_weak_margin_flip_penalty",
        "untempered_post_update_approx_kl",
        "untempered_post_update_clip_fraction",
        "recovery_extra_epochs",
        "recovery_attempted_epochs",
        "recovery_rejected_epochs",
        "recovery_pre_top1_changed",
        "recovery_pre_tempered_kl",
        "recovery_pre_tempered_clip",
        "recovery_pre_untempered_clip",
        "train_movement_quality_pass",
    }


def _summary_markdown(args: argparse.Namespace, rows: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# KeqingRL Tempered-Ratio PPO Diagnostic",
        "",
        "source_type: `checkpoint`",
        "ratio_mode: `tempered_current_logits`",
        f"candidate_summary: `{args.candidate_summary}`",
        f"source_config_ids: `{','.join(str(int(value)) for value in args.source_config_ids)}`",
        f"episodes: `{args.episodes}`",
        f"iterations: `{args.iterations}`",
        f"rule_score_scales: `{','.join(str(float(value)) for value in _rule_score_scales(args))}`",
        f"rule_score_scale_version: `{RULE_SCORE_SCALE_VERSION}`",
        f"temperatures: `{','.join(str(float(value)) for value in args.temperatures)}`",
        f"lrs: `{','.join(str(float(value)) for value in args.lrs)}`",
        f"update_epochs_values: `{','.join(str(int(value)) for value in _update_epochs_values(args))}`",
        f"clip_eps_values: `{','.join(str(float(value)) for value in args.clip_eps_values)}`",
        f"rule_kl_coef_values: `{','.join(str(float(value)) for value in _rule_kl_coef_values(args))}`",
        f"delta_l2_coef_values: `{','.join(str(float(value)) for value in _nonnegative_float_values(args.delta_l2_coef_values, name='delta_l2_coef'))}`",
        f"delta_clip_values: `{','.join(str(float(value)) for value in _nonnegative_float_values(args.delta_clip_values, name='delta_clip'))}`",
        f"delta_clip_coef_values: `{','.join(str(float(value)) for value in _nonnegative_float_values(args.delta_clip_coef_values, name='delta_clip_coef'))}`",
        f"low_rank_flip_topk_values: `{','.join(str(int(value)) for value in _positive_int_values(args.low_rank_flip_topk_values, name='low_rank_flip_topk'))}`",
        f"low_rank_flip_penalty_coef_values: `{','.join(str(float(value)) for value in _nonnegative_float_values(args.low_rank_flip_penalty_coef_values, name='low_rank_flip_penalty_coef'))}`",
        f"weak_margin_threshold_values: `{','.join(str(float(value)) for value in _nonnegative_float_values(args.weak_margin_threshold_values, name='weak_margin_threshold'))}`",
        f"weak_margin_flip_penalty_coef_values: `{','.join(str(float(value)) for value in _nonnegative_float_values(args.weak_margin_flip_penalty_coef_values, name='weak_margin_flip_penalty_coef'))}`",
        f"support_policy: `{_support_policy_config(args)}`",
        f"delta_support_projection: `{_delta_support_projection_config(args)}`",
        f"actor_update_support: `{_actor_update_support_config(args)}`",
        f"entropy_coef: `{args.entropy_coef}`",
        f"pass_criteria: `{_pass_criteria(args)}`",
        f"adaptive_recovery: `{_adaptive_recovery_config(args)}`",
        f"movement_regularization: `{_movement_regularization_config(args)}`",
        f"movement_quality_gate: `{_movement_quality_gate_config(args)}`",
        f"fresh_validation: `{_fresh_validation_config(args)}`",
        f"eval_seed_registry_id: `{_eval_seed_registry_id(args)}`",
        f"eval_seed_hash: `{seed_registry_hash(_eval_seed_registry(args))}`",
        f"eval_scope: `{_eval_scope(args)}`",
        "eval_strength_note: `sanity check only; not duplicate strength evidence`",
        "",
        "## Results",
        "",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            int(item["source_config_id"]),
            float(item["rule_score_scale"]),
            float(item["behavior_temperature"]),
            float(item["lr"]),
            int(item["update_epochs"]),
            float(item["clip_eps"]),
            float(item["rule_kl_coef"]),
            float(item["delta_l2_coef"]),
            float(item["delta_clip"]),
            float(item["delta_clip_coef"]),
            int(item["low_rank_flip_topk"]),
            float(item["low_rank_flip_penalty_coef"]),
            float(item["weak_margin_threshold"]),
            float(item["weak_margin_flip_penalty_coef"]),
            str(item["support_policy_mode"]),
            str(item["delta_support_mode"]),
            int(item["delta_support_topk"]),
            float(item["delta_support_margin_threshold"]),
            str(item["outside_support_delta_mode"]),
            str(item["actor_update_support_mode"]),
            int(item["actor_update_topk"]),
            float(item["actor_update_margin_threshold"]),
        ),
    ):
        lines.append(
            "- "
            f"cfg={row['pilot_config_id']} "
            f"source={row['source_config_id']} "
            f"scale={row['rule_score_scale']:g} "
            f"temp={row['behavior_temperature']:g} "
            f"lr={row['lr']:g} "
            f"epochs={row['update_epochs']} "
            f"clip={row['clip_eps']:g} "
            f"rule_kl={row['rule_kl_coef']:g} "
            f"delta_l2={row['delta_l2_coef']:g} "
            f"delta_clip={row['delta_clip']:g}/{row['delta_clip_coef']:g} "
            f"low_rank={row['low_rank_flip_topk']}/{row['low_rank_flip_penalty_coef']:g} "
            f"weak_margin={row['weak_margin_threshold']:g}/{row['weak_margin_flip_penalty_coef']:g} "
            f"support_policy={row['support_policy_mode']} "
            f"delta_support={row['delta_support_mode']}/{row['delta_support_topk']}/{row['delta_support_margin_threshold']:g}/{row['outside_support_delta_mode']} "
            f"actor_support={row['actor_update_support_mode']}/{row['actor_update_topk']}/{row['actor_update_margin_threshold']:g} "
            f"pass={row.get('step38a_pass')} "
            f"non_top1={row['final_non_top1_selected_count']} "
            f"non_top1_pos={row['final_non_top1_positive_advantage_count']} "
            f"actor_kept={row['final_actor_update_kept_rate']:.6g} "
            f"dropped_pos={row['final_actor_update_dropped_positive_advantage_count']} "
            f"kept_non_top1_pos={row['final_actor_update_kept_non_top1_positive_advantage_count']} "
            f"top1_changed={row['final_untempered_post_top1_action_changed_rate']:.6g} "
            f"train_quality={row.get('train_movement_quality_gate_pass')} "
            f"changed_rank={row['final_untempered_post_changed_action_prior_rank_mean']:.6g} "
            f"rank_ge5={row['final_untempered_post_changed_to_rank_ge5_rate']:.6g} "
            f"margin_p50={row['final_untempered_post_changed_state_prior_margin_p50']:.6g} "
            f"low_rank_pen={row['final_tempered_post_update_low_rank_flip_penalty']:.6g} "
            f"weak_margin_pen={row['final_tempered_post_update_weak_margin_flip_penalty']:.6g} "
            f"fresh_top1={row.get('fresh_validation_top1_action_changed_rate', 0.0):.6g} "
            f"fresh_quality={row.get('fresh_validation_gate_pass')} "
            f"qualified_eval={row.get('qualified_for_eval')} "
            f"effective_margin={row['final_untempered_post_effective_margin_to_flip_mean']:.6g} "
            f"scaled_prior_margin={row['final_untempered_post_scaled_prior_margin_mean']:.6g} "
            f"t_kl={row['final_tempered_post_update_approx_kl']:.6g} "
            f"t_clip={row['final_tempered_post_update_clip_fraction']:.6g} "
            f"u_kl={row['final_untempered_post_update_approx_kl']:.6g} "
            f"u_clip={row['final_untempered_post_update_clip_fraction']:.6g} "
            f"delta_max={row['final_untempered_post_neural_delta_abs_max']:.6g} "
            f"recovery_extra={row.get('final_recovery_extra_epochs', 0)} "
            f"recovery_stop={row.get('final_recovery_stop_reason', '')} "
            f"eval_fourth={row['eval_fourth_rate']:.6g} "
            f"deal_in={row['eval_learner_deal_in_rate']:.6g}"
        )
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `tempered_ratio_pilot.json`",
            "- `summary.csv`",
            "- `iterations.csv`",
            "- `batch_steps.csv`",
            "- `advantage_audit.csv`",
        ]
    )
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
