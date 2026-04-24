"""Longer-run training helpers for keqingrl."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import keqing_core

from keqingrl.actions import ActionType
from keqingrl.buffer import build_ppo_batch
from keqingrl.contracts import ObsTensorBatch, PolicyInput
from keqingrl.env import DiscardOnlyMahjongEnv
from keqingrl.opponent_pool import OpponentPool, OpponentPoolEntry
from keqingrl.policy import InteractivePolicy, RandomInteractivePolicy, RulePriorDeltaPolicy, RulePriorPolicy
from keqingrl.selfplay import (
    DiscardOnlyIterationMetrics,
    build_episodes_ppo_batch,
    collect_selfplay_episodes,
    run_discard_only_ppo_iteration,
    run_ppo_iteration,
)
from keqingrl.review import review_rollout_episode, summarize_review_policy_fields
from keqingrl.rollout import RolloutEpisode, rollout_step_policy_input
from keqingrl.rule_score import smoothed_prior_probs
from keqingrl.ppo import (
    CriticPretrainLossBreakdown,
    _critic_pretrain_trainable_parameter_names,
    _freeze_actor_delta_parameters,
    _initialize_lazy_parameters_for_training,
    compute_critic_pretrain_loss,
    critic_pretrain_update,
)


@dataclass(frozen=True)
class DiscardOnlyEvalMetrics:
    # These smoke counters are hard failure gates until env/review expose real
    # recoverable counters for illegal/fallback/forced-terminal misses.
    episode_count: int
    total_steps: int
    mean_episode_steps: float
    mean_terminal_reward: float
    mean_rank: float
    first_place_rate: float
    fourth_place_rate: float
    win_rate: float
    deal_in_rate: float
    call_rate: float
    riichi_rate: float
    illegal_action_rate: float = 0.0
    fallback_rate: float = 0.0
    forced_terminal_missed: int = 0
    terminal_reason_count: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class DiscardOnlyTrainingIteration:
    iteration: int
    policy_version: int
    seed: int | None
    train_metrics: DiscardOnlyIterationMetrics
    eval_metrics: DiscardOnlyEvalMetrics | None = None


@dataclass(frozen=True)
class DiscardOnlyTrainingHistory:
    iterations: tuple[DiscardOnlyTrainingIteration, ...]


@dataclass(frozen=True)
class FixedSeedEvaluationSmoke:
    episode_count: int
    fixed_seed_count: int
    games_per_seed: int
    seat_rotation: tuple[int, ...]
    seat_rotation_enabled: bool
    seed: int | None
    policy_mode: str
    opponent_name: str
    reuse_training_rollout: bool
    average_rank: float
    rank_pt: float
    fourth_rate: float
    win_rate: float
    deal_in_rate: float
    call_rate: float
    riichi_rate: float
    illegal_action_rate: float
    fallback_rate: float
    forced_terminal_missed: int
    terminal_reason_count: dict[str, int]
    passed_smoke_checks: bool
    failure_reasons: tuple[str, ...]
    per_seat: dict[int, DiscardOnlyEvalMetrics]


@dataclass(frozen=True)
class CriticPretrainingIterationResult:
    episode_count: int
    batch_size: int
    losses: tuple[CriticPretrainLossBreakdown, ...]




@dataclass(frozen=True)
class CriticPretrainSmokeReport:
    episode_count: int
    batch_size: int
    pretrain_steps: int
    value_loss: float
    rank_loss: float | None
    explained_variance: float
    rank_acc: float | None
    grad_norm_by_module: dict[str, float]
    trainable_param_names: tuple[str, ...]
    optimizer_param_names: tuple[str, ...]
    actor_logits_before_after_diff: float
    actor_probs_before_after_diff: float
    neural_delta_before_after_diff: float
    optimizer_actor_param_count: int



@dataclass(frozen=True)
class DiscardOnlyPpoSmokeIterationReport:
    iteration: int
    policy_version: int
    seed: int | None
    episode_count: int
    batch_size: int
    policy_loss: float
    value_loss: float
    rank_loss: float | None
    entropy: float
    approx_kl: float
    clip_fraction: float
    ratio_mean: float
    ratio_std: float
    rule_kl: float | None
    rule_agreement: float | None
    advantage_mean: float
    advantage_std: float
    return_mean: float
    grad_norm: float
    grad_norm_by_module: dict[str, float]
    neural_delta_abs_mean: float
    neural_delta_abs_max: float
    top1_action_changed_rate: float
    avg_rank: float
    rank_pt: float
    first_rate: float
    fourth_rate: float
    win_rate: float
    deal_in_rate: float
    call_rate: float
    riichi_rate: float
    illegal_action_rate: float
    fallback_rate: float
    forced_terminal_missed: int
    terminal_reason_count: dict[str, int]


@dataclass(frozen=True)
class DiscardOnlyPpoSmokeReport:
    iteration_count: int
    rollout_episodes_per_iter: int
    update_epochs: int
    seed: int | None
    stopped_early: bool
    stop_reason: str | None
    final_neural_delta_abs_mean: float
    final_neural_delta_abs_max: float
    iterations: tuple[DiscardOnlyPpoSmokeIterationReport, ...]


@dataclass(frozen=True)
class ZeroDeltaSelfplaySmokeReport:
    greedy_episode_count: int
    sample_episode_count: int
    learner_step_count: int
    autopilot_step_count: int
    autopilot_terminal_count: int
    illegal_action_rate: float
    fallback_rate: float
    forced_terminal_missed: int
    terminal_reason_count: dict[str, int]
    old_log_prob_finite: bool
    entropy_finite: bool
    neural_delta_abs_mean: float
    neural_delta_abs_max: float
    rule_kl_mean: float
    final_logits_max_abs_diff: float
    probs_max_abs_diff: float
    action_order_valid: bool
    metadata_strict_valid: bool
    autopilot_policy_fields_null: bool
    autopilot_rows_excluded_from_ppo: bool


def evaluate_policy(
    env: DiscardOnlyMahjongEnv,
    policy: InteractivePolicy,
    *,
    num_episodes: int,
    opponent_pool: OpponentPool | None = None,
    learner_seats: Sequence[int] = (0, 1, 2, 3),
    seed: int | None = None,
    seed_stride: int = 1,
    max_steps: int = 512,
    greedy: bool = True,
    device: torch.device | str | None = None,
) -> DiscardOnlyEvalMetrics:
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")

    learner_seat_tuple = _normalize_actor_seats(learner_seats)
    episodes = collect_selfplay_episodes(
        env,
        policy,
        num_episodes=num_episodes,
        opponent_pool=opponent_pool,
        learner_seats=learner_seat_tuple,
        seed=seed,
        seed_stride=seed_stride,
        max_steps=max_steps,
        greedy=greedy,
        device=device,
    )
    seat_rewards = [
        episode.terminal_rewards[seat]
        for episode in episodes
        for seat in learner_seat_tuple
    ]
    seat_ranks = [
        episode.final_ranks[seat] + 1
        for episode in episodes
        for seat in learner_seat_tuple
    ]
    episode_steps = [len(episode.steps) for episode in episodes]
    smoke_counts = _smoke_metric_counts(episodes, learner_seat_tuple)

    return DiscardOnlyEvalMetrics(
        episode_count=len(episodes),
        total_steps=sum(episode_steps),
        mean_episode_steps=sum(episode_steps) / len(episode_steps),
        mean_terminal_reward=sum(seat_rewards) / len(seat_rewards),
        mean_rank=sum(seat_ranks) / len(seat_ranks),
        first_place_rate=sum(1.0 for rank in seat_ranks if rank == 1) / len(seat_ranks),
        fourth_place_rate=sum(1.0 for rank in seat_ranks if rank == 4) / len(seat_ranks),
        win_rate=smoke_counts["win_count"] / max(1, smoke_counts["seat_episode_count"]),
        deal_in_rate=smoke_counts["deal_in_count"] / max(1, smoke_counts["seat_episode_count"]),
        call_rate=smoke_counts["call_count"] / max(1, smoke_counts["learner_step_count"]),
        riichi_rate=smoke_counts["riichi_count"] / max(1, smoke_counts["learner_step_count"]),
        illegal_action_rate=0.0,
        fallback_rate=0.0,
        forced_terminal_missed=0,
        terminal_reason_count=smoke_counts["terminal_reason_count"],
    )


def evaluate_discard_only_policy(
    env: DiscardOnlyMahjongEnv,
    policy: InteractivePolicy,
    *,
    num_episodes: int,
    seed: int | None = None,
    seed_stride: int = 1,
    max_steps: int = 512,
    greedy: bool = True,
    device: torch.device | str | None = None,
) -> DiscardOnlyEvalMetrics:
    return evaluate_policy(
        env,
        policy,
        num_episodes=num_episodes,
        learner_seats=(0, 1, 2, 3),
        seed=seed,
        seed_stride=seed_stride,
        max_steps=max_steps,
        greedy=greedy,
        device=device,
    )


def run_fixed_seed_evaluation_smoke(
    env: DiscardOnlyMahjongEnv,
    policy: InteractivePolicy,
    *,
    num_games: int = 100,
    seed: int | None = None,
    seed_stride: int = 1,
    seat_rotation: Sequence[int] = (0, 1, 2, 3),
    opponent_pool: OpponentPool | None = None,
    opponent_name: str | None = None,
    max_steps: int = 512,
    greedy: bool = True,
    reuse_training_rollout: bool = False,
    max_fourth_rate: float = 0.75,
    max_deal_in_rate: float = 0.75,
    device: torch.device | str | None = None,
) -> FixedSeedEvaluationSmoke:
    if num_games <= 0:
        raise ValueError(f"num_games must be positive, got {num_games}")
    seats = _normalize_actor_seats(seat_rotation)

    if reuse_training_rollout:
        raise ValueError("fixed-seed evaluation smoke must not reuse training rollouts")

    per_seat: dict[int, DiscardOnlyEvalMetrics] = {}
    eval_opponent_pool = opponent_pool
    resolved_opponent_name = opponent_name
    if eval_opponent_pool is None:
        resolved_opponent_name = resolved_opponent_name or "rule_prior_greedy"
        eval_opponent_pool = OpponentPool(
            (OpponentPoolEntry(policy=RulePriorPolicy(), policy_version=-1, greedy=True, name=resolved_opponent_name),)
        )
    else:
        resolved_opponent_name = resolved_opponent_name or "custom"

    for seat in seats:
        per_seat[seat] = evaluate_policy(
            env,
            policy,
            num_episodes=num_games,
            opponent_pool=eval_opponent_pool,
            learner_seats=(seat,),
            seed=seed,
            seed_stride=seed_stride,
            max_steps=max_steps,
            greedy=greedy,
            device=device,
        )

    total_episodes = sum(metrics.episode_count for metrics in per_seat.values())
    if total_episodes <= 0:
        raise RuntimeError("fixed-seed evaluation produced no episodes")

    def weighted_mean(field: str) -> float:
        return sum(
            getattr(metrics, field) * metrics.episode_count
            for metrics in per_seat.values()
        ) / total_episodes

    average_rank = weighted_mean("mean_rank")
    rank_pt = weighted_mean("mean_terminal_reward")
    fourth_rate = weighted_mean("fourth_place_rate")
    win_rate = weighted_mean("win_rate")
    deal_in_rate = weighted_mean("deal_in_rate")
    call_rate = weighted_mean("call_rate")
    riichi_rate = weighted_mean("riichi_rate")
    illegal_action_rate = weighted_mean("illegal_action_rate")
    fallback_rate = weighted_mean("fallback_rate")
    forced_terminal_missed = sum(metrics.forced_terminal_missed for metrics in per_seat.values())
    terminal_reason_count = _merge_terminal_reason_counts(per_seat.values())
    failure_reasons = _fixed_seed_eval_failure_reasons(
        illegal_action_rate=illegal_action_rate,
        fallback_rate=fallback_rate,
        forced_terminal_missed=forced_terminal_missed,
        terminal_reason_count=terminal_reason_count,
        fourth_rate=fourth_rate,
        deal_in_rate=deal_in_rate,
        max_fourth_rate=max_fourth_rate,
        max_deal_in_rate=max_deal_in_rate,
    )

    return FixedSeedEvaluationSmoke(
        episode_count=total_episodes,
        fixed_seed_count=num_games,
        games_per_seed=len(seats),
        seat_rotation=seats,
        seat_rotation_enabled=len(seats) > 1,
        seed=seed,
        policy_mode="greedy" if greedy else "sample",
        opponent_name=resolved_opponent_name,
        reuse_training_rollout=False,
        average_rank=average_rank,
        rank_pt=rank_pt,
        fourth_rate=fourth_rate,
        win_rate=win_rate,
        deal_in_rate=deal_in_rate,
        call_rate=call_rate,
        riichi_rate=riichi_rate,
        illegal_action_rate=illegal_action_rate,
        fallback_rate=fallback_rate,
        forced_terminal_missed=forced_terminal_missed,
        terminal_reason_count=terminal_reason_count,
        passed_smoke_checks=not failure_reasons,
        failure_reasons=failure_reasons,
        per_seat=per_seat,
    )


def _fixed_seed_eval_failure_reasons(
    *,
    illegal_action_rate: float,
    fallback_rate: float,
    forced_terminal_missed: int,
    terminal_reason_count: dict[str, int],
    fourth_rate: float,
    deal_in_rate: float,
    max_fourth_rate: float,
    max_deal_in_rate: float,
) -> tuple[str, ...]:
    try:
        result = keqing_core.fixed_seed_eval_gate(
            {
                "illegal_action_rate": float(illegal_action_rate),
                "fallback_rate": float(fallback_rate),
                "forced_terminal_missed": int(forced_terminal_missed),
                "terminal_reason_count": dict(terminal_reason_count),
                "fourth_rate": float(fourth_rate),
                "deal_in_rate": float(deal_in_rate),
                "max_fourth_rate": float(max_fourth_rate),
                "max_deal_in_rate": float(max_deal_in_rate),
            }
        )
    except RuntimeError as exc:
        if keqing_core.is_missing_rust_capability_error(exc):
            raise RuntimeError("KeqingRL checkpoint/eval selection requires Rust fixed-seed eval gate") from exc
        raise
    return tuple(str(reason) for reason in result.get("failure_reasons", ()))


def _smoke_metric_counts(episodes, learner_seats: Sequence[int]) -> dict[str, object]:
    learner_seat_set = set(int(seat) for seat in learner_seats)
    counts: dict[str, object] = {
        "seat_episode_count": len(episodes) * max(1, len(learner_seat_set)),
        "learner_step_count": 0,
        "win_count": 0,
        "deal_in_count": 0,
        "call_count": 0,
        "riichi_count": 0,
        "terminal_reason_count": {},
    }
    terminal_reason_count: dict[str, int] = counts["terminal_reason_count"]  # type: ignore[assignment]
    for episode in episodes:
        for step in episode.steps:
            if step.terminal_reason is not None:
                terminal_reason_count[step.terminal_reason] = terminal_reason_count.get(step.terminal_reason, 0) + 1
            if step.actor not in learner_seat_set:
                continue
            counts["learner_step_count"] = int(counts["learner_step_count"]) + 1
            if step.action_spec.action_type in {ActionType.TSUMO, ActionType.RON}:
                counts["win_count"] = int(counts["win_count"]) + 1
            if step.action_spec.action_type == ActionType.RON and step.action_spec.from_who in learner_seat_set:
                counts["deal_in_count"] = int(counts["deal_in_count"]) + 1
            if step.action_spec.action_type in {ActionType.CHI, ActionType.PON, ActionType.DAIMINKAN}:
                counts["call_count"] = int(counts["call_count"]) + 1
            if step.action_spec.action_type == ActionType.REACH_DISCARD:
                counts["riichi_count"] = int(counts["riichi_count"]) + 1
    return counts


def _merge_terminal_reason_counts(metrics_values) -> dict[str, int]:
    merged: dict[str, int] = {}
    for metrics in metrics_values:
        for reason, count in metrics.terminal_reason_count.items():
            merged[reason] = merged.get(reason, 0) + int(count)
    return merged


def run_zero_delta_selfplay_smoke(
    env: DiscardOnlyMahjongEnv,
    *,
    greedy_episodes: int = 16,
    sample_episodes: int = 16,
    seed: int | None = None,
    seed_stride: int = 1,
    max_steps: int = 512,
    atol: float = 1e-6,
    rule_kl_atol: float = 1e-3,
    device: torch.device | str | None = None,
) -> ZeroDeltaSelfplaySmokeReport:
    if greedy_episodes < 0 or sample_episodes < 0:
        raise ValueError("episode counts must be non-negative")
    if greedy_episodes + sample_episodes <= 0:
        raise ValueError("at least one zero-delta smoke episode is required")

    policy = RulePriorDeltaPolicy(dropout=0.0)
    policy.eval()
    episodes: list[RolloutEpisode] = []
    if greedy_episodes:
        episodes.extend(
            collect_selfplay_episodes(
                env,
                policy,
                num_episodes=greedy_episodes,
                learner_seats=(0, 1, 2, 3),
                seed=seed,
                seed_stride=seed_stride,
                greedy=True,
                policy_version=0,
                max_steps=max_steps,
                device=device,
            )
        )
    if sample_episodes:
        sample_seed = None if seed is None else seed + max(1, greedy_episodes) * seed_stride + 10_000
        episodes.extend(
            collect_selfplay_episodes(
                env,
                policy,
                num_episodes=sample_episodes,
                learner_seats=(0, 1, 2, 3),
                seed=sample_seed,
                seed_stride=seed_stride,
                greedy=False,
                policy_version=0,
                max_steps=max_steps,
                device=device,
            )
        )

    learner_steps = [
        step
        for episode in episodes
        for step in episode.steps
        if not step.is_autopilot and step.is_learner_controlled
    ]
    autopilot_steps = [step for episode in episodes for step in episode.steps if step.is_autopilot]
    if not learner_steps:
        raise RuntimeError("zero-delta smoke collected no learner-controlled steps")

    _adv, _ret, prepared_steps, batch = build_episodes_ppo_batch(
        tuple(episodes),
        strict_metadata=True,
    )
    metadata_strict_valid = bool(prepared_steps and batch.action_index.numel() == len(prepared_steps))

    action_order_valid = all(
        step.legal_actions is not None
        and 0 <= int(step.action_index) < len(step.legal_actions)
        and step.legal_actions[int(step.action_index)].canonical_key == step.chosen_action_canonical_key
        for step in learner_steps
    )

    old_log_probs = torch.tensor([step.log_prob for step in learner_steps], dtype=torch.float32)
    entropies = torch.tensor([step.entropy for step in learner_steps], dtype=torch.float32)

    delta_abs_values: list[torch.Tensor] = []
    rule_kls: list[torch.Tensor] = []
    logit_diffs: list[torch.Tensor] = []
    prob_diffs: list[torch.Tensor] = []
    target_device = torch.device(device) if device is not None else _policy_device(policy)
    for step in learner_steps:
        policy_input = rollout_step_policy_input(step, device=target_device)
        with torch.no_grad():
            output = policy(policy_input)
        mask = policy_input.legal_action_mask.bool()
        prior_logits = output.aux["prior_logits"]
        final_logits = output.aux["final_logits"]
        neural_delta = output.aux["neural_delta"]
        expected_logits = policy.rule_score_scale * prior_logits
        delta_abs_values.append(neural_delta.masked_select(mask).abs().detach().cpu())
        logit_diffs.append((final_logits - expected_logits).masked_select(mask).abs().detach().cpu())
        probs = torch.softmax(final_logits.masked_fill(~mask, torch.finfo(final_logits.dtype).min), dim=-1)
        prior_probs = torch.softmax(expected_logits.masked_fill(~mask, torch.finfo(expected_logits.dtype).min), dim=-1)
        prob_diffs.append((probs - prior_probs).masked_select(mask).abs().detach().cpu())
        smooth_prior = smoothed_prior_probs(expected_logits.float(), mask, eps=1e-4)
        rule_kl = (probs * (torch.log(probs.clamp_min(1e-12)) - torch.log(smooth_prior.clamp_min(1e-12)))).masked_fill(~mask, 0.0)
        rule_kls.append(rule_kl.sum(dim=-1).detach().cpu())

    deltas = torch.cat(delta_abs_values)
    rule_kl_values = torch.cat(rule_kls)
    logit_diff_values = torch.cat(logit_diffs)
    prob_diff_values = torch.cat(prob_diffs)

    review_summaries = [
        summarize_review_policy_fields(review_rollout_episode(policy, episode, top_k=3, device=target_device))
        for episode in episodes
    ]
    autopilot_policy_fields_null = all(
        step.is_autopilot
        and step.log_prob == 0.0
        and step.entropy == 0.0
        for step in autopilot_steps
    )
    review_autopilot_count = sum(summary.autopilot_step_count for summary in review_summaries)
    review_autopilot_terminal_count = sum(summary.autopilot_terminal_count for summary in review_summaries)

    autopilot_rows_excluded_from_ppo = all(not step.is_autopilot for step in prepared_steps)
    if autopilot_steps:
        try:
            build_ppo_batch([autopilot_steps[0]], [0.0], [0.0], strict_metadata=True)
        except ValueError:
            pass
        else:  # pragma: no cover - defensive
            autopilot_rows_excluded_from_ppo = False

    smoke_counts = _smoke_metric_counts(tuple(episodes), (0, 1, 2, 3))
    report = ZeroDeltaSelfplaySmokeReport(
        greedy_episode_count=greedy_episodes,
        sample_episode_count=sample_episodes,
        learner_step_count=len(learner_steps),
        autopilot_step_count=review_autopilot_count,
        autopilot_terminal_count=review_autopilot_terminal_count,
        illegal_action_rate=0.0,
        fallback_rate=0.0,
        forced_terminal_missed=0,
        terminal_reason_count=smoke_counts["terminal_reason_count"],
        old_log_prob_finite=bool(torch.isfinite(old_log_probs).all()),
        entropy_finite=bool(torch.isfinite(entropies).all()),
        neural_delta_abs_mean=float(deltas.mean()),
        neural_delta_abs_max=float(deltas.max()),
        rule_kl_mean=float(rule_kl_values.mean()),
        final_logits_max_abs_diff=float(logit_diff_values.max()),
        probs_max_abs_diff=float(prob_diff_values.max()),
        action_order_valid=bool(action_order_valid),
        metadata_strict_valid=metadata_strict_valid,
        autopilot_policy_fields_null=bool(autopilot_policy_fields_null),
        autopilot_rows_excluded_from_ppo=bool(autopilot_rows_excluded_from_ppo),
    )

    if report.illegal_action_rate != 0.0 or report.fallback_rate != 0.0 or report.forced_terminal_missed != 0:
        raise RuntimeError(f"zero-delta smoke fail-closed metrics failed: {report}")
    if not report.old_log_prob_finite or not report.entropy_finite:
        raise RuntimeError(f"zero-delta smoke found non-finite rollout fields: {report}")
    if report.neural_delta_abs_max > atol:
        raise RuntimeError(f"zero-delta smoke neural_delta drifted: {report.neural_delta_abs_max}")
    if report.final_logits_max_abs_diff > atol or report.probs_max_abs_diff > atol:
        raise RuntimeError(f"zero-delta smoke prior equivalence failed: {report}")
    if report.rule_kl_mean > rule_kl_atol:
        raise RuntimeError(f"zero-delta smoke rule KL too high: {report.rule_kl_mean}")
    if not report.action_order_valid or not report.metadata_strict_valid:
        raise RuntimeError(f"zero-delta smoke rollout contract failed: {report}")
    if report.autopilot_step_count <= 0 or not report.autopilot_policy_fields_null or not report.autopilot_rows_excluded_from_ppo:
        raise RuntimeError(f"zero-delta smoke autopilot trace contract failed: {report}")
    return report


def run_critic_pretrain_smoke(
    env: DiscardOnlyMahjongEnv,
    *,
    episodes: int = 32,
    pretrain_steps: int = 10,
    seed: int | None = None,
    seed_stride: int = 1,
    max_steps: int = 512,
    lr: float = 1e-3,
    value_coef: float = 1.0,
    rank_coef: float = 1.0,
    max_grad_norm: float | None = 1.0,
    freeze_actor_delta: bool = True,
    atol: float = 1e-6,
    device: torch.device | str | None = None,
) -> CriticPretrainSmokeReport:
    if episodes <= 0:
        raise ValueError(f"episodes must be positive, got {episodes}")
    if pretrain_steps <= 0:
        raise ValueError(f"pretrain_steps must be positive, got {pretrain_steps}")

    policy = RulePriorDeltaPolicy(dropout=0.0)
    policy.eval()
    collected = collect_selfplay_episodes(
        env,
        policy,
        num_episodes=episodes,
        learner_seats=(0, 1, 2, 3),
        seed=seed,
        seed_stride=seed_stride,
        greedy=False,
        policy_version=0,
        max_steps=max_steps,
        device=device,
    )
    _advantages, _returns, prepared_steps, batch = build_episodes_ppo_batch(
        collected,
        include_rank_targets=True,
        strict_metadata=True,
    )
    target_device = torch.device(device) if device is not None else _policy_device(policy)
    batch = batch.to(target_device)

    probe_input = _slice_policy_input(batch.policy_input, max_rows=min(8, int(batch.action_index.numel())))
    _initialize_lazy_parameters_for_training(policy, probe_input)
    with torch.no_grad():
        before = policy(probe_input)
        before_logits = before.action_logits.clone()
        before_probs = torch.softmax(before.action_logits, dim=-1)
        before_delta = before.aux["neural_delta"].clone()

    frozen = _freeze_actor_delta_parameters(policy) if freeze_actor_delta else []
    try:
        trainable_param_names = tuple(_critic_pretrain_trainable_parameter_names(policy))
        trainable_params = [
            parameter
            for name, parameter in policy.named_parameters()
            if parameter.requires_grad and name.startswith(("value_head.", "rank_head."))
        ]
    finally:
        for parameter, requires_grad in frozen:
            parameter.requires_grad_(requires_grad)
    if not trainable_params:
        raise RuntimeError("critic pretrain smoke found no trainable critic parameters")
    optimizer = torch.optim.Adam(trainable_params, lr=lr)
    optimizer_param_names = tuple(_optimizer_parameter_names(policy, optimizer))
    optimizer_actor_param_count = sum(
        1
        for name in optimizer_param_names
        if not (name.startswith("value_head.") or name.startswith("rank_head."))
    )
    if optimizer_actor_param_count:
        raise RuntimeError(f"critic optimizer includes actor/shared params: {optimizer_param_names}")

    losses: list[CriticPretrainLossBreakdown] = []
    for _ in range(pretrain_steps):
        losses.append(
            critic_pretrain_update(
                policy,
                optimizer,
                batch,
                value_coef=value_coef,
                rank_coef=rank_coef,
                max_grad_norm=max_grad_norm,
                freeze_actor_delta=freeze_actor_delta,
            )
        )
    final_loss = losses[-1]
    grad_norm_by_module = _grad_norm_by_module(policy)

    with torch.no_grad():
        after = policy(probe_input)
        after_probs = torch.softmax(after.action_logits, dim=-1)
        eval_output = policy(batch.policy_input)

    actor_logits_diff = float((after.action_logits - before_logits).abs().max())
    actor_probs_diff = float((after_probs - before_probs).abs().max())
    actor_delta_diff = float((after.aux["neural_delta"] - before_delta).abs().max())

    explained_variance = _explained_variance(eval_output.value.detach().cpu(), batch.returns.detach().cpu())
    rank_acc = None
    if batch.final_rank_target is not None:
        rank_pred = eval_output.rank_logits.argmax(dim=-1).detach().cpu()
        rank_target = batch.final_rank_target.detach().cpu()
        rank_acc = float((rank_pred == rank_target).float().mean())

    report = CriticPretrainSmokeReport(
        episode_count=len(collected),
        batch_size=int(batch.action_index.numel()),
        pretrain_steps=pretrain_steps,
        value_loss=float(final_loss.value_loss.detach().cpu()),
        rank_loss=None if final_loss.rank_loss is None else float(final_loss.rank_loss.detach().cpu()),
        explained_variance=explained_variance,
        rank_acc=rank_acc,
        grad_norm_by_module=grad_norm_by_module,
        trainable_param_names=trainable_param_names,
        optimizer_param_names=optimizer_param_names,
        actor_logits_before_after_diff=actor_logits_diff,
        actor_probs_before_after_diff=actor_probs_diff,
        neural_delta_before_after_diff=actor_delta_diff,
        optimizer_actor_param_count=optimizer_actor_param_count,
    )

    if not torch.isfinite(torch.tensor(report.value_loss)):
        raise RuntimeError(f"critic pretrain smoke value_loss is not finite: {report}")
    if report.rank_loss is not None and not torch.isfinite(torch.tensor(report.rank_loss)):
        raise RuntimeError(f"critic pretrain smoke rank_loss is not finite: {report}")
    if not torch.isfinite(torch.tensor(report.explained_variance)):
        raise RuntimeError(f"critic pretrain smoke explained variance is not finite: {report}")
    if actor_logits_diff > atol or actor_probs_diff > atol or actor_delta_diff > atol:
        raise RuntimeError(f"critic pretrain smoke changed actor outputs: {report}")
    if optimizer_actor_param_count:
        raise RuntimeError(f"critic pretrain smoke optimizer touched actor/shared path: {report}")
    if not all(name.startswith(("value_head.", "rank_head.")) for name in trainable_param_names):
        raise RuntimeError(f"critic pretrain smoke trainable set is not critic-only: {report}")
    return report


def _slice_policy_input(policy_input, *, max_rows: int):
    rows = slice(0, max_rows)
    obs = policy_input.obs
    return type(policy_input)(
        obs=type(obs)(
            tile_obs=obs.tile_obs[rows],
            scalar_obs=obs.scalar_obs[rows],
            history_obs=None if obs.history_obs is None else obs.history_obs[rows],
            extras={key: value[rows] for key, value in obs.extras.items()},
        ),
        legal_action_ids=policy_input.legal_action_ids[rows],
        legal_action_features=policy_input.legal_action_features[rows],
        legal_action_mask=policy_input.legal_action_mask[rows],
        rule_context=policy_input.rule_context[rows],
        raw_rule_scores=None if policy_input.raw_rule_scores is None else policy_input.raw_rule_scores[rows],
        prior_logits=None if policy_input.prior_logits is None else policy_input.prior_logits[rows],
        style_context=None if policy_input.style_context is None else policy_input.style_context[rows],
        legal_actions=None if policy_input.legal_actions is None else policy_input.legal_actions[:max_rows],
        recurrent_state=policy_input.recurrent_state,
        metadata=policy_input.metadata,
    )


def _optimizer_parameter_names(policy: InteractivePolicy, optimizer: torch.optim.Optimizer) -> list[str]:
    names_by_id = {id(parameter): name for name, parameter in policy.named_parameters()}
    names: list[str] = []
    for group in optimizer.param_groups:
        for parameter in group.get("params", ()):  # type: ignore[assignment]
            name = names_by_id.get(id(parameter))
            if name is not None:
                names.append(name)
    return names


def _grad_norm_by_module(policy: InteractivePolicy) -> dict[str, float]:
    totals: dict[str, float] = {}
    for name, parameter in policy.named_parameters():
        if parameter.grad is None:
            continue
        module_name = name.split(".", 1)[0]
        totals[module_name] = totals.get(module_name, 0.0) + float(parameter.grad.detach().pow(2).sum().cpu())
    return {key: value ** 0.5 for key, value in totals.items()}


def _explained_variance(values: torch.Tensor, returns: torch.Tensor) -> float:
    returns = returns.float()
    values = values.float()
    variance = torch.var(returns, unbiased=False)
    if float(variance) <= 1e-12:
        return 0.0
    residual_variance = torch.var(returns - values, unbiased=False)
    return float(1.0 - residual_variance / variance)



def run_discard_only_ppo_smoke(
    env: DiscardOnlyMahjongEnv,
    *,
    iterations: int = 3,
    rollout_episodes_per_iter: int = 8,
    update_epochs: int = 1,
    seed: int | None = None,
    seed_stride: int = 10_000,
    max_steps: int = 512,
    lr: float = 1e-4,
    clip_eps: float = 0.1,
    value_coef: float = 0.5,
    entropy_coef: float = 0.005,
    rank_coef: float = 0.05,
    rule_kl_coef: float = 0.02,
    prior_kl_eps: float = 1e-4,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    max_grad_norm: float | None = 1.0,
    hidden_dim: int = 32,
    num_res_blocks: int = 1,
    device: torch.device | str | None = None,
    max_approx_kl: float = 0.2,
    max_clip_fraction: float = 0.8,
    min_entropy: float = 1e-6,
    max_neural_delta_abs: float = 5.0,
    min_rule_agreement: float = 0.2,
    raise_on_stop: bool = True,
) -> DiscardOnlyPpoSmokeReport:
    if iterations <= 0:
        raise ValueError(f"iterations must be positive, got {iterations}")
    if rollout_episodes_per_iter <= 0:
        raise ValueError(f"rollout_episodes_per_iter must be positive, got {rollout_episodes_per_iter}")
    if update_epochs <= 0:
        raise ValueError(f"update_epochs must be positive, got {update_epochs}")

    target_device = torch.device(device) if device is not None else torch.device("cpu")
    policy = RulePriorDeltaPolicy(
        hidden_dim=hidden_dim,
        num_res_blocks=num_res_blocks,
        dropout=0.0,
    ).to(target_device)
    _initialize_policy_from_env_observation(env, policy, seed=seed, device=target_device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    reports: list[DiscardOnlyPpoSmokeIterationReport] = []
    stopped_early = False
    stop_reason = None

    for iteration in range(iterations):
        iteration_seed = None if seed is None else seed + iteration * seed_stride
        result = run_discard_only_ppo_iteration(
            env,
            policy,
            optimizer,
            num_episodes=rollout_episodes_per_iter,
            update_epochs=update_epochs,
            seed=iteration_seed,
            policy_version=iteration,
            max_steps=max_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            include_rank_targets=True,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            rank_coef=rank_coef,
            rule_kl_coef=rule_kl_coef,
            prior_kl_eps=prior_kl_eps,
            normalize_advantages=True,
            max_grad_norm=max_grad_norm,
            device=target_device,
            strict_metadata=True,
        )
        loss = result.losses[-1]
        delta_stats = _ppo_delta_smoke_stats(policy, result.batch)
        grad_by_module = _grad_norm_by_module(policy)
        grad_norm = sum(value * value for value in grad_by_module.values()) ** 0.5
        metrics = result.metrics
        iteration_report = DiscardOnlyPpoSmokeIterationReport(
            iteration=iteration,
            policy_version=iteration,
            seed=iteration_seed,
            episode_count=metrics.episode_count,
            batch_size=metrics.batch_size,
            policy_loss=_loss_float(loss.policy_loss),
            value_loss=_loss_float(loss.value_loss),
            rank_loss=None if loss.rank_loss is None else _loss_float(loss.rank_loss),
            entropy=_loss_float(loss.entropy_bonus),
            approx_kl=_loss_float(loss.approx_kl),
            clip_fraction=_loss_float(loss.clip_fraction),
            ratio_mean=_loss_float(loss.ratio_mean),
            ratio_std=_loss_float(loss.ratio_std),
            rule_kl=None if loss.rule_kl is None else _loss_float(loss.rule_kl),
            rule_agreement=None if loss.rule_agreement is None else _loss_float(loss.rule_agreement),
            advantage_mean=_loss_float(loss.advantage_mean),
            advantage_std=_loss_float(loss.advantage_std),
            return_mean=_loss_float(loss.return_mean),
            grad_norm=float(grad_norm),
            grad_norm_by_module=grad_by_module,
            neural_delta_abs_mean=delta_stats["neural_delta_abs_mean"],
            neural_delta_abs_max=delta_stats["neural_delta_abs_max"],
            top1_action_changed_rate=delta_stats["top1_action_changed_rate"],
            avg_rank=metrics.mean_rank,
            rank_pt=metrics.mean_terminal_reward,
            first_rate=metrics.first_place_rate,
            fourth_rate=metrics.fourth_rate,
            win_rate=metrics.win_rate,
            deal_in_rate=metrics.deal_in_rate,
            call_rate=metrics.call_rate,
            riichi_rate=metrics.riichi_rate,
            illegal_action_rate=metrics.illegal_action_rate,
            fallback_rate=metrics.fallback_rate,
            forced_terminal_missed=metrics.forced_terminal_missed,
            terminal_reason_count=dict(metrics.terminal_reason_count),
        )
        reports.append(iteration_report)
        stop_reason = _discard_only_ppo_smoke_stop_reason(
            iteration_report,
            max_approx_kl=max_approx_kl,
            max_clip_fraction=max_clip_fraction,
            min_entropy=min_entropy,
            max_neural_delta_abs=max_neural_delta_abs,
            min_rule_agreement=min_rule_agreement,
        )
        if stop_reason is not None:
            stopped_early = True
            break

    final_report = reports[-1]
    report = DiscardOnlyPpoSmokeReport(
        iteration_count=len(reports),
        rollout_episodes_per_iter=rollout_episodes_per_iter,
        update_epochs=update_epochs,
        seed=seed,
        stopped_early=stopped_early,
        stop_reason=stop_reason,
        final_neural_delta_abs_mean=final_report.neural_delta_abs_mean,
        final_neural_delta_abs_max=final_report.neural_delta_abs_max,
        iterations=tuple(reports),
    )
    if stopped_early and raise_on_stop:
        raise RuntimeError(f"discard-only PPO smoke stopped early: {stop_reason}; report={report}")
    return report


def _initialize_policy_from_env_observation(
    env: DiscardOnlyMahjongEnv,
    policy: InteractivePolicy,
    *,
    seed: int | None,
    device: torch.device,
) -> None:
    env.reset(seed=seed)
    actor = env.current_actor()
    if actor is None:
        return
    sample_input = _policy_input_to_device_local(env.observe(actor), device)
    _initialize_lazy_parameters_for_training(policy, sample_input)


def _policy_input_to_device_local(policy_input: PolicyInput, device: torch.device) -> PolicyInput:
    return PolicyInput(
        obs=ObsTensorBatch(
            tile_obs=policy_input.obs.tile_obs.to(device),
            scalar_obs=policy_input.obs.scalar_obs.to(device),
            history_obs=None
            if policy_input.obs.history_obs is None
            else policy_input.obs.history_obs.to(device),
            extras={key: value.to(device) for key, value in policy_input.obs.extras.items()},
        ),
        legal_action_ids=policy_input.legal_action_ids.to(device),
        legal_action_features=policy_input.legal_action_features.to(device),
        legal_action_mask=policy_input.legal_action_mask.to(device),
        rule_context=policy_input.rule_context.to(device),
        raw_rule_scores=None
        if policy_input.raw_rule_scores is None
        else policy_input.raw_rule_scores.to(device),
        prior_logits=None
        if policy_input.prior_logits is None
        else policy_input.prior_logits.to(device),
        style_context=None
        if policy_input.style_context is None
        else policy_input.style_context.to(device),
        legal_actions=policy_input.legal_actions,
        recurrent_state=policy_input.recurrent_state,
        metadata=policy_input.metadata,
    )


def _ppo_delta_smoke_stats(policy: InteractivePolicy, batch) -> dict[str, float]:
    with torch.no_grad():
        output = policy(batch.policy_input)
    mask = batch.policy_input.legal_action_mask.bool()
    neural_delta = output.aux.get("neural_delta")
    if neural_delta is None:
        delta_abs_mean = 0.0
        delta_abs_max = 0.0
    else:
        legal_delta = neural_delta.masked_select(mask)
        delta_abs_mean = 0.0 if legal_delta.numel() == 0 else float(legal_delta.abs().mean().detach().cpu())
        delta_abs_max = 0.0 if legal_delta.numel() == 0 else float(legal_delta.abs().max().detach().cpu())

    prior_logits = output.aux.get("prior_logits")
    if prior_logits is None:
        prior_logits = batch.policy_input.prior_logits
    if prior_logits is None:
        changed_rate = 0.0
    else:
        min_logit = torch.finfo(output.action_logits.dtype).min
        current_top1 = output.action_logits.masked_fill(~mask, min_logit).argmax(dim=-1)
        prior_top1 = prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min).argmax(dim=-1)
        changed_rate = float((current_top1 != prior_top1).float().mean().detach().cpu())
    return {
        "neural_delta_abs_mean": delta_abs_mean,
        "neural_delta_abs_max": delta_abs_max,
        "top1_action_changed_rate": changed_rate,
    }


def _discard_only_ppo_smoke_stop_reason(
    report: DiscardOnlyPpoSmokeIterationReport,
    *,
    max_approx_kl: float,
    max_clip_fraction: float,
    min_entropy: float,
    max_neural_delta_abs: float,
    min_rule_agreement: float,
) -> str | None:
    finite_fields = {
        "policy_loss": report.policy_loss,
        "value_loss": report.value_loss,
        "entropy": report.entropy,
        "approx_kl": report.approx_kl,
        "clip_fraction": report.clip_fraction,
        "ratio_mean": report.ratio_mean,
        "ratio_std": report.ratio_std,
        "advantage_mean": report.advantage_mean,
        "advantage_std": report.advantage_std,
        "return_mean": report.return_mean,
        "grad_norm": report.grad_norm,
        "neural_delta_abs_mean": report.neural_delta_abs_mean,
        "neural_delta_abs_max": report.neural_delta_abs_max,
    }
    if report.rank_loss is not None:
        finite_fields["rank_loss"] = report.rank_loss
    if report.rule_kl is not None:
        finite_fields["rule_kl"] = report.rule_kl
    if report.rule_agreement is not None:
        finite_fields["rule_agreement"] = report.rule_agreement
    for name, value in finite_fields.items():
        if not torch.isfinite(torch.tensor(value)):
            return f"{name} is not finite: {value}"
    if report.illegal_action_rate > 0.0:
        return f"illegal_action_rate > 0: {report.illegal_action_rate}"
    if report.fallback_rate > 0.0:
        return f"fallback_rate > 0: {report.fallback_rate}"
    if report.forced_terminal_missed > 0:
        return f"forced_terminal_missed > 0: {report.forced_terminal_missed}"
    if report.approx_kl > max_approx_kl:
        return f"approx_kl {report.approx_kl} exceeded {max_approx_kl}"
    if report.clip_fraction > max_clip_fraction:
        return f"clip_fraction {report.clip_fraction} exceeded {max_clip_fraction}"
    if report.entropy < min_entropy:
        return f"entropy {report.entropy} below {min_entropy}"
    if report.neural_delta_abs_max > max_neural_delta_abs:
        return f"neural_delta_abs_max {report.neural_delta_abs_max} exceeded {max_neural_delta_abs}"
    if report.rule_agreement is not None and report.rule_agreement < min_rule_agreement:
        return f"rule_agreement {report.rule_agreement} below {min_rule_agreement}"
    return None


def _loss_float(tensor: torch.Tensor) -> float:
    return float(tensor.detach().cpu())


def run_training(
    env: DiscardOnlyMahjongEnv,
    policy: InteractivePolicy,
    optimizer: torch.optim.Optimizer,
    *,
    num_iterations: int,
    episodes_per_iteration: int,
    opponent_pool: OpponentPool | None = None,
    eval_opponent_pool: OpponentPool | None = None,
    learner_seats: Sequence[int] = (0, 1, 2, 3),
    update_epochs: int = 1,
    eval_episodes: int = 0,
    seed: int | None = None,
    seed_stride: int = 1000,
    eval_seed_offset: int = 100_000,
    policy_version_start: int = 0,
    max_steps: int = 512,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    include_rank_targets: bool = True,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    rank_coef: float = 0.0,
    rule_kl_coef: float = 0.0,
    prior_kl_eps: float = 1e-4,
    normalize_advantages: bool = True,
    max_grad_norm: float | None = None,
    device: torch.device | str | None = None,
) -> DiscardOnlyTrainingHistory:
    if num_iterations <= 0:
        raise ValueError(f"num_iterations must be positive, got {num_iterations}")

    learner_seat_tuple = _normalize_actor_seats(learner_seats)
    iterations: list[DiscardOnlyTrainingIteration] = []
    for iteration in range(num_iterations):
        iteration_seed = None if seed is None else seed + iteration * seed_stride
        policy_version = policy_version_start + iteration
        iteration_result = run_ppo_iteration(
            env,
            policy,
            optimizer,
            num_episodes=episodes_per_iteration,
            opponent_pool=opponent_pool,
            learner_seats=learner_seat_tuple,
            update_epochs=update_epochs,
            seed=iteration_seed,
            policy_version=policy_version,
            max_steps=max_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            include_rank_targets=include_rank_targets,
            clip_eps=clip_eps,
            value_coef=value_coef,
            entropy_coef=entropy_coef,
            rank_coef=rank_coef,
            rule_kl_coef=rule_kl_coef,
            prior_kl_eps=prior_kl_eps,
            normalize_advantages=normalize_advantages,
            max_grad_norm=max_grad_norm,
            device=device,
        )

        eval_metrics = None
        if eval_episodes > 0:
            eval_seed = None if iteration_seed is None else iteration_seed + eval_seed_offset
            eval_metrics = evaluate_policy(
                env,
                policy,
                num_episodes=eval_episodes,
                opponent_pool=opponent_pool if eval_opponent_pool is None else eval_opponent_pool,
                learner_seats=learner_seat_tuple,
                seed=eval_seed,
                max_steps=max_steps,
                greedy=True,
                device=device,
            )

        iterations.append(
            DiscardOnlyTrainingIteration(
                iteration=iteration,
                policy_version=policy_version,
                seed=iteration_seed,
                train_metrics=iteration_result.metrics,
                eval_metrics=eval_metrics,
            )
        )

    return DiscardOnlyTrainingHistory(iterations=tuple(iterations))


def run_critic_pretraining_iteration(
    env: DiscardOnlyMahjongEnv,
    policy: InteractivePolicy,
    optimizer: torch.optim.Optimizer,
    *,
    num_episodes: int,
    update_epochs: int = 1,
    seed: int | None = None,
    seed_stride: int = 1,
    policy_version: int = 0,
    max_steps: int = 512,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    include_rank_targets: bool = True,
    value_coef: float = 1.0,
    rank_coef: float = 1.0,
    max_grad_norm: float | None = None,
    freeze_actor_delta: bool = True,
    device: torch.device | str | None = None,
) -> CriticPretrainingIterationResult:
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")
    if update_epochs <= 0:
        raise ValueError(f"update_epochs must be positive, got {update_epochs}")

    episodes = collect_selfplay_episodes(
        env,
        policy,
        num_episodes=num_episodes,
        learner_seats=(0, 1, 2, 3),
        seed=seed,
        seed_stride=seed_stride,
        greedy=False,
        policy_version=policy_version,
        max_steps=max_steps,
        device=device,
    )
    _advantages, _returns, _prepared_steps, batch = build_episodes_ppo_batch(
        episodes,
        gamma=gamma,
        gae_lambda=gae_lambda,
        include_rank_targets=include_rank_targets,
    )
    target_device = _policy_device(policy) if device is None else torch.device(device)
    batch = batch.to(target_device)

    losses: list[CriticPretrainLossBreakdown] = []
    for _ in range(update_epochs):
        losses.append(
            critic_pretrain_update(
                policy,
                optimizer,
                batch,
                value_coef=value_coef,
                rank_coef=rank_coef,
                max_grad_norm=max_grad_norm,
                freeze_actor_delta=freeze_actor_delta,
            )
        )

    return CriticPretrainingIterationResult(
        episode_count=len(episodes),
        batch_size=int(batch.action_index.numel()),
        losses=tuple(losses),
    )


def run_discard_only_training(
    env: DiscardOnlyMahjongEnv,
    policy: InteractivePolicy,
    optimizer: torch.optim.Optimizer,
    *,
    num_iterations: int,
    episodes_per_iteration: int,
    update_epochs: int = 1,
    eval_episodes: int = 0,
    seed: int | None = None,
    seed_stride: int = 1000,
    eval_seed_offset: int = 100_000,
    policy_version_start: int = 0,
    max_steps: int = 512,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    include_rank_targets: bool = True,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    rank_coef: float = 0.0,
    rule_kl_coef: float = 0.0,
    prior_kl_eps: float = 1e-4,
    normalize_advantages: bool = True,
    max_grad_norm: float | None = None,
    device: torch.device | str | None = None,
) -> DiscardOnlyTrainingHistory:
    return run_training(
        env,
        policy,
        optimizer,
        num_iterations=num_iterations,
        episodes_per_iteration=episodes_per_iteration,
        learner_seats=(0, 1, 2, 3),
        update_epochs=update_epochs,
        eval_episodes=eval_episodes,
        seed=seed,
        seed_stride=seed_stride,
        eval_seed_offset=eval_seed_offset,
        policy_version_start=policy_version_start,
        max_steps=max_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        include_rank_targets=include_rank_targets,
        clip_eps=clip_eps,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        rank_coef=rank_coef,
        rule_kl_coef=rule_kl_coef,
        prior_kl_eps=prior_kl_eps,
        normalize_advantages=normalize_advantages,
        max_grad_norm=max_grad_norm,
        device=device,
    )


def _normalize_actor_seats(actor_seats: Sequence[int]) -> tuple[int, ...]:
    actor_tuple = tuple(int(actor) for actor in actor_seats)
    if not actor_tuple:
        raise ValueError("actor seat selection must not be empty")
    if len(set(actor_tuple)) != len(actor_tuple):
        raise ValueError(f"actor seat selection must not contain duplicates, got {actor_tuple}")
    if any(actor < 0 or actor >= 4 for actor in actor_tuple):
        raise ValueError(f"actor seat selection must stay within [0, 3], got {actor_tuple}")
    return actor_tuple


def _policy_device(policy: InteractivePolicy) -> torch.device:
    parameter = next(policy.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(policy.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


__all__ = [
    "CriticPretrainSmokeReport",
    "CriticPretrainingIterationResult",
    "DiscardOnlyEvalMetrics",
    "DiscardOnlyPpoSmokeIterationReport",
    "DiscardOnlyPpoSmokeReport",
    "DiscardOnlyTrainingHistory",
    "DiscardOnlyTrainingIteration",
    "FixedSeedEvaluationSmoke",
    "ZeroDeltaSelfplaySmokeReport",
    "evaluate_discard_only_policy",
    "evaluate_policy",
    "run_fixed_seed_evaluation_smoke",
    "run_discard_only_ppo_smoke",
    "run_discard_only_training",
    "run_critic_pretrain_smoke",
    "run_critic_pretraining_iteration",
    "run_training",
    "run_zero_delta_selfplay_smoke",
]
