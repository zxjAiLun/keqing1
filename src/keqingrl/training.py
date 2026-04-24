"""Longer-run training helpers for keqingrl."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from keqingrl.env import DiscardOnlyMahjongEnv
from keqingrl.opponent_pool import OpponentPool, OpponentPoolEntry
from keqingrl.policy import InteractivePolicy, RandomInteractivePolicy
from keqingrl.selfplay import (
    DiscardOnlyIterationMetrics,
    build_episodes_ppo_batch,
    collect_selfplay_episodes,
    run_ppo_iteration,
)
from keqingrl.ppo import CriticPretrainLossBreakdown, critic_pretrain_update


@dataclass(frozen=True)
class DiscardOnlyEvalMetrics:
    episode_count: int
    total_steps: int
    mean_episode_steps: float
    mean_terminal_reward: float
    mean_rank: float
    first_place_rate: float
    fourth_place_rate: float
    win_rate: float | None = None
    deal_in_rate: float | None = None
    call_rate: float | None = None
    riichi_rate: float | None = None


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
    seat_rotation: tuple[int, ...]
    seed: int | None
    average_rank: float
    rank_pt: float
    fourth_rate: float
    win_rate: float | None
    deal_in_rate: float | None
    call_rate: float | None
    riichi_rate: float | None
    per_seat: dict[int, DiscardOnlyEvalMetrics]


@dataclass(frozen=True)
class CriticPretrainingIterationResult:
    episode_count: int
    batch_size: int
    losses: tuple[CriticPretrainLossBreakdown, ...]


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

    return DiscardOnlyEvalMetrics(
        episode_count=len(episodes),
        total_steps=sum(episode_steps),
        mean_episode_steps=sum(episode_steps) / len(episode_steps),
        mean_terminal_reward=sum(seat_rewards) / len(seat_rewards),
        mean_rank=sum(seat_ranks) / len(seat_ranks),
        first_place_rate=sum(1.0 for rank in seat_ranks if rank == 1) / len(seat_ranks),
        fourth_place_rate=sum(1.0 for rank in seat_ranks if rank == 4) / len(seat_ranks),
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
    max_steps: int = 512,
    greedy: bool = True,
    device: torch.device | str | None = None,
) -> FixedSeedEvaluationSmoke:
    if num_games <= 0:
        raise ValueError(f"num_games must be positive, got {num_games}")
    seats = _normalize_actor_seats(seat_rotation)

    per_seat: dict[int, DiscardOnlyEvalMetrics] = {}
    eval_opponent_pool = opponent_pool
    if eval_opponent_pool is None and len(seats) < 4:
        eval_opponent_pool = OpponentPool(
            (OpponentPoolEntry(policy=RandomInteractivePolicy(), policy_version=-1, greedy=True),)
        )
    for offset, seat in enumerate(seats):
        seat_seed = None if seed is None else seed + offset * max(1, num_games) * seed_stride
        per_seat[seat] = evaluate_policy(
            env,
            policy,
            num_episodes=num_games,
            opponent_pool=eval_opponent_pool,
            learner_seats=(seat,),
            seed=seat_seed,
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

    return FixedSeedEvaluationSmoke(
        episode_count=total_episodes,
        seat_rotation=seats,
        seed=seed,
        average_rank=weighted_mean("mean_rank"),
        rank_pt=weighted_mean("mean_terminal_reward"),
        fourth_rate=weighted_mean("fourth_place_rate"),
        win_rate=None,
        deal_in_rate=None,
        call_rate=None,
        riichi_rate=None,
        per_seat=per_seat,
    )


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
    "CriticPretrainingIterationResult",
    "DiscardOnlyEvalMetrics",
    "DiscardOnlyTrainingHistory",
    "DiscardOnlyTrainingIteration",
    "FixedSeedEvaluationSmoke",
    "evaluate_discard_only_policy",
    "evaluate_policy",
    "run_fixed_seed_evaluation_smoke",
    "run_discard_only_training",
    "run_critic_pretraining_iteration",
    "run_training",
]
