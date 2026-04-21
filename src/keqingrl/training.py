"""Longer-run training helpers for keqingrl."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from keqingrl.env import DiscardOnlyMahjongEnv
from keqingrl.opponent_pool import OpponentPool
from keqingrl.policy import InteractivePolicy
from keqingrl.selfplay import (
    DiscardOnlyIterationMetrics,
    collect_selfplay_episodes,
    run_ppo_iteration,
)


@dataclass(frozen=True)
class DiscardOnlyEvalMetrics:
    episode_count: int
    total_steps: int
    mean_episode_steps: float
    mean_terminal_reward: float
    mean_rank: float
    first_place_rate: float


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
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    include_rank_targets: bool = True,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    rank_coef: float = 0.0,
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
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    include_rank_targets: bool = True,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    rank_coef: float = 0.0,
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


__all__ = [
    "DiscardOnlyEvalMetrics",
    "DiscardOnlyTrainingHistory",
    "DiscardOnlyTrainingIteration",
    "evaluate_discard_only_policy",
    "evaluate_policy",
    "run_discard_only_training",
    "run_training",
]
