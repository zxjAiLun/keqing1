"""Rollout collection and PPO prep helpers for keqingrl."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Callable, Sequence

import torch

from keqingrl.buffer import PPOBatch, build_ppo_batch, compute_returns_and_advantages
from keqingrl.contracts import ObsTensorBatch, PolicyInput
from keqingrl.env import DiscardOnlyMahjongEnv
from keqingrl.opponent_pool import (
    OpponentPool,
    SeatPolicyAssignment,
    build_selfplay_seat_assignments,
)
from keqingrl.policy import InteractivePolicy
from keqingrl.ppo import PPOLossBreakdown, ppo_update
from keqingrl.rollout import (
    RolloutEpisode,
    RolloutStep,
    backfill_actor_terminal_rewards,
)


@dataclass(frozen=True)
class DiscardOnlyIterationMetrics:
    episode_count: int
    total_steps: int
    batch_size: int
    mean_episode_steps: float
    mean_terminal_reward: float
    mean_rank: float
    first_place_rate: float
    mean_total_loss: float
    mean_policy_loss: float
    mean_value_loss: float
    mean_entropy_bonus: float
    mean_approx_kl: float
    mean_clip_fraction: float
    mean_rank_loss: float | None = None


@dataclass(frozen=True)
class DiscardOnlyIterationResult:
    episodes: tuple[RolloutEpisode, ...]
    batch: PPOBatch
    losses: tuple[PPOLossBreakdown, ...]
    metrics: DiscardOnlyIterationMetrics


_SeatPolicyFactory = Callable[[int, int | None], Sequence[SeatPolicyAssignment]]


def collect_policy_episode(
    env: DiscardOnlyMahjongEnv,
    seat_policies: Sequence[SeatPolicyAssignment],
    *,
    seed: int | None = None,
    max_steps: int = 512,
    device: torch.device | str | None = None,
) -> RolloutEpisode:
    seat_policy_tuple = tuple(seat_policies)
    state = env.reset(seed=seed)
    game_id = state.game_id
    rollout_steps: list[RolloutStep] = []
    final_result = None

    for step_id in range(max_steps):
        actor = env.current_actor()
        if actor is None:
            if env.is_done():
                break
            raise RuntimeError("environment has no current actor before episode termination")
        if actor >= len(seat_policy_tuple):
            raise ValueError(
                f"seat_policies length {len(seat_policy_tuple)} does not cover actor {actor}"
            )

        seat_policy = seat_policy_tuple[actor]
        target_device = torch.device(device) if device is not None else _policy_device(seat_policy.policy)
        policy_input_cpu = env.observe(actor)
        policy_input = _policy_input_to_device(policy_input_cpu, target_device)
        with torch.no_grad():
            sample = seat_policy.policy.sample_action(policy_input, greedy=seat_policy.greedy)

        chosen_index = int(sample.action_index[0].detach().cpu())
        chosen_action = policy_input_cpu.legal_actions[0][chosen_index]
        final_result = env.step(actor, chosen_action)

        rollout_steps.append(
            RolloutStep(
                obs=_single_obs(policy_input_cpu.obs),
                legal_action_ids=policy_input_cpu.legal_action_ids[0].clone(),
                legal_action_features=policy_input_cpu.legal_action_features[0].clone(),
                legal_action_mask=policy_input_cpu.legal_action_mask[0].clone(),
                action_index=chosen_index,
                action_spec=chosen_action,
                log_prob=float(sample.log_prob[0].detach().cpu()),
                value=float(sample.value[0].detach().cpu()),
                entropy=float(sample.entropy[0].detach().cpu()),
                reward=0.0 if final_result.done else float(final_result.reward),
                done=bool(final_result.done),
                actor=actor,
                policy_version=seat_policy.policy_version,
                rule_context=policy_input_cpu.rule_context[0].clone(),
                policy_name=seat_policy.name,
                legal_actions=tuple(policy_input_cpu.legal_actions[0]),
                game_id=game_id,
                step_id=step_id,
            )
        )

        if final_result.done:
            break
    else:
        raise RuntimeError(f"policy episode exceeded max_steps={max_steps}")

    if final_result is None or not final_result.done:
        raise RuntimeError("collect_policy_episode finished without a terminal StepResult")
    if final_result.terminal_rewards is None or final_result.final_ranks is None or final_result.scores is None:
        raise RuntimeError("terminal StepResult is missing episode metadata")

    return RolloutEpisode(
        steps=tuple(rollout_steps),
        terminal_rewards=tuple(float(value) for value in final_result.terminal_rewards),  # type: ignore[arg-type]
        final_ranks=tuple(int(rank) for rank in final_result.final_ranks),  # type: ignore[arg-type]
        scores=tuple(int(score) for score in final_result.scores),  # type: ignore[arg-type]
        game_id=game_id,
        seed=seed,
    )


def collect_policy_episodes(
    env: DiscardOnlyMahjongEnv,
    *,
    num_episodes: int,
    seat_policies: Sequence[SeatPolicyAssignment] | None = None,
    seat_policy_factory: _SeatPolicyFactory | None = None,
    seed: int | None = None,
    seed_stride: int = 1,
    max_steps: int = 512,
    device: torch.device | str | None = None,
) -> tuple[RolloutEpisode, ...]:
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")
    if (seat_policies is None) == (seat_policy_factory is None):
        raise ValueError("exactly one of seat_policies or seat_policy_factory must be provided")

    episodes: list[RolloutEpisode] = []
    for episode_idx in range(num_episodes):
        episode_seed = None if seed is None else seed + episode_idx * seed_stride
        episode_policies = (
            tuple(seat_policies)
            if seat_policies is not None
            else tuple(seat_policy_factory(episode_idx, episode_seed))
        )
        episodes.append(
            collect_policy_episode(
                env,
                episode_policies,
                seed=episode_seed,
                max_steps=max_steps,
                device=device,
            )
        )
    return tuple(episodes)


def collect_selfplay_episode(
    env: DiscardOnlyMahjongEnv,
    policy: InteractivePolicy,
    *,
    opponent_pool: OpponentPool | None = None,
    learner_seats: Sequence[int] = (0, 1, 2, 3),
    seed: int | None = None,
    greedy: bool = False,
    policy_version: int = 0,
    max_steps: int = 512,
    device: torch.device | str | None = None,
) -> RolloutEpisode:
    seat_policies = build_selfplay_seat_assignments(
        learner_policy=policy,
        learner_policy_version=policy_version,
        learner_greedy=greedy,
        learner_seats=learner_seats,
        opponent_pool=opponent_pool,
        rng=_assignment_rng(seed),
    )
    return collect_policy_episode(
        env,
        seat_policies,
        seed=seed,
        max_steps=max_steps,
        device=device,
    )


def collect_selfplay_episodes(
    env: DiscardOnlyMahjongEnv,
    policy: InteractivePolicy,
    *,
    num_episodes: int,
    opponent_pool: OpponentPool | None = None,
    learner_seats: Sequence[int] = (0, 1, 2, 3),
    seed: int | None = None,
    seed_stride: int = 1,
    greedy: bool = False,
    policy_version: int = 0,
    max_steps: int = 512,
    device: torch.device | str | None = None,
) -> tuple[RolloutEpisode, ...]:
    return collect_policy_episodes(
        env,
        num_episodes=num_episodes,
        seat_policy_factory=lambda _episode_idx, episode_seed: build_selfplay_seat_assignments(
            learner_policy=policy,
            learner_policy_version=policy_version,
            learner_greedy=greedy,
            learner_seats=learner_seats,
            opponent_pool=opponent_pool,
            rng=_assignment_rng(episode_seed),
        ),
        seed=seed,
        seed_stride=seed_stride,
        max_steps=max_steps,
        device=device,
    )


def collect_discard_only_episode(
    env: DiscardOnlyMahjongEnv,
    policy: InteractivePolicy,
    *,
    seed: int | None = None,
    greedy: bool = False,
    policy_version: int = 0,
    max_steps: int = 512,
    device: torch.device | str | None = None,
) -> RolloutEpisode:
    return collect_selfplay_episode(
        env,
        policy,
        learner_seats=(0, 1, 2, 3),
        seed=seed,
        greedy=greedy,
        policy_version=policy_version,
        max_steps=max_steps,
        device=device,
    )


def collect_discard_only_episodes(
    env: DiscardOnlyMahjongEnv,
    policy: InteractivePolicy,
    *,
    num_episodes: int,
    seed: int | None = None,
    seed_stride: int = 1,
    greedy: bool = False,
    policy_version: int = 0,
    max_steps: int = 512,
    device: torch.device | str | None = None,
) -> tuple[RolloutEpisode, ...]:
    return collect_selfplay_episodes(
        env,
        policy,
        num_episodes=num_episodes,
        learner_seats=(0, 1, 2, 3),
        seed=seed,
        seed_stride=seed_stride,
        greedy=greedy,
        policy_version=policy_version,
        max_steps=max_steps,
        device=device,
    )


def build_episode_ppo_batch(
    episode: RolloutEpisode,
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    include_rank_targets: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[RolloutStep], PPOBatch]:
    if not episode.steps:
        raise ValueError("episode.steps must not be empty")

    prepared_steps = backfill_actor_terminal_rewards(list(episode.steps), episode.terminal_rewards)
    advantages = torch.zeros((len(prepared_steps),), dtype=torch.float32)
    returns = torch.zeros((len(prepared_steps),), dtype=torch.float32)

    step_indices_by_actor: dict[int, list[int]] = {}
    for index, step in enumerate(prepared_steps):
        step_indices_by_actor.setdefault(step.actor, []).append(index)

    for indices in step_indices_by_actor.values():
        actor_rewards = [prepared_steps[index].reward for index in indices]
        actor_values = [prepared_steps[index].value for index in indices]
        actor_dones = [prepared_steps[index].done for index in indices]
        actor_advantages, actor_returns = compute_returns_and_advantages(
            actor_rewards,
            actor_values,
            actor_dones,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        for offset, index in enumerate(indices):
            advantages[index] = actor_advantages[offset]
            returns[index] = actor_returns[offset]

    final_rank_target = None
    if include_rank_targets:
        final_rank_target = [episode.final_ranks[step.actor] for step in prepared_steps]

    batch = build_ppo_batch(
        prepared_steps,
        advantages,
        returns,
        final_rank_target=final_rank_target,
    )
    return advantages, returns, prepared_steps, batch


def build_episodes_ppo_batch(
    episodes: list[RolloutEpisode] | tuple[RolloutEpisode, ...],
    *,
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
    include_rank_targets: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, list[RolloutStep], PPOBatch]:
    if not episodes:
        raise ValueError("episodes must not be empty")

    all_advantages: list[torch.Tensor] = []
    all_returns: list[torch.Tensor] = []
    prepared_steps: list[RolloutStep] = []
    final_rank_target: list[int] | None = [] if include_rank_targets else None

    for episode in episodes:
        episode_advantages, episode_returns, episode_steps, _episode_batch = build_episode_ppo_batch(
            episode,
            gamma=gamma,
            gae_lambda=gae_lambda,
            include_rank_targets=include_rank_targets,
        )
        all_advantages.append(episode_advantages)
        all_returns.append(episode_returns)
        prepared_steps.extend(episode_steps)
        if final_rank_target is not None:
            final_rank_target.extend(episode.final_ranks[step.actor] for step in episode_steps)

    advantages = torch.cat(all_advantages, dim=0)
    returns = torch.cat(all_returns, dim=0)
    batch = build_ppo_batch(
        prepared_steps,
        advantages,
        returns,
        final_rank_target=final_rank_target,
    )
    return advantages, returns, prepared_steps, batch


def run_ppo_iteration(
    env: DiscardOnlyMahjongEnv,
    policy: InteractivePolicy,
    optimizer: torch.optim.Optimizer,
    *,
    num_episodes: int,
    opponent_pool: OpponentPool | None = None,
    learner_seats: Sequence[int] = (0, 1, 2, 3),
    update_epochs: int = 1,
    seed: int | None = None,
    seed_stride: int = 1,
    greedy: bool = False,
    policy_version: int = 0,
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
) -> DiscardOnlyIterationResult:
    if update_epochs <= 0:
        raise ValueError(f"update_epochs must be positive, got {update_epochs}")

    learner_seat_tuple = _normalize_actor_seats(learner_seats)
    episodes = collect_selfplay_episodes(
        env,
        policy,
        num_episodes=num_episodes,
        opponent_pool=opponent_pool,
        learner_seats=learner_seat_tuple,
        seed=seed,
        seed_stride=seed_stride,
        greedy=greedy,
        policy_version=policy_version,
        max_steps=max_steps,
        device=device,
    )
    learner_episodes = _filter_episodes_for_actors(episodes, learner_seat_tuple)
    _advantages, _returns, _prepared_steps, batch = build_episodes_ppo_batch(
        learner_episodes,
        gamma=gamma,
        gae_lambda=gae_lambda,
        include_rank_targets=include_rank_targets,
    )
    target_device = _policy_device(policy) if device is None else torch.device(device)
    batch = batch.to(target_device)

    losses: list[PPOLossBreakdown] = []
    for _ in range(update_epochs):
        losses.append(
            ppo_update(
                policy,
                optimizer,
                batch,
                clip_eps=clip_eps,
                value_coef=value_coef,
                entropy_coef=entropy_coef,
                rank_coef=rank_coef,
                normalize_advantages=normalize_advantages,
                max_grad_norm=max_grad_norm,
            )
        )

    metrics = summarize_iteration(episodes, losses, batch, learner_seats=learner_seat_tuple)
    return DiscardOnlyIterationResult(
        episodes=episodes,
        batch=batch,
        losses=tuple(losses),
        metrics=metrics,
    )


def run_discard_only_ppo_iteration(
    env: DiscardOnlyMahjongEnv,
    policy: InteractivePolicy,
    optimizer: torch.optim.Optimizer,
    *,
    num_episodes: int,
    update_epochs: int = 1,
    seed: int | None = None,
    seed_stride: int = 1,
    greedy: bool = False,
    policy_version: int = 0,
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
) -> DiscardOnlyIterationResult:
    return run_ppo_iteration(
        env,
        policy,
        optimizer,
        num_episodes=num_episodes,
        learner_seats=(0, 1, 2, 3),
        update_epochs=update_epochs,
        seed=seed,
        seed_stride=seed_stride,
        greedy=greedy,
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


def summarize_iteration(
    episodes: list[RolloutEpisode] | tuple[RolloutEpisode, ...],
    losses: list[PPOLossBreakdown] | tuple[PPOLossBreakdown, ...],
    batch: PPOBatch,
    *,
    learner_seats: Sequence[int] = (0, 1, 2, 3),
) -> DiscardOnlyIterationMetrics:
    if not episodes:
        raise ValueError("episodes must not be empty")
    if not losses:
        raise ValueError("losses must not be empty")

    learner_seat_tuple = _normalize_actor_seats(learner_seats)
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

    rank_losses = [loss.rank_loss.item() for loss in losses if loss.rank_loss is not None]
    mean_rank_loss = None if not rank_losses else sum(rank_losses) / len(rank_losses)

    return DiscardOnlyIterationMetrics(
        episode_count=len(episodes),
        total_steps=sum(episode_steps),
        batch_size=int(batch.action_index.numel()),
        mean_episode_steps=sum(episode_steps) / len(episode_steps),
        mean_terminal_reward=sum(seat_rewards) / len(seat_rewards),
        mean_rank=sum(seat_ranks) / len(seat_ranks),
        first_place_rate=sum(1.0 for rank in seat_ranks if rank == 1) / len(seat_ranks),
        mean_total_loss=sum(loss.total_loss.item() for loss in losses) / len(losses),
        mean_policy_loss=sum(loss.policy_loss.item() for loss in losses) / len(losses),
        mean_value_loss=sum(loss.value_loss.item() for loss in losses) / len(losses),
        mean_entropy_bonus=sum(loss.entropy_bonus.item() for loss in losses) / len(losses),
        mean_approx_kl=sum(loss.approx_kl.item() for loss in losses) / len(losses),
        mean_clip_fraction=sum(loss.clip_fraction.item() for loss in losses) / len(losses),
        mean_rank_loss=mean_rank_loss,
    )


def summarize_discard_only_iteration(
    episodes: list[RolloutEpisode] | tuple[RolloutEpisode, ...],
    losses: list[PPOLossBreakdown] | tuple[PPOLossBreakdown, ...],
    batch: PPOBatch,
) -> DiscardOnlyIterationMetrics:
    return summarize_iteration(
        episodes,
        losses,
        batch,
        learner_seats=(0, 1, 2, 3),
    )


def _filter_episodes_for_actors(
    episodes: Sequence[RolloutEpisode],
    actor_seats: Sequence[int],
) -> tuple[RolloutEpisode, ...]:
    actor_seat_set = set(_normalize_actor_seats(actor_seats))
    filtered_episodes: list[RolloutEpisode] = []
    for episode in episodes:
        filtered_steps = tuple(step for step in episode.steps if step.actor in actor_seat_set)
        if not filtered_steps:
            continue
        filtered_episodes.append(
            RolloutEpisode(
                steps=filtered_steps,
                terminal_rewards=episode.terminal_rewards,
                final_ranks=episode.final_ranks,
                scores=episode.scores,
                game_id=episode.game_id,
                seed=episode.seed,
            )
        )
    if not filtered_episodes:
        raise RuntimeError("no learner-controlled steps were collected for PPO batching")
    return tuple(filtered_episodes)


def _normalize_actor_seats(actor_seats: Sequence[int]) -> tuple[int, ...]:
    actor_tuple = tuple(int(actor) for actor in actor_seats)
    if not actor_tuple:
        raise ValueError("actor seat selection must not be empty")
    if len(set(actor_tuple)) != len(actor_tuple):
        raise ValueError(f"actor seat selection must not contain duplicates, got {actor_tuple}")
    if any(actor < 0 or actor >= 4 for actor in actor_tuple):
        raise ValueError(f"actor seat selection must stay within [0, 3], got {actor_tuple}")
    return actor_tuple


def _assignment_rng(seed: int | None) -> random.Random | None:
    if seed is None:
        return None
    return random.Random(seed)


def _single_obs(obs: ObsTensorBatch) -> ObsTensorBatch:
    history = None if obs.history_obs is None else obs.history_obs[0].clone()
    extras = {key: value[0].clone() for key, value in obs.extras.items()}
    return ObsTensorBatch(
        tile_obs=obs.tile_obs[0].clone(),
        scalar_obs=obs.scalar_obs[0].clone(),
        history_obs=history,
        extras=extras,
    )


def _policy_input_to_device(policy_input: PolicyInput, device: torch.device) -> PolicyInput:
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
        legal_actions=policy_input.legal_actions,
        recurrent_state=policy_input.recurrent_state,
    )


def _policy_device(policy: InteractivePolicy) -> torch.device:
    parameter = next(policy.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(policy.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


__all__ = [
    "DiscardOnlyIterationMetrics",
    "DiscardOnlyIterationResult",
    "build_episode_ppo_batch",
    "build_episodes_ppo_batch",
    "collect_discard_only_episode",
    "collect_discard_only_episodes",
    "collect_policy_episode",
    "collect_policy_episodes",
    "collect_selfplay_episode",
    "collect_selfplay_episodes",
    "run_discard_only_ppo_iteration",
    "run_ppo_iteration",
    "summarize_discard_only_iteration",
    "summarize_iteration",
]
