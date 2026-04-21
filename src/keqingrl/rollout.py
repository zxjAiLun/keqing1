"""Rollout-native trajectory records for keqingrl."""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch

from keqingrl.actions import ActionSpec
from keqingrl.contracts import ObsTensorBatch, PolicyInput


@dataclass(frozen=True)
class RolloutStep:
    obs: ObsTensorBatch
    legal_action_ids: torch.Tensor
    legal_action_features: torch.Tensor
    legal_action_mask: torch.Tensor
    action_index: int
    action_spec: ActionSpec
    log_prob: float
    value: float
    entropy: float
    reward: float
    done: bool
    actor: int
    policy_version: int
    rule_context: torch.Tensor
    policy_name: str | None = None
    legal_actions: tuple[ActionSpec, ...] | None = None
    game_id: str | None = None
    step_id: int | None = None


@dataclass(frozen=True)
class RolloutEpisode:
    steps: tuple[RolloutStep, ...]
    terminal_rewards: tuple[float, float, float, float]
    final_ranks: tuple[int, int, int, int]
    scores: tuple[int, int, int, int]
    game_id: str | None = None
    seed: int | None = None


def backfill_terminal_rewards(
    steps: list[RolloutStep],
    terminal_rewards: list[float] | tuple[float, ...],
) -> list[RolloutStep]:
    updated: list[RolloutStep] = []
    for step in steps:
        reward = float(terminal_rewards[step.actor]) if step.done else step.reward
        updated.append(replace(step, reward=float(reward)))
    return updated


def backfill_actor_terminal_rewards(
    steps: list[RolloutStep],
    terminal_rewards: list[float] | tuple[float, ...],
) -> list[RolloutStep]:
    last_step_for_actor: dict[int, int] = {}
    for index, step in enumerate(steps):
        last_step_for_actor[step.actor] = index

    updated: list[RolloutStep] = []
    for index, step in enumerate(steps):
        actor_terminal = last_step_for_actor.get(step.actor) == index
        reward = float(step.reward)
        if actor_terminal:
            reward += float(terminal_rewards[step.actor])
        updated.append(replace(step, reward=reward, done=actor_terminal))
    return updated


def rollout_step_policy_input(
    step: RolloutStep,
    *,
    device: torch.device | str | None = None,
) -> PolicyInput:
    target_device = None if device is None else torch.device(device)

    def _move(tensor: torch.Tensor) -> torch.Tensor:
        return tensor if target_device is None else tensor.to(target_device)

    history = None
    if step.obs.history_obs is not None:
        history = _move(step.obs.history_obs.unsqueeze(0))

    extras = {
        key: _move(value.unsqueeze(0))
        for key, value in step.obs.extras.items()
    }
    return PolicyInput(
        obs=ObsTensorBatch(
            tile_obs=_move(step.obs.tile_obs.unsqueeze(0)),
            scalar_obs=_move(step.obs.scalar_obs.unsqueeze(0)),
            history_obs=history,
            extras=extras,
        ),
        legal_action_ids=_move(step.legal_action_ids.unsqueeze(0)).long(),
        legal_action_features=_move(step.legal_action_features.unsqueeze(0)).float(),
        legal_action_mask=_move(step.legal_action_mask.unsqueeze(0)).bool(),
        rule_context=_move(step.rule_context.unsqueeze(0)).float(),
        legal_actions=None if step.legal_actions is None else (step.legal_actions,),
    )


__all__ = [
    "RolloutEpisode",
    "RolloutStep",
    "backfill_actor_terminal_rewards",
    "backfill_terminal_rewards",
    "rollout_step_policy_input",
]
