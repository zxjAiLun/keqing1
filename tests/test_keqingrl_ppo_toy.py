from __future__ import annotations

import torch
import torch.nn.functional as F

from keqingrl.actions import ActionSpec, ActionType
from keqingrl.buffer import build_ppo_batch, compute_returns_and_advantages
from keqingrl.contracts import ObsTensorBatch, PolicyInput
from keqingrl.policy import NeuralInteractivePolicy
from keqingrl.ppo import ppo_update
from keqingrl.rollout import RolloutStep
from mahjong_env.action_space import TILE_NAME_TO_IDX


def _make_bandit_policy_input(states: torch.Tensor) -> PolicyInput:
    batch_size = int(states.shape[0])
    obs = ObsTensorBatch(
        tile_obs=torch.zeros((batch_size, 4, 34), dtype=torch.float32),
        scalar_obs=F.one_hot(states, num_classes=2).float(),
    )
    legal_actions = tuple(
        (
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["2m"]),
        )
        for _ in range(batch_size)
    )
    return PolicyInput(
        obs=obs,
        legal_action_ids=torch.tensor([[11, 29]] * batch_size, dtype=torch.long),
        legal_action_features=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]] * batch_size, dtype=torch.float32),
        legal_action_mask=torch.ones((batch_size, 2), dtype=torch.bool),
        rule_context=torch.zeros((batch_size, 3), dtype=torch.float32),
        legal_actions=legal_actions,
    )


def _build_steps(policy_input: PolicyInput, sample, rewards: torch.Tensor) -> list[RolloutStep]:
    steps: list[RolloutStep] = []
    for idx in range(int(rewards.shape[0])):
        steps.append(
            RolloutStep(
                obs=ObsTensorBatch(
                    tile_obs=policy_input.obs.tile_obs[idx],
                    scalar_obs=policy_input.obs.scalar_obs[idx],
                ),
                legal_action_ids=policy_input.legal_action_ids[idx],
                legal_action_features=policy_input.legal_action_features[idx],
                legal_action_mask=policy_input.legal_action_mask[idx],
                action_index=int(sample.action_index[idx]),
                action_spec=sample.action_spec[idx],
                log_prob=float(sample.log_prob[idx]),
                value=float(sample.value[idx]),
                entropy=float(sample.entropy[idx]),
                reward=float(rewards[idx]),
                done=True,
                actor=0,
                policy_version=0,
                rule_context=policy_input.rule_context[idx],
            )
        )
    return steps


def test_compute_returns_and_advantages_one_step_terminal_bandit() -> None:
    rewards = torch.tensor([1.0, -1.0, 0.5])
    values = torch.tensor([0.2, -0.1, 0.0])
    dones = torch.tensor([True, True, True])

    advantages, returns = compute_returns_and_advantages(
        rewards,
        values,
        dones,
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert torch.allclose(advantages, rewards - values)
    assert torch.allclose(returns, rewards)


def test_ppo_update_learns_toy_bandit_policy() -> None:
    torch.manual_seed(0)
    policy = NeuralInteractivePolicy(
        hidden_dim=32,
        num_res_blocks=1,
        c_tile=4,
        n_scalar=2,
        action_id_buckets=64,
        action_id_dim=8,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=3e-3)
    rollout_states = torch.tensor([0, 1] * 32, dtype=torch.long)
    eval_states = torch.tensor([0, 1], dtype=torch.long)

    for _ in range(80):
        policy_input = _make_bandit_policy_input(rollout_states)
        with torch.no_grad():
            sample = policy.sample_action(policy_input)
        rewards = torch.where(sample.action_index == rollout_states, 1.0, -1.0)
        advantages, returns = compute_returns_and_advantages(
            rewards,
            sample.value,
            torch.ones_like(rewards, dtype=torch.bool),
            gamma=1.0,
            gae_lambda=1.0,
        )
        steps = _build_steps(policy_input, sample, rewards)
        batch = build_ppo_batch(steps, advantages, returns)
        for _ in range(4):
            losses = ppo_update(
                policy,
                optimizer,
                batch,
                clip_eps=0.2,
                value_coef=0.5,
                entropy_coef=0.01,
                max_grad_norm=1.0,
            )
            assert torch.isfinite(losses.total_loss)

    eval_input = _make_bandit_policy_input(eval_states)
    with torch.no_grad():
        greedy = policy.sample_action(eval_input, greedy=True).action_index
    assert greedy.tolist() == [0, 1]
