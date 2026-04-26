from __future__ import annotations

import torch
import torch.nn.functional as F

from keqingrl.actions import ActionSpec, ActionType
from keqingrl.buffer import PPOBatch, build_ppo_batch, compute_returns_and_advantages
from keqingrl.contracts import ObsTensorBatch, PolicyInput, PolicyOutput
from keqingrl.metadata import RULE_SCORE_SCALE_VERSION
from keqingrl.policy import InteractivePolicy, NeuralInteractivePolicy, RulePriorDeltaPolicy
from keqingrl.ppo import (
    _assert_critic_pretrain_optimizer_param_groups,
    _critic_pretrain_trainable_parameter_names,
    _freeze_actor_delta_parameters,
    _initialize_lazy_parameters_for_training,
    _uninitialized_parameter_names,
    compute_ppo_loss,
    critic_pretrain_update,
    ppo_update,
)
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


def _make_two_action_policy_input() -> PolicyInput:
    actions = (
        (
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["2m"]),
        ),
    )
    return PolicyInput(
        obs=ObsTensorBatch(
            tile_obs=torch.zeros((1, 4, 34), dtype=torch.float32),
            scalar_obs=torch.zeros((1, 6), dtype=torch.float32),
        ),
        legal_action_ids=torch.tensor([[11, 29]], dtype=torch.long),
        legal_action_features=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
        legal_action_mask=torch.tensor([[True, True]], dtype=torch.bool),
        rule_context=torch.zeros((1, 6), dtype=torch.float32),
        prior_logits=torch.tensor([[0.0, -1.0]], dtype=torch.float32),
        legal_actions=actions,
        metadata={
            "rule_score_scale": 1.0,
            "rule_score_scale_version": RULE_SCORE_SCALE_VERSION,
        },
    )


def _make_two_action_ppo_batch(policy_input: PolicyInput) -> PPOBatch:
    return PPOBatch(
        policy_input=policy_input,
        action_index=torch.tensor([0], dtype=torch.long),
        old_log_prob=torch.tensor([-0.5], dtype=torch.float32),
        old_value=torch.zeros((1,), dtype=torch.float32),
        advantages=torch.ones((1,), dtype=torch.float32),
        returns=torch.tensor([1.0], dtype=torch.float32),
        final_rank_target=torch.tensor([0], dtype=torch.long),
    )


def _make_two_action_rule_prior_delta_policy() -> RulePriorDeltaPolicy:
    return RulePriorDeltaPolicy(
        hidden_dim=32,
        num_res_blocks=1,
        c_tile=4,
        n_scalar=6,
        action_id_buckets=64,
        action_id_dim=8,
        dropout=0.0,
    )


def test_rule_kl_uses_same_filtered_learner_action_set() -> None:
    class _TwoActionPolicy(InteractivePolicy):
        def forward(self, policy_input: PolicyInput) -> PolicyOutput:
            assert policy_input.legal_action_ids.shape == (1, 2)
            return PolicyOutput(
                action_logits=torch.tensor([[0.25, -0.25]], dtype=torch.float32),
                value=torch.zeros((1,), dtype=torch.float32),
                rank_logits=torch.zeros((1, 4), dtype=torch.float32),
                aux={"prior_logits": policy_input.prior_logits},
            )

    actions = (
        ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
        ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["2m"]),
    )
    policy_input = PolicyInput(
        obs=ObsTensorBatch(
            tile_obs=torch.zeros((1, 4, 34), dtype=torch.float32),
            scalar_obs=torch.zeros((1, 2), dtype=torch.float32),
        ),
        legal_action_ids=torch.tensor([[11, 29]], dtype=torch.long),
        legal_action_features=torch.zeros((1, 2, 2), dtype=torch.float32),
        legal_action_mask=torch.tensor([[True, True]], dtype=torch.bool),
        rule_context=torch.zeros((1, 3), dtype=torch.float32),
        prior_logits=torch.tensor([[0.0, -1.0]], dtype=torch.float32),
        legal_actions=(actions,),
    )
    batch = PPOBatch(
        policy_input=policy_input,
        action_index=torch.tensor([0], dtype=torch.long),
        old_log_prob=torch.tensor([-0.5], dtype=torch.float32),
        old_value=torch.zeros((1,), dtype=torch.float32),
        advantages=torch.ones((1,), dtype=torch.float32),
        returns=torch.ones((1,), dtype=torch.float32),
    )

    losses = compute_ppo_loss(_TwoActionPolicy(), batch, rule_kl_coef=0.02)

    assert losses.rule_kl is not None
    assert torch.isfinite(losses.rule_kl)
    assert batch.policy_input.prior_logits.shape == batch.policy_input.legal_action_mask.shape == (1, 2)
    assert [action.action_type for action in batch.policy_input.legal_actions[0]] == [
        ActionType.DISCARD,
        ActionType.DISCARD,
    ]


def test_compute_ppo_loss_rejects_rule_score_scale_mismatch() -> None:
    policy_input = _make_two_action_policy_input()
    batch = _make_two_action_ppo_batch(policy_input)
    policy = _make_two_action_rule_prior_delta_policy()
    policy.rule_score_scale = 0.25

    try:
        compute_ppo_loss(policy, batch)
    except ValueError as exc:
        assert "rule score scale mismatch" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected rule_score_scale mismatch")


def test_critic_pretrain_freeze_keeps_actor_logits_and_probs_unchanged() -> None:
    torch.manual_seed(2)
    policy_input = _make_two_action_policy_input()
    policy = _make_two_action_rule_prior_delta_policy()
    with torch.no_grad():
        before = policy(policy_input)
        before_logits = before.action_logits.clone()
        before_delta = before.aux["neural_delta"].clone()
        before_probs = torch.softmax(before.action_logits, dim=-1)

    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    critic_pretrain_update(policy, optimizer, _make_two_action_ppo_batch(policy_input), freeze_actor_delta=True)

    with torch.no_grad():
        after = policy(policy_input)
        after_probs = torch.softmax(after.action_logits, dim=-1)

    assert torch.allclose(after.aux["neural_delta"], before_delta, atol=1e-6)
    assert torch.allclose(after.action_logits, before_logits, atol=1e-6)
    assert torch.allclose(after_probs, before_probs, atol=1e-6)


def test_critic_pretrain_dry_run_initializes_active_lazy_policy_before_optimizer_step() -> None:
    torch.manual_seed(3)
    policy_input = _make_two_action_policy_input()
    policy = _make_two_action_rule_prior_delta_policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    critic_pretrain_update(policy, optimizer, _make_two_action_ppo_batch(policy_input), freeze_actor_delta=True)

    remaining = _uninitialized_parameter_names(policy)
    assert remaining == []


def test_critic_pretrain_freeze_trainable_param_names_and_optimizer_gate() -> None:
    torch.manual_seed(4)
    policy_input = _make_two_action_policy_input()
    policy = _make_two_action_rule_prior_delta_policy()
    optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)

    _initialize_lazy_parameters_for_training(policy, policy_input)
    frozen = _freeze_actor_delta_parameters(policy)
    try:
        _assert_critic_pretrain_optimizer_param_groups(policy, optimizer)
        trainable_names = _critic_pretrain_trainable_parameter_names(policy)
        assert trainable_names
        assert all(name.startswith(("value_head.", "rank_head.")) for name in trainable_names)
    finally:
        for parameter, requires_grad in frozen:
            parameter.requires_grad_(requires_grad)


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
