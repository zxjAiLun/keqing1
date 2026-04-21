from __future__ import annotations

import torch

from keqingrl.actions import ActionSpec, ActionType, encode_action_id
from keqingrl.contracts import ObsTensorBatch, PolicyInput
from keqingrl.policy import RandomInteractivePolicy
from keqingrl.rollout import RolloutStep, backfill_terminal_rewards
from mahjong_env.action_space import TILE_NAME_TO_IDX


def _make_policy_input() -> PolicyInput:
    legal_actions = (
        (
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["2m"]),
            ActionSpec(ActionType.PASS),
        ),
        (
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["3p"]),
            ActionSpec(ActionType.TSUMO, tile=TILE_NAME_TO_IDX["C"]),
            ActionSpec(ActionType.PASS),
        ),
    )
    legal_action_ids = torch.tensor([[encode_action_id(spec) for spec in row] for row in legal_actions], dtype=torch.long)
    legal_action_features = torch.zeros((2, 3, 4), dtype=torch.float32)
    legal_action_mask = torch.tensor([[True, True, False], [True, True, True]], dtype=torch.bool)
    rule_context = torch.zeros((2, 6), dtype=torch.float32)
    obs = ObsTensorBatch(
        tile_obs=torch.zeros((2, 4, 34), dtype=torch.float32),
        scalar_obs=torch.zeros((2, 8), dtype=torch.float32),
    )
    return PolicyInput(
        obs=obs,
        legal_action_ids=legal_action_ids,
        legal_action_features=legal_action_features,
        legal_action_mask=legal_action_mask,
        rule_context=rule_context,
        legal_actions=legal_actions,
    )


def test_random_policy_forward_and_sampling_respect_legal_actions() -> None:
    policy_input = _make_policy_input()
    policy = RandomInteractivePolicy()

    output = policy(policy_input)
    sample = policy.sample_action(policy_input)
    greedy = policy.sample_action(policy_input, greedy=True)

    assert output.action_logits.shape == (2, 3)
    assert output.value.shape == (2,)
    assert output.rank_logits.shape == (2, 4)
    assert sample.action_index.shape == (2,)
    assert greedy.action_index.tolist() == [0, 0]
    assert torch.all(policy_input.legal_action_mask[torch.arange(2), sample.action_index])
    assert sample.action_spec[0] == policy_input.legal_actions[0][sample.action_index[0]]
    assert sample.action_spec[1] == policy_input.legal_actions[1][sample.action_index[1]]
    assert torch.allclose(sample.rank_probs.sum(dim=-1), torch.ones(2))


def test_backfill_terminal_rewards_only_updates_done_steps() -> None:
    policy_input = _make_policy_input()
    step_open = RolloutStep(
        obs=policy_input.obs,
        legal_action_ids=policy_input.legal_action_ids[0],
        legal_action_features=policy_input.legal_action_features[0],
        legal_action_mask=policy_input.legal_action_mask[0],
        action_index=0,
        action_spec=policy_input.legal_actions[0][0],
        log_prob=-0.7,
        value=0.1,
        entropy=0.5,
        reward=0.0,
        done=False,
        actor=0,
        policy_version=3,
        rule_context=policy_input.rule_context[0],
    )
    step_done = RolloutStep(
        obs=policy_input.obs,
        legal_action_ids=policy_input.legal_action_ids[1],
        legal_action_features=policy_input.legal_action_features[1],
        legal_action_mask=policy_input.legal_action_mask[1],
        action_index=1,
        action_spec=policy_input.legal_actions[1][1],
        log_prob=-0.2,
        value=0.3,
        entropy=0.4,
        reward=0.0,
        done=True,
        actor=1,
        policy_version=3,
        rule_context=policy_input.rule_context[1],
    )

    updated = backfill_terminal_rewards([step_open, step_done], [1.0, 0.5, 0.0, -1.5])
    assert updated[0].reward == 0.0
    assert updated[1].reward == 0.5
