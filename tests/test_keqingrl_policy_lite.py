from __future__ import annotations

import torch

from keqingrl.actions import ActionSpec, ActionType, encode_action_id
from keqingrl.contracts import ObsTensorBatch, PolicyInput
from keqingrl.policy import RulePriorDeltaPolicy, RulePriorPolicy
from keqingrl.rule_score import RuleScoreConfig, prior_logits_from_raw_scores


def _policy_input() -> PolicyInput:
    actions = (
        (
            ActionSpec(ActionType.DISCARD, tile=0),
            ActionSpec(ActionType.DISCARD, tile=1),
            ActionSpec(ActionType.PASS),
        ),
    )
    raw_rule_scores = torch.tensor([[100.0, 90.0, 0.0]], dtype=torch.float32)
    prior_logits = prior_logits_from_raw_scores(
        raw_rule_scores,
        mask=torch.tensor([[True, True, True]]),
        config=RuleScoreConfig(clip_min=-10.0),
    )
    return PolicyInput(
        obs=ObsTensorBatch(
            tile_obs=torch.zeros((1, 4, 34), dtype=torch.float32),
            scalar_obs=torch.zeros((1, 6), dtype=torch.float32),
        ),
        legal_action_ids=torch.tensor([[encode_action_id(spec) for spec in actions[0]]]),
        legal_action_features=torch.zeros((1, 3, 8), dtype=torch.float32),
        legal_action_mask=torch.tensor([[True, True, True]]),
        rule_context=torch.zeros((1, 6), dtype=torch.float32),
        raw_rule_scores=raw_rule_scores,
        prior_logits=prior_logits,
        style_context=torch.zeros((1, 5), dtype=torch.float32),
        legal_actions=actions,
    )


def test_rule_prior_policy_uses_prior_logits_distribution() -> None:
    policy_input = _policy_input()
    policy = RulePriorPolicy()

    out = policy(policy_input)

    assert torch.allclose(out.action_logits, policy_input.prior_logits)
    assert policy.sample_action(policy_input, greedy=True).action_index.tolist() == [0]


def test_zero_delta_policy_logits_and_probs_match_rule_prior() -> None:
    torch.manual_seed(0)
    policy_input = _policy_input()
    policy = RulePriorDeltaPolicy(
        hidden_dim=32,
        num_res_blocks=1,
        c_tile=4,
        n_scalar=6,
        action_id_buckets=512,
        action_id_dim=8,
        dropout=0.0,
    )

    out = policy(policy_input)
    expected_logits = policy_input.prior_logits

    assert torch.allclose(out.aux["neural_delta"], torch.zeros_like(out.aux["neural_delta"]), atol=1e-6)
    assert torch.allclose(out.action_logits, expected_logits, atol=1e-6)
    assert torch.allclose(torch.softmax(out.action_logits, dim=-1), torch.softmax(expected_logits, dim=-1), atol=1e-6)
    assert policy.sample_action(policy_input, greedy=True).action_index.tolist() == [0]
