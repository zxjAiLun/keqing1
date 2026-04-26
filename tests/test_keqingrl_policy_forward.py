from __future__ import annotations

import torch

from keqingrl.actions import ActionSpec, ActionType, encode_action_id
from keqingrl.contracts import ObsTensorBatch, PolicyInput
from keqingrl.policy import NeuralInteractivePolicy
from mahjong_env.action_space import TILE_NAME_TO_IDX


def _make_policy_input() -> PolicyInput:
    legal_actions = (
        (
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["2m"]),
            ActionSpec(ActionType.PASS),
            ActionSpec(ActionType.RYUKYOKU),
        ),
        (
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["3p"]),
            ActionSpec(ActionType.TSUMO, tile=TILE_NAME_TO_IDX["C"]),
            ActionSpec(ActionType.PASS),
            ActionSpec(ActionType.RYUKYOKU),
        ),
        (
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["7s"]),
            ActionSpec(ActionType.RON, tile=TILE_NAME_TO_IDX["7s"], from_who=1),
            ActionSpec(ActionType.PASS),
            ActionSpec(ActionType.RYUKYOKU),
        ),
    )
    obs = ObsTensorBatch(
        tile_obs=torch.randn((3, 4, 34), dtype=torch.float32),
        scalar_obs=torch.randn((3, 6), dtype=torch.float32),
        history_obs=torch.randn((3, 5, 3), dtype=torch.float32),
    )
    return PolicyInput(
        obs=obs,
        legal_action_ids=torch.tensor(
            [[encode_action_id(spec) for spec in row] for row in legal_actions],
            dtype=torch.long,
        ),
        legal_action_features=torch.randn((3, 4, 5), dtype=torch.float32),
        legal_action_mask=torch.tensor(
            [[True, True, False, False], [True, True, True, False], [True, True, True, True]],
            dtype=torch.bool,
        ),
        rule_context=torch.randn((3, 6), dtype=torch.float32),
        legal_actions=legal_actions,
    )


def test_neural_policy_forward_shapes_and_masking() -> None:
    torch.manual_seed(0)
    policy = NeuralInteractivePolicy(
        hidden_dim=32,
        num_res_blocks=1,
        c_tile=4,
        n_scalar=6,
        action_id_buckets=128,
        action_id_dim=8,
        dropout=0.0,
    )
    policy_input = _make_policy_input()

    output = policy(policy_input)
    sample = policy.sample_action(policy_input)
    greedy = policy.sample_action(policy_input, greedy=True)

    assert output.action_logits.shape == (3, 4)
    assert output.value.shape == (3,)
    assert output.rank_logits.shape == (3, 4)
    assert torch.isfinite(output.value).all()
    assert torch.isfinite(output.rank_logits).all()
    assert torch.isfinite(output.entropy).all()
    assert torch.isfinite(output.action_logits[policy_input.legal_action_mask]).all()
    assert torch.all(output.action_logits[~policy_input.legal_action_mask] == torch.finfo(output.action_logits.dtype).min)
    assert sample.action_index.shape == (3,)
    assert greedy.action_index.shape == (3,)
    assert torch.all(policy_input.legal_action_mask[torch.arange(3), sample.action_index])
    assert torch.all(policy_input.legal_action_mask[torch.arange(3), greedy.action_index])
    assert torch.allclose(sample.rank_probs.sum(dim=-1), torch.ones(3), atol=1e-5)


def test_neural_policy_rule_context_affects_value_path() -> None:
    torch.manual_seed(1)
    policy = NeuralInteractivePolicy(
        hidden_dim=32,
        num_res_blocks=1,
        c_tile=4,
        n_scalar=6,
        action_id_buckets=128,
        action_id_dim=8,
        dropout=0.0,
    )
    policy_input = _make_policy_input()
    alt_input = PolicyInput(
        obs=policy_input.obs,
        legal_action_ids=policy_input.legal_action_ids,
        legal_action_features=policy_input.legal_action_features,
        legal_action_mask=policy_input.legal_action_mask,
        legal_actions=policy_input.legal_actions,
        rule_context=policy_input.rule_context + 0.5,
    )

    out_a = policy(policy_input)
    out_b = policy(alt_input)

    assert not torch.allclose(out_a.value, out_b.value)
