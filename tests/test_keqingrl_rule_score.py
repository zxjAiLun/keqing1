from __future__ import annotations

import torch

from keqingrl.actions import ActionSpec, ActionType
from keqingrl.metadata import default_checkpoint_metadata, validate_checkpoint_metadata
from keqingrl.rewards import RuleContext, build_rule_context
from keqingrl.rule_score import RuleScoreConfig, prior_logits_from_raw_scores, smoothed_prior_probs


def test_prior_logits_center_and_clip_without_changing_best_identity() -> None:
    raw = torch.tensor([[100.0, 90.0, 0.0], [1.0, 1.0, -50.0]])
    prior = prior_logits_from_raw_scores(
        raw,
        mask=torch.ones_like(raw, dtype=torch.bool),
        config=RuleScoreConfig(clip_min=-10.0, clip_max=0.0, prior_temperature=1.0),
    )

    assert prior.tolist() == [[0.0, -10.0, -10.0], [0.0, 0.0, -10.0]]
    assert prior.argmax(dim=-1).tolist() == [0, 0]


def test_smoothed_prior_probs_keeps_all_legal_actions_positive() -> None:
    prior = torch.tensor([[0.0, -10.0, -10.0]])
    mask = torch.tensor([[True, True, True]])
    probs = smoothed_prior_probs(prior, mask, eps=1e-4)

    assert torch.all(probs > 0)
    assert torch.allclose(probs.sum(dim=-1), torch.ones(1))


def test_rule_context_tensor_is_normalized() -> None:
    tensor = build_rule_context(RuleContext(pt_map=(90.0, 45.0, 0.0, -135.0), pt_norm=90.0))

    assert tensor.tolist() == [1.0, 0.5, 0.0, -1.5, 0.0, 1.0]


def test_action_spec_canonical_key_can_guard_rollout_order() -> None:
    actions = (
        ActionSpec(ActionType.DISCARD, tile=0),
        ActionSpec(ActionType.DISCARD, tile=1),
    )
    chosen_key = actions[1].canonical_key
    reordered = (actions[1], actions[0])

    assert reordered[1].canonical_key != chosen_key


def test_checkpoint_metadata_rejects_contract_version_mismatch() -> None:
    metadata = default_checkpoint_metadata()
    validate_checkpoint_metadata(metadata)

    metadata["action_contract_version"] = "old-action-contract"
    try:
        validate_checkpoint_metadata(metadata)
    except ValueError as exc:
        assert "action_contract_version" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected checkpoint contract-version mismatch")
