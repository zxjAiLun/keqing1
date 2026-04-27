from __future__ import annotations

import pytest
import torch

from keqingrl.actions import ACTION_FLAG_TSUMOGIRI, ActionSpec, ActionType
from keqingrl.mortal_teacher import (
    MORTAL_ACTION_MASK_EXTRA_KEY,
    MORTAL_ACTION_SPACE,
    MORTAL_Q_VALUES_EXTRA_KEY,
    MortalTeacherMappingError,
    assert_mortal_discard_mask_parity,
    mortal_discard_action_spec,
    mortal_discard_ids_for_tile_id,
    mortal_discard_scores_for_legal_actions,
    mortal_discard_teacher_tensors_from_extras,
    mortal_discard_tile_id,
    mortal_discard_tile_name,
    mortal_discard_topk_teacher_context,
)
from mahjong_env.action_space import TILE_NAME_TO_IDX


def _q_values() -> torch.Tensor:
    return torch.arange(MORTAL_ACTION_SPACE, dtype=torch.float32)


def _mask(*ids: int) -> torch.BoolTensor:
    mask = torch.zeros((MORTAL_ACTION_SPACE,), dtype=torch.bool)
    mask[list(ids)] = True
    return mask


def test_mortal_discard_id_maps_normal_and_red_fives_to_keqing_tile_ids() -> None:
    assert mortal_discard_tile_name(0) == "1m"
    assert mortal_discard_tile_id(0) == TILE_NAME_TO_IDX["1m"]
    assert mortal_discard_tile_name(34) == "5mr"
    assert mortal_discard_tile_id(34) == TILE_NAME_TO_IDX["5m"]
    assert mortal_discard_tile_name(35) == "5pr"
    assert mortal_discard_tile_id(35) == TILE_NAME_TO_IDX["5p"]
    assert mortal_discard_tile_name(36) == "5sr"
    assert mortal_discard_tile_id(36) == TILE_NAME_TO_IDX["5s"]


def test_mortal_discard_action_spec_preserves_tsumogiri_flag_when_requested() -> None:
    assert mortal_discard_action_spec(8) == ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["9m"])
    assert mortal_discard_action_spec(8, tsumogiri=True) == ActionSpec(
        ActionType.DISCARD,
        tile=TILE_NAME_TO_IDX["9m"],
        flags=ACTION_FLAG_TSUMOGIRI,
    )


def test_mortal_discard_ids_for_tile_collapses_red_five_family() -> None:
    assert mortal_discard_ids_for_tile_id(TILE_NAME_TO_IDX["4m"]) == (3,)
    assert mortal_discard_ids_for_tile_id(TILE_NAME_TO_IDX["5m"]) == (4, 34)
    assert mortal_discard_ids_for_tile_id(TILE_NAME_TO_IDX["5p"]) == (13, 35)
    assert mortal_discard_ids_for_tile_id(TILE_NAME_TO_IDX["5s"]) == (22, 36)


def test_mask_parity_accepts_red_five_collapse() -> None:
    legal_actions = (
        ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["5m"]),
        ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["9m"]),
    )

    assert_mortal_discard_mask_parity(_mask(34, 8), legal_actions)


def test_mask_parity_reports_missing_and_extra_tiles() -> None:
    legal_actions = (
        ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["5m"]),
        ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["9m"]),
    )

    with pytest.raises(MortalTeacherMappingError, match="missing_keqing_tiles"):
        assert_mortal_discard_mask_parity(_mask(34, 0), legal_actions)


def test_scores_for_legal_actions_uses_only_masked_mortal_ids_and_keeps_non_discard_unscored() -> None:
    legal_actions = (
        ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["5m"]),
        ActionSpec(ActionType.PASS),
        ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["9m"], flags=ACTION_FLAG_TSUMOGIRI),
    )

    mapped = mortal_discard_scores_for_legal_actions(
        _q_values(),
        _mask(4, 34, 8),
        legal_actions,
        strict_mask=True,
    )

    assert mapped.score_mask.tolist() == [True, False, True]
    assert mapped.scores[0].item() == 34.0
    assert torch.isneginf(mapped.scores[1])
    assert mapped.scores[2].item() == 8.0
    assert mapped.source_action_ids == ((4, 34), (), (8,))


def test_scores_for_legal_actions_can_report_missing_legal_action_without_strict_failure() -> None:
    legal_actions = (ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["5m"]),)

    mapped = mortal_discard_scores_for_legal_actions(
        _q_values(),
        _mask(0),
        legal_actions,
        strict_mask=False,
    )

    assert mapped.score_mask.tolist() == [False]
    assert mapped.missing_legal_keys == (legal_actions[0].canonical_key,)
    assert mapped.extra_mortal_discard_ids == (0,)


def test_mortal_discard_topk_teacher_context_builds_distribution_on_rule_prior_topk() -> None:
    legal_actions = (
        (
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["5m"]),
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["9m"]),
        ),
    )
    q_values = _q_values().unsqueeze(0)
    mortal_masks = _mask(0, 34, 8).unsqueeze(0)

    teacher = mortal_discard_topk_teacher_context(
        prior_logits=torch.tensor([[1.0, 3.0, 2.0]], dtype=torch.float32),
        legal_action_mask=torch.ones((1, 3), dtype=torch.bool),
        legal_actions=legal_actions,
        q_values=q_values,
        mortal_masks=mortal_masks,
        topk=2,
        teacher_temperature=1.0,
    )

    assert teacher["topk_indices"].tolist() == [[1, 2]]
    assert teacher["teacher_topk_scores"].tolist() == [[34.0, 8.0]]
    assert teacher["teacher_argmax"].tolist() == [0]
    assert teacher["teacher_prior_agreement"].tolist() == [1.0]
    assert torch.allclose(teacher["teacher_probs"].sum(dim=-1), torch.ones(1))


def test_mortal_discard_topk_teacher_context_fails_when_topk_contains_unmapped_action() -> None:
    legal_actions = (
        (
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["5m"]),
        ),
    )

    with pytest.raises(MortalTeacherMappingError, match="missing"):
        mortal_discard_topk_teacher_context(
            prior_logits=torch.tensor([[1.0, 3.0]], dtype=torch.float32),
            legal_action_mask=torch.ones((1, 2), dtype=torch.bool),
            legal_actions=legal_actions,
            q_values=_q_values().unsqueeze(0),
            mortal_masks=_mask(0).unsqueeze(0),
            topk=2,
            teacher_temperature=1.0,
        )


def test_mortal_discard_teacher_tensors_from_extras_validates_standard_keys() -> None:
    q_values = _q_values().unsqueeze(0)
    mortal_masks = _mask(0, 8).unsqueeze(0)

    loaded_q, loaded_mask = mortal_discard_teacher_tensors_from_extras(
        {
            MORTAL_Q_VALUES_EXTRA_KEY: q_values,
            MORTAL_ACTION_MASK_EXTRA_KEY: mortal_masks,
        }
    )

    assert torch.equal(loaded_q, q_values)
    assert torch.equal(loaded_mask, mortal_masks)


def test_mortal_discard_teacher_tensors_from_extras_fails_closed_when_missing() -> None:
    with pytest.raises(MortalTeacherMappingError, match="missing required keys"):
        mortal_discard_teacher_tensors_from_extras({MORTAL_Q_VALUES_EXTRA_KEY: _q_values().unsqueeze(0)})
