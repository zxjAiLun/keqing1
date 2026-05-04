from __future__ import annotations

import json

import pytest
import torch

from keqingrl.actions import ACTION_FLAG_REACH, ActionSpec, ActionType
from keqingrl.mortal_teacher import (
    MORTAL_AGARI_ACTION_ID,
    MORTAL_CHI_HIGH_ACTION_ID,
    MORTAL_CHI_LOW_ACTION_ID,
    MORTAL_CHI_MID_ACTION_ID,
    MORTAL_KAN_ACTION_ID,
    MORTAL_PASS_ACTION_ID,
    MORTAL_PON_ACTION_ID,
    MORTAL_RIICHI_ACTION_ID,
    MORTAL_RYUKYOKU_ACTION_ID,
    MORTAL_ACTION_SPACE,
    MortalTeacherMappingError,
    assert_mortal_action_mask_compatible,
    mortal_action_mapping_audit_row,
    mortal_action_ids_for_action_spec,
    mortal_scores_for_legal_actions,
)
from mahjong_env.action_space import TILE_NAME_TO_IDX


def _q_values() -> torch.Tensor:
    return torch.arange(MORTAL_ACTION_SPACE, dtype=torch.float32)


def _mask(*ids: int) -> torch.BoolTensor:
    mask = torch.zeros((MORTAL_ACTION_SPACE,), dtype=torch.bool)
    mask[list(ids)] = True
    return mask


def _tile(name: str) -> int:
    return int(TILE_NAME_TO_IDX[name])


def test_mortal_full_action_ids_cover_reach_terminal_pass_and_calls() -> None:
    assert mortal_action_ids_for_action_spec(
        ActionSpec(ActionType.REACH_DISCARD, tile=_tile("5m"), flags=ACTION_FLAG_REACH)
    ) == (MORTAL_RIICHI_ACTION_ID, _tile("5m"), 34)
    assert mortal_action_ids_for_action_spec(ActionSpec(ActionType.TSUMO, tile=_tile("7p"))) == (
        MORTAL_AGARI_ACTION_ID,
    )
    assert mortal_action_ids_for_action_spec(ActionSpec(ActionType.RON, tile=_tile("7p"), from_who=1)) == (
        MORTAL_AGARI_ACTION_ID,
    )
    assert mortal_action_ids_for_action_spec(ActionSpec(ActionType.PON, tile=_tile("E"), consumed=(_tile("E"), _tile("E")), from_who=2)) == (
        MORTAL_PON_ACTION_ID,
    )
    assert mortal_action_ids_for_action_spec(ActionSpec(ActionType.PASS)) == (MORTAL_PASS_ACTION_ID,)
    assert mortal_action_ids_for_action_spec(ActionSpec(ActionType.RYUKYOKU)) == (MORTAL_RYUKYOKU_ACTION_ID,)


def test_mortal_chi_action_ids_follow_low_mid_high_called_tile_position() -> None:
    low = ActionSpec(
        ActionType.CHI,
        tile=_tile("3m"),
        consumed=(_tile("4m"), _tile("5m")),
        from_who=0,
    )
    mid = ActionSpec(
        ActionType.CHI,
        tile=_tile("5p"),
        consumed=(_tile("4p"), _tile("6p")),
        from_who=0,
    )
    high = ActionSpec(
        ActionType.CHI,
        tile=_tile("7s"),
        consumed=(_tile("5s"), _tile("6s")),
        from_who=0,
    )

    assert mortal_action_ids_for_action_spec(low) == (MORTAL_CHI_LOW_ACTION_ID,)
    assert mortal_action_ids_for_action_spec(mid) == (MORTAL_CHI_MID_ACTION_ID,)
    assert mortal_action_ids_for_action_spec(high) == (MORTAL_CHI_HIGH_ACTION_ID,)


def test_mortal_scores_for_legal_actions_scores_current_keqing_legal_set_only() -> None:
    legal_actions = (
        ActionSpec(ActionType.DISCARD, tile=_tile("5m")),
        ActionSpec(ActionType.REACH_DISCARD, tile=_tile("5m"), flags=ACTION_FLAG_REACH),
        ActionSpec(ActionType.TSUMO, tile=_tile("7p")),
        ActionSpec(ActionType.PASS),
        ActionSpec(ActionType.RYUKYOKU),
    )

    mapped = mortal_scores_for_legal_actions(
        _q_values(),
        _mask(4, 34, MORTAL_RIICHI_ACTION_ID, MORTAL_AGARI_ACTION_ID, MORTAL_PASS_ACTION_ID, MORTAL_RYUKYOKU_ACTION_ID),
        legal_actions,
    )

    assert mapped.score_mask.tolist() == [True, True, True, True, True]
    assert mapped.scores.tolist() == [34.0, 71.0, 43.0, 45.0, 44.0]
    assert mapped.source_action_ids == (
        (4, 34),
        (MORTAL_RIICHI_ACTION_ID, 4, 34),
        (MORTAL_AGARI_ACTION_ID,),
        (MORTAL_PASS_ACTION_ID,),
        (MORTAL_RYUKYOKU_ACTION_ID,),
    )


def test_mortal_reach_discard_scores_include_tile_level_discard_q() -> None:
    legal_actions = (
        ActionSpec(ActionType.REACH_DISCARD, tile=_tile("2m"), flags=ACTION_FLAG_REACH),
        ActionSpec(ActionType.REACH_DISCARD, tile=_tile("7p"), flags=ACTION_FLAG_REACH),
    )
    q_values = _q_values()
    q_values[_tile("2m")] = 0.5
    q_values[_tile("7p")] = 4.0
    q_values[MORTAL_RIICHI_ACTION_ID] = 10.0

    mapped = mortal_scores_for_legal_actions(
        q_values,
        _mask(MORTAL_RIICHI_ACTION_ID, _tile("2m"), _tile("7p")),
        legal_actions,
    )

    assert mapped.score_mask.tolist() == [True, True]
    assert mapped.scores.tolist() == [10.5, 14.0]
    assert mapped.source_action_ids == (
        (MORTAL_RIICHI_ACTION_ID, _tile("2m")),
        (MORTAL_RIICHI_ACTION_ID, _tile("7p")),
    )


def test_mortal_reach_discard_requires_riichi_and_discard_tile_mask() -> None:
    legal_actions = (
        ActionSpec(ActionType.REACH_DISCARD, tile=_tile("2m"), flags=ACTION_FLAG_REACH),
    )

    missing_riichi = mortal_scores_for_legal_actions(
        _q_values(),
        _mask(_tile("2m")),
        legal_actions,
        strict_mask=False,
    )
    missing_tile = mortal_scores_for_legal_actions(
        _q_values(),
        _mask(MORTAL_RIICHI_ACTION_ID),
        legal_actions,
        strict_mask=False,
    )

    assert missing_riichi.score_mask.tolist() == [False]
    assert missing_tile.score_mask.tolist() == [False]
    assert missing_riichi.missing_legal_keys == (legal_actions[0].canonical_key,)
    assert missing_tile.missing_legal_keys == (legal_actions[0].canonical_key,)


def test_mortal_action_mask_compatibility_fails_for_missing_or_extra_actions() -> None:
    legal_actions = (ActionSpec(ActionType.PASS),)

    with pytest.raises(MortalTeacherMappingError, match="missing_legal_keys"):
        assert_mortal_action_mask_compatible(_mask(), legal_actions)

    with pytest.raises(MortalTeacherMappingError, match="extra_mortal_action_ids"):
        assert_mortal_action_mask_compatible(_mask(MORTAL_PASS_ACTION_ID, MORTAL_PON_ACTION_ID), legal_actions)


def test_mortal_scores_can_report_missing_without_strict_failure() -> None:
    legal_actions = (ActionSpec(ActionType.PASS),)

    mapped = mortal_scores_for_legal_actions(_q_values(), _mask(MORTAL_PON_ACTION_ID), legal_actions, strict_mask=False)

    assert mapped.score_mask.tolist() == [False]
    assert mapped.missing_legal_keys == (legal_actions[0].canonical_key,)
    assert mapped.extra_mortal_action_ids == (MORTAL_PON_ACTION_ID,)


def test_mortal_action_audit_reports_missing_pass() -> None:
    legal_actions = (ActionSpec(ActionType.PASS),)

    row = mortal_action_mapping_audit_row(_q_values(), _mask(MORTAL_PON_ACTION_ID), legal_actions)

    assert row is not None
    assert row["mismatch_kind"] == "missing_and_extra"
    assert json.loads(str(row["missing_legal_keys_json"])) == [legal_actions[0].canonical_key]
    assert json.loads(str(row["missing_legal_types_json"])) == ["PASS"]
    assert json.loads(str(row["extra_mortal_action_ids_json"])) == [MORTAL_PON_ACTION_ID]


def test_mortal_action_audit_reports_missing_pon_and_extra_ids() -> None:
    legal_actions = (
        ActionSpec(ActionType.PON, tile=_tile("E"), consumed=(_tile("E"), _tile("E")), from_who=2),
    )

    row = mortal_action_mapping_audit_row(_q_values(), _mask(MORTAL_PASS_ACTION_ID), legal_actions)

    assert row is not None
    assert row["mismatch_kind"] == "missing_and_extra"
    assert json.loads(str(row["missing_legal_types_json"])) == ["PON"]
    assert json.loads(str(row["extra_mortal_action_ids_json"])) == [MORTAL_PASS_ACTION_ID]


def test_mortal_action_mapping_fails_closed_on_ambiguous_or_unsupported_specs() -> None:
    with pytest.raises(MortalTeacherMappingError, match="from_who"):
        mortal_action_ids_for_action_spec(ActionSpec(ActionType.PON, tile=_tile("E"), consumed=(_tile("E"), _tile("E"))))

    with pytest.raises(MortalTeacherMappingError, match="sequence"):
        mortal_action_ids_for_action_spec(
            ActionSpec(
                ActionType.CHI,
                tile=_tile("3m"),
                consumed=(_tile("5m"), _tile("6m")),
                from_who=0,
            )
        )

    with pytest.raises(MortalTeacherMappingError, match="no action id"):
        mortal_action_ids_for_action_spec(ActionSpec(ActionType.NUKI, tile=_tile("5m")))


def test_mortal_action_mapping_fails_closed_on_multiple_kan_choices() -> None:
    legal_actions = (
        ActionSpec(ActionType.ANKAN, consumed=(_tile("5m"), _tile("5m"), _tile("5m"), _tile("5m"))),
        ActionSpec(ActionType.KAKAN, tile=_tile("5p")),
    )

    with pytest.raises(MortalTeacherMappingError, match="ambiguous"):
        mortal_scores_for_legal_actions(_q_values(), _mask(MORTAL_KAN_ACTION_ID), legal_actions)


def test_mortal_action_mapping_scores_multiple_kan_choices_with_kan_select_q() -> None:
    legal_actions = (
        ActionSpec(ActionType.ANKAN, consumed=(_tile("5m"), _tile("5m"), _tile("5m"), _tile("5m"))),
        ActionSpec(ActionType.KAKAN, tile=_tile("5p")),
    )
    kan_q = torch.zeros((MORTAL_ACTION_SPACE,), dtype=torch.float32)
    kan_q[_tile("5m")] = 0.25
    kan_q[_tile("5p")] = 2.5

    mapped = mortal_scores_for_legal_actions(
        _q_values(),
        _mask(MORTAL_KAN_ACTION_ID),
        legal_actions,
        kan_select_q_values=kan_q,
        kan_select_mask=_mask(_tile("5m"), _tile("5p")),
    )

    assert mapped.score_mask.tolist() == [True, True]
    assert mapped.scores.tolist() == [42.25, 44.5]
    assert mapped.missing_legal_keys == ()


def test_mortal_action_mapping_fails_closed_when_kan_select_mask_missing_tile() -> None:
    legal_actions = (
        ActionSpec(ActionType.ANKAN, consumed=(_tile("5m"), _tile("5m"), _tile("5m"), _tile("5m"))),
        ActionSpec(ActionType.KAKAN, tile=_tile("5p")),
    )

    mapped = mortal_scores_for_legal_actions(
        _q_values(),
        _mask(MORTAL_KAN_ACTION_ID),
        legal_actions,
        strict_mask=False,
        kan_select_q_values=_q_values(),
        kan_select_mask=_mask(_tile("5m")),
    )

    assert mapped.score_mask.tolist() == [True, False]
    assert mapped.missing_legal_keys == (legal_actions[1].canonical_key,)
