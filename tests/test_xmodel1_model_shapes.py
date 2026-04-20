from __future__ import annotations

import pytest
import torch

from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_HISTORY_SUMMARY_DIM,
)
from xmodel1.model import Xmodel1Model
from xmodel1.schema import (
    XMODEL1_SPECIAL_TYPE_ANKAN,
    XMODEL1_SPECIAL_TYPE_CHI_HIGH,
    XMODEL1_SPECIAL_TYPE_CHI_LOW,
    XMODEL1_SPECIAL_TYPE_CHI_MID,
    XMODEL1_SPECIAL_TYPE_DAIMINKAN,
    XMODEL1_SPECIAL_TYPE_DAMA,
    XMODEL1_SPECIAL_TYPE_HORA,
    XMODEL1_SPECIAL_TYPE_KAKAN,
    XMODEL1_SPECIAL_TYPE_NONE,
    XMODEL1_SPECIAL_TYPE_PON,
    XMODEL1_SPECIAL_TYPE_REACH,
    XMODEL1_SPECIAL_TYPE_RYUKYOKU,
)


def test_xmodel1_model_forward_shapes():
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=64,
        candidate_feature_dim=XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim=XMODEL1_CANDIDATE_FLAG_DIM,
        hidden_dim=64,
        num_res_blocks=2,
    )
    batch_size = 3
    max_candidates = 14
    out = model(
        torch.randn(batch_size, 57, 34),
        torch.randn(batch_size, 64),
        torch.randn(batch_size, max_candidates, XMODEL1_CANDIDATE_FEATURE_DIM),
        torch.randint(0, 34, (batch_size, max_candidates), dtype=torch.long),
        torch.randint(0, 2, (batch_size, max_candidates, XMODEL1_CANDIDATE_FLAG_DIM), dtype=torch.uint8),
        torch.ones(batch_size, max_candidates, dtype=torch.uint8),
        history_summary=torch.zeros(batch_size, XMODEL1_HISTORY_SUMMARY_DIM, dtype=torch.float32),
    )
    assert out.discard_logits.shape == (batch_size, max_candidates)
    assert out.special_logits.shape == (batch_size, 12)
    assert out.action_logits.shape == (batch_size, 45)
    assert out.win_logit.shape == (batch_size, 1)
    assert out.dealin_logit.shape == (batch_size, 1)
    assert out.pts_given_win.shape == (batch_size, 1)
    assert out.pts_given_dealin.shape == (batch_size, 1)
    assert out.opp_tenpai_logits.shape == (batch_size, 3)


def test_xmodel1_model_event_history_changes_state_encoding():
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=64,
        candidate_feature_dim=XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim=XMODEL1_CANDIDATE_FLAG_DIM,
        hidden_dim=64,
        num_res_blocks=1,
        dropout=0.0,
    )
    model.eval()
    batch_size = 1
    max_candidates = 14
    common_args = (
        torch.randn(batch_size, 57, 34),
        torch.randn(batch_size, 64),
        torch.randn(batch_size, max_candidates, XMODEL1_CANDIDATE_FEATURE_DIM),
        torch.randint(0, 34, (batch_size, max_candidates), dtype=torch.long),
        torch.randint(0, 2, (batch_size, max_candidates, XMODEL1_CANDIDATE_FLAG_DIM), dtype=torch.uint8),
        torch.ones(batch_size, max_candidates, dtype=torch.uint8),
    )
    pad_history = torch.zeros(batch_size, XMODEL1_HISTORY_SUMMARY_DIM, dtype=torch.float32)
    active_history = pad_history.clone()
    active_history[0, -1] = 1.0

    out_pad = model(*common_args, history_summary=pad_history)
    out_active = model(*common_args, history_summary=active_history)

    assert not torch.allclose(out_pad.discard_logits, out_active.discard_logits)


def test_xmodel1_action_logits_use_special_candidate_path_for_reach_and_none():
    model = Xmodel1Model(57, 64, XMODEL1_CANDIDATE_FEATURE_DIM, XMODEL1_CANDIDATE_FLAG_DIM, hidden_dim=32, num_res_blocks=1)
    discard_logits = torch.zeros(1, 14)
    candidate_tile_id = torch.full((1, 14), -1, dtype=torch.long)
    candidate_mask = torch.zeros(1, 14, dtype=torch.uint8)
    special_logits = torch.tensor([[3.5, 1.0, -1e4, -1e4, -1e4, -1e4, -1e4, -1e4, -1e4, -1e4, -1e4, 2.25]])
    special_type_id = torch.tensor([[XMODEL1_SPECIAL_TYPE_REACH, XMODEL1_SPECIAL_TYPE_DAMA, -1, -1, -1, -1, -1, -1, -1, -1, -1, XMODEL1_SPECIAL_TYPE_NONE]])
    special_mask = torch.tensor([[1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]], dtype=torch.uint8)

    action_logits = model.to_action_logits(
        discard_logits,
        candidate_tile_id,
        candidate_mask,
        special_logits,
        special_type_id,
        special_mask,
    )

    assert float(action_logits[0, 34]) == 3.5
    assert float(action_logits[0, 44]) == 2.25


def test_xmodel1_action_logits_use_special_candidate_path_for_call_families():
    model = Xmodel1Model(57, 64, XMODEL1_CANDIDATE_FEATURE_DIM, XMODEL1_CANDIDATE_FLAG_DIM, hidden_dim=32, num_res_blocks=1)
    discard_logits = torch.zeros(1, 14)
    candidate_tile_id = torch.full((1, 14), -1, dtype=torch.long)
    candidate_mask = torch.zeros(1, 14, dtype=torch.uint8)
    special_logits = torch.tensor([[-1e4, -1e4, -1e4, 1.75, 0.5, 0.25, 2.5, 1.1, 0.8, 0.6, -1e4, -1e4]])
    special_type_id = torch.tensor([[-1, -1, -1, XMODEL1_SPECIAL_TYPE_CHI_LOW, XMODEL1_SPECIAL_TYPE_CHI_MID, XMODEL1_SPECIAL_TYPE_CHI_HIGH, XMODEL1_SPECIAL_TYPE_PON, XMODEL1_SPECIAL_TYPE_DAIMINKAN, XMODEL1_SPECIAL_TYPE_ANKAN, XMODEL1_SPECIAL_TYPE_KAKAN, -1, -1]])
    special_mask = torch.tensor([[0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0]], dtype=torch.uint8)

    action_logits = model.to_action_logits(
        discard_logits,
        candidate_tile_id,
        candidate_mask,
        special_logits,
        special_type_id,
        special_mask,
    )

    assert float(action_logits[0, 35]) == 1.75
    assert float(action_logits[0, 36]) == 0.5
    assert float(action_logits[0, 37]) == 0.25
    assert float(action_logits[0, 38]) == 2.5
    assert float(action_logits[0, 39]) == pytest.approx(1.1)
    assert float(action_logits[0, 40]) == pytest.approx(0.8)
    assert float(action_logits[0, 41]) == pytest.approx(0.6)


def test_xmodel1_action_logits_use_special_candidate_path_for_hora_and_ryukyoku():
    model = Xmodel1Model(57, 64, XMODEL1_CANDIDATE_FEATURE_DIM, XMODEL1_CANDIDATE_FLAG_DIM, hidden_dim=32, num_res_blocks=1)
    discard_logits = torch.zeros(1, 14)
    candidate_tile_id = torch.full((1, 14), -1, dtype=torch.long)
    candidate_mask = torch.zeros(1, 14, dtype=torch.uint8)
    special_logits = torch.tensor([[-1e4, -1e4, 4.0, -1e4, -1e4, -1e4, -1e4, -1e4, -1e4, -1e4, 1.25, -1e4]])
    special_type_id = torch.tensor([[-1, -1, XMODEL1_SPECIAL_TYPE_HORA, -1, -1, -1, -1, -1, -1, -1, XMODEL1_SPECIAL_TYPE_RYUKYOKU, -1]])
    special_mask = torch.tensor([[0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0]], dtype=torch.uint8)

    action_logits = model.to_action_logits(
        discard_logits,
        candidate_tile_id,
        candidate_mask,
        special_logits,
        special_type_id,
        special_mask,
    )

    assert float(action_logits[0, 42]) == 4.0
    assert float(action_logits[0, 43]) == 1.25
