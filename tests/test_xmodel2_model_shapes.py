from __future__ import annotations

import torch

from mahjong_env.action_space import ACTION_SPACE
from training.state_features import C_TILE, N_SCALAR
from xmodel2.model import Xmodel2Model


def test_xmodel2_model_forward_shapes():
    model = Xmodel2Model(
        hidden_dim=64,
        num_res_blocks=2,
        dropout=0.0,
    )
    batch_size = 3
    logits, value = model(
        torch.randn(batch_size, C_TILE, 34),
        torch.randn(batch_size, N_SCALAR),
    )

    aux = model.get_last_aux_outputs()
    assert logits.shape == (batch_size, ACTION_SPACE)
    assert value.shape == (batch_size, 1)
    assert aux["win_logit"].shape == (batch_size, 1)
    assert aux["dealin_logit"].shape == (batch_size, 1)
    assert aux["pts_given_win"].shape == (batch_size, 1)
    assert aux["pts_given_dealin"].shape == (batch_size, 1)
    assert aux["opp_tenpai_logits"].shape == (batch_size, 3)
    assert aux["rank_logits"].shape == (batch_size, 4)
    assert aux["final_score_delta"].shape == (batch_size, 1)


def test_xmodel2_model_scalar_changes_affect_rank_logits():
    model = Xmodel2Model(
        hidden_dim=64,
        num_res_blocks=1,
        dropout=0.0,
    )
    model.eval()
    tile_feat = torch.randn(1, C_TILE, 34)
    scalar = torch.randn(1, N_SCALAR)
    scalar_alt = scalar.clone()
    scalar_alt[0, 7] += 0.75

    model(tile_feat, scalar)
    logits_a = model.get_last_aux_outputs()["rank_logits"]
    model(tile_feat, scalar_alt)
    logits_b = model.get_last_aux_outputs()["rank_logits"]

    assert not torch.allclose(logits_a, logits_b)
