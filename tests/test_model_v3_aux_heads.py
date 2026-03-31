import torch

from keqingv3.features import C_TILE, N_SCALAR
from keqingv3.model import MahjongModel


def test_v3_model_exposes_aux_heads_after_forward():
    model = MahjongModel()
    tile_feat = torch.zeros(2, C_TILE, 34, dtype=torch.float32)
    scalar = torch.zeros(2, N_SCALAR, dtype=torch.float32)

    policy, value = model(tile_feat, scalar)
    aux = model.get_last_aux_outputs()

    assert policy.shape == (2, 45)
    assert value.shape == (2, 1)
    assert aux["score_delta"].shape == (2, 1)
    assert aux["win_prob"].shape == (2, 1)
    assert aux["dealin_prob"].shape == (2, 1)
