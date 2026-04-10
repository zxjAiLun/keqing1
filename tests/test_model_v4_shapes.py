import torch

from keqingv1.action_space import ACTION_SPACE
from keqingv3.features import C_TILE, N_SCALAR
from keqingv31.model import KeqingV31Model


def test_keqingv31_forward_shapes():
    model = KeqingV31Model(hidden_dim=64, num_res_blocks=2)
    tile = torch.randn(3, C_TILE, 34)
    scalar = torch.randn(3, N_SCALAR)
    logits, value = model(tile, scalar)

    assert logits.shape == (3, ACTION_SPACE)
    assert value.shape == (3, 1)

    aux = model.get_last_aux_outputs()
    assert aux['global_value'].shape == (3, 1)
    assert aux['offense_value'].shape == (3, 1)
    assert aux['defense_risk'].shape == (3, 1)
    assert aux['score_delta'].shape == (3, 1)
    assert aux['win_prob'].shape == (3, 1)
    assert aux['dealin_prob'].shape == (3, 1)
    assert aux['stream_gate'].shape == (3, 3)
