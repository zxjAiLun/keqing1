from __future__ import annotations

import torch

from xmodel1.model import Xmodel1Model


def test_xmodel1_model_forward_shapes():
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=64,
        candidate_feature_dim=21,
        candidate_flag_dim=10,
        hidden_dim=64,
        num_res_blocks=2,
    )
    batch_size = 3
    max_candidates = 14
    out = model(
        torch.randn(batch_size, 57, 34),
        torch.randn(batch_size, 64),
        torch.randn(batch_size, max_candidates, 21),
        torch.randint(0, 34, (batch_size, max_candidates), dtype=torch.long),
        torch.randint(0, 2, (batch_size, max_candidates, 10), dtype=torch.uint8),
        torch.ones(batch_size, max_candidates, dtype=torch.uint8),
    )
    assert out.discard_logits.shape == (batch_size, max_candidates)
    assert out.global_value.shape == (batch_size, 1)
    assert out.score_delta.shape == (batch_size, 1)
    assert out.win_logit.shape == (batch_size, 1)
    assert out.dealin_logit.shape == (batch_size, 1)
    assert out.offense_quality.shape == (batch_size, 1)
