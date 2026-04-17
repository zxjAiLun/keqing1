from pathlib import Path

import numpy as np
import torch

from xmodel1.cached_dataset import Xmodel1DiscardDataset
from xmodel1.model import Xmodel1Model


def _write_fixture(path: Path) -> None:
    n = 4
    k = 14
    d = 35
    f = 10
    mask = np.zeros((n, k), dtype=np.uint8)
    mask[:, :4] = 1
    np.savez(
        path,
        schema_name=np.array("xmodel1_discard_v2", dtype=np.str_),
        schema_version=np.array(2, dtype=np.int32),
        state_tile_feat=np.zeros((n, 57, 34), dtype=np.float16),
        state_scalar=np.zeros((n, 56), dtype=np.float16),
        candidate_feat=np.zeros((n, k, d), dtype=np.float16),
        candidate_tile_id=np.full((n, k), -1, dtype=np.int16),
        candidate_mask=mask,
        candidate_flags=np.zeros((n, k, f), dtype=np.uint8),
        chosen_candidate_idx=np.zeros((n,), dtype=np.int16),
        action_idx_target=np.zeros((n,), dtype=np.int16),
        candidate_quality_score=np.zeros((n, k), dtype=np.float32),
        candidate_rank_bucket=np.zeros((n, k), dtype=np.int8),
        candidate_hard_bad_flag=np.zeros((n, k), dtype=np.uint8),
        special_candidate_feat=np.zeros((n, 12, 25), dtype=np.float16),
        special_candidate_type_id=np.full((n, 12), -1, dtype=np.int16),
        special_candidate_mask=np.zeros((n, 12), dtype=np.uint8),
        special_candidate_quality_score=np.zeros((n, 12), dtype=np.float32),
        special_candidate_rank_bucket=np.zeros((n, 12), dtype=np.int8),
        special_candidate_hard_bad_flag=np.zeros((n, 12), dtype=np.uint8),
        chosen_special_candidate_idx=np.full((n,), -1, dtype=np.int16),
        score_delta_target=np.zeros((n,), dtype=np.float32),
        win_target=np.zeros((n,), dtype=np.float32),
        dealin_target=np.zeros((n,), dtype=np.float32),
        pts_given_win_target=np.zeros((n,), dtype=np.float32),
        pts_given_dealin_target=np.zeros((n,), dtype=np.float32),
        opp_tenpai_target=np.zeros((n, 3), dtype=np.float32),
        event_history=np.zeros((n, 48, 5), dtype=np.int16),
        sample_type=np.zeros((n,), dtype=np.int8),
        actor=np.zeros((n,), dtype=np.int8),
        event_index=np.arange(n, dtype=np.int32),
        kyoku=np.ones((n,), dtype=np.int8),
        honba=np.zeros((n,), dtype=np.int8),
        is_open_hand=np.zeros((n,), dtype=np.uint8),
    )


def test_xmodel1_model_forward_smoke(tmp_path: Path):
    fixture = tmp_path / "sample.npz"
    _write_fixture(fixture)
    ds = Xmodel1DiscardDataset([fixture], shuffle=False)
    batch = Xmodel1DiscardDataset.collate(list(ds))
    model = Xmodel1Model(
        state_tile_channels=batch["state_tile_feat"].shape[1],
        state_scalar_dim=batch["state_scalar"].shape[1],
        candidate_feature_dim=batch["candidate_feat"].shape[2],
        candidate_flag_dim=batch["candidate_flags"].shape[2],
    )
    out = model(
        batch["state_tile_feat"].float(),
        batch["state_scalar"].float(),
        batch["candidate_feat"].float(),
        batch["candidate_tile_id"],
        batch["candidate_flags"].float(),
        batch["candidate_mask"],
    )
    assert out.discard_logits.shape == (4, 14)
    assert out.win_logit.shape == (4, 1)
    assert out.dealin_logit.shape == (4, 1)
    assert out.pts_given_win.shape == (4, 1)
    assert out.pts_given_dealin.shape == (4, 1)
    assert out.opp_tenpai_logits.shape == (4, 3)
    assert torch.isfinite(out.discard_logits).all()
    assert torch.isfinite(out.pts_given_win).all()
    assert torch.isfinite(out.pts_given_dealin).all()


def test_xmodel1_model_backward_smoke(tmp_path: Path):
    fixture = tmp_path / "sample.npz"
    _write_fixture(fixture)
    ds = Xmodel1DiscardDataset([fixture], shuffle=False)
    batch = Xmodel1DiscardDataset.collate(list(ds))
    model = Xmodel1Model(
        state_tile_channels=batch["state_tile_feat"].shape[1],
        state_scalar_dim=batch["state_scalar"].shape[1],
        candidate_feature_dim=batch["candidate_feat"].shape[2],
        candidate_flag_dim=batch["candidate_flags"].shape[2],
    )
    out = model(
        batch["state_tile_feat"].float(),
        batch["state_scalar"].float(),
        batch["candidate_feat"].float(),
        batch["candidate_tile_id"],
        batch["candidate_flags"].float(),
        batch["candidate_mask"],
    )
    loss = torch.nn.functional.cross_entropy(out.discard_logits, batch["chosen_candidate_idx"])
    loss.backward()
    grad_sum = sum(
        float(param.grad.abs().sum().item())
        for param in model.parameters()
        if param.grad is not None
    )
    assert grad_sum > 0.0
