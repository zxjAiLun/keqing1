from pathlib import Path

import numpy as np
import torch

from tests.xmodel1_test_utils import write_xmodel1_v3_npz
from xmodel1.cached_dataset import Xmodel1DiscardDataset
from xmodel1.model import Xmodel1Model


def _write_fixture(path: Path) -> None:
    write_xmodel1_v3_npz(path, n=4, state_scalar_dim=56, candidate_active=4)


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
        history_summary=batch["history_summary"].float(),
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
        history_summary=batch["history_summary"].float(),
    )
    loss = torch.nn.functional.cross_entropy(out.discard_logits, batch["chosen_candidate_idx"])
    loss.backward()
    grad_sum = sum(
        float(param.grad.abs().sum().item())
        for param in model.parameters()
        if param.grad is not None
    )
    assert grad_sum > 0.0
