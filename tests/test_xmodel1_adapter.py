from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
)
from tests.xmodel1_test_utils import write_xmodel1_v3_npz
from xmodel1.adapter import Xmodel1Adapter
from xmodel1.cached_dataset import Xmodel1DiscardDataset
from xmodel1.model import Xmodel1Model
from xmodel1.schema import (
    XMODEL1_SAMPLE_TYPE_CALL,
    XMODEL1_SAMPLE_TYPE_DISCARD,
)


def _write_fixture(path: Path, n: int = 2) -> None:
    k = 14
    candidate_mask = np.zeros((n, k), dtype=np.uint8)
    candidate_mask[:, :3] = 1
    candidate_tile_id = np.full((n, k), -1, dtype=np.int16)
    candidate_tile_id[:, 0] = 0
    candidate_tile_id[:, 1] = 1
    candidate_tile_id[:, 2] = 27
    payload = write_xmodel1_v3_npz(
        path,
        n=n,
        sample_type=np.array([XMODEL1_SAMPLE_TYPE_DISCARD, XMODEL1_SAMPLE_TYPE_CALL], dtype=np.int8),
        chosen_candidate_idx=np.array([0, -1], dtype=np.int16),
        replay_ids=["r0", "r1"],
        sample_ids=["r0:0", "r1:1"],
    )
    payload["state_tile_feat"][:] = np.random.randn(n, 57, 34).astype(np.float16)
    payload["state_scalar"][:] = np.random.randn(n, 64).astype(np.float16)
    payload["candidate_feat"][:] = np.random.randn(n, k, XMODEL1_CANDIDATE_FEATURE_DIM).astype(np.float16)
    payload["candidate_tile_id"][:] = candidate_tile_id
    payload["candidate_mask"][:] = candidate_mask
    payload["candidate_quality_score"][:] = np.random.randn(n, k).astype(np.float32)
    payload["response_action_idx"][1, 0] = 44
    payload["response_action_mask"][1, 0] = 1
    payload["chosen_response_action_idx"][1] = 0
    np.savez(path, **payload)


def test_xmodel1_adapter_scores_batch_and_returns_discard_review_row(tmp_path: Path):
    fixture = tmp_path / "sample.npz"
    _write_fixture(fixture)
    ds = Xmodel1DiscardDataset([fixture], shuffle=False)
    batch = Xmodel1DiscardDataset.collate(list(ds))
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=64,
        candidate_feature_dim=XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim=XMODEL1_CANDIDATE_FLAG_DIM,
        hidden_dim=32,
        num_res_blocks=1,
    )
    adapter = Xmodel1Adapter(model, device="cpu")
    scored = adapter.score_batch(batch)
    review = adapter.scored_row_to_review(scored, 0, k=2)
    assert len(review.top_k) == 2
    assert review.chosen_action.startswith("dahai:")
    assert 0.0 <= review.win_prob <= 1.0
    assert 0.0 <= review.dealin_prob <= 1.0
    assert isinstance(review.pts_given_win, float)
    assert isinstance(review.pts_given_dealin, float)
    assert isinstance(review.composed_ev, float)
    assert scored["sample_id"] == ["r0:0", "r1:1"]
    assert scored["replay_id"] == ["r0", "r1"]


def test_xmodel1_adapter_returns_response_review_row_for_call_samples(tmp_path: Path):
    fixture = tmp_path / "sample.npz"
    _write_fixture(fixture)
    ds = Xmodel1DiscardDataset([fixture], shuffle=False)
    batch = Xmodel1DiscardDataset.collate(list(ds))
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=64,
        candidate_feature_dim=XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim=XMODEL1_CANDIDATE_FLAG_DIM,
        hidden_dim=32,
        num_res_blocks=1,
    )
    adapter = Xmodel1Adapter(model, device="cpu")
    scored = adapter.score_batch(batch)
    review = adapter.scored_row_to_review(scored, 1, k=2)
    assert review.chosen_action == "none"
    assert review.top_k
    assert review.top_k[0].action == "none"


def test_xmodel1_adapter_rejects_partial_checkpoint_load(tmp_path: Path):
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=56,
        candidate_feature_dim=XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim=XMODEL1_CANDIDATE_FLAG_DIM,
        hidden_dim=32,
        num_res_blocks=1,
    )
    state_dict = model.state_dict()
    state_dict.pop("win_head.2.weight")
    ckpt = tmp_path / "partial_xmodel1.pth"
    torch.save(
        {
            "model": state_dict,
            "cfg": {
                "state_tile_channels": 57,
                "state_scalar_dim": 56,
                "candidate_feature_dim": XMODEL1_CANDIDATE_FEATURE_DIM,
                "candidate_flag_dim": XMODEL1_CANDIDATE_FLAG_DIM,
                "schema_name": XMODEL1_SCHEMA_NAME,
                "schema_version": XMODEL1_SCHEMA_VERSION,
                "hidden_dim": 32,
                "num_res_blocks": 1,
                "dropout": 0.1,
            },
            "schema_name": XMODEL1_SCHEMA_NAME,
            "schema_version": XMODEL1_SCHEMA_VERSION,
        },
        ckpt,
    )

    with pytest.raises(RuntimeError, match="Refusing partial load"):
        Xmodel1Adapter.from_checkpoint(ckpt, device="cpu")
