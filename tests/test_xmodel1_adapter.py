from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from xmodel1.adapter import Xmodel1Adapter
from xmodel1.cached_dataset import Xmodel1DiscardDataset
from xmodel1.model import Xmodel1Model
from xmodel1.schema import (
    XMODEL1_SAMPLE_TYPE_CALL,
    XMODEL1_SAMPLE_TYPE_DISCARD,
    XMODEL1_SPECIAL_TYPE_NONE,
)


def _write_fixture(path: Path, n: int = 2) -> None:
    k = 14
    d = 35
    f = 10
    candidate_mask = np.zeros((n, k), dtype=np.uint8)
    candidate_mask[:, :3] = 1
    candidate_tile_id = np.full((n, k), -1, dtype=np.int16)
    candidate_tile_id[:, 0] = 0
    candidate_tile_id[:, 1] = 1
    candidate_tile_id[:, 2] = 27
    special_mask = np.zeros((n, 12), dtype=np.uint8)
    special_mask[1, 0] = 1
    special_type_id = np.full((n, 12), -1, dtype=np.int16)
    special_type_id[1, 0] = XMODEL1_SPECIAL_TYPE_NONE
    np.savez(
        path,
        schema_name=np.array("xmodel1_discard_v2", dtype=np.str_),
        schema_version=np.array(2, dtype=np.int32),
        state_tile_feat=np.random.randn(n, 57, 34).astype(np.float16),
        state_scalar=np.random.randn(n, 64).astype(np.float16),
        candidate_feat=np.random.randn(n, k, d).astype(np.float16),
        candidate_tile_id=candidate_tile_id,
        candidate_mask=candidate_mask,
        candidate_flags=np.zeros((n, k, f), dtype=np.uint8),
        chosen_candidate_idx=np.array([0, -1], dtype=np.int16),
        sample_type=np.array([XMODEL1_SAMPLE_TYPE_DISCARD, XMODEL1_SAMPLE_TYPE_CALL], dtype=np.int8),
        action_idx_target=np.array([0, 44], dtype=np.int16),
        candidate_quality_score=np.random.randn(n, k).astype(np.float32),
        candidate_rank_bucket=np.zeros((n, k), dtype=np.int8),
        candidate_hard_bad_flag=np.zeros((n, k), dtype=np.uint8),
        special_candidate_feat=np.zeros((n, 12, 25), dtype=np.float16),
        special_candidate_type_id=special_type_id,
        special_candidate_mask=special_mask,
        special_candidate_quality_score=np.zeros((n, 12), dtype=np.float32),
        special_candidate_rank_bucket=np.zeros((n, 12), dtype=np.int8),
        special_candidate_hard_bad_flag=np.zeros((n, 12), dtype=np.uint8),
        chosen_special_candidate_idx=np.array([-1, 0], dtype=np.int16),
        score_delta_target=np.zeros((n,), dtype=np.float32),
        win_target=np.zeros((n,), dtype=np.float32),
        dealin_target=np.zeros((n,), dtype=np.float32),
        pts_given_win_target=np.zeros((n,), dtype=np.float32),
        pts_given_dealin_target=np.zeros((n,), dtype=np.float32),
        opp_tenpai_target=np.zeros((n, 3), dtype=np.float32),
        event_history=np.zeros((n, 48, 5), dtype=np.int16),
        actor=np.zeros((n,), dtype=np.int8),
        event_index=np.arange(n, dtype=np.int32),
        kyoku=np.ones((n,), dtype=np.int8),
        honba=np.zeros((n,), dtype=np.int8),
        is_open_hand=np.zeros((n,), dtype=np.uint8),
        replay_id=np.array(["r0", "r1"], dtype=np.str_),
        sample_id=np.array(["r0:0", "r1:1"], dtype=np.str_),
    )


def test_xmodel1_adapter_scores_batch_and_returns_discard_review_row(tmp_path: Path):
    fixture = tmp_path / "sample.npz"
    _write_fixture(fixture)
    ds = Xmodel1DiscardDataset([fixture], shuffle=False)
    batch = Xmodel1DiscardDataset.collate(list(ds))
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=64,
        candidate_feature_dim=35,
        candidate_flag_dim=10,
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


def test_xmodel1_adapter_returns_special_review_row_for_call_samples(tmp_path: Path):
    fixture = tmp_path / "sample.npz"
    _write_fixture(fixture)
    ds = Xmodel1DiscardDataset([fixture], shuffle=False)
    batch = Xmodel1DiscardDataset.collate(list(ds))
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=64,
        candidate_feature_dim=35,
        candidate_flag_dim=10,
        hidden_dim=32,
        num_res_blocks=1,
    )
    adapter = Xmodel1Adapter(model, device="cpu")
    scored = adapter.score_batch(batch)
    review = adapter.scored_row_to_review(scored, 1, k=2)
    assert review.chosen_action == "none"
    assert review.top_k
    assert review.top_k[0].action == "none"
