from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from xmodel1.adapter import Xmodel1Adapter
from xmodel1.cached_dataset import Xmodel1DiscardDataset
from xmodel1.model import Xmodel1Model


def _write_fixture(path: Path, n: int = 2) -> None:
    k = 14
    d = 21
    f = 10
    candidate_mask = np.zeros((n, k), dtype=np.uint8)
    candidate_mask[:, :3] = 1
    candidate_tile_id = np.full((n, k), -1, dtype=np.int16)
    candidate_tile_id[:, 0] = 0
    candidate_tile_id[:, 1] = 1
    candidate_tile_id[:, 2] = 27
    np.savez(
        path,
        state_tile_feat=np.random.randn(n, 57, 34).astype(np.float16),
        state_scalar=np.random.randn(n, 64).astype(np.float16),
        candidate_feat=np.random.randn(n, k, d).astype(np.float16),
        candidate_tile_id=candidate_tile_id,
        candidate_mask=candidate_mask,
        candidate_flags=np.zeros((n, k, f), dtype=np.uint8),
        chosen_candidate_idx=np.zeros((n,), dtype=np.int16),
        candidate_quality_score=np.random.randn(n, k).astype(np.float32),
        candidate_rank_bucket=np.zeros((n, k), dtype=np.int8),
        candidate_hard_bad_flag=np.zeros((n, k), dtype=np.uint8),
        global_value_target=np.zeros((n,), dtype=np.float32),
        score_delta_target=np.zeros((n,), dtype=np.float32),
        win_target=np.zeros((n,), dtype=np.float32),
        dealin_target=np.zeros((n,), dtype=np.float32),
        offense_quality_target=np.zeros((n,), dtype=np.float32),
        sample_type=np.zeros((n,), dtype=np.int8),
        actor=np.zeros((n,), dtype=np.int8),
        event_index=np.arange(n, dtype=np.int32),
        kyoku=np.ones((n,), dtype=np.int8),
        honba=np.zeros((n,), dtype=np.int8),
        is_open_hand=np.zeros((n,), dtype=np.uint8),
    )


def test_xmodel1_adapter_scores_batch_and_returns_review_row(tmp_path: Path):
    fixture = tmp_path / "sample.npz"
    _write_fixture(fixture)
    ds = Xmodel1DiscardDataset([fixture], shuffle=False)
    batch = Xmodel1DiscardDataset.collate(list(ds))
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=64,
        candidate_feature_dim=21,
        candidate_flag_dim=10,
        hidden_dim=32,
        num_res_blocks=1,
    )
    adapter = Xmodel1Adapter(model, device="cpu")
    scored = adapter.score_batch(batch)
    review = adapter.scored_row_to_review(scored, 0, k=2)
    assert len(review.top_k) == 2
    assert review.chosen_action.startswith("dahai:")

