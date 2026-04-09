from __future__ import annotations

from pathlib import Path

import numpy as np

from xmodel1.cached_dataset import Xmodel1DiscardDataset


def _write_sample_npz(path: Path) -> None:
    n = 2
    np.savez(
        path,
        state_tile_feat=np.zeros((n, 57, 34), dtype=np.float16),
        state_scalar=np.zeros((n, 64), dtype=np.float16),
        candidate_feat=np.zeros((n, 14, 21), dtype=np.float16),
        candidate_tile_id=np.full((n, 14), -1, dtype=np.int16),
        candidate_mask=np.concatenate(
            [np.ones((n, 3), dtype=np.uint8), np.zeros((n, 11), dtype=np.uint8)],
            axis=1,
        ),
        candidate_flags=np.zeros((n, 14, 10), dtype=np.uint8),
        chosen_candidate_idx=np.zeros((n,), dtype=np.int16),
        candidate_quality_score=np.zeros((n, 14), dtype=np.float32),
        candidate_rank_bucket=np.zeros((n, 14), dtype=np.int8),
        candidate_hard_bad_flag=np.zeros((n, 14), dtype=np.uint8),
        global_value_target=np.zeros((n,), dtype=np.float32),
        score_delta_target=np.zeros((n,), dtype=np.float32),
        win_target=np.zeros((n,), dtype=np.float32),
        dealin_target=np.zeros((n,), dtype=np.float32),
        offense_quality_target=np.zeros((n,), dtype=np.float32),
        sample_type=np.zeros((n,), dtype=np.int8),
        actor=np.zeros((n,), dtype=np.int8),
        event_index=np.zeros((n,), dtype=np.int32),
        kyoku=np.ones((n,), dtype=np.int8),
        honba=np.zeros((n,), dtype=np.int8),
        is_open_hand=np.zeros((n,), dtype=np.uint8),
    )


def test_xmodel1_cached_dataset_iterates_and_collates(tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    _write_sample_npz(npz_path)
    ds = Xmodel1DiscardDataset([npz_path], shuffle=False, buffer_size=4, seed=1)
    rows = list(ds)
    assert len(rows) == 2
    batch = Xmodel1DiscardDataset.collate(rows)
    assert batch["state_tile_feat"].shape == (2, 57, 34)
    assert batch["state_scalar"].shape == (2, 64)
    assert batch["candidate_feat"].shape == (2, 14, 21)
    assert batch["candidate_flags"].shape == (2, 14, 10)
    assert batch["candidate_mask"].shape == (2, 14)
