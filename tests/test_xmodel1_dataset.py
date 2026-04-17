from pathlib import Path

import numpy as np

from xmodel1.cached_dataset import Xmodel1DiscardDataset


def _write_fixture(path: Path) -> None:
    n = 3
    k = 14
    d = 35
    f = 10
    np.savez(
        path,
        state_tile_feat=np.zeros((n, 57, 34), dtype=np.float16),
        state_scalar=np.zeros((n, 56), dtype=np.float16),
        candidate_feat=np.zeros((n, k, d), dtype=np.float16),
        candidate_tile_id=np.full((n, k), -1, dtype=np.int16),
        candidate_mask=np.zeros((n, k), dtype=np.uint8),
        candidate_flags=np.zeros((n, k, f), dtype=np.uint8),
        chosen_candidate_idx=np.zeros((n,), dtype=np.int16),
        candidate_quality_score=np.zeros((n, k), dtype=np.float32),
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


def test_xmodel1_dataset_reads_fixture(tmp_path: Path):
    fixture = tmp_path / "sample.npz"
    _write_fixture(fixture)
    ds = Xmodel1DiscardDataset([fixture], shuffle=False)
    rows = list(ds)
    assert len(rows) == 3
    assert rows[0][0].shape == (57, 34)
    assert rows[0][2].shape == (14, 35)


def test_xmodel1_collate_returns_expected_keys(tmp_path: Path):
    fixture = tmp_path / "sample.npz"
    _write_fixture(fixture)
    ds = Xmodel1DiscardDataset([fixture], shuffle=False)
    batch = Xmodel1DiscardDataset.collate(list(ds))
    assert batch["state_tile_feat"].shape == (3, 57, 34)
    assert batch["candidate_feat"].shape == (3, 14, 35)
    assert batch["candidate_flags"].shape == (3, 14, 10)
    assert batch["chosen_candidate_idx"].shape == (3,)
