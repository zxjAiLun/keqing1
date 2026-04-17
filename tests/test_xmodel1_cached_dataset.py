from __future__ import annotations

from pathlib import Path

import numpy as np

from xmodel1.cached_dataset import Xmodel1DiscardDataset, discover_cached_files, infer_cached_dimensions, summarize_cached_files


def _write_sample_npz(path: Path) -> None:
    n = 2
    np.savez(
        path,
        schema_name=np.array("xmodel1_discard_v2", dtype=np.str_),
        schema_version=np.array(2, dtype=np.int32),
        state_tile_feat=np.zeros((n, 57, 34), dtype=np.float16),
        state_scalar=np.zeros((n, 64), dtype=np.float16),
        candidate_feat=np.zeros((n, 14, 35), dtype=np.float16),
        candidate_tile_id=np.full((n, 14), -1, dtype=np.int16),
        candidate_mask=np.concatenate(
            [np.ones((n, 3), dtype=np.uint8), np.zeros((n, 11), dtype=np.uint8)],
            axis=1,
        ),
        candidate_flags=np.zeros((n, 14, 10), dtype=np.uint8),
        chosen_candidate_idx=np.zeros((n,), dtype=np.int16),
        sample_type=np.zeros((n,), dtype=np.int8),
        action_idx_target=np.zeros((n,), dtype=np.int16),
        candidate_quality_score=np.zeros((n, 14), dtype=np.float32),
        candidate_rank_bucket=np.zeros((n, 14), dtype=np.int8),
        candidate_hard_bad_flag=np.zeros((n, 14), dtype=np.uint8),
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
    assert batch["candidate_feat"].shape == (2, 14, 35)
    assert batch["candidate_flags"].shape == (2, 14, 10)
    assert batch["candidate_mask"].shape == (2, 14)
    assert batch["action_idx_target"].shape == (2,)
    assert batch["special_candidate_feat"].shape == (2, 12, 25)
    assert batch["special_candidate_mask"].shape == (2, 12)
    assert batch["pts_given_win_target"].shape == (2,)
    assert batch["pts_given_dealin_target"].shape == (2,)
    assert float(batch["pts_given_win_target"].sum()) == 0.0
    assert float(batch["pts_given_dealin_target"].sum()) == 0.0


def test_xmodel1_infer_cached_dimensions_reads_real_shapes(tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    _write_sample_npz(npz_path)
    dims = infer_cached_dimensions([npz_path])
    assert dims == {
        "state_tile_channels": 57,
        "state_scalar_dim": 64,
        "candidate_feature_dim": 35,
        "candidate_flag_dim": 10,
        "max_candidates": 14,
        "special_candidate_feature_dim": 25,
        "max_special_candidates": 12,
    }


def test_discover_cached_files_accepts_processed_root_layout(tmp_path: Path):
    root = tmp_path / "processed_xmodel1"
    ds1 = root / "ds1"
    ds2 = root / "ds2"
    ds1.mkdir(parents=True)
    ds2.mkdir(parents=True)
    first = ds1 / "a.npz"
    second = ds2 / "b.npz"
    _write_sample_npz(first)
    _write_sample_npz(second)

    files = discover_cached_files([root])

    assert files == [first, second]


def test_summarize_cached_files_reports_file_and_sample_counts(tmp_path: Path):
    ds1 = tmp_path / "ds1"
    ds2 = tmp_path / "ds2"
    ds1.mkdir()
    ds2.mkdir()
    first = ds1 / "a.npz"
    second = ds2 / "b.npz"
    _write_sample_npz(first)
    _write_sample_npz(second)

    summary = summarize_cached_files([first, second])

    assert summary["num_files"] == 2
    assert summary["num_samples"] == 4
    assert summary["shard_file_counts"] == {"ds1": 1, "ds2": 1}
    assert summary["shard_sample_counts"] == {"ds1": 2, "ds2": 2}


def test_xmodel1_cached_dataset_prefers_exported_pts_given_targets(tmp_path: Path):
    npz_path = tmp_path / "sample_true_pts.npz"
    _write_sample_npz(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload["pts_given_win_target"] = np.array([0.25, 0.0], dtype=np.float32)
    payload["pts_given_dealin_target"] = np.array([0.0, 0.5], dtype=np.float32)
    np.savez(npz_path, **payload)

    ds = Xmodel1DiscardDataset([npz_path], shuffle=False, buffer_size=4, seed=1)
    batch = Xmodel1DiscardDataset.collate(list(ds))

    assert batch["pts_given_win_target"].tolist() == [0.25, 0.0]
    assert batch["pts_given_dealin_target"].tolist() == [0.0, 0.5]


def test_xmodel1_cached_dataset_rejects_v1_cache(tmp_path: Path):
    npz_path = tmp_path / "legacy.npz"
    _write_sample_npz(npz_path)
    with np.load(npz_path, allow_pickle=False) as data:
        payload = {key: data[key] for key in data.files}
    payload["schema_name"] = np.array("xmodel1_discard_v1", dtype=np.str_)
    payload["schema_version"] = np.array(1, dtype=np.int32)
    np.savez(npz_path, **payload)

    ds = Xmodel1DiscardDataset([npz_path], shuffle=False, buffer_size=4, seed=1)
    try:
        list(ds)
    except ValueError as exc:
        assert "rerun preprocess" in str(exc)
    else:
        raise AssertionError("expected v1 cache to fail")
