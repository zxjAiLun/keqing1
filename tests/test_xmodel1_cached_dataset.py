from __future__ import annotations

from pathlib import Path
import json

import numpy as np
import pytest
import torch
import xmodel1.cached_dataset as cached_dataset_mod

from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_HISTORY_SUMMARY_DIM,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_MAX_RESPONSE_CANDIDATES,
    XMODEL1_RULE_CONTEXT_DIM,
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
)
from tests.xmodel1_test_utils import make_xmodel1_v3_payload, write_xmodel1_v3_npz
from xmodel1.cached_dataset import (
    Xmodel1DiscardDataset,
    discover_cached_files,
    infer_cached_dimensions,
    probe_cached_samples,
    summarize_cached_files,
    validate_export_manifest,
)


def _write_sample_npz(path: Path) -> None:
    write_xmodel1_v3_npz(path, n=2)


def test_xmodel1_cached_dataset_iterates_and_collates(tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    _write_sample_npz(npz_path)
    ds = Xmodel1DiscardDataset([npz_path], shuffle=False, buffer_size=4, seed=1)
    rows = list(ds)
    assert len(rows) == 2
    batch = Xmodel1DiscardDataset.collate(rows)
    assert batch["state_tile_feat"].shape == (2, 57, 34)
    assert batch["state_scalar"].shape == (2, 64)
    assert batch["candidate_feat"].shape == (2, 14, XMODEL1_CANDIDATE_FEATURE_DIM)
    assert batch["candidate_flags"].shape == (2, 14, XMODEL1_CANDIDATE_FLAG_DIM)
    assert batch["candidate_mask"].shape == (2, 14)
    assert batch["action_idx_target"].shape == (2,)
    assert batch["response_action_idx"].shape == (2, XMODEL1_MAX_RESPONSE_CANDIDATES)
    assert batch["response_post_candidate_feat"].shape == (
        2,
        XMODEL1_MAX_RESPONSE_CANDIDATES,
        XMODEL1_MAX_CANDIDATES,
        XMODEL1_CANDIDATE_FEATURE_DIM,
    )
    assert batch["history_summary"].shape == (2, XMODEL1_HISTORY_SUMMARY_DIM)
    assert batch["pts_given_win_target"].shape == (2,)
    assert batch["pts_given_dealin_target"].shape == (2,)
    assert batch["final_rank_target"].shape == (2,)
    assert batch["final_score_delta_points_target"].shape == (2,)
    assert batch["response_human_discard_idx"].shape == (2, XMODEL1_MAX_RESPONSE_CANDIDATES)
    assert batch["rule_context"].shape == (2, XMODEL1_RULE_CONTEXT_DIM)
    assert float(batch["pts_given_win_target"].sum()) == 0.0
    assert float(batch["pts_given_dealin_target"].sum()) == 0.0
    assert batch["replay_id"] == ["sample", "sample"]
    assert batch["sample_id"] == ["sample:0", "sample:1"]


def test_xmodel1_infer_cached_dimensions_reads_real_shapes(tmp_path: Path):
    npz_path = tmp_path / "sample.npz"
    _write_sample_npz(npz_path)
    dims = infer_cached_dimensions([npz_path])
    assert dims == {
        "state_tile_channels": 57,
        "state_scalar_dim": 64,
        "candidate_feature_dim": XMODEL1_CANDIDATE_FEATURE_DIM,
        "candidate_flag_dim": XMODEL1_CANDIDATE_FLAG_DIM,
        "max_candidates": 14,
    }


def test_xmodel1_infer_cached_dimensions_logs_skipped_invalid_files(tmp_path: Path, capsys):
    bad = tmp_path / "broken.npz"
    bad.write_bytes(b"not-a-zip")
    good = tmp_path / "sample.npz"
    _write_sample_npz(good)

    dims = infer_cached_dimensions([bad, good])

    captured = capsys.readouterr()
    assert dims["state_scalar_dim"] == 64
    assert "skipping unreadable cache" in captured.err
    assert "skipped 1 unreadable cache file(s)" in captured.err


def test_xmodel1_infer_cached_dimensions_surfaces_all_skipped_errors_when_no_file_is_readable(tmp_path: Path):
    bad = tmp_path / "broken.npz"
    bad.write_bytes(b"not-a-zip")

    with pytest.raises(FileNotFoundError, match="skipped errors"):
        infer_cached_dimensions([bad])


def test_xmodel1_infer_cached_dimensions_can_fail_closed_in_strict_mode(tmp_path: Path):
    bad = tmp_path / "broken.npz"
    bad.write_bytes(b"not-a-zip")
    good = tmp_path / "sample.npz"
    _write_sample_npz(good)

    with pytest.raises(RuntimeError, match="strict cache scan failed"):
        infer_cached_dimensions([bad, good], strict=True)


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


def test_xmodel1_cached_dataset_loads_each_npz_array_once_per_file(monkeypatch):
    n = 3
    payload = make_xmodel1_v3_payload(n=n)
    access_counts: dict[str, int] = {key: 0 for key in payload}

    class _FakeNpz:
        def __contains__(self, key):
            return key in payload

        def __getitem__(self, key):
            access_counts[key] += 1
            return payload[key]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

    monkeypatch.setattr(cached_dataset_mod.np, "load", lambda *args, **kwargs: _FakeNpz())

    ds = Xmodel1DiscardDataset([Path("fake.npz")], shuffle=False, buffer_size=8, seed=1)
    rows = list(ds)

    assert len(rows) == n
    assert access_counts["candidate_feat"] == 1
    assert access_counts["candidate_tile_id"] == 1
    assert access_counts["candidate_mask"] == 1
    assert access_counts["candidate_flags"] == 1
    assert access_counts["state_scalar"] == 1
    assert access_counts["opp_tenpai_target"] == 1
    assert access_counts["history_summary"] == 2


def test_xmodel1_cached_dataset_backfills_optional_rule_context_and_human_response(tmp_path: Path):
    npz_path = tmp_path / "legacy_missing_optional.npz"
    payload = make_xmodel1_v3_payload(n=2)
    payload.pop("rule_context")
    payload.pop("response_human_discard_idx")
    np.savez(npz_path, **payload)

    batch = Xmodel1DiscardDataset.collate(list(Xmodel1DiscardDataset([npz_path], shuffle=False, buffer_size=4, seed=1)))

    assert batch["response_human_discard_idx"].shape == (2, XMODEL1_MAX_RESPONSE_CANDIDATES)
    assert torch.equal(batch["response_human_discard_idx"], torch.full((2, XMODEL1_MAX_RESPONSE_CANDIDATES), -1))
    assert batch["rule_context"].shape == (2, XMODEL1_RULE_CONTEXT_DIM)


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


def test_validate_export_manifest_checks_schema_and_requested_shards(tmp_path: Path):
    manifest_path = tmp_path / "xmodel1_export_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_name": XMODEL1_SCHEMA_NAME,
                "schema_version": XMODEL1_SCHEMA_VERSION,
                "file_count": 2,
                "exported_file_count": 2,
                "exported_sample_count": 4,
                "processed_file_count": 2,
                "skipped_existing_file_count": 0,
                "shard_file_counts": {"ds1": 1, "ds2": 1},
                "shard_sample_counts": {"ds1": 2, "ds2": 2},
            }
        ),
        encoding="utf-8",
    )

    manifest = validate_export_manifest(manifest_path, required_shards=["ds1", "ds2"])

    assert manifest["schema_name"] == XMODEL1_SCHEMA_NAME


def test_probe_cached_samples_checks_required_shapes(tmp_path: Path):
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    _write_sample_npz(first)
    _write_sample_npz(second)

    summary = probe_cached_samples([first, second], max_files=2, rows_per_file=1)

    assert summary["num_files"] == 2
    assert summary["rows_probed"] == 2
    assert summary["dims"]["candidate_feature_dim"] == XMODEL1_CANDIDATE_FEATURE_DIM
