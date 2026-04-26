from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np


REPO_ROOT = "/media/bailan/DISK1/AUbuntuProject/project/keqing1"


def _write_legacy_npz(path: Path) -> None:
    np.savez(
        path,
        state_tile_feat=np.zeros((2, 57, 34), dtype=np.float16),
        state_scalar=np.zeros((2, 56), dtype=np.float16),
        candidate_feat=np.zeros((2, 14, 35), dtype=np.float16),
        candidate_tile_id=np.full((2, 14), -1, dtype=np.int16),
        candidate_mask=np.zeros((2, 14), dtype=np.uint8),
        candidate_flags=np.zeros((2, 14, 10), dtype=np.uint8),
        chosen_candidate_idx=np.zeros((2,), dtype=np.int16),
        candidate_quality_score=np.zeros((2, 14), dtype=np.float32),
        candidate_rank_bucket=np.zeros((2, 14), dtype=np.int8),
        candidate_hard_bad_flag=np.zeros((2, 14), dtype=np.uint8),
        special_candidate_feat=np.zeros((2, 12, 25), dtype=np.float16),
        special_candidate_type_id=np.full((2, 12), -1, dtype=np.int16),
        special_candidate_mask=np.zeros((2, 12), dtype=np.uint8),
        special_candidate_quality_score=np.zeros((2, 12), dtype=np.float32),
        special_candidate_rank_bucket=np.zeros((2, 12), dtype=np.int8),
        special_candidate_hard_bad_flag=np.zeros((2, 12), dtype=np.uint8),
        chosen_special_candidate_idx=np.full((2,), -1, dtype=np.int16),
        score_delta_target=np.zeros((2,), dtype=np.float32),
        win_target=np.zeros((2,), dtype=np.float32),
        dealin_target=np.zeros((2,), dtype=np.float32),
        pts_given_win_target=np.zeros((2,), dtype=np.float32),
        pts_given_dealin_target=np.zeros((2,), dtype=np.float32),
        opp_tenpai_target=np.zeros((2, 3), dtype=np.float32),
        event_history=np.zeros((2, 48, 5), dtype=np.int16),
        sample_type=np.zeros((2,), dtype=np.int8),
        action_idx_target=np.zeros((2,), dtype=np.int16),
        actor=np.zeros((2,), dtype=np.int8),
        event_index=np.zeros((2,), dtype=np.int32),
        kyoku=np.zeros((2,), dtype=np.int8),
        honba=np.zeros((2,), dtype=np.int8),
        is_open_hand=np.zeros((2,), dtype=np.uint8),
    )


def test_repair_xmodel1_cache_schema_script_repairs_missing_metadata(tmp_path: Path):
    npz_path = tmp_path / "legacy.npz"
    _write_legacy_npz(npz_path)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/repair_xmodel1_cache_schema.py",
            str(npz_path),
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "repaired=1" in result.stdout
    with np.load(npz_path, allow_pickle=False) as data:
        assert data["schema_name"].item() == "xmodel1_discard_v2"
        assert int(data["schema_version"].item()) == 2
        assert data["state_tile_feat"].shape == (2, 57, 34)


def test_repair_xmodel1_cache_schema_script_check_only_reports_missing(tmp_path: Path):
    npz_path = tmp_path / "legacy.npz"
    _write_legacy_npz(npz_path)

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/repair_xmodel1_cache_schema.py",
            str(npz_path),
            "--check-only",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "missing=1" in result.stdout
    with np.load(npz_path, allow_pickle=False) as data:
        assert "schema_name" not in data.files
