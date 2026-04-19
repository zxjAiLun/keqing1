from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np


REPO_ROOT = "/media/bailan/DISK1/AUbuntuProject/project/keqing1"


def _write_sample_npz(path: Path, n: int) -> None:
    candidate_mask = np.concatenate(
        [np.ones((n, 3), dtype=np.uint8), np.zeros((n, 11), dtype=np.uint8)],
        axis=1,
    )
    np.savez(
        path,
        schema_name=np.array("xmodel1_discard_v2", dtype=np.str_),
        schema_version=np.array(2, dtype=np.int32),
        state_tile_feat=np.zeros((n, 57, 34), dtype=np.float16),
        state_scalar=np.zeros((n, 56), dtype=np.float16),
        candidate_feat=np.zeros((n, 14, 35), dtype=np.float16),
        candidate_tile_id=np.full((n, 14), -1, dtype=np.int16),
        candidate_mask=candidate_mask,
        candidate_flags=np.zeros((n, 14, 10), dtype=np.uint8),
        chosen_candidate_idx=np.zeros((n,), dtype=np.int16),
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
        sample_type=np.zeros((n,), dtype=np.int8),
        actor=np.zeros((n,), dtype=np.int8),
        event_index=np.zeros((n,), dtype=np.int32),
        kyoku=np.ones((n,), dtype=np.int8),
        honba=np.zeros((n,), dtype=np.int8),
        is_open_hand=np.zeros((n,), dtype=np.uint8),
    )


def test_pack_xmodel1_shards_merges_small_npz_files(tmp_path: Path):
    input_root = tmp_path / "processed_xmodel1"
    (input_root / "ds1").mkdir(parents=True)
    _write_sample_npz(input_root / "ds1" / "a.npz", 3)
    _write_sample_npz(input_root / "ds1" / "b.npz", 4)
    _write_sample_npz(input_root / "ds1" / "c.npz", 5)
    output_root = tmp_path / "packed"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/pack_xmodel1_shards.py",
            "--input_dir",
            str(input_root),
            "--output_dir",
            str(output_root),
            "--max-samples-per-shard",
            "7",
            "--max-files-per-shard",
            "2",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "xmodel1 shard pack:" in result.stdout
    shard_files = sorted((output_root / "ds1").glob("*.npz"))
    assert len(shard_files) == 2
    with np.load(shard_files[0], allow_pickle=False) as data0:
        assert int(data0["state_tile_feat"].shape[0]) == 7
    with np.load(shard_files[1], allow_pickle=False) as data1:
        assert int(data1["state_tile_feat"].shape[0]) == 5
    manifest = json.loads((output_root / "xmodel1_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["export_mode"] == "xmodel1_packed_shards"
    assert manifest["exported_file_count"] == 2
    assert manifest["exported_sample_count"] == 12
    assert manifest["shard_file_counts"] == {"ds1": 2}
    assert manifest["shard_sample_counts"] == {"ds1": 12}
