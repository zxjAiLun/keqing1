from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np


def _write_sample_npz(path: Path) -> None:
    n = 8
    candidate_mask = np.concatenate(
        [np.ones((n, 3), dtype=np.uint8), np.zeros((n, 11), dtype=np.uint8)],
        axis=1,
    )
    candidate_tile_id = np.full((n, 14), -1, dtype=np.int16)
    candidate_tile_id[:, 0] = 1
    candidate_tile_id[:, 1] = 2
    candidate_tile_id[:, 2] = 3
    np.savez(
        path,
        state_tile_feat=np.random.randn(n, 57, 34).astype(np.float16),
        state_scalar=np.random.randn(n, 56).astype(np.float16),
        candidate_feat=np.random.randn(n, 14, 35).astype(np.float16),
        candidate_tile_id=candidate_tile_id,
        candidate_mask=candidate_mask,
        candidate_flags=np.zeros((n, 14, 10), dtype=np.uint8),
        chosen_candidate_idx=np.zeros((n,), dtype=np.int16),
        action_idx_target=np.ones((n,), dtype=np.int16),
        candidate_quality_score=np.zeros((n, 14), dtype=np.float32),
        candidate_rank_bucket=np.zeros((n, 14), dtype=np.int8),
        candidate_hard_bad_flag=np.zeros((n, 14), dtype=np.uint8),
        special_candidate_feat=np.zeros((n, 8, 25), dtype=np.float16),
        special_candidate_type_id=np.full((n, 8), -1, dtype=np.int16),
        special_candidate_mask=np.zeros((n, 8), dtype=np.uint8),
        special_candidate_quality_score=np.zeros((n, 8), dtype=np.float32),
        special_candidate_rank_bucket=np.zeros((n, 8), dtype=np.int8),
        special_candidate_hard_bad_flag=np.zeros((n, 8), dtype=np.uint8),
        chosen_special_candidate_idx=np.full((n,), -1, dtype=np.int16),
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


def _write_manifest(root: Path) -> None:
    (root / "xmodel1_export_manifest.json").write_text(
        json.dumps(
            {
                "schema_name": "xmodel1_discard_v1",
                "schema_version": 1,
                "max_candidates": 14,
                "candidate_feature_dim": 35,
                "candidate_flag_dim": 10,
                "file_count": 2,
                "exported_file_count": 2,
                "exported_sample_count": 16,
                "processed_file_count": 2,
                "skipped_existing_file_count": 0,
                "shard_file_counts": {"ds1": 1, "ds2": 1},
                "shard_sample_counts": {"ds1": 8, "ds2": 8},
                "used_fallback": False,
                "export_mode": "rust_full_npz_export",
                "files": ["ds1/a.mjson", "ds2/b.mjson"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def test_train_xmodel1_script_accepts_cli_data_dir_and_output_dir_overrides(tmp_path: Path):
    data_root = tmp_path / "processed_xmodel1"
    ds1 = data_root / "ds1"
    ds2 = data_root / "ds2"
    ds1.mkdir(parents=True)
    ds2.mkdir(parents=True)
    _write_sample_npz(ds1 / "a.npz")
    _write_sample_npz(ds2 / "b.npz")
    _write_manifest(data_root)
    output_dir = tmp_path / "model_out"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/train_xmodel1.py",
            "--config",
            "configs/xmodel1_default.yaml",
            "--data_dirs",
            str(ds1),
            str(ds2),
            "--output_dir",
            str(output_dir),
            "--device",
            "cpu",
            "--smoke",
        ],
        cwd="/media/bailan/DISK1/AUbuntuProject/project/keqing1",
        check=True,
        capture_output=True,
        text=True,
    )
    smoke_output = Path(str(output_dir) + "_smoke")
    assert smoke_output.exists()
    assert (smoke_output / "last.pth").exists()
    train_inputs = json.loads((smoke_output / "train_inputs.json").read_text(encoding="utf-8"))
    training_summary = json.loads((smoke_output / "training_summary.json").read_text(encoding="utf-8"))
    assert train_inputs["manifest_paths"]
    assert str(ds1) in train_inputs["data_dirs"]
    assert train_inputs["dataset_summary"]["selected"]["num_files"] == 2
    assert train_inputs["dataset_summary"]["selected"]["num_samples"] == 16
    assert training_summary["model_version"] == "xmodel1"
    assert training_summary["dataset_summary"]["selected"]["num_files"] == 2
    assert "data_dirs=" in result.stdout
    assert str(ds1) in result.stdout
    assert str(ds2) in result.stdout
    assert "manifests=[" in result.stdout
    assert "steps_per_epoch=" in result.stdout
    assert "files_per_epoch_ratio=" in result.stdout
    assert "files_per_epoch_count=" in result.stdout
    assert "[Epoch 1/1]" in result.stdout
    assert "[epoch-plan] sampled_files=" in result.stdout
    assert "[epoch-summary] train_batches=" in result.stdout
    assert "best_updated=" in result.stdout
    assert "global_step=0/" in result.stdout
    assert "[train] batch=1/" in result.stdout
    assert "[val] batch=1/" in result.stdout


def test_train_xmodel1_script_accepts_processed_root_and_resume(tmp_path: Path):
    data_root = tmp_path / "processed_xmodel1"
    ds1 = data_root / "ds1"
    ds2 = data_root / "ds2"
    ds1.mkdir(parents=True)
    ds2.mkdir(parents=True)
    _write_sample_npz(ds1 / "a.npz")
    _write_sample_npz(ds2 / "b.npz")
    _write_manifest(data_root)
    output_dir = tmp_path / "model_out"

    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/train_xmodel1.py",
            "--config",
            "configs/xmodel1_default.yaml",
            "--data_dirs",
            str(data_root),
            "--output_dir",
            str(output_dir),
            "--device",
            "cpu",
            "--smoke",
        ],
        cwd="/media/bailan/DISK1/AUbuntuProject/project/keqing1",
        check=True,
        capture_output=True,
        text=True,
    )
    smoke_output = Path(str(output_dir) + "_smoke")
    resume_path = smoke_output / "last.pth"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/train_xmodel1.py",
            "--config",
            "configs/xmodel1_default.yaml",
            "--data_dirs",
            str(data_root),
            "--output_dir",
            str(output_dir),
            "--device",
            "cpu",
            "--smoke",
            "--resume",
            str(resume_path),
        ],
        cwd="/media/bailan/DISK1/AUbuntuProject/project/keqing1",
        check=True,
        capture_output=True,
        text=True,
    )

    assert resume_path.exists()
    assert "resume=" in result.stdout
    assert str(data_root) in result.stdout
    assert "resumed checkpoint=" in result.stdout or "already satisfies num_epochs=1" in result.stdout
    train_inputs = json.loads((smoke_output / "train_inputs.json").read_text(encoding="utf-8"))
    training_summary = json.loads((smoke_output / "training_summary.json").read_text(encoding="utf-8"))
    assert train_inputs["resume_path"] == str(resume_path)
    assert train_inputs["dataset_summary"]["train"]["num_samples"] >= 8
    assert training_summary["resume_path"] == str(resume_path)
    assert training_summary["dataset_summary"]["val"]["num_files"] >= 1


def test_train_xmodel1_script_auto_resumes_from_output_last_checkpoint(tmp_path: Path):
    data_root = tmp_path / "processed_xmodel1"
    ds1 = data_root / "ds1"
    ds2 = data_root / "ds2"
    ds1.mkdir(parents=True)
    ds2.mkdir(parents=True)
    _write_sample_npz(ds1 / "a.npz")
    _write_sample_npz(ds2 / "b.npz")
    _write_manifest(data_root)
    output_dir = tmp_path / "model_out"

    subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/train_xmodel1.py",
            "--config",
            "configs/xmodel1_default.yaml",
            "--data_dirs",
            str(data_root),
            "--output_dir",
            str(output_dir),
            "--device",
            "cpu",
            "--smoke",
        ],
        cwd="/media/bailan/DISK1/AUbuntuProject/project/keqing1",
        check=True,
        capture_output=True,
        text=True,
    )

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/train_xmodel1.py",
            "--config",
            "configs/xmodel1_default.yaml",
            "--data_dirs",
            str(data_root),
            "--output_dir",
            str(output_dir),
            "--device",
            "cpu",
            "--smoke",
        ],
        cwd="/media/bailan/DISK1/AUbuntuProject/project/keqing1",
        check=True,
        capture_output=True,
        text=True,
    )

    smoke_output = Path(str(output_dir) + "_smoke")
    resume_path = smoke_output / "last.pth"
    train_inputs = json.loads((smoke_output / "train_inputs.json").read_text(encoding="utf-8"))
    assert resume_path.exists()
    assert train_inputs["resume_path"] == str(resume_path)
    assert "resume=" in result.stdout
    assert str(resume_path) in result.stdout
    assert "resumed checkpoint=" in result.stdout or "already satisfies num_epochs=1" in result.stdout


def test_train_xmodel1_script_rejects_manifest_cache_stat_mismatch(tmp_path: Path):
    data_root = tmp_path / "processed_xmodel1"
    ds1 = data_root / "ds1"
    ds1.mkdir(parents=True)
    _write_sample_npz(ds1 / "a.npz")
    (data_root / "xmodel1_export_manifest.json").write_text(
        json.dumps(
            {
                "schema_name": "xmodel1_discard_v1",
                "schema_version": 1,
                "max_candidates": 14,
                "candidate_feature_dim": 35,
                "candidate_flag_dim": 10,
                "file_count": 1,
                "exported_file_count": 1,
                "exported_sample_count": 99,
                "processed_file_count": 1,
                "skipped_existing_file_count": 0,
                "shard_file_counts": {"ds1": 1},
                "shard_sample_counts": {"ds1": 99},
                "used_fallback": False,
                "export_mode": "rust_full_npz_export",
                "files": ["ds1/a.mjson"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/train_xmodel1.py",
            "--config",
            "configs/xmodel1_default.yaml",
            "--data_dirs",
            str(data_root),
            "--output_dir",
            str(tmp_path / "model_out"),
            "--device",
            "cpu",
            "--smoke",
        ],
        cwd="/media/bailan/DISK1/AUbuntuProject/project/keqing1",
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "manifest/cache mismatch" in result.stderr or "manifest/cache mismatch" in result.stdout
