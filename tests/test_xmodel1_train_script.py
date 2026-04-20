from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

pytest.importorskip("torch")

from tests.xmodel1_test_utils import write_xmodel1_v3_manifest, write_xmodel1_v3_npz

REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_sample_npz(path: Path) -> None:
    payload = write_xmodel1_v3_npz(path, n=8, state_scalar_dim=56)
    payload["state_tile_feat"][:] = np.random.randn(8, 57, 34).astype(np.float16)
    payload["state_scalar"][:] = np.random.randn(8, 56).astype(np.float16)
    payload["action_idx_target"][:] = 1
    np.savez(path, **payload)


def _run_train(tmp_path: Path, *extra_args: str, check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_xmodel1.py"),
            "--config",
            "configs/xmodel1_default.yaml",
            *extra_args,
        ],
        cwd=str(REPO_ROOT),
        check=check,
        capture_output=True,
        text=True,
    )


def test_train_xmodel1_script_accepts_cli_data_dir_and_output_dir_overrides(tmp_path: Path):
    data_root = tmp_path / "processed_xmodel1"
    ds1 = data_root / "ds1"
    ds2 = data_root / "ds2"
    ds1.mkdir(parents=True)
    ds2.mkdir(parents=True)
    _write_sample_npz(ds1 / "a.npz")
    _write_sample_npz(ds2 / "b.npz")
    write_xmodel1_v3_manifest(data_root, shard_file_counts={"ds1": 1, "ds2": 1}, shard_sample_counts={"ds1": 8, "ds2": 8})
    output_dir = tmp_path / "model_out"

    result = _run_train(
        tmp_path,
        "--data_dirs",
        str(ds1),
        str(ds2),
        "--output_dir",
        str(output_dir),
        "--device",
        "cpu",
        "--smoke",
        check=True,
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
    assert train_inputs["dataset_summary"]["selected"]["sample_count_source"] == "manifest_exact"
    assert not train_inputs["dataset_summary"]["selected"]["is_sample_count_estimated"]
    assert training_summary["model_version"] == "xmodel1"
    assert training_summary["dataset_summary"]["selected"]["num_files"] == 2
    assert "data_dirs=" in result.stdout
    assert str(ds1) in result.stdout
    assert str(ds2) in result.stdout
    assert "manifests=[" in result.stdout
    assert "strict_cache_scan=False" in result.stdout
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


def test_train_xmodel1_script_estimates_subset_counts_from_manifest(tmp_path: Path):
    data_root = tmp_path / "processed_xmodel1"
    ds1 = data_root / "ds1"
    ds1.mkdir(parents=True)
    for name in ["a.npz", "b.npz", "c.npz", "d.npz"]:
        _write_sample_npz(ds1 / name)
    write_xmodel1_v3_manifest(data_root, shard_file_counts={"ds1": 4}, shard_sample_counts={"ds1": 32})
    output_dir = tmp_path / "model_out"

    _run_train(
        tmp_path,
        "--data_dirs",
        str(data_root),
        "--output_dir",
        str(output_dir),
        "--device",
        "cpu",
        "--smoke",
        check=True,
    )

    smoke_output = Path(str(output_dir) + "_smoke")
    train_inputs = json.loads((smoke_output / "train_inputs.json").read_text(encoding="utf-8"))
    selected = train_inputs["dataset_summary"]["selected"]
    train_summary = train_inputs["dataset_summary"]["train"]
    val_summary = train_inputs["dataset_summary"]["val"]
    assert selected["num_files"] == 3
    assert selected["num_samples"] == 24
    assert selected["sample_count_source"] == "manifest_estimate"
    assert selected["is_sample_count_estimated"] is True
    assert train_summary["num_files"] == 2
    assert train_summary["num_samples"] == 16
    assert train_summary["sample_count_source"] == "manifest_estimate"
    assert val_summary["num_files"] == 1
    assert val_summary["num_samples"] == 8
    assert val_summary["sample_count_source"] == "manifest_estimate"


def test_train_xmodel1_script_accepts_processed_root_and_resume(tmp_path: Path):
    data_root = tmp_path / "processed_xmodel1"
    ds1 = data_root / "ds1"
    ds2 = data_root / "ds2"
    ds1.mkdir(parents=True)
    ds2.mkdir(parents=True)
    _write_sample_npz(ds1 / "a.npz")
    _write_sample_npz(ds2 / "b.npz")
    write_xmodel1_v3_manifest(data_root, shard_file_counts={"ds1": 1, "ds2": 1}, shard_sample_counts={"ds1": 8, "ds2": 8})
    output_dir = tmp_path / "model_out"

    _run_train(
        tmp_path,
        "--data_dirs",
        str(data_root),
        "--output_dir",
        str(output_dir),
        "--device",
        "cpu",
        "--smoke",
        check=True,
    )
    smoke_output = Path(str(output_dir) + "_smoke")
    resume_path = smoke_output / "last.pth"
    result = _run_train(
        tmp_path,
        "--data_dirs",
        str(data_root),
        "--output_dir",
        str(output_dir),
        "--device",
        "cpu",
        "--smoke",
        "--resume",
        str(resume_path),
        check=True,
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
    write_xmodel1_v3_manifest(data_root, shard_file_counts={"ds1": 1, "ds2": 1}, shard_sample_counts={"ds1": 8, "ds2": 8})
    output_dir = tmp_path / "model_out"

    _run_train(
        tmp_path,
        "--data_dirs",
        str(data_root),
        "--output_dir",
        str(output_dir),
        "--device",
        "cpu",
        "--smoke",
        check=True,
    )

    result = _run_train(
        tmp_path,
        "--data_dirs",
        str(data_root),
        "--output_dir",
        str(output_dir),
        "--device",
        "cpu",
        "--smoke",
        check=True,
    )

    smoke_output = Path(str(output_dir) + "_smoke")
    resume_path = smoke_output / "last.pth"
    train_inputs = json.loads((smoke_output / "train_inputs.json").read_text(encoding="utf-8"))
    assert resume_path.exists()
    assert train_inputs["resume_path"] == str(resume_path)
    assert "resume=" in result.stdout
    assert str(resume_path) in result.stdout
    assert "resumed checkpoint=" in result.stdout or "already satisfies num_epochs=1" in result.stdout


def test_train_xmodel1_script_skips_strict_manifest_validation_by_default(tmp_path: Path):
    data_root = tmp_path / "processed_xmodel1"
    ds1 = data_root / "ds1"
    ds1.mkdir(parents=True)
    _write_sample_npz(ds1 / "a.npz")
    write_xmodel1_v3_manifest(data_root, shard_file_counts={"ds1": 1}, shard_sample_counts={"ds1": 99})

    result = _run_train(
        tmp_path,
        "--data_dirs",
        str(data_root),
        "--output_dir",
        str(tmp_path / "model_out"),
        "--device",
        "cpu",
        "--smoke",
        check=True,
    )

    assert result.returncode == 0
    smoke_output = tmp_path / "model_out_smoke"
    train_inputs = json.loads((smoke_output / "train_inputs.json").read_text(encoding="utf-8"))
    assert train_inputs["dataset_summary"]["selected"]["num_samples"] == 99
    assert train_inputs["dataset_summary"]["selected"]["sample_count_source"] == "manifest_exact"


def test_train_xmodel1_script_rejects_manifest_cache_stat_mismatch_in_strict_mode(tmp_path: Path):
    data_root = tmp_path / "processed_xmodel1"
    ds1 = data_root / "ds1"
    ds1.mkdir(parents=True)
    _write_sample_npz(ds1 / "a.npz")
    write_xmodel1_v3_manifest(data_root, shard_file_counts={"ds1": 1}, shard_sample_counts={"ds1": 99})

    result = _run_train(
        tmp_path,
        "--data_dirs",
        str(data_root),
        "--output_dir",
        str(tmp_path / "model_out"),
        "--device",
        "cpu",
        "--smoke",
        "--strict-cache-scan",
        check=False,
    )

    assert result.returncode != 0
    assert "manifest/cache mismatch" in result.stderr or "manifest/cache mismatch" in result.stdout


def test_train_xmodel1_script_auto_derives_steps_per_epoch_from_sampled_files(tmp_path: Path):
    data_root = tmp_path / "processed_xmodel1"
    ds1 = data_root / "ds1"
    ds1.mkdir(parents=True)
    for name in ["a.npz", "b.npz", "c.npz", "d.npz"]:
        _write_sample_npz(ds1 / name)
    write_xmodel1_v3_manifest(data_root, shard_file_counts={"ds1": 4}, shard_sample_counts={"ds1": 32})

    cfg_path = tmp_path / "auto_steps.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data_dirs:",
                f"  - {data_root}",
                f"output_dir: {tmp_path / 'model_out'}",
                "device: cpu",
                "state_tile_channels: 57",
                "state_scalar_dim: 56",
                "candidate_feature_dim: 22",
                "candidate_flag_dim: 8",
                "special_candidate_feature_dim: 19",
                "hidden_dim: 32",
                "num_res_blocks: 1",
                "dropout: 0.1",
                "num_epochs: 1",
                "batch_size: 4",
                "learning_rate: 0.0003",
                "weight_decay: 0.0001",
                "warmup_steps: 4",
                "steps_per_epoch: 0",
                "val_steps_per_epoch: 1",
                "files_per_epoch_ratio: 0.0",
                "files_per_epoch_count: 2",
                "val_ratio: 0.25",
                "num_workers: 0",
                "pin_memory: false",
                "persistent_workers: false",
                "buffer_size: 16",
                "log_interval: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "scripts" / "train_xmodel1.py"),
            "--config",
            str(cfg_path),
            "--device",
            "cpu",
        ],
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
    )

    assert "auto steps_per_epoch=4" in result.stdout
    assert "steps_per_epoch=4" in result.stdout
    training_summary = json.loads(((tmp_path / "model_out") / "training_summary.json").read_text(encoding="utf-8"))
    assert training_summary["cfg"]["steps_per_epoch"] == 4
    rows = [
        json.loads(line)
        for line in ((tmp_path / "model_out") / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["train"]["num_batches"] == 4
