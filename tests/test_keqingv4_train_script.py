from __future__ import annotations

from pathlib import Path
import subprocess

import numpy as np

from mahjong_env.action_space import ACTION_SPACE
from training.state_features import C_TILE, N_SCALAR
from training.cache_schema import (
    KEQINGV4_CALL_SUMMARY_SLOTS,
    KEQINGV4_EVENT_HISTORY_DIM,
    KEQINGV4_EVENT_HISTORY_LEN,
    KEQINGV4_SPECIAL_SUMMARY_SLOTS,
    KEQINGV4_SUMMARY_DIM,
)


def _write_sample_npz(path: Path) -> None:
    n = 8
    mask = np.zeros((n, ACTION_SPACE), dtype=np.uint8)
    mask[:, :5] = 1
    action_idx = np.arange(n, dtype=np.int16) % 5
    np.savez(
        path,
        tile_feat=np.random.randn(n, C_TILE, 34).astype(np.float16),
        scalar=np.random.randn(n, N_SCALAR).astype(np.float16),
        mask=mask,
        action_idx=action_idx,
        value=np.zeros((n,), dtype=np.float32),
        score_delta_target=np.zeros((n,), dtype=np.float32),
        win_target=np.zeros((n,), dtype=np.float32),
        dealin_target=np.zeros((n,), dtype=np.float32),
    )


def _write_keqingv4_contract_npz(path: Path) -> None:
    n = 8
    mask = np.zeros((n, ACTION_SPACE), dtype=np.uint8)
    mask[:, :5] = 1
    action_idx = np.arange(n, dtype=np.int16) % 5
    np.savez(
        path,
        tile_feat=np.random.randn(n, C_TILE, 34).astype(np.float16),
        scalar=np.random.randn(n, N_SCALAR).astype(np.float16),
        mask=mask,
        action_idx=action_idx,
        value=np.zeros((n,), dtype=np.float32),
        score_delta_target=np.zeros((n,), dtype=np.float32),
        win_target=np.zeros((n,), dtype=np.float32),
        dealin_target=np.zeros((n,), dtype=np.float32),
        pts_given_win_target=np.zeros((n,), dtype=np.float32),
        pts_given_dealin_target=np.zeros((n,), dtype=np.float32),
        opp_tenpai_target=np.zeros((n, 3), dtype=np.float32),
        event_history=np.zeros((n, KEQINGV4_EVENT_HISTORY_LEN, KEQINGV4_EVENT_HISTORY_DIM), dtype=np.int16),
        v4_discard_summary=np.zeros((n, 34, KEQINGV4_SUMMARY_DIM), dtype=np.float16),
        v4_call_summary=np.zeros((n, KEQINGV4_CALL_SUMMARY_SLOTS, KEQINGV4_SUMMARY_DIM), dtype=np.float16),
        v4_special_summary=np.zeros((n, KEQINGV4_SPECIAL_SUMMARY_SLOTS, KEQINGV4_SUMMARY_DIM), dtype=np.float16),
    )


def test_train_keqingv4_script_accepts_cli_data_dir_and_output_dir_overrides(tmp_path: Path):
    data_root = tmp_path / "processed_v4"
    ds1 = data_root / "ds1"
    ds2 = data_root / "ds2"
    ds1.mkdir(parents=True)
    ds2.mkdir(parents=True)
    _write_sample_npz(ds1 / "a.npz")
    _write_sample_npz(ds2 / "b.npz")
    output_dir = tmp_path / "model_out"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/train_keqingv4.py",
            "--config",
            "configs/keqingv4_default.yaml",
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
    assert "data_dirs=" in result.stdout
    assert "keqingv4 cache-inspect:" in result.stdout
    assert str(ds1) in result.stdout
    assert str(ds2) in result.stdout


def test_train_keqingv4_script_rejects_stale_cache_without_smoke(tmp_path: Path):
    data_root = tmp_path / "processed_v4"
    ds1 = data_root / "ds1"
    ds2 = data_root / "ds2"
    ds1.mkdir(parents=True)
    ds2.mkdir(parents=True)
    _write_sample_npz(ds1 / "a.npz")
    _write_sample_npz(ds2 / "b.npz")
    output_dir = tmp_path / "model_out"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/train_keqingv4.py",
            "--config",
            "configs/keqingv4_default.yaml",
            "--data_dirs",
            str(ds1),
            str(ds2),
            "--output_dir",
            str(output_dir),
            "--device",
            "cpu",
        ],
        cwd="/media/bailan/DISK1/AUbuntuProject/project/keqing1",
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "cache contract check failed" in combined
    assert "rerun preprocess_keqingv4" in combined


def test_train_keqingv4_script_rejects_manifest_contract_mismatch(tmp_path: Path):
    data_root = tmp_path / "processed_v4"
    ds1 = data_root / "ds1"
    ds2 = data_root / "ds2"
    ds1.mkdir(parents=True)
    ds2.mkdir(parents=True)
    _write_keqingv4_contract_npz(ds1 / "a.npz")
    _write_keqingv4_contract_npz(ds2 / "b.npz")
    (data_root / "keqingv4_export_manifest.json").write_text(
        """{
  "schema_name": "keqingv4_cached_v1",
  "schema_version": 5,
  "summary_dim": 28,
  "call_summary_slots": 8,
  "special_summary_slots": 3,
  "export_mode": "python_fallback"
}""",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/train_keqingv4.py",
            "--config",
            "configs/keqingv4_default.yaml",
            "--data_dirs",
            str(ds1),
            str(ds2),
            "--output_dir",
            str(tmp_path / "model_out"),
            "--device",
            "cpu",
        ],
        cwd="/media/bailan/DISK1/AUbuntuProject/project/keqing1",
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "cache contract check failed" in combined
    assert "export_mode mismatch" in combined
