from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np


REPO_ROOT = "/media/bailan/DISK1/AUbuntuProject/project/keqing1"


def _write_mjson(path: Path) -> None:
    events = [
        {"type": "start_game", "names": ["A", "B", "C", "D"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
                ["1s"] * 13,
                ["2s"] * 13,
                ["3s"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "4p"},
        {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
    ]
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _run_preprocess_script(input_dir: Path, output_dir: Path, *extra_args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/preprocess_xmodel1.py",
            "--config",
            "configs/xmodel1_preprocess.yaml",
            "--data_dirs",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            *extra_args,
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_preprocess_xmodel1_script_runs_native_v2_export(tmp_path: Path):
    input_dir = tmp_path / "converted" / "ds1"
    input_dir.mkdir(parents=True)
    _write_mjson(input_dir / "sample.mjson")
    output_dir = tmp_path / "processed"
    result = _run_preprocess_script(input_dir, output_dir)

    exported = output_dir / "ds1" / "sample.npz"
    assert exported.exists()
    assert "xmodel1 preprocess launcher -> native-v2-export" in result.stdout
    assert "xmodel1 preprocess complete:" in result.stdout
    with np.load(exported, allow_pickle=False) as data:
        assert np.any(data["state_tile_feat"] != 0)
        assert np.any(data["state_scalar"] != 0)
    manifest = json.loads((output_dir / "xmodel1_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["export_mode"] == "rust_full_npz_export"
    assert manifest["used_fallback"] is False
    assert manifest["processed_file_count"] == 1
    assert manifest["skipped_existing_file_count"] == 0


def test_preprocess_xmodel1_script_resumes_and_skips_existing_outputs(tmp_path: Path):
    input_dir = tmp_path / "converted" / "ds1"
    input_dir.mkdir(parents=True)
    _write_mjson(input_dir / "sample.mjson")
    output_dir = tmp_path / "processed"

    _run_preprocess_script(input_dir, output_dir)
    second = _run_preprocess_script(input_dir, output_dir)

    manifest = json.loads((output_dir / "xmodel1_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["processed_file_count"] == 0
    assert manifest["skipped_existing_file_count"] == 1
    assert manifest["exported_file_count"] == 1
    assert manifest["exported_sample_count"] >= 1
    assert "xmodel1 preprocess complete:" in second.stdout


def test_preprocess_xmodel1_script_rebuilds_corrupt_existing_output(tmp_path: Path):
    input_dir = tmp_path / "converted" / "ds1"
    input_dir.mkdir(parents=True)
    _write_mjson(input_dir / "sample.mjson")
    output_dir = tmp_path / "processed"
    corrupt_output = output_dir / "ds1" / "sample.npz"
    corrupt_output.parent.mkdir(parents=True, exist_ok=True)
    corrupt_output.write_bytes(b"not-a-valid-zip")

    result = _run_preprocess_script(input_dir, output_dir, "--progress-every", "1")

    manifest = json.loads((output_dir / "xmodel1_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["processed_file_count"] == 1
    assert manifest["skipped_existing_file_count"] == 0
    assert "xmodel1 preprocess complete:" in result.stdout
    with np.load(corrupt_output, allow_pickle=False) as data:
        assert data["state_tile_feat"].shape[0] >= 1


def test_preprocess_xmodel1_script_uses_native_worker_count_for_full_export(tmp_path: Path):
    input_dir = tmp_path / "converted" / "ds1"
    input_dir.mkdir(parents=True)
    _write_mjson(input_dir / "sample_a.mjson")
    _write_mjson(input_dir / "sample_b.mjson")
    output_dir = tmp_path / "processed"

    result = _run_preprocess_script(
        input_dir,
        output_dir,
        "--jobs",
        "2",
        "--progress-every",
        "1",
    )

    manifest = json.loads((output_dir / "xmodel1_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["processed_file_count"] == 2
    assert manifest["exported_file_count"] == 2
    assert manifest["export_mode"] == "rust_full_npz_export"
    assert "workers=2" in result.stdout
