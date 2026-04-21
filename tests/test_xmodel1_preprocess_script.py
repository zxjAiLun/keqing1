from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import signal
import time

import numpy as np
from training.cache_schema import XMODEL1_SCHEMA_NAME, XMODEL1_SCHEMA_VERSION


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
    assert "xmodel1 preprocess preflight selection:" in result.stdout
    assert "xmodel1 preprocess preflight gate:" in result.stdout
    assert "xmodel1 preprocess launcher -> native-v3-export" in result.stdout
    assert "xmodel1 preprocess complete:" in result.stdout
    with np.load(exported, allow_pickle=False) as data:
        assert np.any(data["state_tile_feat"] != 0)
        assert np.any(data["state_scalar"] != 0)
        assert data["history_summary"].shape[1] == 20
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


def test_preprocess_xmodel1_script_can_skip_preflight(tmp_path: Path):
    input_dir = tmp_path / "converted" / "ds1"
    input_dir.mkdir(parents=True)
    _write_mjson(input_dir / "sample.mjson")
    output_dir = tmp_path / "processed"

    result = _run_preprocess_script(input_dir, output_dir, "--skip-preflight")

    assert "xmodel1 preprocess preflight selection:" not in result.stdout
    assert "xmodel1 preprocess preflight gate:" not in result.stdout
    assert "xmodel1 preprocess complete:" in result.stdout


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


def test_preprocess_xmodel1_script_rebuilds_existing_output_missing_schema_metadata(tmp_path: Path):
    input_dir = tmp_path / "converted" / "ds1"
    input_dir.mkdir(parents=True)
    _write_mjson(input_dir / "sample.mjson")
    output_dir = tmp_path / "processed"
    stale_output = output_dir / "ds1" / "sample.npz"
    stale_output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        stale_output,
        state_tile_feat=np.zeros((1, 57, 34), dtype=np.float16),
    )

    result = _run_preprocess_script(input_dir, output_dir, "--progress-every", "1")

    manifest = json.loads((output_dir / "xmodel1_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["processed_file_count"] == 1
    assert manifest["skipped_existing_file_count"] == 0
    assert "xmodel1 preprocess complete:" in result.stdout
    with np.load(stale_output, allow_pickle=False) as data:
        assert data["schema_name"].item() == XMODEL1_SCHEMA_NAME
        assert int(data["schema_version"].item()) == XMODEL1_SCHEMA_VERSION
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


def test_preprocess_xmodel1_script_limit_files_does_not_require_all_requested_shards(tmp_path: Path):
    input_root = tmp_path / "converted"
    ds1 = input_root / "ds1"
    ds2 = input_root / "ds2"
    ds1.mkdir(parents=True)
    ds2.mkdir(parents=True)
    _write_mjson(ds1 / "sample_a.mjson")
    _write_mjson(ds2 / "sample_b.mjson")
    output_dir = tmp_path / "processed"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/preprocess_xmodel1.py",
            "--config",
            "configs/xmodel1_preprocess.yaml",
            "--data_dirs",
            str(ds1),
            str(ds2),
            "--output_dir",
            str(output_dir),
            "--skip-preflight",
            "--limit-files",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads((output_dir / "xmodel1_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["processed_file_count"] == 1
    assert manifest["exported_file_count"] == 1
    assert "xmodel1 preprocess complete:" in result.stdout


def test_preprocess_xmodel1_script_reads_limit_files_from_config(tmp_path: Path):
    input_dir = tmp_path / "converted" / "ds3"
    input_dir.mkdir(parents=True)
    _write_mjson(input_dir / "sample_a.mjson")
    _write_mjson(input_dir / "sample_b.mjson")
    output_dir = tmp_path / "processed"
    cfg_path = tmp_path / "preprocess_probe.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "data_dirs:",
                f"  - {input_dir}",
                f"output_dir: {output_dir}",
                "limit_files: 1",
                "jobs: 1",
                "progress_every: 1",
                "preflight_files_per_shard: 0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/preprocess_xmodel1.py",
            "--config",
            str(cfg_path),
            "--skip-preflight",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    manifest = json.loads((output_dir / "xmodel1_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["processed_file_count"] == 1
    assert manifest["exported_file_count"] == 1
    assert "limit_files=1" in result.stdout


def test_preprocess_xmodel1_script_handles_sigint_and_resume(tmp_path: Path):
    input_dir = tmp_path / "converted" / "ds1"
    input_dir.mkdir(parents=True)
    for idx in range(200):
        _write_mjson(input_dir / f"sample_{idx:03d}.mjson")
    output_dir = tmp_path / "processed"
    env = dict(os.environ)
    env["XMODEL1_EXPORT_PROFILE"] = "1"
    env["XMODEL1_EXPORT_TEST_SLEEP_MS"] = "50"
    proc = subprocess.Popen(
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
            "--jobs",
            "1",
            "--progress-every",
            "1",
            "--skip-preflight",
        ],
        cwd=REPO_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        deadline = time.time() + 20.0
        while time.time() < deadline:
            if (output_dir / "ds1").exists():
                break
            if proc.poll() is not None:
                break
            time.sleep(0.1)
        assert proc.poll() is None, proc.communicate(timeout=5)[0]
        proc.send_signal(signal.SIGINT)
        stdout, stderr = proc.communicate(timeout=30)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()

    assert proc.returncode == 130
    combined = stdout + stderr
    assert "interrupted" in combined.lower()
    assert "resume" in combined.lower()
    manifest_path = output_dir / "xmodel1_export_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_name"] == XMODEL1_SCHEMA_NAME
    partial_files = sorted((output_dir / "ds1").glob("*.npz"))
    assert partial_files
    assert not list((output_dir / "ds1").glob("*.tmp"))

    resumed = _run_preprocess_script(input_dir, output_dir, "--skip-preflight", "--jobs", "1", "--progress-every", "1")
    manifest2 = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest2["exported_file_count"] == 200
    assert manifest2["processed_file_count"] + manifest2["skipped_existing_file_count"] == 200
    assert "xmodel1 preprocess complete:" in resumed.stdout
