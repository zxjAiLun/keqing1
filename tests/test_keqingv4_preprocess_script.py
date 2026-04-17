from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
from training.cache_schema import KEQINGV4_SUMMARY_DIM


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
        {"type": "tsumo", "actor": 0, "pai": "5s"},
        {"type": "dahai", "actor": 0, "pai": "5s", "tsumogiri": True},
    ]
    with path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def test_preprocess_keqingv4_script_runs_rust_orchestrator(tmp_path: Path):
    input_dir = tmp_path / "converted" / "ds1"
    input_dir.mkdir(parents=True)
    _write_mjson(input_dir / "sample.mjson")
    output_dir = tmp_path / "processed"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/preprocess_keqingv4.py",
            "--config",
            "configs/keqingv4_preprocess.yaml",
            "--data_dirs",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--jobs",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    exported = output_dir / "ds1" / "sample.npz"
    assert exported.exists()
    assert "keqingv4 preprocess launcher -> cargo run --release" in result.stdout
    assert "Rust keqingv4 export completed:" in result.stdout
    with np.load(exported, allow_pickle=True) as data:
        assert "pts_given_win_target" in data.files
        assert "pts_given_dealin_target" in data.files
        assert "opp_tenpai_target" in data.files
        assert "event_history" in data.files
        assert data["v4_discard_summary"].shape[1:] == (34, KEQINGV4_SUMMARY_DIM)
        assert data["v4_call_summary"].shape[1:] == (8, KEQINGV4_SUMMARY_DIM)
        assert data["v4_special_summary"].shape[1:] == (3, KEQINGV4_SUMMARY_DIM)
        assert data["opp_tenpai_target"].shape[1:] == (3,)
        assert data["event_history"].shape[1:] == (48, 5)
    manifest = json.loads((output_dir / "keqingv4_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["export_mode"] == "rust_semantic_core"
    assert manifest["used_python_semantics"] is False
    assert manifest["schema_version"] == 5
    assert manifest["processed_file_count"] == 1
    assert manifest["skipped_existing_file_count"] == 0


def test_rust_keqingv4_export_cli_runs_directly(tmp_path: Path):
    input_dir = tmp_path / "converted" / "ds1"
    input_dir.mkdir(parents=True)
    _write_mjson(input_dir / "sample.mjson")
    output_dir = tmp_path / "processed"
    result = subprocess.run(
        [
            "cargo",
            "run",
            "--quiet",
            "--manifest-path",
            str(Path(REPO_ROOT) / "rust/keqing_core/Cargo.toml"),
            "--bin",
            "keqingv4_export",
            "--",
            "--data-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--jobs",
            "1",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert (output_dir / "ds1" / "sample.npz").exists()
    assert "Rust keqingv4 export completed:" in result.stdout
