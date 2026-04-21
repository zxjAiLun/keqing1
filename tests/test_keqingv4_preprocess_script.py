from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest
from keqingv4.cache_contract import KEQINGV4_SCHEMA_NAME, KEQINGV4_SCHEMA_VERSION
from training.cache_schema import KEQINGV4_OPPORTUNITY_DIM, KEQINGV4_SUMMARY_DIM


REPO_ROOT = Path(__file__).resolve().parents[1]


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
    if shutil.which("uv") is None:
        pytest.skip("uv is not available in this environment")
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
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    exported = output_dir / "ds1" / "sample.npz"
    assert exported.exists()
    assert "keqingv4 preprocess launcher -> cargo run --release" in result.stdout
    assert "Rust keqingv4 export completed:" in result.stdout
    with np.load(exported, allow_pickle=True) as data:
        assert "pts_given_win_target" in data.files
        assert "pts_given_dealin_target" in data.files
        assert "opp_tenpai_target" in data.files
        assert "final_rank_target" in data.files
        assert "final_score_delta_points_target" in data.files
        assert "event_history" in data.files
        assert "v4_opportunity" in data.files
        assert data["v4_discard_summary"].shape[1:] == (34, KEQINGV4_SUMMARY_DIM)
        assert data["v4_call_summary"].shape[1:] == (8, KEQINGV4_SUMMARY_DIM)
        assert data["v4_special_summary"].shape[1:] == (3, KEQINGV4_SUMMARY_DIM)
        assert data["opp_tenpai_target"].shape[1:] == (3,)
        assert data["event_history"].shape[1:] == (48, 5)
        assert data["v4_opportunity"].shape[1:] == (KEQINGV4_OPPORTUNITY_DIM,)
    manifest = json.loads((output_dir / "keqingv4_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_name"] == KEQINGV4_SCHEMA_NAME
    assert manifest["export_mode"] == "rust_semantic_core"
    assert manifest["used_python_semantics"] is False
    assert manifest["schema_version"] == KEQINGV4_SCHEMA_VERSION
    assert manifest["opportunity_dim"] == 3
    assert manifest["processed_file_count"] == 1
    assert manifest["skipped_existing_file_count"] == 0


def test_rust_keqingv4_export_cli_runs_directly(tmp_path: Path):
    if shutil.which("cargo") is None:
        pytest.skip("cargo is not available in this environment")
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
                str(REPO_ROOT / "rust/keqing_core/Cargo.toml"),
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
        cwd=str(REPO_ROOT),
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    assert (output_dir / "ds1" / "sample.npz").exists()
    assert "Rust keqingv4 export completed:" in result.stdout


def test_preprocess_keqingv4_script_accepts_legacy_workers_config(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "converted" / "ds1"
    input_dir.mkdir(parents=True)
    output_dir = tmp_path / "processed"
    config_path = tmp_path / "keqingv4_preprocess.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data_dirs:",
                f"  - {input_dir.as_posix()}",
                f"output_dir: {output_dir.as_posix()}",
                "workers: 7",
                "progress_every: 11",
            ]
        ),
        encoding="utf-8",
    )

    captured: dict[str, object] = {}

    def _fake_run(cmd, cwd, check):
        captured["cmd"] = list(cmd)
        captured["cwd"] = cwd
        captured["check"] = check
        out_index = cmd.index("--output-dir") + 1
        current_output = Path(cmd[out_index])
        current_output.mkdir(parents=True, exist_ok=True)
        (current_output / "ds1").mkdir(parents=True, exist_ok=True)
        _write_contract_npz(current_output / "ds1" / "sample.npz")
        (current_output / "keqingv4_export_manifest.json").write_text(
            json.dumps(
                {
                    "schema_name": KEQINGV4_SCHEMA_NAME,
                    "schema_version": KEQINGV4_SCHEMA_VERSION,
                    "summary_dim": KEQINGV4_SUMMARY_DIM,
                    "call_summary_slots": 8,
                    "special_summary_slots": 3,
                    "opportunity_dim": 3,
                    "export_mode": "rust_semantic_core",
                    "used_python_semantics": False,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    script_path = REPO_ROOT / "scripts" / "preprocess_keqingv4.py"
    spec = importlib.util.spec_from_file_location("preprocess_keqingv4_testmod", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocess_keqingv4.py",
            "--config",
            str(config_path),
            "--skip-preflight",
        ],
    )

    module.main()

    cmd = captured["cmd"]
    assert "--jobs" in cmd
    assert cmd[cmd.index("--jobs") + 1] == "7"
    assert cmd[cmd.index("--progress-every") + 1] == "11"


def test_preprocess_keqingv4_script_runs_preflight_before_full_export(tmp_path: Path, monkeypatch):
    input_dir = tmp_path / "converted" / "ds1"
    input_dir.mkdir(parents=True)
    _write_mjson(input_dir / "a.mjson")
    _write_mjson(input_dir / "b.mjson")
    output_dir = tmp_path / "processed"
    config_path = tmp_path / "keqingv4_preprocess.yaml"
    config_path.write_text(
        "\n".join(
            [
                "data_dirs:",
                f"  - {input_dir.as_posix()}",
                f"output_dir: {output_dir.as_posix()}",
                "jobs: 3",
                "progress_every: 9",
                "preflight_files_per_shard: 1",
                "preflight_seed: 20260419",
            ]
        ),
        encoding="utf-8",
    )

    calls: list[list[str]] = []

    def _fake_run(cmd, cwd, check):
        calls.append(list(cmd))
        out_index = cmd.index("--output-dir") + 1
        current_output = Path(cmd[out_index])
        current_output.mkdir(parents=True, exist_ok=True)
        (current_output / "ds1").mkdir(parents=True, exist_ok=True)
        _write_contract_npz(current_output / "ds1" / "sample.npz")
        (current_output / "keqingv4_export_manifest.json").write_text(
            json.dumps(
                {
                    "schema_name": KEQINGV4_SCHEMA_NAME,
                    "schema_version": KEQINGV4_SCHEMA_VERSION,
                    "summary_dim": KEQINGV4_SUMMARY_DIM,
                    "call_summary_slots": 8,
                    "special_summary_slots": 3,
                    "opportunity_dim": 3,
                    "export_mode": "rust_semantic_core",
                    "used_python_semantics": False,
                }
            ),
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    script_path = REPO_ROOT / "scripts" / "preprocess_keqingv4.py"
    spec = importlib.util.spec_from_file_location("preprocess_keqingv4_preflight_testmod", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.subprocess, "run", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "preprocess_keqingv4.py",
            "--config",
            str(config_path),
        ],
    )

    module.main()

    assert len(calls) == 2
    preflight_cmd, full_cmd = calls
    assert "--force" in preflight_cmd
    assert "--force" not in full_cmd


def _write_contract_npz(path: Path) -> None:
    n = 1
    np.savez(
        path,
        tile_feat=np.zeros((n, 57, 34), dtype=np.float16),
        scalar=np.zeros((n, 56), dtype=np.float16),
        mask=np.zeros((n, 45), dtype=np.uint8),
        action_idx=np.zeros((n,), dtype=np.int16),
        value=np.zeros((n,), dtype=np.float32),
        score_delta_target=np.zeros((n,), dtype=np.float32),
        win_target=np.zeros((n,), dtype=np.float32),
        dealin_target=np.zeros((n,), dtype=np.float32),
        pts_given_win_target=np.zeros((n,), dtype=np.float32),
        pts_given_dealin_target=np.zeros((n,), dtype=np.float32),
        opp_tenpai_target=np.zeros((n, 3), dtype=np.float32),
        final_rank_target=np.zeros((n,), dtype=np.int8),
        final_score_delta_points_target=np.zeros((n,), dtype=np.int32),
        event_history=np.zeros((n, 48, 5), dtype=np.int16),
        v4_opportunity=np.zeros((n, KEQINGV4_OPPORTUNITY_DIM), dtype=np.uint8),
        v4_discard_summary=np.zeros((n, 34, KEQINGV4_SUMMARY_DIM), dtype=np.float16),
        v4_call_summary=np.zeros((n, 8, KEQINGV4_SUMMARY_DIM), dtype=np.float16),
        v4_special_summary=np.zeros((n, 3, KEQINGV4_SUMMARY_DIM), dtype=np.float16),
    )
