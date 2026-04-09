from __future__ import annotations

import json
from pathlib import Path
import subprocess


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


def test_preprocess_xmodel1_script_fallback_smoke(tmp_path: Path):
    input_dir = tmp_path / "converted" / "ds1"
    input_dir.mkdir(parents=True)
    _write_mjson(input_dir / "sample.mjson")
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
            str(input_dir),
            "--output_dir",
            str(output_dir),
        ],
        cwd="/media/bailan/DISK1/AUbuntuProject/project/keqing1",
        check=True,
        capture_output=True,
        text=True,
    )
    exported = output_dir / "ds1" / "sample.npz"
    assert exported.exists()
    assert ("Python fallback" in result.stdout) or ("Rust Xmodel1 export completed:" in result.stdout)
