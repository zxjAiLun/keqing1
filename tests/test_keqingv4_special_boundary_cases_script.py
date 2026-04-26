from __future__ import annotations

import json
from pathlib import Path
import subprocess


def test_mine_keqingv4_special_boundary_cases_script_smoke(tmp_path: Path):
    replay_dir = tmp_path / "replay_demo"
    replay_dir.mkdir()
    decisions_path = replay_dir / "decisions.json"
    payload = {
        "log": [
            {
                "step": 7,
                "bakaze": "E",
                "kyoku": 1,
                "honba": 0,
                "hand": ["4m"],
                "tsumo_pai": "5m",
                "chosen": {"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": False},
                "candidates": [
                    {
                        "action": {"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": False},
                        "final_score": 1.0,
                    },
                    {
                        "action": {"type": "reach", "actor": 0},
                        "final_score": 0.3,
                    },
                ],
            }
        ]
    }
    decisions_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    output_path = tmp_path / "boundary_cases.json"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/mine_keqingv4_special_boundary_cases.py",
            "--inputs",
            str(replay_dir),
            "--output",
            str(output_path),
            "--max-per-type",
            "5",
            "--margin-threshold",
            "1.0",
        ],
        cwd="/media/bailan/DISK1/AUbuntuProject/project/keqing1",
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.exists()
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["summary"]["total_cases"] == 1
    assert report["cases"][0]["special_type"] == "reach"
    assert report["cases"][0]["margin_vs_chosen"] == 0.7
    assert report["cases"][0]["candidate_is_chosen"] is False
    assert "wrote special boundary cases" in result.stdout
