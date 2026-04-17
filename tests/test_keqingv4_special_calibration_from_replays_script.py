from __future__ import annotations

import json
from pathlib import Path
import subprocess


def test_evaluate_keqingv4_special_calibration_from_replays_script_smoke(tmp_path: Path):
    replay_dir = tmp_path / "replay_demo"
    replay_dir.mkdir()
    decisions_path = replay_dir / "decisions.json"
    payload = {
        "log": [
            {
                "step": 12,
                "bakaze": "E",
                "kyoku": 1,
                "honba": 0,
                "oya": 0,
                "scores": [25000, 25000, 25000, 25000],
                "hand": ["1m", "9m"],
                "discards": [[], [], [], []],
                "melds": [[], [], [], []],
                "dora_markers": ["1p"],
                "reached": [False, False, False, False],
                "actor_to_move": 0,
                "tsumo_pai": "5m",
                "candidates": [
                    {
                        "action": {"type": "ryukyoku"},
                        "logit": 0.1,
                        "final_score": 0.1,
                    },
                    {
                        "action": {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False},
                        "logit": 0.4,
                        "final_score": 0.4,
                    },
                ],
                "chosen": {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False},
            }
        ]
    }
    decisions_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    output_path = tmp_path / "report.json"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/evaluate_keqingv4_special_calibration_from_replays.py",
            "--inputs",
            str(replay_dir),
            "--output",
            str(output_path),
            "--max-per-type",
            "5",
        ],
        cwd="/media/bailan/DISK1/AUbuntuProject/project/keqing1",
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.exists()
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["summary"]["total_cases"] == 1
    assert report["cases"][0]["special_type"] == "ryukyoku"
    assert "before" in report["cases"][0]
    assert "after" in report["cases"][0]
    assert "wrote replay special calibration evidence" in result.stdout

