from __future__ import annotations

import json
from pathlib import Path
import subprocess


def test_evaluate_keqingv4_special_calibration_script_smoke(tmp_path: Path):
    output_path = tmp_path / "special_calibration_report.json"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/evaluate_keqingv4_special_calibration.py",
            "--output",
            str(output_path),
        ],
        cwd="/media/bailan/DISK1/AUbuntuProject/project/keqing1",
        check=True,
        capture_output=True,
        text=True,
    )

    assert output_path.exists()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert len(payload["cases"]) == 3
    names = {case["case"] for case in payload["cases"]}
    assert names == {"reach_decl", "hora_finish", "abortive_ryukyoku"}
    for case in payload["cases"]:
        assert "before" in case and "after" in case
        assert "chosen" in case["before"] and "chosen" in case["after"]
        assert case["before"]["candidates"]
        assert case["after"]["candidates"]
    assert "wrote special calibration evidence" in result.stdout

