from __future__ import annotations

import subprocess


REPO_ROOT = "/media/bailan/DISK1/AUbuntuProject/project/keqing1"


def test_xmodel1_preprocess_cli_smoke_reports_rust_cli_path():
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/preprocess_xmodel1.py",
            "--smoke",
        ],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "xmodel1 preprocess launcher -> native-v3-export" in result.stdout
    assert "xmodel1 preprocess complete:" in result.stdout
