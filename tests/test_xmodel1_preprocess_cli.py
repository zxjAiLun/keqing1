from __future__ import annotations

import subprocess


def test_xmodel1_preprocess_cli_smoke_reports_rust_or_fallback_path():
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/preprocess_xmodel1.py",
            "--smoke",
        ],
        cwd="/media/bailan/DISK1/AUbuntuProject/project/keqing1",
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert (
        "Rust Xmodel1 export unavailable, using Python fallback" in result.stdout
        or "Rust Xmodel1 export completed:" in result.stdout
    )
