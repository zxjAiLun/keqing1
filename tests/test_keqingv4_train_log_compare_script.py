from __future__ import annotations

from pathlib import Path
import subprocess


def test_compare_keqingv4_train_logs_script_smoke(tmp_path: Path):
    baseline = tmp_path / "baseline.jsonl"
    probe = tmp_path / "probe.jsonl"
    baseline.write_text(
        "\n".join(
            [
                '{"epoch": 1, "val_objective": 1.2, "val_typed_rank_loss": 0.9, "val_reach_chosen_rate": 0.10, "val_meld_chosen_rate": 0.05, "val_reach_slice_acc": 0.20, "val_meld_slice_acc": 0.30, "val_special_slice_acc": 0.25}',
                '{"epoch": 2, "val_objective": 1.1, "val_typed_rank_loss": 0.8, "val_reach_chosen_rate": 0.12, "val_meld_chosen_rate": 0.05, "val_reach_slice_acc": 0.25, "val_meld_slice_acc": 0.31, "val_special_slice_acc": 0.28}',
            ]
        ),
        encoding="utf-8",
    )
    probe.write_text(
        "\n".join(
            [
                '{"epoch": 1, "val_objective": 1.05, "val_typed_rank_loss": 0.7, "val_reach_chosen_rate": 0.14, "val_meld_chosen_rate": 0.07, "val_reach_slice_acc": 0.32, "val_meld_slice_acc": 0.36, "val_special_slice_acc": 0.38}',
                '{"epoch": 2, "val_objective": 0.98, "val_typed_rank_loss": 0.55, "val_reach_chosen_rate": 0.18, "val_meld_chosen_rate": 0.08, "val_reach_slice_acc": 0.40, "val_meld_slice_acc": 0.41, "val_special_slice_acc": 0.48}',
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/compare_keqingv4_train_logs.py",
            str(baseline),
            str(probe),
        ],
        cwd="/media/bailan/DISK1/AUbuntuProject/project/keqing1",
        check=True,
        capture_output=True,
        text=True,
    )
    assert "# keqingv4 B3 comparison" in result.stdout
    assert "val_special_slice_acc" in result.stdout
    assert "next-iteration recommendations:" in result.stdout
    assert "keep typed ranking loss on" in result.stdout

