from __future__ import annotations

from pathlib import Path
import subprocess


def test_summarize_keqingv4_train_log_script_smoke(tmp_path: Path):
    log_path = tmp_path / "train_log.jsonl"
    log_path.write_text(
        "\n".join(
            [
                '{"epoch": 1, "train_objective": 1.2, "val_objective": 1.1, "val_reach_slice_acc": 0.2, "val_meld_slice_acc": 0.3, "val_special_slice_acc": 0.25, "val_typed_rank_loss": 0.8, "val_reach_chosen_rate": 0.1, "val_meld_chosen_rate": 0.05}',
                '{"epoch": 2, "train_objective": 1.0, "val_objective": 0.9, "val_reach_slice_acc": 0.4, "val_meld_slice_acc": 0.35, "val_special_slice_acc": 0.45, "val_typed_rank_loss": 0.6, "val_reach_chosen_rate": 0.2, "val_meld_chosen_rate": 0.08}',
            ]
        ),
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/summarize_keqingv4_train_log.py",
            str(log_path),
        ],
        cwd="/media/bailan/DISK1/AUbuntuProject/project/keqing1",
        check=True,
        capture_output=True,
        text=True,
    )
    assert "epochs_logged: 2" in result.stdout
    assert "best_val_objective: epoch=2 value=0.9000" in result.stdout
    assert "best_val_special_slice_acc: epoch=2 value=0.4500" in result.stdout
    assert "best_val_typed_rank_loss: epoch=2 value=0.6000" in result.stdout

