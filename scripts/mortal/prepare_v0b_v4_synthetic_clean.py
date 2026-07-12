#!/usr/bin/env python3
"""Prepare the clean, resumable no-teacher-CE V0b synthetic scratch experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.prepare_reward_pt_experiments import dump_toml
from scripts.mortal.prepare_v0a_v4_synthetic_scratch import load_toml, prepare_config


EXPERIMENT_ID = "V0b_v4_synthetic_clean_2026_07"
DEFAULT_OUTPUT_ROOT = Path("artifacts/experiments/v4_synthetic_2026_06")
DEFAULT_DATA_ROOT = DEFAULT_OUTPUT_ROOT / "V1_data"
TRAIN_LABELS = ("challenger", "champion", "v4")
CHECKPOINT_70K = "artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth"
CHECKPOINT_T1 = "artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth"
CHECKPOINT_V1 = "artifacts/experiments/v4_synthetic_2026_06/V1_v4_synthetic_warmstart_2026_06/checkpoints/mortal_v1_74000.pth"
CHECKPOINT_V4 = "artifacts/model_v4_20240308_best_min.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path("artifacts/mortal_training/config.toml"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--target-steps", type=int, default=15000)
    parser.add_argument("--archive-steps", default="2000,5000,10000,15000")
    parser.add_argument("--model-seed", type=int, default=20260711)
    parser.add_argument("--data-seed", type=int, default=20260711)
    parser.add_argument("--reset-state", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def training_command(config_path: Path, exp_dir: Path, args: argparse.Namespace) -> list[str]:
    return [
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/run_mortal_dqn_offline.py",
        "--config",
        str(config_path),
        "--target-steps",
        str(int(args.target_steps)),
        "--device",
        "cuda",
        "--num-workers",
        "0",
        "--seed",
        str(int(args.model_seed)),
        "--data-seed",
        str(int(args.data_seed)),
        "--archive-steps",
        str(args.archive_steps),
        "--archive-dir",
        str(exp_dir / "checkpoints"),
        "--log-every",
        "50",
    ]


def final_eval_command(exp_dir: Path) -> list[str]:
    return [
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/mortal/four_player_native.py",
        "--require-cuda",
        "--device",
        "cuda",
        "--seat-mode",
        "random",
        "--seed-start",
        "930000",
        "--seed-key",
        "8192",
        "--games",
        "500",
        "--native-batch-games",
        "100",
        "--progress-every",
        "100",
        "--rank-points",
        "90,45,0,-135",
        "--model",
        f"model_v4={CHECKPOINT_V4}",
        "--model",
        f"V1_74000={CHECKPOINT_V1}",
        "--model",
        f"T1_71000={CHECKPOINT_T1}",
        "--model",
        f"V0b={exp_dir / 'checkpoints' / 'mortal_15000.pth'}",
        "--output-dir",
        str(exp_dir / "eval_500h_v0b_15000"),
    ]


def reset_state(exp_dir: Path) -> None:
    for filename in ("mortal.pth", "mortal_best.pth", "file_index.pth", "data_exposure.json"):
        path = exp_dir / filename
        if path.exists():
            path.unlink()
    for dirname in ("tb_mortal", "checkpoints"):
        path = exp_dir / dirname
        if path.exists():
            shutil.rmtree(path)


def main() -> None:
    args = parse_args()
    if int(args.target_steps) <= 0:
        raise ValueError("--target-steps must be positive")
    exp_dir = args.output_root / EXPERIMENT_ID
    config_path = exp_dir / "config.toml"
    config = prepare_config(load_toml(args.base_config), exp_dir=exp_dir, data_root=args.data_root)

    manifest: dict[str, Any] = {
        "schema": "keqing.mortal.v0b_v4_synthetic_clean.v1",
        "experiment_id": EXPERIMENT_ID,
        "init": "random_mortal_dqn_aux_existing_grp_targets",
        "parent_checkpoint": None,
        "target_steps": int(args.target_steps),
        "model_seed": int(args.model_seed),
        "data_seed": int(args.data_seed),
        "archive_steps": str(args.archive_steps),
        "resume_contract": "checkpointed data_stream cursor plus Python/Torch/CUDA RNG; legacy checkpoints are rejected",
        "dataset_globs": list(config["dataset"]["globs"]),
        "train_labels": list(TRAIN_LABELS),
        "teacher_ce_weight": 0.0,
        "risk_gate_enabled": False,
        "objective": "offline DQN + CQL + next-rank auxiliary with existing GRP targets",
        "config": str(config_path),
        "state_file": str(config["control"]["state_file"]),
        "archive_dir": str(exp_dir / "checkpoints"),
        "dataset_audit_command": [
            "uv", "run", "--no-sync", "python", "scripts/mortal/audit_v4_synthetic_dataset.py"
        ],
        "training_command": training_command(config_path, exp_dir, args),
        "final_eval_command": final_eval_command(exp_dir),
    }

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    exp_dir.mkdir(parents=True, exist_ok=True)
    if args.reset_state:
        reset_state(exp_dir)
    (exp_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    config_path.write_text(dump_toml(config), encoding="utf-8")
    (exp_dir / "v4_train_labels.txt").write_text("\n".join(TRAIN_LABELS) + "\n", encoding="utf-8")
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
