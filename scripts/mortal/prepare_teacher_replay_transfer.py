#!/usr/bin/env python3
"""Prepare T1 teacher replay transfer config — model_v4 vs 3x70k replays, challenger-only samples."""
from __future__ import annotations

import argparse
import copy
import json
import shutil
from pathlib import Path
import sys
import tomllib
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.prepare_reward_pt_experiments import dump_toml
from scripts.mortal.prepare_reward_pt_experiments import read_checkpoint_steps

DEFAULT_ANCHOR_70K = Path("artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth")
TEACHER_REPLAY_POOL = "artifacts/eval/gate_10000h/Gate_v4_vs_70k_2500/logs/**/*.json.gz"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-config", type=Path, default=Path("artifacts/mortal_training/config.toml"))
    p.add_argument("--output-root", type=Path, default=Path("artifacts/experiments/teacher_transfer_2026_05"))
    p.add_argument("--experiment-id", default="T1_teacher_ce_01")
    p.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR_70K)
    p.add_argument("--parent-steps", type=int, default=None, help="Override parent checkpoint steps, useful for weights-only checkpoints")
    p.add_argument("--train-steps", type=int, default=1000)
    p.add_argument("--teacher-ce-weight", type=float, default=0.1)
    p.add_argument("--copy-parent-checkpoint", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def prepare_t1_config(
    base_config: Mapping[str, Any],
    *,
    exp_dir: Path,
    teacher_ce_weight: float,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))

    control = config.setdefault("control", {})
    control["state_file"] = str((exp_dir / "mortal.pth").resolve())
    control["best_state_file"] = str((exp_dir / "mortal_best.pth").resolve())
    control["tensorboard_dir"] = str((exp_dir / "tb_mortal").resolve())

    dataset = config.setdefault("dataset", {})
    dataset["globs"] = [TEACHER_REPLAY_POOL]
    dataset["file_index"] = str((exp_dir / "file_index.pth").resolve())
    dataset["num_epochs"] = 1
    dataset["enable_augmentation"] = False
    dataset["num_workers"] = 0
    dataset["player_names_files"] = [str((exp_dir / "challenger_only.json").resolve())]

    env = config.setdefault("env", {})
    env["pts"] = [6.0, 4.0, 2.0, 0.0]

    teacher = config.setdefault("teacher", {})
    teacher["ce_weight"] = float(teacher_ce_weight)

    return config


def main():
    args = parse_args()
    base = load_toml(args.base_config)
    parent_steps = int(args.parent_steps) if args.parent_steps is not None else read_checkpoint_steps(args.anchor_checkpoint)
    exp_dir = args.output_root / args.experiment_id
    config_path = exp_dir / "config.toml"

    config = prepare_t1_config(base, exp_dir=exp_dir, teacher_ce_weight=args.teacher_ce_weight)
    target_steps = int(parent_steps) + int(args.train_steps)

    # player_names filter: only challenger
    player_names_path = exp_dir / "challenger_only.json"
    player_names_path.parent.mkdir(parents=True, exist_ok=True)
    player_names_path.write_text("challenger\n", encoding="utf-8")

    manifest = {
        "schema": "keqing.mortal.teacher_transfer_config.v1",
        "experiment_id": args.experiment_id,
        "parent_checkpoint": str(args.anchor_checkpoint),
        "parent_steps": int(parent_steps),
        "train_steps": int(args.train_steps),
        "target_steps": int(target_steps),
        "teacher_ce_weight": float(args.teacher_ce_weight),
        "teacher_replay_pool": TEACHER_REPLAY_POOL,
        "player_filter": "challenger",
        "config": str(config_path),
        "state_file": str(config["control"]["state_file"]),
        "training_command": [
            "uv", "run", "python",
            "scripts/run_mortal_dqn_offline.py",
            "--config", str(config_path),
            "--target-steps", str(int(target_steps)),
            "--num-workers", "0",
        ],
    }

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(dump_toml(config), encoding="utf-8")
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.copy_parent_checkpoint:
        state_file = Path(config["control"]["state_file"])
        if not state_file.exists():
            shutil.copy2(args.anchor_checkpoint, state_file)

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


if __name__ == "__main__":
    main()
