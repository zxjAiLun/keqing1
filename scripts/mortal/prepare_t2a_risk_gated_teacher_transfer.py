#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tomllib
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.prepare_reward_pt_experiments import dump_toml, read_checkpoint_steps


DEFAULT_PARENT = Path("artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth")
DEFAULT_BASE_CONFIG = Path("artifacts/mortal_training/config.toml")
DEFAULT_OUTPUT_ROOT = Path("artifacts/experiments/teacher_transfer_2026_05")
TEACHER_REPLAY_POOL = "artifacts/eval/gate_10000h/Gate_v4_vs_70k_2500/logs/**/*.json.gz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare T2a risk-gated teacher CE config.")
    parser.add_argument("--base-config", type=Path, default=DEFAULT_BASE_CONFIG)
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--experiment-id", default="T2a_risk_gated_teacher_ce_005")
    parser.add_argument("--parent-steps", type=int, default=None)
    parser.add_argument("--target-steps", type=int, default=71400)
    parser.add_argument("--base-weight", type=float, default=0.05)
    parser.add_argument("--risk-weight", type=float, default=0.0)
    parser.add_argument("--copy-parent-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def prepare_config(
    base_config: Mapping[str, Any],
    *,
    exp_dir: Path,
    base_weight: float,
    risk_weight: float,
) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))
    control = config.setdefault("control", {})
    control["state_file"] = str((exp_dir / "mortal.pth").resolve())
    control["best_state_file"] = str((exp_dir / "mortal_best.pth").resolve())
    control["tensorboard_dir"] = str((exp_dir / "tb_mortal").resolve())
    control["device"] = "cuda:0"
    control["save_every"] = 400

    dataset = config.setdefault("dataset", {})
    dataset["globs"] = [TEACHER_REPLAY_POOL]
    dataset["file_index"] = str((exp_dir / "file_index.pth").resolve())
    dataset["file_batch_size"] = 15
    dataset["reserve_ratio"] = 0.0
    dataset["num_workers"] = 0
    dataset["player_names_files"] = [str((exp_dir / "challenger_only.json").resolve())]
    dataset["num_epochs"] = 1
    dataset["enable_augmentation"] = False
    dataset["augmented_first"] = False

    env = config.setdefault("env", {})
    env["pts"] = [6.0, 4.0, 2.0, 0.0]

    grp = config.setdefault("grp", {})
    grp["state_file"] = str(Path("artifacts/mortal_training/grp.pth").resolve())

    teacher = config.setdefault("teacher", {})
    teacher["ce_weight"] = float(base_weight)
    teacher["risk_gate"] = {
        "enabled": True,
        "mode": "sample_weight",
        "base_weight": float(base_weight),
        "risk_weight": float(risk_weight),
        "disable_after_fuuro_discard": True,
        "disable_vs_riichi_discard": True,
        "disable_after_fuuro_vs_riichi_discard": True,
        "disable_dealer_or_leading": True,
        "disable_start_rank_1": True,
    }
    return config


def main() -> None:
    args = parse_args()
    parent_steps = (
        int(args.parent_steps)
        if args.parent_steps is not None
        else int(read_checkpoint_steps(args.parent_checkpoint))
    )
    target_steps = int(args.target_steps)
    if target_steps <= parent_steps:
        raise ValueError(f"target_steps must be greater than parent_steps: {target_steps} <= {parent_steps}")

    exp_dir = args.output_root / args.experiment_id
    config_path = exp_dir / "config.toml"
    base_config = load_toml(args.base_config)
    config = prepare_config(
        base_config,
        exp_dir=exp_dir,
        base_weight=float(args.base_weight),
        risk_weight=float(args.risk_weight),
    )
    state_file = Path(config["control"]["state_file"])
    checkpoints_dir = exp_dir / "checkpoints"
    manifest = {
        "schema": "keqing.mortal.t2a_risk_gated_teacher_transfer_config.v1",
        "experiment_id": args.experiment_id,
        "parent_checkpoint": str(args.parent_checkpoint),
        "parent_steps": int(parent_steps),
        "target_steps": int(target_steps),
        "teacher_replay_pool": TEACHER_REPLAY_POOL,
        "player_filter": "challenger",
        "teacher_ce_base_weight": float(args.base_weight),
        "teacher_ce_risk_weight": float(args.risk_weight),
        "config": str(config_path),
        "state_file": str(state_file),
        "preflight_output": str(exp_dir / "preflight" / "risk_weight_alignment.json"),
        "checkpoint_archive_dir": str(checkpoints_dir),
        "training_command_windows": [
            "uv",
            "run",
            "--no-sync",
            "python",
            "scripts/run_mortal_dqn_offline.py",
            "--config",
            str(config_path),
            "--target-steps",
            str(target_steps),
            "--device",
            "cuda",
            "--num-workers",
            "0",
        ],
    }

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    (exp_dir / "challenger_only.json").write_text("challenger\n", encoding="utf-8")
    config_path.write_text(dump_toml(config), encoding="utf-8")
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if args.copy_parent_checkpoint:
        if not args.parent_checkpoint.exists():
            raise FileNotFoundError(f"parent checkpoint not found: {args.parent_checkpoint}")
        if state_file.exists():
            print(f"state file already exists, leaving in place: {state_file}")
        else:
            shutil.copy2(args.parent_checkpoint, state_file)
            print(f"copied parent checkpoint to {state_file}")

    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
