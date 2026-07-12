#!/usr/bin/env python3
"""Prepare V0a model_v4 synthetic scratch training config."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import sys
import tomllib
from typing import Any, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.prepare_reward_pt_experiments import dump_toml

DEFAULT_OUTPUT_ROOT = Path("artifacts/experiments/v4_synthetic_2026_06")
DEFAULT_DATA_ROOT = DEFAULT_OUTPUT_ROOT / "V1_data"
EXPERIMENT_ID = "V0a_v4_synthetic_scratch_2026_06"
TRAIN_LABELS = ("challenger", "champion", "v4")
CHECKPOINT_70K = "artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth"
CHECKPOINT_80K = "artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth"
CHECKPOINT_T1 = "artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path("artifacts/mortal_training/config.toml"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--smoke-steps", type=int, default=400)
    parser.add_argument("--stage1-steps", type=int, default=2000)
    parser.add_argument("--probe-steps", type=int, default=10000)
    parser.add_argument("--reset-state", action="store_true", help="remove existing V0a state/file index before writing config")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def normalize_host_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: normalize_host_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_host_paths(item) for item in value]
    if not isinstance(value, str) or os.name != "nt":
        return value
    if not value.startswith("/mnt/") or len(value) < 7 or value[6] != "/":
        return value
    drive = value[5].upper()
    rest = value[7:]
    return str(Path(f"{drive}:/{rest}").resolve())


def prepare_config(base_config: Mapping[str, Any], *, exp_dir: Path, data_root: Path) -> dict[str, Any]:
    config = normalize_host_paths(copy.deepcopy(dict(base_config)))

    control = config.setdefault("control", {})
    control["state_file"] = str((exp_dir / "mortal.pth").resolve())
    control["best_state_file"] = str((exp_dir / "mortal_best.pth").resolve())
    control["tensorboard_dir"] = str((exp_dir / "tb_mortal").resolve())

    train_play = config.get("train_play", {}).get("default")
    if isinstance(train_play, dict):
        train_play["log_dir"] = str((exp_dir / "train_play").resolve())
    if isinstance(config.get("test_play"), dict):
        config["test_play"]["log_dir"] = str((exp_dir / "test_play").resolve())
    if isinstance(config.get("online"), dict) and isinstance(config["online"].get("server"), dict):
        config["online"]["server"]["buffer_dir"] = str((exp_dir / "buffer").resolve())
        config["online"]["server"]["drain_dir"] = str((exp_dir / "drain").resolve())
    if isinstance(config.get("1v3"), dict):
        config["1v3"]["log_dir"] = str((exp_dir / "1v3").resolve())
        if isinstance(config["1v3"].get("challenger"), dict):
            config["1v3"]["challenger"]["state_file"] = control["state_file"]

    dataset = config.setdefault("dataset", {})
    dataset["globs"] = [
        str((data_root / "selfplay_v4_12000h_1v3" / "logs" / "**" / "*_a.json.gz").resolve()),
        str((data_root / "selfplay_v4_unique_9000h" / "logs" / "**" / "*.json.gz").resolve()),
    ]
    dataset["file_index"] = str((exp_dir / "file_index.pth").resolve())
    dataset["num_workers"] = 0
    dataset["player_names_files"] = [str((exp_dir / "v4_train_labels.txt").resolve())]
    dataset["num_epochs"] = 1
    dataset["enable_augmentation"] = False

    teacher = config.setdefault("teacher", {})
    teacher["ce_weight"] = 0.0
    teacher["risk_gate_enabled"] = False
    teacher["risk_gate"] = {"enabled": False}

    return config


def training_command(config_path: Path, target_steps: int, *, log_every: int = 50) -> list[str]:
    return [
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/run_mortal_dqn_offline.py",
        "--config",
        str(config_path),
        "--target-steps",
        str(int(target_steps)),
        "--device",
        "cuda",
        "--num-workers",
        "0",
        "--log-every",
        str(int(log_every)),
    ]


def eval_command(*, checkpoint: Path, output_dir: Path, games: int, seed_start: int) -> list[str]:
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
        str(int(seed_start)),
        "--seed-key",
        "8192",
        "--games",
        str(int(games)),
        "--progress-every",
        "25",
        "--rank-points",
        "90,45,0,-135",
        "--model",
        f"70k={CHECKPOINT_70K}",
        "--model",
        f"80k_game={CHECKPOINT_80K}",
        "--model",
        f"T1_71000={CHECKPOINT_T1}",
        "--model",
        f"V0a={checkpoint}",
        "--output-dir",
        str(output_dir),
    ]


def maybe_reset(exp_dir: Path) -> None:
    for path in [
        exp_dir / "mortal.pth",
        exp_dir / "mortal_best.pth",
        exp_dir / "file_index.pth",
    ]:
        if path.exists():
            path.unlink()
    tb_dir = exp_dir / "tb_mortal"
    if tb_dir.exists():
        shutil.rmtree(tb_dir)


def main() -> None:
    args = parse_args()
    exp_dir = args.output_root / EXPERIMENT_ID
    config_path = exp_dir / "config.toml"
    checkpoints_dir = exp_dir / "checkpoints"
    base_config = load_toml(args.base_config)
    config = prepare_config(base_config, exp_dir=exp_dir, data_root=args.data_root)

    manifest = {
        "schema": "keqing.mortal.v0a_v4_synthetic_scratch_config.v1",
        "experiment_id": EXPERIMENT_ID,
        "init": "random_mortal_dqn_aux_existing_grp_targets",
        "parent_checkpoint": None,
        "initial_steps": 0,
        "smoke_steps": int(args.smoke_steps),
        "stage1_steps": int(args.stage1_steps),
        "probe_steps": int(args.probe_steps),
        "data_root": str(args.data_root),
        "dataset_globs": list(config["dataset"]["globs"]),
        "train_labels": list(TRAIN_LABELS),
        "teacher_ce_weight": 0.0,
        "risk_gate_enabled": False,
        "config": str(config_path),
        "state_file": str(config["control"]["state_file"]),
        "dataset_audit_command": [
            "uv",
            "run",
            "--no-sync",
            "python",
            "scripts/mortal/audit_v4_synthetic_dataset.py",
        ],
        "smoke_training_command": training_command(config_path, int(args.smoke_steps), log_every=10),
        "stage1_training_command": training_command(config_path, int(args.stage1_steps)),
        "probe_training_command": training_command(config_path, int(args.probe_steps)),
        "stage1_archive_path": str(checkpoints_dir / "mortal_v0a_2000.pth"),
        "probe_archive_path": str(checkpoints_dir / "mortal_v0a_10000.pth"),
        "stage1_eval_command": eval_command(
            checkpoint=checkpoints_dir / "mortal_v0a_2000.pth",
            output_dir=exp_dir / "eval_100h_v0a_2000",
            games=100,
            seed_start=810000,
        ),
        "probe_eval_command": eval_command(
            checkpoint=checkpoints_dir / "mortal_v0a_10000.pth",
            output_dir=exp_dir / "eval_250h_v0a_10000",
            games=250,
            seed_start=811000,
        ),
    }

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if args.reset_state:
        maybe_reset(exp_dir)
    config_path.write_text(dump_toml(config), encoding="utf-8")
    (exp_dir / "v4_train_labels.txt").write_text("\n".join(TRAIN_LABELS) + "\n", encoding="utf-8")
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
