#!/usr/bin/env python3
"""Prepare V1 model_v4 synthetic warm-start training config."""

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

from scripts.mortal.prepare_reward_pt_experiments import dump_toml
from scripts.mortal.prepare_reward_pt_experiments import read_checkpoint_steps

DEFAULT_PARENT = Path("artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth")
DEFAULT_OUTPUT_ROOT = Path("artifacts/experiments/v4_synthetic_2026_06")
DEFAULT_DATA_ROOT = DEFAULT_OUTPUT_ROOT / "V1_data"
EXPERIMENT_ID = "V1_v4_synthetic_warmstart_2026_06"
TRAIN_LABELS = ("challenger", "champion", "v4_a", "v4_b")
DATASET_GLOBS = (
    "artifacts/experiments/v4_synthetic_2026_06/V1_data/selfplay_v4_12000h_1v3/logs/**/*.json.gz",
    "artifacts/experiments/v4_synthetic_2026_06/V1_data/mix_v4_70k_T1_4000h/logs/**/*.json.gz",
    "artifacts/experiments/v4_synthetic_2026_06/V1_data/mix_v4_80k_T1_4000h/logs/**/*.json.gz",
)
V4_CHECKPOINT = "artifacts/model_v4_20240308_best_min.pth"
CHECKPOINT_70K = "artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth"
CHECKPOINT_80K = "artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth"
CHECKPOINT_T1 = "artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path("artifacts/mortal_training/config.toml"))
    parser.add_argument("--parent-checkpoint", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--parent-steps", type=int, default=None)
    parser.add_argument("--stage1-steps", type=int, default=74000)
    parser.add_argument("--final-steps", type=int, default=80000)
    parser.add_argument("--copy-parent-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def prepare_config(base_config: Mapping[str, Any], *, exp_dir: Path, data_root: Path) -> dict[str, Any]:
    config = copy.deepcopy(dict(base_config))

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
        str((data_root / "selfplay_v4_12000h_1v3" / "logs" / "**" / "*.json.gz").resolve()),
        str((data_root / "mix_v4_70k_T1_4000h" / "logs" / "**" / "*.json.gz").resolve()),
        str((data_root / "mix_v4_80k_T1_4000h" / "logs" / "**" / "*.json.gz").resolve()),
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


def training_command(config_path: Path, target_steps: int) -> list[str]:
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
        f"V1={checkpoint}",
        "--output-dir",
        str(output_dir),
    ]


def generation_command(*, models: list[str], output_dir: Path, games: int, seed_start: int, resume: bool) -> list[str]:
    command = [
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
    ]
    for model in models:
        command.extend(["--model", model])
    command.extend(["--output-dir", str(output_dir)])
    if resume:
        command.append("--resume")
    return command


def one_vs_three_generation_command(*, output_dir: Path, hanchans: int, seed_start: int, resume: bool) -> list[str]:
    if hanchans % 4 != 0:
        raise ValueError(f"OneVsThree generation requires hanchans divisible by 4, got {hanchans}")
    command = [
        "uv",
        "run",
        "--no-sync",
        "python",
        "scripts/mortal/one_vs_three_smoke.py",
        "--challenger",
        V4_CHECKPOINT,
        "--champion",
        V4_CHECKPOINT,
        "--device",
        "cuda",
        "--seed-start",
        str(int(seed_start)),
        "--seed-key",
        "8192",
        "--seed-count",
        str(int(hanchans) // 4),
        "--progress-every",
        "25",
        "--rank-points",
        "90,45,0,-135",
        "--challenger-label",
        "challenger",
        "--champion-label",
        "champion",
        "--platform-model-label",
        "v4",
        "--output-dir",
        str(output_dir),
    ]
    if resume:
        command.append("--resume")
    return command


def data_generation_commands(data_root: Path) -> dict[str, Any]:
    pools = {
        "selfplay_v4_12000h_1v3": {
            "models": [
                f"challenger={V4_CHECKPOINT}",
                f"champion={V4_CHECKPOINT}",
            ],
            "output_dir": data_root / "selfplay_v4_12000h_1v3",
            "seed_start": 810000,
            "smoke_games": 28,
            "full_games": 12000,
            "backend": "one_vs_three",
        },
        "mix_v4_70k_T1_4000h": {
            "models": [
                f"v4_a={V4_CHECKPOINT}",
                f"v4_b={V4_CHECKPOINT}",
                f"70k={CHECKPOINT_70K}",
                f"T1_71000={CHECKPOINT_T1}",
            ],
            "output_dir": data_root / "mix_v4_70k_T1_4000h",
            "seed_start": 830000,
            "smoke_games": 25,
            "full_games": 4000,
            "backend": "four_player",
        },
        "mix_v4_80k_T1_4000h": {
            "models": [
                f"v4_a={V4_CHECKPOINT}",
                f"v4_b={V4_CHECKPOINT}",
                f"80k_game={CHECKPOINT_80K}",
                f"T1_71000={CHECKPOINT_T1}",
            ],
            "output_dir": data_root / "mix_v4_80k_T1_4000h",
            "seed_start": 850000,
            "smoke_games": 25,
            "full_games": 4000,
            "backend": "four_player",
        },
    }
    commands: dict[str, Any] = {}
    for pool_id, spec in pools.items():
        if spec["backend"] == "one_vs_three":
            smoke_command = one_vs_three_generation_command(
                output_dir=Path(spec["output_dir"]),
                hanchans=int(spec["smoke_games"]),
                seed_start=int(spec["seed_start"]),
                resume=False,
            )
            full_command = one_vs_three_generation_command(
                output_dir=Path(spec["output_dir"]),
                hanchans=int(spec["full_games"]),
                seed_start=int(spec["seed_start"]),
                resume=True,
            )
        else:
            smoke_command = generation_command(
                models=list(spec["models"]),
                output_dir=Path(spec["output_dir"]),
                games=int(spec["smoke_games"]),
                seed_start=int(spec["seed_start"]),
                resume=False,
            )
            full_command = generation_command(
                models=list(spec["models"]),
                output_dir=Path(spec["output_dir"]),
                games=int(spec["full_games"]),
                seed_start=int(spec["seed_start"]),
                resume=True,
            )
        commands[pool_id] = {
            "backend": spec["backend"],
            "smoke_command": smoke_command,
            "full_resume_command": full_command,
        }
    return commands


def main() -> None:
    args = parse_args()
    exp_dir = args.output_root / EXPERIMENT_ID
    config_path = exp_dir / "config.toml"
    checkpoints_dir = exp_dir / "checkpoints"
    parent_steps = int(args.parent_steps) if args.parent_steps is not None else read_checkpoint_steps(args.parent_checkpoint)
    base_config = load_toml(args.base_config)
    config = prepare_config(base_config, exp_dir=exp_dir, data_root=args.data_root)

    manifest = {
        "schema": "keqing.mortal.v1_v4_synthetic_warmstart_config.v1",
        "experiment_id": EXPERIMENT_ID,
        "parent_checkpoint": str(args.parent_checkpoint),
        "parent_steps": parent_steps,
        "stage1_steps": int(args.stage1_steps),
        "final_steps": int(args.final_steps),
        "data_root": str(args.data_root),
        "dataset_globs": list(config["dataset"]["globs"]),
        "train_labels": list(TRAIN_LABELS),
        "teacher_ce_weight": 0.0,
        "risk_gate_enabled": False,
        "config": str(config_path),
        "state_file": str(config["control"]["state_file"]),
        "data_generation_commands": data_generation_commands(args.data_root),
        "dataset_audit_command": [
            "uv",
            "run",
            "--no-sync",
            "python",
            "scripts/mortal/audit_v4_synthetic_dataset.py",
        ],
        "stage1_training_command": training_command(config_path, int(args.stage1_steps)),
        "final_training_command": training_command(config_path, int(args.final_steps)),
        "stage1_archive_path": str(checkpoints_dir / "mortal_v1_74000.pth"),
        "final_archive_path": str(checkpoints_dir / "mortal_v1_80000.pth"),
        "stage1_eval_command": eval_command(
            checkpoint=checkpoints_dir / "mortal_v1_74000.pth",
            output_dir=exp_dir / "eval_250h_v1_74000",
            games=250,
            seed_start=790000,
        ),
        "final_eval_command": eval_command(
            checkpoint=checkpoints_dir / "mortal_v1_80000.pth",
            output_dir=exp_dir / "eval_1000h_v1_80000",
            games=1000,
            seed_start=800000,
        ),
    }

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return

    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(dump_toml(config), encoding="utf-8")
    (exp_dir / "v4_train_labels.txt").write_text("\n".join(TRAIN_LABELS) + "\n", encoding="utf-8")
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.copy_parent_checkpoint:
        state_file = Path(config["control"]["state_file"])
        if not state_file.exists():
            shutil.copy2(args.parent_checkpoint, state_file)
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
