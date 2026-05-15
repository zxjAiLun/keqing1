#!/usr/bin/env python3
"""Prepare isolated Mortal configs for selfplay fine-tune phase 1."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import shutil
import sys
import tomllib
from typing import Any, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.eval_metrics import RANK_POINT_PROFILES
from scripts.mortal.prepare_reward_pt_experiments import dump_toml
from scripts.mortal.prepare_reward_pt_experiments import read_checkpoint_steps


DEFAULT_70K_CHECKPOINT = Path("artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth")
DEFAULT_80K_CHECKPOINT = Path("artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth")
DEFAULT_70K_LOGS = "artifacts/experiments/grp_audit/selfplay_70k_base_500h_arena/logs/**/*.json.gz"
DEFAULT_80K_LOGS = "artifacts/experiments/grp_audit/selfplay_80k_base_500h_arena/logs/**/*.json.gz"

DEFAULT_MATRIX: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("S1_standard_selfplay", "70k", "standard selfplay fine-tune", ("70k",)),
    ("A1_aggressive_data_transfer", "70k", "aggressive selfplay data transfer", ("80k",)),
    ("A2_aggressive_lineage_continuation", "80k", "aggressive lineage continuation", ("80k",)),
    ("M1_mixed_selfplay", "70k", "mixed standard/aggressive selfplay", ("70k", "80k")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path("artifacts/mortal_training/config.toml"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments/selfplay_finetune_2026_05"))
    parser.add_argument("--checkpoint-70k", type=Path, default=DEFAULT_70K_CHECKPOINT)
    parser.add_argument("--checkpoint-80k", type=Path, default=DEFAULT_80K_CHECKPOINT)
    parser.add_argument("--logs-70k", default=DEFAULT_70K_LOGS)
    parser.add_argument("--logs-80k", default=DEFAULT_80K_LOGS)
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--target-steps", type=int, default=None, help="Optional common target; defaults to parent steps + train steps per experiment")
    parser.add_argument("--matrix", default=",".join(row[0] for row in DEFAULT_MATRIX))
    parser.add_argument("--copy-parent-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def matrix_by_id() -> dict[str, tuple[str, str, tuple[str, ...]]]:
    return {experiment_id: (parent_label, notes, data_labels) for experiment_id, parent_label, notes, data_labels in DEFAULT_MATRIX}


def parse_matrix(value: str) -> list[str]:
    known = matrix_by_id()
    rows: list[str] = []
    for raw in value.split(","):
        experiment_id = raw.strip()
        if not experiment_id:
            continue
        if experiment_id not in known:
            raise ValueError(f"unknown experiment {experiment_id!r}; known: {', '.join(sorted(known))}")
        rows.append(experiment_id)
    if not rows:
        raise ValueError("matrix must contain at least one experiment")
    return rows


def checkpoint_map(checkpoint_70k: Path, checkpoint_80k: Path) -> dict[str, Path]:
    return {"70k": checkpoint_70k, "80k": checkpoint_80k}


def log_glob_map(logs_70k: str, logs_80k: str) -> dict[str, str]:
    return {"70k": logs_70k, "80k": logs_80k}


def prepare_config(
    base_config: Mapping[str, Any],
    *,
    experiment_id: str,
    output_root: Path,
    dataset_globs: Sequence[str],
) -> tuple[dict[str, Any], Path]:
    exp_dir = output_root / experiment_id
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
    dataset = config.setdefault("dataset", {})
    dataset["globs"] = [str(glob) for glob in dataset_globs]
    dataset["file_index"] = str((exp_dir / "file_index.pth").resolve())
    env = config.setdefault("env", {})
    env["pts"] = [float(value) for value in RANK_POINT_PROFILES["mortal_default"]]
    if isinstance(config.get("online"), dict) and isinstance(config["online"].get("server"), dict):
        config["online"]["server"]["buffer_dir"] = str((exp_dir / "buffer").resolve())
        config["online"]["server"]["drain_dir"] = str((exp_dir / "drain").resolve())
    if isinstance(config.get("1v3"), dict):
        config["1v3"]["log_dir"] = str((exp_dir / "1v3").resolve())
        if isinstance(config["1v3"].get("challenger"), dict):
            config["1v3"]["challenger"]["state_file"] = control["state_file"]
    return config, exp_dir


def build_training_command(*, config_path: Path, target_steps: int, num_workers: int = 0) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "scripts/run_mortal_dqn_offline.py",
        "--config",
        str(config_path),
        "--target-steps",
        str(int(target_steps)),
        "--num-workers",
        str(int(num_workers)),
    ]


def write_experiment_configs(
    *,
    base_config_path: Path,
    output_root: Path,
    experiment_ids: Sequence[str],
    checkpoint_paths: Mapping[str, Path],
    log_globs: Mapping[str, str],
    train_steps: int,
    target_steps: int | None,
    copy_parent_checkpoint: bool,
    dry_run: bool,
) -> dict[str, Any]:
    base_config = load_toml(base_config_path)
    matrix = matrix_by_id()
    output_rows: list[dict[str, Any]] = []
    for experiment_id in experiment_ids:
        parent_label, notes, data_labels = matrix[experiment_id]
        parent_checkpoint = checkpoint_paths[parent_label]
        parent_steps = read_checkpoint_steps(parent_checkpoint)
        effective_target = int(target_steps) if target_steps is not None else int(parent_steps) + int(train_steps)
        if effective_target <= parent_steps:
            raise ValueError(
                f"target_steps must be greater than parent_steps for {experiment_id}: "
                f"target_steps={effective_target}, parent_steps={parent_steps}"
            )
        dataset_globs = [log_globs[label] for label in data_labels]
        config, exp_dir = prepare_config(
            base_config,
            experiment_id=experiment_id,
            output_root=output_root,
            dataset_globs=dataset_globs,
        )
        config_path = exp_dir / "config.toml"
        state_file = Path(config["control"]["state_file"])
        manifest = {
            "schema": "keqing.mortal.selfplay_finetune_config.v1",
            "experiment_id": experiment_id,
            "parent_label": parent_label,
            "parent_checkpoint": str(parent_checkpoint),
            "parent_steps": int(parent_steps),
            "train_steps": int(train_steps),
            "effective_train_steps": int(effective_target) - int(parent_steps),
            "target_steps": int(effective_target),
            "reward_profile": "mortal_default",
            "pt_table": [float(value) for value in RANK_POINT_PROFILES["mortal_default"]],
            "grp_checkpoint": str(config.get("grp", {}).get("state_file", "")),
            "training_data_labels": list(data_labels),
            "training_data": dataset_globs,
            "style_data": "selfplay_default_70k_80k_behavior_domains",
            "config": str(config_path),
            "state_file": str(state_file),
            "training_command": build_training_command(config_path=config_path, target_steps=effective_target),
            "notes": notes,
        }
        output_rows.append(manifest)
        if dry_run:
            continue
        exp_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(dump_toml(config), encoding="utf-8")
        (exp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        if copy_parent_checkpoint:
            if not parent_checkpoint.exists():
                raise FileNotFoundError(f"parent checkpoint not found: {parent_checkpoint}")
            if state_file.exists():
                raise FileExistsError(f"refusing to overwrite existing experiment checkpoint: {state_file}")
            shutil.copy2(parent_checkpoint, state_file)
    return {
        "schema": "keqing.mortal.selfplay_finetune_prepare.v1",
        "base_config": str(base_config_path),
        "output_root": str(output_root),
        "dry_run": bool(dry_run),
        "copy_parent_checkpoint": bool(copy_parent_checkpoint),
        "experiments": output_rows,
    }


def main() -> None:
    args = parse_args()
    report = write_experiment_configs(
        base_config_path=args.base_config,
        output_root=args.output_root,
        experiment_ids=parse_matrix(args.matrix),
        checkpoint_paths=checkpoint_map(args.checkpoint_70k, args.checkpoint_80k),
        log_globs=log_glob_map(args.logs_70k, args.logs_80k),
        train_steps=args.train_steps,
        target_steps=args.target_steps,
        copy_parent_checkpoint=bool(args.copy_parent_checkpoint),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
