#!/usr/bin/env python3
"""Prepare isolated Mortal DQN configs for reward/pt-table experiments."""
# ruff: noqa: E402

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

DEFAULT_MATRIX: tuple[tuple[str, str], ...] = (
    ("R0_base", "base"),
    ("R1_avoid4_strong", "avoid4_strong"),
    ("R2_top1_heavy", "top1_heavy"),
    ("R3_zero_sum_balanced", "zero_sum_balanced"),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Mortal reward/pt-table experiment configs")
    parser.add_argument("--base-config", type=Path, default=Path("artifacts/mortal_training/config.toml"))
    parser.add_argument("--parent-checkpoint", type=Path, default=Path("artifacts/mortal_training/mortal.pth"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments/reward_pt_2026_05"))
    parser.add_argument("--target-steps", type=int, default=65000)
    parser.add_argument("--train-steps", type=int, default=5000)
    parser.add_argument("--matrix", default=",".join(f"{exp}:{profile}" for exp, profile in DEFAULT_MATRIX))
    parser.add_argument("--copy-parent-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def parse_matrix(value: str) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if ":" not in raw:
            raise ValueError(f"matrix item must be EXPERIMENT_ID:PROFILE, got {raw!r}")
        experiment_id, profile = [part.strip() for part in raw.split(":", 1)]
        if not experiment_id or not profile:
            raise ValueError(f"matrix item must be EXPERIMENT_ID:PROFILE, got {raw!r}")
        if profile not in RANK_POINT_PROFILES or profile == "tenhou_reference":
            known = ", ".join(profile for profile in sorted(RANK_POINT_PROFILES) if profile != "tenhou_reference")
            raise ValueError(f"unknown training reward profile {profile!r}; known profiles: {known}")
        rows.append((experiment_id, profile))
    if not rows:
        raise ValueError("matrix must contain at least one experiment")
    return rows


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def prepare_config(
    base_config: Mapping[str, Any],
    *,
    experiment_id: str,
    reward_profile: str,
    output_root: Path,
) -> tuple[dict[str, Any], Path]:
    if reward_profile not in RANK_POINT_PROFILES or reward_profile == "tenhou_reference":
        raise ValueError(f"unsupported reward profile for training: {reward_profile!r}")
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
    dataset["file_index"] = str((exp_dir / "file_index.pth").resolve())
    env = config.setdefault("env", {})
    env["pts"] = [float(value) for value in RANK_POINT_PROFILES[reward_profile]]
    if isinstance(config.get("online"), dict) and isinstance(config["online"].get("server"), dict):
        config["online"]["server"]["buffer_dir"] = str((exp_dir / "buffer").resolve())
        config["online"]["server"]["drain_dir"] = str((exp_dir / "drain").resolve())
    if isinstance(config.get("1v3"), dict):
        config["1v3"]["log_dir"] = str((exp_dir / "1v3").resolve())
        if isinstance(config["1v3"].get("challenger"), dict):
            config["1v3"]["challenger"]["state_file"] = control["state_file"]
    return config, exp_dir


def build_training_command(*, config_path: Path, target_steps: int) -> list[str]:
    return [
        "uv",
        "run",
        "python",
        "scripts/run_mortal_dqn_offline.py",
        "--config",
        str(config_path),
        "--target-steps",
        str(int(target_steps)),
    ]


def write_experiment_configs(
    *,
    base_config_path: Path,
    parent_checkpoint: Path,
    output_root: Path,
    matrix: Sequence[tuple[str, str]],
    target_steps: int,
    train_steps: int,
    copy_parent_checkpoint: bool,
    dry_run: bool,
) -> dict[str, Any]:
    base_config = load_toml(base_config_path)
    output_rows: list[dict[str, Any]] = []
    for experiment_id, reward_profile in matrix:
        config, exp_dir = prepare_config(
            base_config,
            experiment_id=experiment_id,
            reward_profile=reward_profile,
            output_root=output_root,
        )
        config_path = exp_dir / "config.toml"
        state_file = Path(config["control"]["state_file"])
        manifest = {
            "schema": "keqing.mortal.reward_pt_experiment_config.v1",
            "experiment_id": experiment_id,
            "parent_checkpoint": str(parent_checkpoint),
            "reward_profile": reward_profile,
            "pt_table": [float(value) for value in RANK_POINT_PROFILES[reward_profile]],
            "grp_checkpoint": str(config.get("grp", {}).get("state_file", "")),
            "training_data": list(config.get("dataset", {}).get("globs", [])),
            "style_data": None,
            "train_steps": int(train_steps),
            "target_steps": int(target_steps),
            "config": str(config_path),
            "state_file": str(state_file),
            "training_command": build_training_command(config_path=config_path, target_steps=target_steps),
            "notes": "Mortal/libriichi reward-only pt-table short run; no obs/model-head changes.",
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
        "schema": "keqing.mortal.reward_pt_experiment_prepare.v1",
        "base_config": str(base_config_path),
        "parent_checkpoint": str(parent_checkpoint),
        "output_root": str(output_root),
        "dry_run": bool(dry_run),
        "copy_parent_checkpoint": bool(copy_parent_checkpoint),
        "experiments": output_rows,
    }


def dump_toml(data: Mapping[str, Any]) -> str:
    lines: list[str] = []
    _dump_toml_table(lines, [], data)
    return "\n".join(lines).rstrip() + "\n"


def _dump_toml_table(lines: list[str], prefix: list[str], table: Mapping[str, Any]) -> None:
    scalars: list[tuple[str, Any]] = []
    children: list[tuple[str, Mapping[str, Any]]] = []
    for key, value in table.items():
        if isinstance(value, Mapping):
            children.append((str(key), value))
        else:
            scalars.append((str(key), value))
    if prefix:
        if lines and lines[-1] != "":
            lines.append("")
        lines.append("[" + ".".join(prefix) + "]")
    for key, value in scalars:
        lines.append(f"{key} = {_toml_value(value)}")
    for key, child in children:
        _dump_toml_table(lines, [*prefix, key], child)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, float):
        return repr(float(value))
    if isinstance(value, str):
        return json.dumps(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    if value is None:
        raise TypeError("TOML does not support null values")
    raise TypeError(f"unsupported TOML value type: {type(value).__name__}")


def main() -> None:
    args = _parse_args()
    report = write_experiment_configs(
        base_config_path=args.base_config,
        parent_checkpoint=args.parent_checkpoint,
        output_root=args.output_root,
        matrix=parse_matrix(str(args.matrix)),
        target_steps=int(args.target_steps),
        train_steps=int(args.train_steps),
        copy_parent_checkpoint=bool(args.copy_parent_checkpoint),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
