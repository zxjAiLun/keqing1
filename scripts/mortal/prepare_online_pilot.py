#!/usr/bin/env python3
"""Prepare isolated Mortal online pilot configs."""

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


DEFAULT_ANCHOR_70K = Path("artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path("artifacts/mortal_training/config.toml"))
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments/online_phase2_2026_05"))
    parser.add_argument("--experiment-id", default="O1_70k_online_control")
    parser.add_argument("--anchor-checkpoint", type=Path, default=DEFAULT_ANCHOR_70K)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--train-play-games", type=int, default=800)
    parser.add_argument("--server-capacity", type=int, default=1600)
    parser.add_argument("--save-every", type=int, default=400)
    parser.add_argument("--submit-every", type=int, default=400)
    parser.add_argument("--test-every", type=int, default=20000)
    parser.add_argument("--copy-parent-checkpoint", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def prepare_online_config(
    base_config: Mapping[str, Any],
    *,
    exp_dir: Path,
    anchor_checkpoint: Path,
    host: str,
    port: int,
    train_play_games: int,
    server_capacity: int,
    save_every: int,
    submit_every: int,
    test_every: int,
) -> dict[str, Any]:
    if train_play_games <= 0 or train_play_games % 4 != 0:
        raise ValueError(f"train_play_games must be a positive multiple of 4, got {train_play_games}")
    if server_capacity <= 0:
        raise ValueError(f"server_capacity must be positive, got {server_capacity}")
    if save_every <= 0 or submit_every <= 0 or test_every <= 0:
        raise ValueError("save_every, submit_every, and test_every must be positive")
    if test_every % save_every != 0:
        raise ValueError(f"test_every must be a multiple of save_every, got {test_every} and {save_every}")

    config = copy.deepcopy(dict(base_config))
    control = config.setdefault("control", {})
    control["online"] = True
    control["state_file"] = str((exp_dir / "mortal.pth").resolve())
    control["best_state_file"] = str((exp_dir / "mortal_best.pth").resolve())
    control["tensorboard_dir"] = str((exp_dir / "tb_mortal").resolve())
    control["save_every"] = int(save_every)
    control["submit_every"] = int(submit_every)
    control["test_every"] = int(test_every)

    freeze_bn = config.setdefault("freeze_bn", {})
    freeze_bn["mortal"] = True

    train_play = config.setdefault("train_play", {}).setdefault("default", {})
    train_play["games"] = int(train_play_games)
    train_play["log_dir"] = str((exp_dir / "train_play").resolve())

    if isinstance(config.get("test_play"), dict):
        config["test_play"]["log_dir"] = str((exp_dir / "test_play").resolve())

    dataset = config.setdefault("dataset", {})
    dataset["file_index"] = str((exp_dir / "file_index.pth").resolve())

    baseline = config.setdefault("baseline", {})
    for key in ("train", "test"):
        branch = baseline.setdefault(key, {})
        branch["state_file"] = str(anchor_checkpoint.resolve())

    online = config.setdefault("online", {})
    remote = online.setdefault("remote", {})
    remote["host"] = str(host)
    remote["port"] = int(port)
    server = online.setdefault("server", {})
    server["buffer_dir"] = str((exp_dir / "buffer").resolve())
    server["drain_dir"] = str((exp_dir / "drain").resolve())
    server["capacity"] = int(server_capacity)

    if isinstance(config.get("1v3"), dict):
        config["1v3"]["log_dir"] = str((exp_dir / "1v3").resolve())
        if isinstance(config["1v3"].get("challenger"), dict):
            config["1v3"]["challenger"]["state_file"] = control["state_file"]
        if isinstance(config["1v3"].get("champion"), dict):
            config["1v3"]["champion"]["state_file"] = str(anchor_checkpoint.resolve())

    return config


def command_with_env(config_path: Path, script_path: str) -> list[str]:
    return ["env", f"MORTAL_CFG={config_path.resolve()}", "uv", "run", "python", script_path]


def prepare_online_pilot(
    *,
    base_config_path: Path,
    output_root: Path,
    experiment_id: str,
    anchor_checkpoint: Path,
    host: str,
    port: int,
    train_play_games: int,
    server_capacity: int,
    save_every: int,
    submit_every: int,
    test_every: int,
    copy_parent_checkpoint: bool,
    dry_run: bool,
) -> dict[str, Any]:
    parent_steps = read_checkpoint_steps(anchor_checkpoint)
    exp_dir = output_root / experiment_id
    config_path = exp_dir / "config.toml"
    config = prepare_online_config(
        load_toml(base_config_path),
        exp_dir=exp_dir,
        anchor_checkpoint=anchor_checkpoint,
        host=host,
        port=port,
        train_play_games=train_play_games,
        server_capacity=server_capacity,
        save_every=save_every,
        submit_every=submit_every,
        test_every=test_every,
    )
    state_file = Path(config["control"]["state_file"])
    manifest = {
        "schema": "keqing.mortal.online_pilot_config.v1",
        "experiment_id": experiment_id,
        "parent_checkpoint": str(anchor_checkpoint),
        "parent_steps": int(parent_steps),
        "role": "online_control",
        "research_question": "Does Mortal online replay refresh avoid the negative drift seen in static selfplay fine-tune?",
        "config": str(config_path),
        "state_file": str(state_file),
        "baseline_checkpoint": str(anchor_checkpoint.resolve()),
        "control_online": True,
        "freeze_bn_mortal": True,
        "train_play_games": int(train_play_games),
        "server_capacity": int(server_capacity),
        "save_every": int(save_every),
        "submit_every": int(submit_every),
        "test_every": int(test_every),
        "checkpoints_to_read": [int(parent_steps), int(parent_steps) + 400, int(parent_steps) + 800, int(parent_steps) + 1200],
        "commands": {
            "server": command_with_env(config_path, "third_party/Mortal/mortal/server.py"),
            "trainer": command_with_env(config_path, "third_party/Mortal/mortal/train.py"),
            "client": command_with_env(config_path, "third_party/Mortal/mortal/client.py"),
        },
        "notes": [
            "Start server first, then trainer, then one or more clients.",
            "When switching an offline checkpoint into online mode, Mortal loads model weights but resets optimizer/scheduler state.",
            "The baseline train/test checkpoint is fixed to the 70k anchor.",
        ],
    }
    if dry_run:
        return manifest

    exp_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(dump_toml(config), encoding="utf-8")
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if copy_parent_checkpoint:
        if state_file.exists():
            raise FileExistsError(f"refusing to overwrite existing online pilot checkpoint: {state_file}")
        shutil.copy2(anchor_checkpoint, state_file)
    return manifest


def main() -> None:
    args = parse_args()
    manifest = prepare_online_pilot(
        base_config_path=args.base_config,
        output_root=args.output_root,
        experiment_id=str(args.experiment_id),
        anchor_checkpoint=args.anchor_checkpoint,
        host=str(args.host),
        port=int(args.port),
        train_play_games=int(args.train_play_games),
        server_capacity=int(args.server_capacity),
        save_every=int(args.save_every),
        submit_every=int(args.submit_every),
        test_every=int(args.test_every),
        copy_parent_checkpoint=bool(args.copy_parent_checkpoint),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
