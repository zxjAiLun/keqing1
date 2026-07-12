#!/usr/bin/env python3
"""Prepare the no-teacher-CE V2 mixed-ecology model_v4 warm-start experiment."""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import sys
import tomllib
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.prepare_reward_pt_experiments import dump_toml, read_checkpoint_steps


EXPERIMENT_ID = "V2_population_mixed_v4_warmstart_2026_07"
DEFAULT_OUTPUT_ROOT = Path("artifacts/experiments/model_pool_2026_07")
DEFAULT_DATA_ROOT = DEFAULT_OUTPUT_ROOT / "V2_data"
PARENT_CHECKPOINT = Path("artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth")
CHECKPOINTS = {
    "model_v4": "artifacts/model_v4_20240308_best_min.pth",
    "70k": "artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth",
    "80k_game": "artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth",
    "T1_71000": "artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth",
    "V0b_15000": "artifacts/experiments/v4_synthetic_2026_06/V0b_v4_synthetic_clean_2026_07/checkpoints/mortal_15000.pth",
    "V1_74000": "artifacts/experiments/v4_synthetic_2026_06/V1_v4_synthetic_warmstart_2026_06/checkpoints/mortal_v1_74000.pth",
}
POOL_SPECS = (
    ("v4_70k_t1_v0b_2000h", 960000, ("model_v4", "70k", "T1_71000", "V0b_15000")),
    ("v4_70k_v1_80k_2000h", 962000, ("model_v4", "70k", "V1_74000", "80k_game")),
    ("v4_v0b_v1_t1_2000h", 964000, ("model_v4", "V0b_15000", "V1_74000", "T1_71000")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-config", type=Path, default=Path("artifacts/mortal_training/config.toml"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--parent-checkpoint", type=Path, default=PARENT_CHECKPOINT)
    parser.add_argument("--initial-steps", type=int, default=70000)
    parser.add_argument("--stage1-steps", type=int, default=72000)
    parser.add_argument("--final-steps", type=int, default=74000)
    parser.add_argument("--model-seed", type=int, default=20260712)
    parser.add_argument("--data-seed", type=int, default=20260712)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _normalize_host_paths(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _normalize_host_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_host_paths(item) for item in value]
    if not isinstance(value, str) or os.name != "nt":
        return value
    if not value.startswith("/mnt/") or len(value) < 7 or value[6] != "/":
        return value
    return str(Path(f"{value[5].upper()}:/{value[7:]}").resolve())


def _load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def prepare_config(base_config: dict[str, Any], *, exp_dir: Path, data_root: Path) -> dict[str, Any]:
    config = _normalize_host_paths(copy.deepcopy(base_config))
    control = config.setdefault("control", {})
    control["state_file"] = str((exp_dir / "mortal.pth").resolve())
    control["best_state_file"] = str((exp_dir / "mortal_best.pth").resolve())
    control["tensorboard_dir"] = str((exp_dir / "tb_mortal").resolve())

    dataset = config.setdefault("dataset", {})
    dataset["globs"] = [str((data_root / pool_id / "logs" / "**" / "*.json.gz").resolve()) for pool_id, _, _ in POOL_SPECS]
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


def _native_command(*, pool_id: str, seed_start: int, labels: tuple[str, ...], output_dir: Path, games: int, resume: bool) -> list[str]:
    command = [
        "uv", "run", "--no-sync", "python", "scripts/mortal/four_player_native.py",
        "--require-cuda", "--device", "cuda", "--seat-mode", "random",
        "--seed-start", str(seed_start), "--seed-key", "8192", "--games", str(games),
        "--native-batch-games", "100", "--progress-every", "100", "--rank-points", "90,45,0,-135",
    ]
    for label in labels:
        command.extend(["--model", f"{label}={CHECKPOINTS[label]}"])
    command.extend(["--output-dir", str(output_dir)])
    if resume:
        command.append("--resume")
    return command


def _eval_command(*, checkpoint: Path, output_dir: Path, games: int, seed_start: int) -> list[str]:
    command = [
        "uv", "run", "--no-sync", "python", "scripts/mortal/four_player_native.py",
        "--require-cuda", "--device", "cuda", "--seat-mode", "random",
        "--seed-start", str(seed_start), "--seed-key", "8192", "--games", str(games),
        "--native-batch-games", "100", "--progress-every", "100", "--rank-points", "90,45,0,-135",
    ]
    for label in ("model_v4", "70k", "T1_71000"):
        command.extend(["--model", f"{label}={CHECKPOINTS[label]}"])
    command.extend(["--model", f"V2={checkpoint}", "--output-dir", str(output_dir)])
    return command


def main() -> None:
    args = parse_args()
    if not 0 <= int(args.initial_steps) < int(args.stage1_steps) < int(args.final_steps):
        raise ValueError("steps must satisfy 0 <= initial < stage1 < final")
    exp_dir = args.output_root / EXPERIMENT_ID
    config = prepare_config(_load_toml(args.base_config), exp_dir=exp_dir, data_root=args.data_root)
    config_path = exp_dir / "config.toml"
    checkpoints_dir = exp_dir / "checkpoints"
    parent_steps = read_checkpoint_steps(args.parent_checkpoint)
    pools = []
    for pool_id, seed_start, labels in POOL_SPECS:
        output_dir = args.data_root / pool_id
        pools.append({
            "pool_id": pool_id,
            "seed_start": seed_start,
            "games": 2000,
            "models": list(labels),
            "train_label": "model_v4",
            "smoke_command": _native_command(pool_id=pool_id, seed_start=seed_start, labels=labels, output_dir=output_dir, games=25, resume=False),
            "full_resume_command": _native_command(pool_id=pool_id, seed_start=seed_start, labels=labels, output_dir=output_dir, games=2000, resume=True),
        })
    train_base = [
        "uv", "run", "--no-sync", "python", "scripts/run_mortal_dqn_offline.py",
        "--config", str(config_path), "--device", "cuda", "--num-workers", "0",
        "--seed", str(args.model_seed), "--data-seed", str(args.data_seed),
        "--initialize-from", str(args.parent_checkpoint), "--initial-steps", str(args.initial_steps),
        "--archive-steps", f"{args.stage1_steps},{args.final_steps}", "--archive-dir", str(checkpoints_dir), "--log-every", "50",
    ]
    manifest = {
        "schema": "keqing.mortal.v2_population_mixed_warmstart.v1",
        "experiment_id": EXPERIMENT_ID,
        "initialization": {"parent_checkpoint": str(args.parent_checkpoint), "parent_steps": parent_steps, "initial_steps": args.initial_steps, "optimizer": "fresh", "data_stream": "fresh"},
        "objective": "offline DQN + CQL + next-rank auxiliary; teacher CE disabled",
        "train_labels": ["model_v4"],
        "pools": pools,
        "expected_unique_hanchans": 6000,
        "expected_trainable_v4_seat_hanchans": 6000,
        "config": str(config_path),
        "state_file": str(config["control"]["state_file"]),
        "dataset_audit_command": ["uv", "run", "--no-sync", "python", "scripts/mortal/audit_population_synthetic_dataset.py", "--data-root", str(args.data_root), "--output", str(exp_dir / "dataset_audit.json")],
        "training_command": [*train_base, "--target-steps", str(args.final_steps)],
        "stage1_archive": str(checkpoints_dir / f"mortal_{args.stage1_steps}.pth"),
        "final_archive": str(checkpoints_dir / f"mortal_{args.final_steps}.pth"),
        "stage1_eval": _eval_command(checkpoint=checkpoints_dir / f"mortal_{args.stage1_steps}.pth", output_dir=exp_dir / f"eval_250h_v2_{args.stage1_steps}", games=250, seed_start=966000),
        "final_eval": _eval_command(checkpoint=checkpoints_dir / f"mortal_{args.final_steps}.pth", output_dir=exp_dir / f"eval_500h_v2_{args.final_steps}", games=500, seed_start=967000),
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return
    exp_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(dump_toml(config), encoding="utf-8")
    (exp_dir / "v4_train_labels.txt").write_text("model_v4\n", encoding="utf-8")
    (exp_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
