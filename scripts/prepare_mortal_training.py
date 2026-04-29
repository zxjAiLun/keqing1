#!/usr/bin/env python3
"""Prepare local mjai logs for Mortal offline training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import gzip
import json
from pathlib import Path
import random
import shutil
from typing import Any, Sequence


MORTAL_TRAINING_DATASET_CONTRACT_VERSION = "mortal_training_dataset_v1"


@dataclass(frozen=True)
class PreparedMortalFile:
    split: str
    source: str
    target: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare gzip mjai dataset and config for Mortal training")
    parser.add_argument("--source-dir", type=Path, default=Path("artifacts/converted_mjai"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/mortal_mjai_gz"))
    parser.add_argument("--training-dir", type=Path, default=Path("artifacts/mortal_training"))
    parser.add_argument(
        "--exclude-dir",
        action="append",
        default=[],
        help="Relative source subdirectory to exclude from packaging; may be repeated, e.g. --exclude-dir ds3",
    )
    parser.add_argument("--config-path", type=Path, default=None)
    parser.add_argument("--manifest-path", type=Path, default=None)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--grp-batch-size", type=int, default=512)
    parser.add_argument("--mortal-save-every", type=int, default=400)
    parser.add_argument("--mortal-test-every", type=int, default=20000)
    parser.add_argument("--grp-save-every", type=int, default=2000)
    parser.add_argument("--grp-val-steps", type=int, default=400)
    parser.add_argument("--baseline-state-file", type=Path, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def prepare_mortal_training(
    *,
    source_dir: Path,
    output_dir: Path,
    training_dir: Path,
    config_path: Path | None = None,
    manifest_path: Path | None = None,
    val_ratio: float = 0.05,
    seed: int = 20260428,
    limit: int | None = None,
    device: str = "cuda:0",
    batch_size: int = 512,
    grp_batch_size: int = 512,
    mortal_save_every: int = 400,
    mortal_test_every: int = 20000,
    grp_save_every: int = 2000,
    grp_val_steps: int = 400,
    baseline_state_file: Path | None = None,
    exclude_dirs: Sequence[str | Path] = (),
    skip_existing: bool = True,
    dry_run: bool = False,
) -> dict[str, Any]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    training_dir = training_dir.resolve()
    config_path = (training_dir / "config.toml" if config_path is None else config_path).resolve()
    manifest_path = (training_dir / "manifest.json" if manifest_path is None else manifest_path).resolve()
    baseline_state_file = (
        training_dir / "mortal_baseline_required.pth" if baseline_state_file is None else baseline_state_file
    ).resolve()

    normalized_exclude_dirs = _normalize_exclude_dirs(exclude_dirs)
    source_files = discover_mjson_files(source_dir, exclude_dirs=normalized_exclude_dirs)
    if limit is not None:
        if int(limit) <= 0:
            raise ValueError(f"limit must be positive when provided, got {limit}")
        source_files = source_files[: int(limit)]
    train_rel, val_rel = split_relative_paths(source_files, source_dir=source_dir, val_ratio=val_ratio, seed=seed)

    prepared_files: list[PreparedMortalFile] = []
    prepared_files.extend(
        _prepare_split(
            source_dir=source_dir,
            output_dir=output_dir,
            split="train",
            rel_paths=train_rel,
            skip_existing=skip_existing,
            dry_run=dry_run,
        )
    )
    prepared_files.extend(
        _prepare_split(
            source_dir=source_dir,
            output_dir=output_dir,
            split="val",
            rel_paths=val_rel,
            skip_existing=skip_existing,
            dry_run=dry_run,
        )
    )

    config_text = render_mortal_config(
        output_dir=output_dir,
        training_dir=training_dir,
        baseline_state_file=baseline_state_file,
        device=device,
        batch_size=batch_size,
        grp_batch_size=grp_batch_size,
        mortal_save_every=mortal_save_every,
        mortal_test_every=mortal_test_every,
        grp_save_every=grp_save_every,
        grp_val_steps=grp_val_steps,
    )
    manifest = {
        "contract_version": MORTAL_TRAINING_DATASET_CONTRACT_VERSION,
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "training_dir": str(training_dir),
        "config_path": str(config_path),
        "manifest_path": str(manifest_path),
        "seed": int(seed),
        "val_ratio": float(val_ratio),
        "source_count": len(source_files),
        "exclude_dirs": [path.as_posix() for path in normalized_exclude_dirs],
        "train_count": len(train_rel),
        "val_count": len(val_rel),
        "skip_existing": bool(skip_existing),
        "dry_run": bool(dry_run),
        "baseline_state_file": str(baseline_state_file),
        "baseline_state_file_exists": baseline_state_file.exists(),
        "files": [file.__dict__ for file in prepared_files],
    }

    if not dry_run:
        training_dir.mkdir(parents=True, exist_ok=True)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(config_text, encoding="utf-8")
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        manifest["config_text"] = config_text
    return manifest


def discover_mjson_files(source_dir: Path, *, exclude_dirs: Sequence[str | Path] = ()) -> list[Path]:
    if not source_dir.exists():
        raise FileNotFoundError(f"source directory does not exist: {source_dir}")
    normalized_exclude_dirs = _normalize_exclude_dirs(exclude_dirs)
    files = sorted(
        path
        for path in source_dir.rglob("*.mjson")
        if path.is_file() and not _is_excluded_source(path, source_dir=source_dir, exclude_dirs=normalized_exclude_dirs)
    )
    if not files:
        raise FileNotFoundError(f"no .mjson files found under {source_dir}")
    return files


def _normalize_exclude_dirs(exclude_dirs: Sequence[str | Path]) -> tuple[Path, ...]:
    normalized: list[Path] = []
    for raw in exclude_dirs:
        path = Path(raw)
        if path.is_absolute():
            raise ValueError(f"exclude-dir must be relative to source-dir, got absolute path: {path}")
        parts = tuple(part for part in path.parts if part not in {"", "."})
        if not parts:
            continue
        if any(part == ".." for part in parts):
            raise ValueError(f"exclude-dir must stay under source-dir, got: {path}")
        normalized.append(Path(*parts))
    return tuple(dict.fromkeys(normalized))


def _is_excluded_source(path: Path, *, source_dir: Path, exclude_dirs: Sequence[Path]) -> bool:
    if not exclude_dirs:
        return False
    rel_path = path.relative_to(source_dir)
    rel_parts = rel_path.parts
    for excluded in exclude_dirs:
        excluded_parts = excluded.parts
        if rel_parts[: len(excluded_parts)] == excluded_parts:
            return True
    return False


def split_relative_paths(
    files: Sequence[Path],
    *,
    source_dir: Path,
    val_ratio: float,
    seed: int,
) -> tuple[list[Path], list[Path]]:
    if not 0.0 <= float(val_ratio) < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    rel_paths = [path.relative_to(source_dir) for path in files]
    shuffled = list(rel_paths)
    random.Random(int(seed)).shuffle(shuffled)
    if len(shuffled) <= 1 or float(val_ratio) <= 0.0:
        val_count = 0
    else:
        val_count = max(1, round(len(shuffled) * float(val_ratio)))
        val_count = min(val_count, len(shuffled) - 1)
    val_set = set(shuffled[:val_count])
    train_rel = sorted(path for path in rel_paths if path not in val_set)
    val_rel = sorted(path for path in rel_paths if path in val_set)
    return train_rel, val_rel


def render_mortal_config(
    *,
    output_dir: Path,
    training_dir: Path,
    baseline_state_file: Path,
    device: str,
    batch_size: int,
    grp_batch_size: int,
    mortal_save_every: int,
    mortal_test_every: int,
    grp_save_every: int,
    grp_val_steps: int,
) -> str:
    train_glob = output_dir / "train" / "**" / "*.json.gz"
    val_glob = output_dir / "val" / "**" / "*.json.gz"
    return f"""# Generated by scripts/prepare_mortal_training.py.
# Mortal is the only allowed teacher source for KeqingRL.

[control]
version = 4
online = false
state_file = {_toml_string(training_dir / "mortal.pth")}
best_state_file = {_toml_string(training_dir / "mortal_best.pth")}
tensorboard_dir = {_toml_string(training_dir / "tb_mortal")}
device = {_toml_string(device)}
enable_cudnn_benchmark = false
enable_amp = false
enable_compile = false
batch_size = {int(batch_size)}
opt_step_every = 1
save_every = {int(mortal_save_every)}
test_every = {int(mortal_test_every)}
submit_every = {int(mortal_save_every)}

[train_play.default]
games = 800
log_dir = {_toml_string(training_dir / "train_play")}
boltzmann_epsilon = 0.005
boltzmann_temp = 0.05
top_p = 1.0
repeats = 1

[test_play]
games = 3000
log_dir = {_toml_string(training_dir / "test_play")}

[dataset]
globs = [{_toml_string(train_glob)}]
file_index = {_toml_string(training_dir / "file_index.pth")}
file_batch_size = 15
reserve_ratio = 0.0
num_workers = 1
player_names_files = []
num_epochs = 1
enable_augmentation = false
augmented_first = false

[env]
gamma = 1
pts = [6.0, 4.0, 2.0, 0.0]

[resnet]
conv_channels = 192
num_blocks = 40

[cql]
min_q_weight = 5

[aux]
next_rank_weight = 0.2

[freeze_bn]
mortal = false

[optim]
eps = 1e-8
betas = [0.9, 0.999]
weight_decay = 0.1
max_grad_norm = 0

[optim.scheduler]
peak = 1e-4
final = 1e-4
warm_up_steps = 0
max_steps = 0

[baseline.train]
device = {_toml_string(device)}
enable_compile = false
state_file = {_toml_string(baseline_state_file)}

[baseline.test]
device = {_toml_string(device)}
enable_compile = false
state_file = {_toml_string(baseline_state_file)}

[online]
history_window = 50
enable_compile = false

[online.remote]
host = "127.0.0.1"
port = 5000

[online.server]
buffer_dir = {_toml_string(training_dir / "buffer")}
drain_dir = {_toml_string(training_dir / "drain")}
sample_reuse_rate = 0
sample_reuse_threshold = 0
capacity = 1600
force_sequential = false

[1v3]
seed_key = -1
games_per_iter = 2000
iters = 500
log_dir = {_toml_string(training_dir / "1v3")}

[1v3.challenger]
device = {_toml_string(device)}
name = "mortal"
state_file = {_toml_string(training_dir / "mortal.pth")}
stochastic_latent = false
enable_compile = false
enable_amp = true
enable_rule_based_agari_guard = true

[1v3.champion]
device = {_toml_string(device)}
name = "baseline"
state_file = {_toml_string(baseline_state_file)}
stochastic_latent = false
enable_compile = false
enable_amp = true
enable_rule_based_agari_guard = true

[1v3.akochan]
enabled = false
dir = {_toml_string(training_dir / "akochan")}
tactics = {_toml_string(training_dir / "akochan" / "tactics.json")}

[grp]
state_file = {_toml_string(training_dir / "grp.pth")}

[grp.network]
hidden_size = 64
num_layers = 2

[grp.control]
device = {_toml_string(device)}
enable_cudnn_benchmark = false
tensorboard_dir = {_toml_string(training_dir / "tb_grp")}
batch_size = {int(grp_batch_size)}
save_every = {int(grp_save_every)}
val_steps = {int(grp_val_steps)}

[grp.dataset]
train_globs = [{_toml_string(train_glob)}]
val_globs = [{_toml_string(val_glob)}]
file_index = {_toml_string(training_dir / "grp_file_index.pth")}
file_batch_size = 50

[grp.optim]
lr = 1e-5
"""


def _prepare_split(
    *,
    source_dir: Path,
    output_dir: Path,
    split: str,
    rel_paths: Sequence[Path],
    skip_existing: bool,
    dry_run: bool,
) -> list[PreparedMortalFile]:
    prepared: list[PreparedMortalFile] = []
    for rel_path in rel_paths:
        source = source_dir / rel_path
        target_rel = rel_path.with_suffix(".json.gz")
        target = output_dir / split / target_rel
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            if not (skip_existing and target.exists()):
                with source.open("rb") as src, gzip.open(target, "wb", compresslevel=6) as dst:
                    shutil.copyfileobj(src, dst)
        prepared.append(
            PreparedMortalFile(
                split=split,
                source=rel_path.as_posix(),
                target=(Path(split) / target_rel).as_posix(),
            )
        )
    return prepared


def _toml_string(value: object) -> str:
    return json.dumps(str(value))


def main() -> None:
    args = _parse_args()
    manifest = prepare_mortal_training(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        training_dir=args.training_dir,
        config_path=args.config_path,
        manifest_path=args.manifest_path,
        val_ratio=args.val_ratio,
        seed=args.seed,
        limit=args.limit,
        device=args.device,
        batch_size=args.batch_size,
        grp_batch_size=args.grp_batch_size,
        mortal_save_every=args.mortal_save_every,
        mortal_test_every=args.mortal_test_every,
        grp_save_every=args.grp_save_every,
        grp_val_steps=args.grp_val_steps,
        baseline_state_file=args.baseline_state_file,
        exclude_dirs=args.exclude_dir,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )
    print(json.dumps({key: value for key, value in manifest.items() if key != "files"}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
