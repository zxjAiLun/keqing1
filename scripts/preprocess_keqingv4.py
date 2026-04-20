#!/usr/bin/env python3
"""keqingv4 preprocess entrypoint.

Rust now owns orchestration/export; Python remains the semantic implementation
backend for the first transition version.
"""

from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path
import sys
import tempfile

import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess keqingv4 caches via Rust orchestrator")
    parser.add_argument("--config", default="configs/keqingv4_preprocess.yaml")
    parser.add_argument("--data-dirs", "--data_dirs", nargs="+", default=None)
    parser.add_argument("--output-dir", "--output_dir", default=None)
    parser.add_argument("--progress-every", "--progress_every", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--preflight-files", "--preflight_files", type=int, default=None)
    parser.add_argument("--preflight-seed", "--preflight_seed", type=int, default=None)
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_paths(root: Path, values: list[str] | None) -> list[Path]:
    resolved: list[Path] = []
    for raw in values or []:
        path = Path(raw)
        resolved.append(path if path.is_absolute() else root / path)
    return resolved


def _resolve_jobs(cfg: dict, cli_jobs: int | None) -> int:
    if cli_jobs is not None:
        return int(cli_jobs)
    if "jobs" in cfg:
        return int(cfg.get("jobs", 0))
    # Keep backward compatibility with older preprocess configs.
    return int(cfg.get("workers", 0))


def _build_preflight_sample_dirs(
    temp_root: Path,
    data_dirs: list[Path],
    *,
    files_per_shard: int,
    seed: int,
) -> tuple[list[Path], dict[str, int]]:
    rng = random.Random(seed)
    sampled_dirs: list[Path] = []
    sampled_counts: dict[str, int] = {}
    input_root = temp_root / "input"
    input_root.mkdir(parents=True, exist_ok=True)
    for data_dir in data_dirs:
        if not data_dir.exists():
            raise RuntimeError(f"keqingv4 preprocess preflight: data dir does not exist: {data_dir}")
        if not data_dir.is_dir():
            raise RuntimeError(f"keqingv4 preprocess preflight: data dir is not a directory: {data_dir}")
        files = sorted(data_dir.glob("*.mjson"))
        if not files:
            continue
        sample_size = min(len(files), max(1, files_per_shard))
        selected = sorted(rng.sample(files, sample_size)) if len(files) > sample_size else files
        shard_dir = input_root / data_dir.name
        shard_dir.mkdir(parents=True, exist_ok=True)
        for source in selected:
            (shard_dir / source.name).symlink_to(source.resolve())
        sampled_dirs.append(shard_dir)
        sampled_counts[data_dir.name] = len(selected)
    return sampled_dirs, sampled_counts


def _run_export_gate(
    *,
    root: Path,
    label: str,
    data_dirs: list[Path],
    output_dir: Path,
    progress_every: int,
    jobs: int,
    force: bool,
    smoke: bool,
    inspect_keqingv4_cached_contract,
    assert_keqingv4_cached_contract,
) -> None:
    cmd = [
        "cargo",
        "run",
        "--release",
        "--quiet",
        "--manifest-path",
        str(root / "rust/keqing_core/Cargo.toml"),
        "--bin",
        "keqingv4_export",
        "--",
        "--output-dir",
        str(output_dir),
        "--progress-every",
        str(progress_every),
        "--jobs",
        str(jobs),
    ]
    for data_dir in data_dirs:
        cmd.extend(["--data-dir", str(data_dir)])
    if force:
        cmd.append("--force")
    if smoke:
        cmd.append("--smoke")

    print(
        f"{label}: data_dirs={[str(p) for p in data_dirs]} output_dir={output_dir} smoke={smoke}",
        flush=True,
    )
    print(f"{label} launcher -> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=root, check=True)

    cache_files = sorted(output_dir.glob("ds*/*.npz"))
    if not cache_files:
        raise RuntimeError(f"{label} gate: no exported cache files found under {output_dir}")
    inspected = inspect_keqingv4_cached_contract(cache_files)
    assert_keqingv4_cached_contract(
        inspected,
        smoke=smoke,
        allow_stale_cache=False,
    )
    print(
        f"{label} gate: files_scanned={inspected['files_scanned']} "
        f"summary_dims={sorted(inspected['summary_dims'])} "
        f"call_slots={sorted(inspected['call_summary_slots'])} "
        f"special_slots={sorted(inspected['special_summary_slots'])} "
        f"event_history_shapes={sorted(inspected['event_history_shapes'])} "
        f"opportunity_shapes={sorted(inspected['opportunity_shapes'])} "
        f"opp_tenpai_shapes={sorted(inspected['opp_tenpai_shapes'])}",
        flush=True,
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = _load_cfg(cfg_path if cfg_path.is_absolute() else root / cfg_path)
    from keqingv4.cache_contract import (
        assert_keqingv4_cached_contract,
        inspect_keqingv4_cached_contract,
    )

    data_dirs = _resolve_paths(root, args.data_dirs or cfg.get("data_dirs", []))
    output_dir = _resolve_paths(root, [args.output_dir or cfg.get("output_dir", "processed_v4")])[0]
    progress_every = (
        args.progress_every
        if args.progress_every is not None
        else int(cfg.get("progress_every", 20))
    )
    jobs = _resolve_jobs(cfg, args.jobs)
    preflight_files = (
        args.preflight_files
        if args.preflight_files is not None
        else int(cfg.get("preflight_files_per_shard", 1))
    )
    preflight_seed = (
        args.preflight_seed
        if args.preflight_seed is not None
        else int(cfg.get("preflight_seed", 20260419))
    )
    if args.smoke:
        data_dirs = [path for path in data_dirs if path.exists()]
    if (not args.smoke) and (not args.skip_preflight) and preflight_files > 0:
        with tempfile.TemporaryDirectory(prefix="keqingv4_preflight_") as temp_dir:
            temp_root = Path(temp_dir)
            sampled_dirs, sampled_counts = _build_preflight_sample_dirs(
                temp_root,
                data_dirs,
                files_per_shard=preflight_files,
                seed=preflight_seed,
            )
            if not sampled_dirs:
                raise RuntimeError("keqingv4 preprocess preflight: no sampleable .mjson files found")
            print(
                "keqingv4 preprocess preflight selection: "
                f"sampled_counts={sampled_counts} "
                f"seed={preflight_seed}",
                flush=True,
            )
            _run_export_gate(
                root=root,
                label="keqingv4 preprocess preflight",
                data_dirs=sampled_dirs,
                output_dir=temp_root / "output",
                progress_every=max(1, min(progress_every, 1)),
                jobs=jobs,
                force=True,
                smoke=False,
                inspect_keqingv4_cached_contract=inspect_keqingv4_cached_contract,
                assert_keqingv4_cached_contract=assert_keqingv4_cached_contract,
            )
    _run_export_gate(
        root=root,
        label="keqingv4 preprocess",
        data_dirs=data_dirs,
        output_dir=output_dir,
        progress_every=progress_every,
        jobs=jobs,
        force=args.force,
        smoke=args.smoke,
        inspect_keqingv4_cached_contract=inspect_keqingv4_cached_contract,
        assert_keqingv4_cached_contract=assert_keqingv4_cached_contract,
    )


if __name__ == "__main__":
    main()
