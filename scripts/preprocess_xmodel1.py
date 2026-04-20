#!/usr/bin/env python3
"""Xmodel1 preprocess entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys
import tempfile

import yaml


def _requested_shards(paths: list[Path]) -> list[str]:
    shards = sorted({path.name for path in paths if path.name.startswith("ds")})
    return shards


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Xmodel1 caches via Rust full exporter")
    parser.add_argument("--config", default="configs/xmodel1_preprocess.yaml")
    parser.add_argument("--data-dirs", "--data_dirs", nargs="+", default=None)
    parser.add_argument("--output-dir", "--output_dir", default=None)
    parser.add_argument("--progress-every", "--progress_every", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--limit-files", "--limit_files", type=int, default=None)
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
            raise RuntimeError(f"xmodel1 preprocess preflight: data dir does not exist: {data_dir}")
        if not data_dir.is_dir():
            raise RuntimeError(f"xmodel1 preprocess preflight: data dir is not a directory: {data_dir}")
        files = sorted(data_dir.glob("*.mjson"))
        if not files:
            continue
        sample_size = min(len(files), max(1, files_per_shard))
        selected = sorted(rng.sample(files, sample_size)) if len(files) > sample_size else files
        shard_dir = input_root / data_dir.name
        shard_dir.mkdir(parents=True, exist_ok=True)
        for source in selected:
            (shard_dir / source.name).symlink_to(source)
        sampled_dirs.append(shard_dir)
        sampled_counts[data_dir.name] = len(selected)
    return sampled_dirs, sampled_counts


def _run_export_gate(
    *,
    label: str,
    build_xmodel1_discard_records,
    data_dirs: list[Path],
    output_dir: Path,
    progress_every: int,
    jobs: int,
    limit_files: int | None,
    smoke: bool,
    resume: bool,
    discover_cached_files,
    find_export_manifests,
    probe_cached_samples,
    summarize_cached_files,
    validate_export_manifest,
) -> None:
    print(
        f"{label}: data_dirs={[str(p) for p in data_dirs]} output_dir={output_dir} smoke={smoke}",
        flush=True,
    )
    print(
        f"{label} launcher -> native-v3-export progress_every={progress_every} jobs={jobs} limit_files={limit_files}",
        flush=True,
    )
    count, manifest_path, produced_npz = build_xmodel1_discard_records(
        data_dirs=[str(path) for path in data_dirs],
        output_dir=str(output_dir),
        smoke=smoke,
        limit_files=limit_files,
        progress_every=progress_every,
        jobs=jobs,
        resume=resume,
    )
    print(
        f"{label} complete: files={count} manifest={manifest_path} produced_npz={produced_npz}",
        flush=True,
    )
    manifest_map = find_export_manifests([output_dir])
    if not manifest_map:
        raise RuntimeError(f"{label} gate: no export manifest found under {output_dir}")
    manifest_path_obj = sorted(manifest_map)[0]
    required_shards = _requested_shards(data_dirs) if limit_files is None else []
    manifest = validate_export_manifest(manifest_path_obj, required_shards=required_shards)
    cache_files = discover_cached_files([output_dir])
    if not cache_files:
        raise RuntimeError(f"{label} gate: no exported cache files found under {output_dir}")
    cache_summary = summarize_cached_files(cache_files)
    if required_shards:
        actual_shards = set(cache_summary["shard_file_counts"])
        missing = [shard for shard in required_shards if shard not in actual_shards]
        if missing:
            raise RuntimeError(f"{label} gate: missing requested shard outputs {missing}")
    if smoke:
        print(
            f"{label} smoke gate: "
            f"manifest_ok={manifest_path_obj} "
            f"cache_files={cache_summary['num_files']} "
            f"cache_samples={cache_summary['num_samples']} "
            f"requested_shards={required_shards or sorted(manifest.get('shard_file_counts', {}))}",
            flush=True,
        )
        return
    probe_summary = probe_cached_samples(cache_files)
    print(
        f"{label} gate: "
        f"manifest_ok={manifest_path_obj} "
        f"cache_files={cache_summary['num_files']} "
        f"cache_samples={cache_summary['num_samples']} "
        f"probe_rows={probe_summary['rows_probed']} "
        f"dims={probe_summary['dims']} "
        f"requested_shards={required_shards or sorted(manifest.get('shard_file_counts', {}))}",
        flush=True,
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent

    sys.path.insert(0, str(root / "src"))
    from keqing_core import build_xmodel1_discard_records
    from xmodel1.cached_dataset import (
        discover_cached_files,
        find_export_manifests,
        probe_cached_samples,
        summarize_cached_files,
        validate_export_manifest,
    )

    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = _load_cfg(cfg_path if cfg_path.is_absolute() else root / cfg_path)
    data_dirs = _resolve_paths(root, args.data_dirs or cfg.get("data_dirs", []))
    output_dir = _resolve_paths(root, [args.output_dir or cfg.get("output_dir", "processed_xmodel1")])[0]
    progress_every = (
        args.progress_every
        if args.progress_every is not None
        else int(cfg.get("progress_every", 20))
    )
    jobs = (
        args.jobs
        if args.jobs is not None
        else int(cfg.get("jobs", 0))
    )
    preflight_files = (
        args.preflight_files
        if args.preflight_files is not None
        else int(cfg.get("preflight_files_per_shard", 2))
    )
    preflight_seed = (
        args.preflight_seed
        if args.preflight_seed is not None
        else int(cfg.get("preflight_seed", 20260419))
    )
    if args.smoke:
        data_dirs = [path for path in data_dirs if path.exists()]
        if args.output_dir is None:
            smoke_root = output_dir.parent / f"{output_dir.name}_smoke"
            smoke_root.mkdir(parents=True, exist_ok=True)
            output_dir = Path(tempfile.mkdtemp(prefix="run_", dir=str(smoke_root)))
    try:
        if (not args.smoke) and (not args.skip_preflight) and preflight_files > 0:
            with tempfile.TemporaryDirectory(prefix="xmodel1_preflight_") as temp_dir:
                temp_root = Path(temp_dir)
                sampled_dirs, sampled_counts = _build_preflight_sample_dirs(
                    temp_root,
                    data_dirs,
                    files_per_shard=preflight_files,
                    seed=preflight_seed,
                )
                if not sampled_dirs:
                    raise RuntimeError("xmodel1 preprocess preflight: no sampleable .mjson files found")
                print(
                    "xmodel1 preprocess preflight selection: "
                    f"sampled_counts={sampled_counts} "
                    f"seed={preflight_seed}",
                    flush=True,
                )
                _run_export_gate(
                    label="xmodel1 preprocess preflight",
                    build_xmodel1_discard_records=build_xmodel1_discard_records,
                    data_dirs=sampled_dirs,
                    output_dir=temp_root / "output",
                    progress_every=max(1, min(progress_every, 1)),
                    jobs=jobs,
                    limit_files=None,
                    smoke=False,
                    resume=False,
                    discover_cached_files=discover_cached_files,
                    find_export_manifests=find_export_manifests,
                    probe_cached_samples=probe_cached_samples,
                    summarize_cached_files=summarize_cached_files,
                    validate_export_manifest=validate_export_manifest,
                )
        _run_export_gate(
            label="xmodel1 preprocess",
            build_xmodel1_discard_records=build_xmodel1_discard_records,
            data_dirs=data_dirs,
            output_dir=output_dir,
            progress_every=progress_every,
            jobs=jobs,
            limit_files=args.limit_files,
            smoke=args.smoke,
            resume=(not args.force) and (not args.smoke),
            discover_cached_files=discover_cached_files,
            find_export_manifests=find_export_manifests,
            probe_cached_samples=probe_cached_samples,
            summarize_cached_files=summarize_cached_files,
            validate_export_manifest=validate_export_manifest,
        )
    except KeyboardInterrupt:
        print(
            "xmodel1 preprocess interrupted by Ctrl-C. Resume by rerunning the same command; "
            "completed shard/file outputs are kept when resume=True.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(130)
    except RuntimeError as exc:
        message = str(exc)
        if "interrupted" not in message.lower():
            raise
        print(message, file=sys.stderr, flush=True)
        print(
            "xmodel1 preprocess resume hint: rerun the same command. Existing completed shard/file outputs "
            "will be skipped while unfinished files are re-exported.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(130)


if __name__ == "__main__":
    main()
