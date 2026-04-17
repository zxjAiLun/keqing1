#!/usr/bin/env python3
"""Xmodel1 preprocess entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Xmodel1 caches via Rust full exporter")
    parser.add_argument("--config", default="configs/xmodel1_preprocess.yaml")
    parser.add_argument("--data-dirs", "--data_dirs", nargs="+", default=None)
    parser.add_argument("--output-dir", "--output_dir", default=None)
    parser.add_argument("--progress-every", "--progress_every", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=None)
    parser.add_argument("--limit-files", "--limit_files", type=int, default=None)
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


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    import sys

    sys.path.insert(0, str(root / "src"))
    from keqing_core import build_xmodel1_discard_records

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
    if args.smoke:
        data_dirs = [path for path in data_dirs if path.exists()]

    print(
        f"xmodel1 preprocess: data_dirs={[str(p) for p in data_dirs]} output_dir={output_dir} smoke={args.smoke}",
        flush=True,
    )
    print(
        f"xmodel1 preprocess launcher -> native-v2-export progress_every={progress_every} jobs={jobs} limit_files={args.limit_files}",
        flush=True,
    )
    count, manifest_path, produced_npz = build_xmodel1_discard_records(
        data_dirs=[str(path) for path in data_dirs],
        output_dir=str(output_dir),
        smoke=args.smoke,
        limit_files=args.limit_files,
        progress_every=progress_every,
        jobs=jobs,
        resume=not args.force,
    )
    print(
        f"xmodel1 preprocess complete: files={count} manifest={manifest_path} produced_npz={produced_npz}",
        flush=True,
    )


if __name__ == "__main__":
    main()
