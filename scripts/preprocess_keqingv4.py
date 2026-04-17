#!/usr/bin/env python3
"""keqingv4 preprocess entrypoint.

Rust now owns orchestration/export; Python remains the semantic implementation
backend for the first transition version.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess keqingv4 caches via Rust orchestrator")
    parser.add_argument("--config", default="configs/keqingv4_preprocess.yaml")
    parser.add_argument("--data-dirs", "--data_dirs", nargs="+", default=None)
    parser.add_argument("--output-dir", "--output_dir", default=None)
    parser.add_argument("--progress-every", "--progress_every", type=int, default=None)
    parser.add_argument("--jobs", type=int, default=None)
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
    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = _load_cfg(cfg_path if cfg_path.is_absolute() else root / cfg_path)
    data_dirs = _resolve_paths(root, args.data_dirs or cfg.get("data_dirs", []))
    output_dir = _resolve_paths(root, [args.output_dir or cfg.get("output_dir", "processed_v4")])[0]
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
    if args.force:
        cmd.append("--force")
    if args.smoke:
        cmd.append("--smoke")

    print(
        f"keqingv4 preprocess: data_dirs={[str(p) for p in data_dirs]} output_dir={output_dir} smoke={args.smoke}",
        flush=True,
    )
    print(f"keqingv4 preprocess launcher -> {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, cwd=root, check=True)


if __name__ == "__main__":
    main()

