#!/usr/bin/env python3
"""Xmodel1 preprocess entrypoint.

Preferred path: Rust `keqing_core` export.
Fallback path: Python candidate-centric export for smoke/e2e development.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess Xmodel1 caches via Rust exporter or Python fallback")
    parser.add_argument("--config", default="configs/xmodel1_preprocess.yaml")
    parser.add_argument("--data-dirs", "--data_dirs", nargs="+", default=None)
    parser.add_argument("--output-dir", "--output_dir", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))

    from keqing_core import build_xmodel1_discard_records
    from xmodel1.preprocess import preprocess_files

    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = _load_cfg(cfg_path if cfg_path.is_absolute() else root / cfg_path)
    data_dirs = [Path(p) for p in (args.data_dirs or cfg.get("data_dirs", []))]
    output_dir = Path(args.output_dir or cfg.get("output_dir", "processed_xmodel1"))

    print(
        f"xmodel1 preprocess: data_dirs={[str(p) for p in data_dirs]} output_dir={output_dir} smoke={args.smoke}",
        flush=True,
    )
    try:
        file_count, manifest_path, produced_npz = build_xmodel1_discard_records(
            data_dirs=[str(p) for p in data_dirs],
            output_dir=str(output_dir),
            smoke=args.smoke,
        )
        print(
            f"Rust Xmodel1 export completed: file_count={file_count} "
            f"manifest={manifest_path} produced_npz={produced_npz}"
        )
        if produced_npz:
            return
        if args.smoke:
            print("Xmodel1 preprocess smoke: Rust request surface verified; skipping Python fallback export.")
            return
        print("Rust export did not produce npz yet; continuing with Python fallback export.")
        preprocess_files(data_dirs=data_dirs, output_dir=output_dir)
        return
    except Exception as exc:
        print(f"Rust Xmodel1 export unavailable, using Python fallback: {exc}")
        if args.smoke:
            print("Xmodel1 preprocess smoke: fallback path reachable; skipping full export.")
            return
        preprocess_files(data_dirs=data_dirs, output_dir=output_dir)


if __name__ == "__main__":
    main()
