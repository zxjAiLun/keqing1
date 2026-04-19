#!/usr/bin/env python3
"""Repair missing Xmodel1 per-file schema metadata in existing `.npz` caches."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import io
from pathlib import Path
import sys
import zipfile

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair missing xmodel1 schema metadata in existing .npz caches")
    parser.add_argument("roots", nargs="+", help="processed cache roots or individual .npz files")
    parser.add_argument("--jobs", type=int, default=8)
    parser.add_argument("--limit-files", "--limit_files", type=int, default=None)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--progress-every", "--progress_every", type=int, default=1000)
    return parser.parse_args()


def _discover_npz_files(roots: list[Path]) -> list[Path]:
    files: list[Path] = []
    seen: set[str] = set()
    for root in roots:
        if root.is_file():
            candidates = [root] if root.suffix == ".npz" else []
        elif root.is_dir():
            candidates = sorted(root.rglob("*.npz"))
        else:
            candidates = []
        for path in candidates:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    return files


def _scalar_npy_bytes(value: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, value, allow_pickle=False)
    return buf.getvalue()


def _inspect_schema_members(path: Path) -> tuple[bool, bool]:
    with zipfile.ZipFile(path, "r") as zf:
        names = set(zf.namelist())
    return "schema_name.npy" in names, "schema_version.npy" in names


def _repair_one(
    path: Path,
    *,
    schema_name_bytes: bytes,
    schema_version_bytes: bytes,
    check_only: bool,
) -> str:
    has_name, has_version = _inspect_schema_members(path)
    if has_name and has_version:
        return "ok"
    if check_only:
        return "missing"
    with zipfile.ZipFile(path, "a", compression=zipfile.ZIP_STORED, allowZip64=True) as zf:
        if not has_name:
            zf.writestr("schema_name.npy", schema_name_bytes)
        if not has_version:
            zf.writestr("schema_version.npy", schema_version_bytes)
    has_name, has_version = _inspect_schema_members(path)
    if has_name and has_version:
        return "repaired"
    raise RuntimeError(f"repair incomplete for {path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(repo_root / "src"))
    from xmodel1.schema import XMODEL1_SCHEMA_NAME, XMODEL1_SCHEMA_VERSION

    args = _parse_args()
    roots = [Path(raw).resolve() for raw in args.roots]
    files = _discover_npz_files(roots)
    if args.limit_files is not None:
        files = files[: max(0, args.limit_files)]
    if not files:
        raise RuntimeError("xmodel1 schema repair: no .npz files found")

    schema_name_bytes = _scalar_npy_bytes(np.array(XMODEL1_SCHEMA_NAME, dtype=np.str_))
    schema_version_bytes = _scalar_npy_bytes(np.array(XMODEL1_SCHEMA_VERSION, dtype=np.int32))
    jobs = min(max(1, int(args.jobs)), len(files))
    progress_every = max(1, int(args.progress_every))
    counts = {"ok": 0, "missing": 0, "repaired": 0, "failed": 0}

    print(
        "xmodel1 schema repair: "
        f"roots={[str(root) for root in roots]} files={len(files)} jobs={jobs} check_only={args.check_only}",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = {
            pool.submit(
                _repair_one,
                path,
                schema_name_bytes=schema_name_bytes,
                schema_version_bytes=schema_version_bytes,
                check_only=args.check_only,
            ): path
            for path in files
        }
        done = 0
        for future in as_completed(futures):
            path = futures[future]
            try:
                status = future.result()
            except Exception as exc:
                counts["failed"] += 1
                print(f"xmodel1 schema repair failed: {path}: {exc}", flush=True)
            else:
                counts[status] += 1
            done += 1
            if done % progress_every == 0 or done == len(files):
                print(
                    "xmodel1 schema repair progress: "
                    f"{done}/{len(files)} ok={counts['ok']} missing={counts['missing']} repaired={counts['repaired']} failed={counts['failed']}",
                    flush=True,
                )

    if counts["failed"] > 0:
        raise RuntimeError(f"xmodel1 schema repair: {counts['failed']} files failed")
    print(
        "xmodel1 schema repair complete: "
        f"ok={counts['ok']} missing={counts['missing']} repaired={counts['repaired']} failed={counts['failed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
