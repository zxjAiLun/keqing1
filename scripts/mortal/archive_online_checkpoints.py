#!/usr/bin/env python3
"""Archive overwritten Mortal online checkpoints at configured read points."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.prepare_reward_pt_experiments import read_checkpoint_steps


def parse_read_points(value: str) -> list[int]:
    points: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        try:
            point = int(raw)
        except ValueError as exc:
            raise ValueError(f"read point must be an integer, got {raw!r}") from exc
        if point <= 0:
            raise ValueError(f"read point must be positive, got {point}")
        points.append(point)
    if not points:
        raise ValueError("at least one read point is required")
    return sorted(set(points))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-file", type=Path, required=True, help="Online checkpoint path that is overwritten by training")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for archived checkpoint copies")
    parser.add_argument("--read-points", required=True, help="Comma-separated step list, e.g. 70400,70800,71200")
    parser.add_argument("--prefix", default="mortal_online", help="Archived checkpoint filename prefix")
    parser.add_argument("--poll-seconds", type=float, default=10.0)
    parser.add_argument("--watch", action="store_true", help="Keep polling until every read point has been archived")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--manifest", type=Path, default=None, help="Optional JSONL archive manifest path")
    return parser.parse_args()


def archive_checkpoint(
    *,
    state_file: Path,
    output_dir: Path,
    step: int,
    prefix: str,
    overwrite: bool,
) -> Path | None:
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / f"{prefix}_{step}.pth"
    if dest.exists() and not overwrite:
        return None
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    shutil.copy2(state_file, tmp)
    tmp.replace(dest)
    return dest


def append_manifest(path: Path | None, row: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def archive_available_checkpoints(
    *,
    state_file: Path,
    output_dir: Path,
    read_points: list[int],
    prefix: str,
    overwrite: bool,
    manifest: Path | None = None,
) -> list[int]:
    current_step = read_checkpoint_steps(state_file)
    archived: list[int] = []
    for step in read_points:
        if current_step < step:
            continue
        dest = archive_checkpoint(
            state_file=state_file,
            output_dir=output_dir,
            step=step,
            prefix=prefix,
            overwrite=overwrite,
        )
        if dest is not None:
            archived.append(step)
            append_manifest(
                manifest,
                {
                    "source": str(state_file),
                    "archive": str(dest),
                    "archived_step": int(step),
                    "observed_state_step": int(current_step),
                    "timestamp": time.time(),
                },
            )
    return archived


def watch_and_archive(
    *,
    state_file: Path,
    output_dir: Path,
    read_points: list[int],
    prefix: str,
    poll_seconds: float,
    overwrite: bool,
    manifest: Path | None,
) -> list[int]:
    remaining = set(read_points)
    completed: list[int] = []
    while remaining:
        try:
            newly_archived = archive_available_checkpoints(
                state_file=state_file,
                output_dir=output_dir,
                read_points=sorted(remaining),
                prefix=prefix,
                overwrite=overwrite,
                manifest=manifest,
            )
        except (FileNotFoundError, RuntimeError, EOFError, KeyError) as exc:
            print(f"archive poll skipped: {exc}", file=sys.stderr)
            time.sleep(poll_seconds)
            continue
        for step in newly_archived:
            if step in remaining:
                remaining.remove(step)
                completed.append(step)
                print(f"archived {step} -> {output_dir / f'{prefix}_{step}.pth'}", flush=True)
        if remaining:
            time.sleep(poll_seconds)
    return completed


def main() -> None:
    args = parse_args()
    read_points = parse_read_points(str(args.read_points))
    state_file = args.state_file
    output_dir = args.output_dir
    manifest = args.manifest
    if args.poll_seconds <= 0:
        raise ValueError(f"poll-seconds must be positive, got {args.poll_seconds}")

    if args.watch:
        completed = watch_and_archive(
            state_file=state_file,
            output_dir=output_dir,
            read_points=read_points,
            prefix=str(args.prefix),
            poll_seconds=float(args.poll_seconds),
            overwrite=bool(args.overwrite),
            manifest=manifest,
        )
    else:
        completed = archive_available_checkpoints(
            state_file=state_file,
            output_dir=output_dir,
            read_points=read_points,
            prefix=str(args.prefix),
            overwrite=bool(args.overwrite),
            manifest=manifest,
        )
    print(json.dumps({"archived_steps": completed}, ensure_ascii=False))


if __name__ == "__main__":
    main()
