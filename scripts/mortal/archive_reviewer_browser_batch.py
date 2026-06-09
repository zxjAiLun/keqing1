#!/usr/bin/env python3
"""Archive a report URL produced by the browser-assisted reviewer workflow."""
from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.archive_reviewer_reports import archive_report
from scripts.mortal.submit_reviewer_teacher_probe import DEFAULT_OUTPUT_DIR
from scripts.mortal.submit_reviewer_teacher_probe import load_jsonl
from scripts.mortal.submit_reviewer_teacher_probe import write_jsonl_row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", type=Path, required=True, help="browser_tasks.jsonl from prepare_reviewer_browser_batch.py")
    parser.add_argument("--task-index", type=int, required=True)
    parser.add_argument("--report", required=True, help="Reviewer report id, page URL, or JSON URL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def select_task(tasks_path: Path, task_index: int) -> dict[str, Any]:
    for task in load_jsonl(tasks_path):
        if int(task.get("task_index", -1)) == task_index:
            return task
    raise IndexError(f"task-index {task_index} not found in {tasks_path}")


def archive_browser_task(
    *,
    tasks_path: Path,
    task_index: int,
    report: str,
    output_dir: Path,
    dry_run: bool,
) -> dict[str, Any]:
    task = select_task(tasks_path, task_index)
    archived = archive_report(
        source_manifest=Path(str(task["source_manifest"])),
        source_index=int(task["source_index"]),
        source_tenhou6=Path(str(task["source_tenhou6_path"])),
        target_player=int(task["target_player"]),
        network=str(task["network"]),
        report=report,
        output_dir=output_dir,
        dry_run=dry_run,
    )
    row = {
        "schema": "keqing.mortal.reviewer_browser_archive.v1",
        "tasks": str(tasks_path),
        "task_index": int(task_index),
        "source_index": int(task["source_index"]),
        "target_player": int(task["target_player"]),
        "network": str(task["network"]),
        "report": str(report),
        "status": "dry_run" if dry_run else "archived",
        "archived_at": datetime.now(timezone.utc).isoformat(),
        "report_id": archived.get("report_id"),
        "report_json_path": archived.get("report_json_path"),
        "report_page_url": archived.get("report_page_url"),
    }
    write_jsonl_row(output_dir / "browser_archive_manifest.jsonl", row)
    return {"task": task, "archive": archived, "browser_archive_row": row}


def main() -> None:
    args = parse_args()
    result = archive_browser_task(
        tasks_path=args.tasks,
        task_index=int(args.task_index),
        report=str(args.report),
        output_dir=args.output_dir,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
