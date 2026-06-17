#!/usr/bin/env python3
"""Submit cURL files emitted by emit_reviewer_curls.py and archive reports."""

from __future__ import annotations

import argparse
from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
import sys
import time
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.archive_reviewer_reports import archive_report
from scripts.mortal.submit_reviewer_teacher_probe import parse_curl_file
from scripts.mortal.submit_reviewer_teacher_probe import submit_form


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--curl-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--poll-attempts", type=int, default=30)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _source_manifest_path(output_dir: Path) -> Path:
    return output_dir / "source_manifest.from_curls.jsonl"


def _submit_manifest_path(output_dir: Path) -> Path:
    return output_dir / "submit_manifest.from_curls.jsonl"


def _existing_keys(output_dir: Path) -> set[tuple[str, int, str]]:
    manifest = output_dir / "report_manifest.jsonl"
    if not manifest.exists():
        return set()
    keys: set[tuple[str, int, str]] = set()
    for row in _load_jsonl(manifest):
        keys.add((str(row["source_tenhou6_path"]), int(row["target_player"]), str(row["network"])))
    return keys


def _write_source_manifest(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    source_manifest = _source_manifest_path(output_dir)
    with source_manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "schema": "keqing.mortal.reviewer_source_from_curl.v1",
                        "source_log": row.get("source_tenhou6_path"),
                        "tenhou6_path": row["source_tenhou6_path"],
                        "target_players": [int(row["target_player"])],
                        "networks": [str(row["network"])],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
                + "\n"
            )
    return source_manifest


def _archive_with_retry(
    *,
    source_manifest: Path,
    source_index: int,
    source_tenhou6: Path,
    target_player: int,
    network: str,
    report_id: str,
    output_dir: Path,
    poll_seconds: float,
    poll_attempts: int,
) -> dict[str, Any]:
    last_error: Exception | None = None
    for _ in range(max(1, poll_attempts)):
        try:
            return archive_report(
                source_manifest=source_manifest,
                source_index=source_index,
                source_tenhou6=source_tenhou6,
                target_player=target_player,
                network=network,
                report=report_id,
                output_dir=output_dir,
                dry_run=False,
            )
        except Exception as exc:  # noqa: BLE001 - remote report may not be ready yet.
            last_error = exc
            time.sleep(max(0.0, poll_seconds))
    raise RuntimeError(f"report {report_id} did not become downloadable") from last_error


def main() -> None:
    args = _parse_args()
    rows = _load_jsonl(args.curl_manifest)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source_manifest = _write_source_manifest(rows, args.output_dir)
    submit_manifest = _submit_manifest_path(args.output_dir)
    existing = _existing_keys(args.output_dir) if bool(args.skip_existing) else set()

    results: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        source_tenhou6 = Path(str(row["source_tenhou6_path"]))
        target_player = int(row["target_player"])
        network = str(row["network"])
        key = (str(source_tenhou6), target_player, network)
        planned = {
            "schema": "keqing.mortal.reviewer_submit_from_curl.v1",
            "source_index": index,
            "source_tenhou6_path": str(source_tenhou6),
            "target_player": target_player,
            "player_name": row.get("player_name"),
            "network": network,
            "curl_path": str(row["curl_path"]),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        }
        if key in existing:
            planned["status"] = "skipped_existing"
            results.append(planned)
            _write_jsonl_row(submit_manifest, planned)
            continue

        curl_template = parse_curl_file(Path(str(row["curl_path"])))
        if args.dry_run:
            planned.update(
                {
                    "status": "dry_run",
                    "submit_url": curl_template["url"],
                    "form_field_count": len(curl_template["form"]),
                    "has_turnstile_response": bool(curl_template["form"].get("cf-turnstile-response")),
                }
            )
            results.append(planned)
            _write_jsonl_row(submit_manifest, planned)
            continue

        try:
            report_id = submit_form(
                url=curl_template["url"],
                headers=curl_template["headers"],
                form=curl_template["form"],
            )
            archived = _archive_with_retry(
                source_manifest=source_manifest,
                source_index=index,
                source_tenhou6=source_tenhou6,
                target_player=target_player,
                network=network,
                report_id=report_id,
                output_dir=args.output_dir,
                poll_seconds=float(args.poll_seconds),
                poll_attempts=int(args.poll_attempts),
            )
            planned.update(
                {
                    "status": "archived",
                    "report_id": report_id,
                    "report_json_path": archived["report_json_path"],
                    "report_page_url": archived["report_page_url"],
                }
            )
        except Exception as exc:  # noqa: BLE001 - keep batch state inspectable.
            planned.update({"status": "failed", "error": repr(exc)})
            results.append(planned)
            _write_jsonl_row(submit_manifest, planned)
            raise

        results.append(planned)
        _write_jsonl_row(submit_manifest, planned)
        time.sleep(max(0.0, float(args.sleep_seconds)))

    print(json.dumps({"source_manifest": str(source_manifest), "submit_manifest": str(submit_manifest), "results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
