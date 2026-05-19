#!/usr/bin/env python3
"""Archive mjai reviewer report JSON for reviewer teacher probes."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime
from datetime import timezone
import json
from pathlib import Path
import re
import sys
from typing import Any
from urllib.request import urlopen

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_OUTPUT_DIR = Path("artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_smoke")
REPORT_URL_RE = re.compile(r"(?:^|/)report/([A-Za-z0-9_-]+)(?:\.json)?(?:[/?#].*)?$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True, help="R0 local manifest.jsonl")
    parser.add_argument("--source-index", type=int, default=0, help="0-based row index in source manifest")
    parser.add_argument("--source-tenhou6", type=Path, default=None, help="Override source Tenhou6 path")
    parser.add_argument("--target-player", type=int, required=True)
    parser.add_argument("--network", required=True, help="Reviewer network, e.g. 3.0 or 4.1b")
    parser.add_argument("--report", required=True, help="Reviewer report id, report URL, or report JSON URL")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_manifest_row(path: Path, index: int) -> dict[str, Any]:
    if index < 0:
        raise ValueError(f"source-index must be non-negative, got {index}")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if index >= len(rows):
        raise IndexError(f"source-index {index} out of range for {path} with {len(rows)} rows")
    return rows[index]


def parse_report_id(value: str) -> str:
    raw = value.strip()
    if not raw:
        raise ValueError("report must not be empty")
    match = REPORT_URL_RE.search(raw)
    if match:
        return match.group(1)
    if re.fullmatch(r"[A-Za-z0-9_-]+", raw):
        return raw
    raise ValueError(f"could not parse reviewer report id from {value!r}")


def report_urls(report_id: str) -> tuple[str, str]:
    page_url = f"https://mjai.ekyu.moe/report/{report_id}"
    json_url = f"{page_url}.json"
    return page_url, json_url


def sanitize_token(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def report_output_path(*, output_dir: Path, source_tenhou6: Path, network: str, target_player: int) -> Path:
    source_stem = source_tenhou6.name
    if source_stem.endswith(".tenhou6.json"):
        source_stem = source_stem[: -len(".tenhou6.json")]
    else:
        source_stem = source_tenhou6.stem
    filename = f"{sanitize_token(source_stem)}__{sanitize_token(network)}__p{target_player}.json"
    return output_dir / "reports" / filename


def default_downloader(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:  # nosec B310 - URL is constrained by report id.
        return response.read()


def summarize_json_shape(value: Any) -> dict[str, Any]:
    found_keys: set[str] = set()
    likely_detail_paths: list[str] = []
    likely_q_paths: list[str] = []
    likely_action_paths: list[str] = []

    def walk(node: Any, path: str, depth: int) -> None:
        if depth > 8:
            return
        if isinstance(node, dict):
            for key, child in node.items():
                key_text = str(key)
                found_keys.add(key_text)
                child_path = f"{path}.{key_text}" if path else key_text
                lower = key_text.lower()
                if "detail" in lower:
                    likely_detail_paths.append(child_path)
                if lower in {"q", "q_value", "q_values", "qvalue", "score", "scores"} or "q_value" in lower:
                    likely_q_paths.append(child_path)
                if "action" in lower or lower in {"type", "pai", "actor"}:
                    likely_action_paths.append(child_path)
                walk(child, child_path, depth + 1)
        elif isinstance(node, list):
            for idx, child in enumerate(node[:5]):
                walk(child, f"{path}[{idx}]", depth + 1)

    walk(value, "", 0)
    return {
        "top_level_type": type(value).__name__,
        "top_level_keys": sorted(value.keys()) if isinstance(value, dict) else None,
        "key_count": len(found_keys),
        "has_detail_like_key": bool(likely_detail_paths),
        "has_q_or_score_like_key": bool(likely_q_paths),
        "has_action_like_key": bool(likely_action_paths),
        "detail_like_paths_sample": likely_detail_paths[:20],
        "q_or_score_like_paths_sample": likely_q_paths[:20],
        "action_like_paths_sample": likely_action_paths[:20],
    }


def archive_report(
    *,
    source_manifest: Path,
    source_index: int,
    source_tenhou6: Path | None,
    target_player: int,
    network: str,
    report: str,
    output_dir: Path,
    dry_run: bool,
    downloader: Callable[[str], bytes] = default_downloader,
) -> dict[str, Any]:
    if target_player < 0 or target_player > 3:
        raise ValueError(f"target-player must be in 0..3, got {target_player}")
    manifest_row = load_manifest_row(source_manifest, source_index)
    resolved_source_tenhou6 = source_tenhou6 or Path(str(manifest_row["tenhou6_path"]))
    report_id = parse_report_id(report)
    page_url, json_url = report_urls(report_id)
    json_path = report_output_path(
        output_dir=output_dir,
        source_tenhou6=resolved_source_tenhou6,
        network=network,
        target_player=target_player,
    )
    manifest_path = output_dir / "report_manifest.jsonl"

    row: dict[str, Any] = {
        "schema": "keqing.mortal.reviewer_teacher_report.v1",
        "source_manifest": str(source_manifest),
        "source_index": source_index,
        "source_log": manifest_row.get("source_log"),
        "source_tenhou6_path": str(resolved_source_tenhou6),
        "target_player": target_player,
        "network": network,
        "report_id": report_id,
        "report_page_url": page_url,
        "report_json_url": json_url,
        "report_json_path": str(json_path),
        "downloaded_at": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run:
        row["dry_run"] = True
        return row

    payload = downloader(json_url)
    report_json = json.loads(payload.decode("utf-8"))
    row["report_schema_summary"] = summarize_json_shape(report_json)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report_json, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    return row


def main() -> None:
    args = parse_args()
    row = archive_report(
        source_manifest=args.source_manifest,
        source_index=args.source_index,
        source_tenhou6=args.source_tenhou6,
        target_player=int(args.target_player),
        network=str(args.network),
        report=str(args.report),
        output_dir=args.output_dir,
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(row, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
