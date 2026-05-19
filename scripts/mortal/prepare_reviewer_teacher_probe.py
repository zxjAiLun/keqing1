#!/usr/bin/env python3
"""Prepare Tenhou6 inputs for reviewer-backed teacher probes."""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _REPO_ROOT / "src"
for candidate in (_REPO_ROOT, _SRC_ROOT):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from tools.mjai_jsonl_to_tenhou6 import convert_mjai_jsonl_to_tenhou6
from tools.mjai_jsonl_to_tenhou6 import load_mjai_jsonl


DEFAULT_NETWORKS = ("3.0", "4.1b")
CONVLOG_BIN = _REPO_ROOT / "third_party" / "mjai-reviewer" / "target" / "release" / "convlog"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", action="append", required=True, help="Glob of mjai JSONL/json.gz hanchan logs; may be repeated")
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/experiments/reviewer_teacher_probe_2026_05"))
    parser.add_argument("--experiment-id", default="R0_reviewer_input_smoke")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--target-players", default="0", help="Comma-separated reviewer target players, e.g. 0 or 0,1,2,3")
    parser.add_argument("--target-player-name", default=None, help="If set, select the unique seat whose player name matches this value for each log")
    parser.add_argument("--networks", default=",".join(DEFAULT_NETWORKS), help="Comma-separated Mortal reviewer networks")
    parser.add_argument("--validate-convlog", action="store_true", help="Round-trip generated Tenhou6 through mjai-reviewer convlog")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def expand_logs(patterns: list[str], limit: int) -> list[Path]:
    if limit <= 0:
        raise ValueError(f"limit must be positive, got {limit}")
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(path) for path in glob.glob(pattern, recursive=True))
    unique = sorted(dict.fromkeys(paths))
    if not unique:
        raise FileNotFoundError(f"no logs matched: {patterns}")
    return unique[:limit]


def parse_csv_ints(value: str, *, field: str) -> list[int]:
    out: list[int] = []
    for raw in value.split(","):
        raw = raw.strip()
        if not raw:
            continue
        parsed = int(raw)
        if parsed < 0 or parsed > 3:
            raise ValueError(f"{field} values must be in 0..3, got {parsed}")
        out.append(parsed)
    if not out:
        raise ValueError(f"{field} must contain at least one value")
    return out


def parse_csv_strings(value: str, *, field: str) -> list[str]:
    out = [part.strip() for part in value.split(",") if part.strip()]
    if not out:
        raise ValueError(f"{field} must contain at least one value")
    return out


def validate_with_convlog(tenhou6_path: Path, mjson_path: Path) -> dict[str, Any]:
    if not CONVLOG_BIN.exists():
        raise FileNotFoundError(f"convlog binary not found: {CONVLOG_BIN}")
    result = subprocess.run(
        [str(CONVLOG_BIN), str(tenhou6_path), str(mjson_path)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return {"ok": False, "stderr": result.stderr.strip(), "stdout": result.stdout.strip()}
    line_count = sum(1 for _ in mjson_path.open("r", encoding="utf-8"))
    return {"ok": True, "mjson_path": str(mjson_path), "mjson_events": line_count}


def resolve_target_players(names: list[Any], *, target_players: list[int], target_player_name: str | None) -> list[int]:
    if target_player_name is None:
        return target_players
    matches = [idx for idx, name in enumerate(names[:4]) if str(name) == target_player_name]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one player named {target_player_name!r}, found seats {matches} in {names}")
    return matches


def prepare_probe(
    *,
    logs: list[str],
    output_root: Path,
    experiment_id: str,
    limit: int,
    target_players: list[int],
    target_player_name: str | None,
    networks: list[str],
    validate_convlog: bool,
    dry_run: bool,
) -> dict[str, Any]:
    source_logs = expand_logs(logs, limit)
    exp_dir = output_root / experiment_id
    input_dir = exp_dir / "input"
    roundtrip_dir = exp_dir / "roundtrip_mjai"
    manifest_path = exp_dir / "manifest.jsonl"
    summary_path = exp_dir / "summary.json"
    rows: list[dict[str, Any]] = []

    for idx, source_path in enumerate(source_logs, 1):
        tenhou6_name = source_path.name
        if tenhou6_name.endswith(".json.gz"):
            tenhou6_name = tenhou6_name[:-8]
        elif tenhou6_name.endswith(".jsonl.gz"):
            tenhou6_name = tenhou6_name[:-9]
        else:
            tenhou6_name = source_path.stem
        tenhou6_path = input_dir / f"{idx:04d}_{tenhou6_name}.tenhou6.json"

        row: dict[str, Any]
        if not dry_run:
            events = load_mjai_jsonl(source_path)
            tenhou6 = convert_mjai_jsonl_to_tenhou6(events)
            row_target_players = resolve_target_players(
                list(tenhou6.get("name", [])),
                target_players=target_players,
                target_player_name=target_player_name,
            )
            input_dir.mkdir(parents=True, exist_ok=True)
            tenhou6_path.write_text(json.dumps(tenhou6, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")
            row = {
                "schema": "keqing.mortal.reviewer_teacher_probe_input.v1",
                "experiment_id": experiment_id,
                "source_log": str(source_path),
                "tenhou6_path": str(tenhou6_path),
                "review_status": "pending_upload",
                "target_players": row_target_players,
                "target_player_name": target_player_name,
                "networks": networks,
                "kyoku_count": len(tenhou6.get("log", [])),
                "player_names": tenhou6.get("name", []),
            }
            if validate_convlog:
                roundtrip_dir.mkdir(parents=True, exist_ok=True)
                row["convlog_validation"] = validate_with_convlog(tenhou6_path, roundtrip_dir / f"{tenhou6_path.stem}.mjson")
        else:
            row = {
                "schema": "keqing.mortal.reviewer_teacher_probe_input.v1",
                "experiment_id": experiment_id,
                "source_log": str(source_path),
                "tenhou6_path": str(tenhou6_path),
                "review_status": "pending_upload",
                "target_players": target_players,
                "target_player_name": target_player_name,
                "networks": networks,
            }
        rows.append(row)

    summary = {
        "schema": "keqing.mortal.reviewer_teacher_probe_manifest.v1",
        "experiment_id": experiment_id,
        "output_dir": str(exp_dir),
        "input_dir": str(input_dir),
        "manifest": str(manifest_path),
        "source_count": len(source_logs),
        "target_players": target_players,
        "target_player_name": target_player_name,
        "networks": networks,
        "validate_convlog": bool(validate_convlog),
        "rows": rows if dry_run else None,
        "notes": [
            "Upload each Tenhou6 JSON as Custom log on mjai.ekyu.moe.",
            "Custom log target player must be set explicitly.",
            "Archive reviewer report JSON immediately; result pages are retained for 15 days.",
        ],
    }
    if dry_run:
        return summary

    exp_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    summary_for_disk = {**summary, "rows": None}
    summary_path.write_text(json.dumps(summary_for_disk, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary_for_disk


def main() -> None:
    args = parse_args()
    summary = prepare_probe(
        logs=list(args.logs),
        output_root=args.output_root,
        experiment_id=str(args.experiment_id),
        limit=int(args.limit),
        target_players=parse_csv_ints(str(args.target_players), field="target_players"),
        target_player_name=args.target_player_name,
        networks=parse_csv_strings(str(args.networks), field="networks"),
        validate_convlog=bool(args.validate_convlog),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
