#!/usr/bin/env python3
"""Emit reviewer submit cURL commands from a captured browser submit template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import sys
from urllib.parse import urlencode

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.submit_reviewer_teacher_probe import build_form
from scripts.mortal.submit_reviewer_teacher_probe import parse_curl_file


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--submit-curl-file", type=Path, required=True)
    parser.add_argument("--tenhou6", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--players", default="0,1,2,3", help="comma-separated target player ids")
    parser.add_argument("--network", default="4.1b")
    parser.add_argument("--prefix", default=None)
    return parser.parse_args()


def _parse_players(value: str) -> list[int]:
    players = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not players:
        raise ValueError("--players must not be empty")
    for player in players:
        if player not in {0, 1, 2, 3}:
            raise ValueError(f"invalid player id {player}; expected 0..3")
    return players


def _quote(value: str) -> str:
    return shlex.quote(value)


def _render_curl(*, url: str, headers: dict[str, str], form: dict[str, str]) -> str:
    parts = ["curl", _quote(url)]
    for key, value in headers.items():
        if key.lower() in {"content-length", "host"}:
            continue
        if key.lower() == "cookie":
            parts.extend(["-b", _quote(value)])
        else:
            parts.extend(["-H", _quote(f"{key}: {value}")])
    parts.extend(["--data-raw", _quote(urlencode(form))])
    return " \\\n  ".join(parts) + "\n"


def main() -> None:
    args = _parse_args()
    template = parse_curl_file(args.submit_curl_file)
    tenhou6_text = args.tenhou6.read_text(encoding="utf-8")
    players = _parse_players(str(args.players))
    prefix = args.prefix or args.tenhou6.stem

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for player in players:
        form = build_form(
            template["form"],
            tenhou6_text=tenhou6_text,
            target_player=player,
            network=str(args.network),
        )
        out_path = args.output_dir / f"{prefix}_p{player}_{args.network}.curl"
        out_path.write_text(
            _render_curl(url=template["url"], headers=template["headers"], form=form),
            encoding="utf-8",
        )
        rows.append(
            {
                "curl_path": str(out_path),
                "source_tenhou6_path": str(args.tenhou6),
                "target_player": player,
                "network": str(args.network),
                "player_name": json.loads(tenhou6_text).get("name", [None, None, None, None])[player],
                "has_turnstile_response": bool(form.get("cf-turnstile-response")),
            }
        )

    manifest = args.output_dir / f"{prefix}_curl_manifest.jsonl"
    with manifest.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    print(json.dumps({"manifest": str(manifest), "tasks": rows}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
