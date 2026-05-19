#!/usr/bin/env python3
"""Submit prepared Tenhou6 reviewer probe inputs and archive report JSON.

This script intentionally does not solve or bypass Turnstile. It can replay a
browser-captured reviewer submit request, replacing the Tenhou6, target player,
and Mortal network fields for each manifest row.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Callable
from datetime import datetime
from datetime import timezone
from http.client import HTTPResponse
import json
from pathlib import Path
import re
import shlex
import sys
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import parse_qsl
from urllib.parse import urlencode
from urllib.parse import urljoin
from urllib.request import HTTPCookieProcessor
from urllib.request import HTTPRedirectHandler
from urllib.request import Request
from urllib.request import build_opener

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.archive_reviewer_reports import archive_report
from scripts.mortal.archive_reviewer_reports import parse_report_id


DEFAULT_INPUT_MANIFEST = Path(
    "artifacts/experiments/reviewer_teacher_probe_2026_05/R0_reviewer_input_smoke/manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_batch")
REPORT_REF_RE = re.compile(r"(?:/report/|data=/report/)([A-Za-z0-9_-]+)(?:\.json)?")


class NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: HTTPResponse,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--submit-curl-file", type=Path, required=True, help="Browser Copy-as-cURL of one successful /review submit")
    parser.add_argument("--networks", default=None, help="Override comma-separated networks; default uses each manifest row")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--poll-seconds", type=float, default=2.0)
    parser.add_argument("--poll-attempts", type=int, default=30)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl_row(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def parse_csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    parsed = [part.strip() for part in value.split(",") if part.strip()]
    if not parsed:
        raise ValueError("CSV value must not be empty")
    return parsed


def parse_curl_file(path: Path) -> dict[str, Any]:
    command = path.read_text(encoding="utf-8").strip()
    if not command:
        raise ValueError(f"empty curl file: {path}")
    tokens = shlex.split(command)
    if not tokens or Path(tokens[0]).name not in {"curl", "curl.exe"}:
        raise ValueError("submit curl file must start with curl")

    url: str | None = None
    headers: dict[str, str] = {}
    data_parts: list[str] = []
    method: str | None = None
    idx = 1
    while idx < len(tokens):
        token = tokens[idx]
        if token in {"-X", "--request"}:
            idx += 1
            method = tokens[idx].upper()
        elif token in {"-H", "--header"}:
            idx += 1
            raw_header = tokens[idx]
            if ":" in raw_header:
                key, value = raw_header.split(":", 1)
                headers[key.strip()] = value.strip()
        elif token in {"-b", "--cookie"}:
            idx += 1
            cookie_value = tokens[idx]
            if "=" in cookie_value:
                headers["Cookie"] = cookie_value
        elif token in {"--data-raw", "--data", "--data-binary", "--data-ascii", "-d"}:
            idx += 1
            data_parts.append(tokens[idx])
        elif token.startswith("http://") or token.startswith("https://"):
            url = token
        idx += 1

    if url is None:
        raise ValueError("could not find submit URL in curl file")
    data = "&".join(data_parts)
    form = dict(parse_qsl(data, keep_blank_values=True))
    return {"url": url, "headers": headers, "form": form, "method": method or "POST"}


def build_form(template: dict[str, str], *, tenhou6_text: str, target_player: int, network: str) -> dict[str, str]:
    form = dict(template)
    form.update(
        {
            "input-method": "tenhou6",
            "tenhou6": tenhou6_text,
            "player-id": str(target_player),
            "engine": "mortal",
            "mortal-model-tag": network,
            "ui": form.get("ui") or "killerducky",
            "lang": form.get("lang") or "en",
        }
    )
    return form


def extract_report_id_from_response(headers: Any, body: str, base_url: str) -> str:
    candidates: list[str] = []
    location = headers.get("Location") if hasattr(headers, "get") else None
    if location:
        candidates.append(urljoin(base_url, location))
    candidates.append(body)
    for candidate in candidates:
        match = REPORT_REF_RE.search(candidate)
        if match:
            return match.group(1)
        try:
            return parse_report_id(candidate)
        except ValueError:
            pass
    raise ValueError("could not find reviewer report id in submit response")


def submit_form(
    *,
    url: str,
    headers: dict[str, str],
    form: dict[str, str],
    opener_factory: Callable[[], Any] | None = None,
) -> str:
    body = urlencode(form).encode("utf-8")
    request_headers = {
        key: value
        for key, value in headers.items()
        if key.lower()
        not in {
            "content-length",
            "host",
            "origin",
        }
    }
    request_headers.setdefault("User-Agent", "Mozilla/5.0 (compatible; keqing-reviewer-teacher-probe/1.0)")
    request_headers.setdefault("Content-Type", "application/x-www-form-urlencoded")
    request_headers.setdefault("Referer", "https://mjai.ekyu.moe/")
    request = Request(url, data=body, headers=request_headers, method="POST")
    opener = opener_factory() if opener_factory else build_opener(NoRedirectHandler, HTTPCookieProcessor)
    try:
        response = opener.open(request, timeout=120)
        response_body = response.read().decode("utf-8", errors="replace")
        return extract_report_id_from_response(response.headers, response_body, url)
    except HTTPError as exc:
        response_body = exc.read().decode("utf-8", errors="replace")
        if exc.code in {301, 302, 303, 307, 308}:
            return extract_report_id_from_response(exc.headers, response_body, url)
        if "captcha" in response_body.lower():
            raise RuntimeError("reviewer submit failed captcha validation; capture a fresh browser submit cURL") from exc
        raise RuntimeError(f"reviewer submit failed HTTP {exc.code}: {response_body[:500]}") from exc


def output_report_manifest_path(output_dir: Path) -> Path:
    return output_dir / "report_manifest.jsonl"


def load_existing_keys(path: Path) -> set[tuple[str, int, str]]:
    if not path.exists():
        return set()
    keys = set()
    for row in load_jsonl(path):
        keys.add((str(row.get("source_tenhou6_path")), int(row.get("target_player", -1)), str(row.get("network"))))
    return keys


def wait_for_report(
    *,
    source_manifest: Path,
    source_index: int,
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
                source_tenhou6=None,
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


def submit_probe(
    *,
    input_manifest: Path,
    output_dir: Path,
    submit_curl_file: Path,
    networks_override: list[str] | None,
    limit: int | None,
    sleep_seconds: float,
    poll_seconds: float,
    poll_attempts: int,
    skip_existing: bool,
    dry_run: bool,
) -> dict[str, Any]:
    submit_template = parse_curl_file(submit_curl_file)
    rows = load_jsonl(input_manifest)
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        rows = rows[:limit]

    existing = load_existing_keys(output_report_manifest_path(output_dir)) if skip_existing else set()
    submit_manifest = output_dir / "submit_manifest.jsonl"
    results: list[dict[str, Any]] = []
    for source_index, row in enumerate(rows):
        tenhou6_path = Path(str(row["tenhou6_path"]))
        tenhou6_text = tenhou6_path.read_text(encoding="utf-8")
        networks = networks_override or list(row["networks"])
        for target_player in row["target_players"]:
            for network in networks:
                key = (str(tenhou6_path), int(target_player), str(network))
                planned = {
                    "schema": "keqing.mortal.reviewer_teacher_submit.v1",
                    "source_manifest": str(input_manifest),
                    "source_index": source_index,
                    "source_log": row.get("source_log"),
                    "source_tenhou6_path": str(tenhou6_path),
                    "target_player": int(target_player),
                    "network": str(network),
                    "submitted_at": datetime.now(timezone.utc).isoformat(),
                }
                if key in existing:
                    planned["status"] = "skipped_existing"
                    results.append(planned)
                    write_jsonl_row(submit_manifest, planned)
                    continue
                form = build_form(
                    submit_template["form"],
                    tenhou6_text=tenhou6_text,
                    target_player=int(target_player),
                    network=str(network),
                )
                if dry_run:
                    planned.update(
                        {
                            "status": "dry_run",
                            "submit_url": submit_template["url"],
                            "form_field_count": len(form),
                            "has_captcha_response": bool(form.get("cf-turnstile-response")),
                        }
                    )
                    results.append(planned)
                    write_jsonl_row(submit_manifest, planned)
                    continue
                try:
                    report_id = submit_form(url=submit_template["url"], headers=submit_template["headers"], form=form)
                    archived = wait_for_report(
                        source_manifest=input_manifest,
                        source_index=source_index,
                        target_player=int(target_player),
                        network=str(network),
                        report_id=report_id,
                        output_dir=output_dir,
                        poll_seconds=poll_seconds,
                        poll_attempts=poll_attempts,
                    )
                    planned.update(
                        {
                            "status": "archived",
                            "report_id": report_id,
                            "report_json_path": archived["report_json_path"],
                            "report_page_url": archived["report_page_url"],
                        }
                    )
                except Exception as exc:  # noqa: BLE001 - keep batch resumable.
                    planned.update({"status": "failed", "error": str(exc)})
                results.append(planned)
                write_jsonl_row(submit_manifest, planned)
                time.sleep(max(0.0, sleep_seconds))
    summary = {
        "schema": "keqing.mortal.reviewer_teacher_submit_summary.v1",
        "input_manifest": str(input_manifest),
        "output_dir": str(output_dir),
        "submit_manifest": str(submit_manifest),
        "report_manifest": str(output_report_manifest_path(output_dir)),
        "dry_run": dry_run,
        "result_counts": dict(sorted(Counter(row["status"] for row in results).items())),
        "results": results,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "submit_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = submit_probe(
        input_manifest=args.input_manifest,
        output_dir=args.output_dir,
        submit_curl_file=args.submit_curl_file,
        networks_override=parse_csv(args.networks),
        limit=args.limit,
        sleep_seconds=float(args.sleep_seconds),
        poll_seconds=float(args.poll_seconds),
        poll_attempts=int(args.poll_attempts),
        skip_existing=bool(args.skip_existing),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
