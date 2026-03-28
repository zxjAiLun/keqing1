#!/usr/bin/env python3
"""Download Tenhou XML logs from CSV links and convert to MJAI JSONL.

Pipeline: CSV -> extract log ID -> download XML -> tenhou6 JSON -> convlog -> .mjson

Usage:
    python scripts/download_and_convert.py --csv 'dataset/links/csv/foo.csv' --output dataset/ds/ds99
    python scripts/download_and_convert.py --csv 'dataset/links/csv/*.csv' --output dataset/ds/ds99
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from urllib.request import urlopen, Request
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).parent))

from tenhou_xml_to_json import convert_xml_to_tenhou6

CONVLOG_BIN = Path(__file__).parent.parent / "third_party" / "mjai-reviewer" / "target" / "release" / "convlog"

# Tenhou XML download endpoint
_TENHOU_LOG_URL = "https://tenhou.net/0/log/?{log_id}"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; mahjong-research-bot/1.0)",
    "Referer": "https://tenhou.net/",
}
_RETRY_DELAY = 2.0  # seconds between requests


def extract_log_id(url: str) -> str | None:
    """Extract log ID from tenhou.net URL like https://tenhou.net/3/?log=XXXX&tw=N."""
    parsed = urlparse(url)
    qs = parse_qs(parsed.query)
    ids = qs.get("log", [])
    return ids[0] if ids else None


def download_xml(log_id: str, retries: int = 3) -> str | None:
    """Download Tenhou XML for the given log ID. Returns XML string or None on failure."""
    url = _TENHOU_LOG_URL.format(log_id=log_id)
    for attempt in range(retries):
        try:
            req = Request(url, headers=_HEADERS)
            with urlopen(req, timeout=30) as resp:
                raw = resp.read()
            # Tenhou returns XML; detect encoding
            text = raw.decode("utf-8", errors="replace")
            if "<mjloggm" in text or "<mjlog" in text.lower():
                return text
            print(f"  WARN: unexpected response for {log_id}: {text[:200]}")
            return None
        except URLError as e:
            if attempt < retries - 1:
                time.sleep(_RETRY_DELAY * (attempt + 1))
            else:
                print(f"  ERROR downloading {log_id}: {e}")
                return None
    return None


def xml_to_tenhou6_json(xml_str: str) -> dict:
    return convert_xml_to_tenhou6(xml_str)


def tenhou6_to_mjson(t6_json: dict, output_path: Path) -> bool:
    """Write tenhou6 JSON to a temp file then run convlog. Returns success."""
    tmp = output_path.with_suffix(".tmp.json")
    try:
        tmp.write_text(json.dumps(t6_json, ensure_ascii=False), encoding="utf-8")
        result = subprocess.run(
            [str(CONVLOG_BIN), str(tmp), str(output_path)],
            capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"  ERROR convlog: {result.stderr.strip()}")
            return False
        return True
    finally:
        if tmp.exists():
            tmp.unlink()


def parse_csv_links(csv_path: Path) -> list[str]:
    """Return list of tenhou.net log URLs from a nodocchi CSV."""
    urls = []
    with open(csv_path, encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            url = row.get("牌谱", "").strip()
            if url.startswith("http"):
                urls.append(url)
    return urls


def process_csv(csv_path: Path, output_dir: Path, delay: float) -> tuple[int, int, int]:
    """Process one CSV file. Returns (success, skipped, failed)."""
    urls = parse_csv_links(csv_path)
    if not urls:
        print(f"  No URLs found in {csv_path.name}")
        return 0, 0, 0

    print(f"  {len(urls)} links in {csv_path.name}")
    output_dir.mkdir(parents=True, exist_ok=True)

    success = skipped = failed = 0
    for url in urls:
        log_id = extract_log_id(url)
        if not log_id:
            print(f"  WARN: cannot parse log ID from {url}")
            failed += 1
            continue

        out_path = output_dir / f"{log_id}.mjson"
        if out_path.exists():
            skipped += 1
            continue

        xml = download_xml(log_id)
        if xml is None:
            failed += 1
            continue

        try:
            t6 = xml_to_tenhou6_json(xml)
        except Exception as e:
            print(f"  ERROR xml->json {log_id}: {e}")
            failed += 1
            continue

        if not tenhou6_to_mjson(t6, out_path):
            failed += 1
            continue

        success += 1
        print(f"  OK {log_id}")
        time.sleep(delay)

    return success, skipped, failed


def main():
    parser = argparse.ArgumentParser(description="Download Tenhou logs from CSV and convert to MJAI JSONL")
    parser.add_argument("--csv", required=True, nargs="+", help="CSV file(s) with tenhou log links")
    parser.add_argument("--output", required=True, help="Output directory for .mjson files")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between downloads in seconds (default: 1.5)")
    args = parser.parse_args()

    if not CONVLOG_BIN.exists():
        print(f"ERROR: convlog not found at {CONVLOG_BIN}")
        sys.exit(1)

    output_dir = Path(args.output)
    total_success = total_skipped = total_failed = 0

    csv_paths = []
    for pattern in args.csv:
        csv_paths.extend(sorted(Path().glob(pattern)) if "*" in pattern else [Path(pattern)])

    for csv_path in csv_paths:
        if not csv_path.exists():
            print(f"ERROR: CSV not found: {csv_path}")
            continue
        print(f"\n=== {csv_path.name} ===")
        s, sk, f = process_csv(csv_path, output_dir, args.delay)
        total_success += s
        total_skipped += sk
        total_failed += f

    print(f"\n=== Total: {total_success} downloaded+converted, {total_skipped} skipped, {total_failed} failed ===")


if __name__ == "__main__":
    main()
