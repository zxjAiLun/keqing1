from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from convert.link_converter import convert_url_to_tenhou6


def _sanitize_name(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)


def _extract_ref(url: str) -> str:
    q = parse_qs(urlparse(url).query)
    log_vals = q.get("log", [])
    if log_vals and log_vals[0].strip():
        return _sanitize_name(log_vals[0].strip())
    return _sanitize_name(url)[:120]


def _next_index_from_output_dir(out_dir: Path) -> int:
    max_idx = -1
    for p in out_dir.glob("*.json"):
        stem = p.stem
        prefix = stem.split("_", 1)[0]
        if prefix.isdigit():
            max_idx = max(max_idx, int(prefix))
    return max_idx + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", action="append", default=[], help="Log URL, can be repeated")
    parser.add_argument("--url-file", default=None, help="Text file containing one URL per line")
    parser.add_argument("--start-line", type=int, default=1, help="Start from line N of --url-file (1-based)")
    parser.add_argument("--output-dir", default="dataset", help="Output directory for tenhou6 json files")
    parser.add_argument("--reviewer-bin", default=None, help="Path to mjai-reviewer binary")
    parser.add_argument("--start-index", type=int, default=None, help="Start output index (e.g. 1235)")
    parser.add_argument("--auto-resume", action="store_true", help="Start from max existing index + 1 in output dir")
    args = parser.parse_args()

    urls = list(args.url)
    start_line = max(1, int(args.start_line))
    if args.url_file:
        lines = Path(args.url_file).read_text(encoding="utf-8").splitlines()
        selected = []
        for i, line in enumerate(lines, start=1):
            if i < start_line:
                continue
            x = line.strip()
            if not x or x.startswith("#"):
                continue
            selected.append(x)
        urls.extend(selected)
    if not urls:
        raise RuntimeError("no urls provided")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.auto_resume:
        start_idx = _next_index_from_output_dir(out_dir)
    elif args.start_index is not None:
        start_idx = int(args.start_index)
    elif args.url_file:
        # Convenient default: align output index with source line number.
        # e.g. start-line=1950 -> first file index 1949.
        start_idx = start_line - 1
    else:
        start_idx = 0

    results = []
    ok_count = 0
    failed_count = 0
    total = len(urls)
    for i, url in enumerate(urls):
        out_idx = start_idx + i
        ref = _extract_ref(url)
        out_file = out_dir / f"{out_idx}_{ref}.json"
        res = convert_url_to_tenhou6(url, str(out_file), args.reviewer_bin)
        if not res.get("ok", False):
            # Keep summary and continue; users may mix tenhou and mjsoul links.
            res["out_file"] = str(out_file)
            failed_count += 1
        else:
            ok_count += 1
        results.append(res)
        print(
            f"\rprogress: {i + 1}/{total} | ok={ok_count} | failed={failed_count} | current={out_idx}_{ref}.json",
            end="",
            flush=True,
        )

    print("")
    out = {"total": len(results), "ok": ok_count, "failed": len(results) - ok_count, "results": results}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

