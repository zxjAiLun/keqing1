#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from download_and_convert import download_xml, xml_to_mjai_jsonl


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-download and reconvert failed ds1 mjson logs")
    parser.add_argument(
        "--bad-list",
        required=True,
        help="Text file listing failed .mjson paths",
    )
    parser.add_argument(
        "--output-dir",
        default="artifacts/converted_mjai/ds1",
        help="Directory to overwrite with regenerated .mjson files",
    )
    parser.add_argument(
        "--xml-cache-dir",
        default="dataset/tenhou_xml_redownload/ds1",
        help="Directory to cache downloaded Tenhou XML",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N failed files",
    )
    return parser.parse_args()


def _load_bad_paths(path: Path) -> list[Path]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [Path(line) for line in lines]


def _write_mjson(events: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")


def _write_mjson_text(mjson_text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(mjson_text, encoding="utf-8")


def main() -> None:
    args = _parse_args()
    bad_list = Path(args.bad_list)
    output_dir = Path(args.output_dir)
    xml_cache_dir = Path(args.xml_cache_dir)

    bad_paths = _load_bad_paths(bad_list)
    if args.limit is not None:
        bad_paths = bad_paths[: args.limit]

    print(f"failed logs to reconvert: {len(bad_paths)}")
    ok = 0
    failed = 0
    for idx, bad_mjson_path in enumerate(bad_paths, 1):
        log_id = bad_mjson_path.stem
        xml_path = xml_cache_dir / f"{log_id}.xml"
        out_path = output_dir / f"{log_id}.mjson"
        try:
            xml = download_xml(log_id)
            if xml is None:
                raise RuntimeError(f"failed to download xml for {log_id}")
            xml_path.parent.mkdir(parents=True, exist_ok=True)
            xml_path.write_text(xml, encoding="utf-8")
            mjson_text = xml_to_mjai_jsonl(xml)
            _write_mjson_text(mjson_text, out_path)
            ok += 1
            print(f"[{idx}/{len(bad_paths)}] OK {log_id}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"[{idx}/{len(bad_paths)}] FAIL {log_id}: {e}")

    print(f"done: ok={ok} failed={failed}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
