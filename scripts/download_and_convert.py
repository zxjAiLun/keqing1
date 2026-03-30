#!/usr/bin/env python3
"""Download Tenhou XML logs from CSV links and convert to MJAI JSONL.

Pipeline: CSV -> extract log ID -> download XML -> mjlog2mjai -> .mjson

Usage (单CSV模式):
    python scripts/download_and_convert.py --csv 'dataset/wait/foo.csv' --output artifacts/converted_mjai/ds99

Usage (批处理模式):
    python scripts/download_and_convert.py --batch --wait-dir dataset/wait --output artifacts/converted_mjai
    # 自动扫描 wait 目录下的 CSV，从已有最高编号+1开始，依次分配 ds6, ds7, ...
    # 每个 CSV 在新 terminal 中启动下载任务
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse, parse_qs
from urllib.request import urlopen, Request
from urllib.error import URLError

sys.path.insert(0, str(Path(__file__).parent))

from mjlog2mjai_parse import parse_mjlog_to_mjai

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


def xml_to_mjai_jsonl(xml_str: str) -> str:
    """Convert Tenhou XML string to mjai JSONL string via mjlog2mjai."""
    root = ET.fromstring(xml_str)
    return parse_mjlog_to_mjai(root)


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
            mjson_str = xml_to_mjai_jsonl(xml)
        except Exception as e:
            print(f"  ERROR xml->mjai {log_id}: {e}")
            failed += 1
            continue

        out_path.write_text(mjson_str, encoding="utf-8")

        success += 1
        print(f"  OK {log_id}")
        time.sleep(delay)

    return success, skipped, failed


def get_existing_ds_numbers(output_dir: Path) -> list[int]:
    """Return sorted list of existing ds directory numbers."""
    if not output_dir.exists():
        return []
    pattern = re.compile(r"^ds(\d+)$")
    numbers = []
    for d in output_dir.iterdir():
        if d.is_dir():
            m = pattern.match(d.name)
            if m:
                numbers.append(int(m.group(1)))
    return sorted(numbers)


def scan_wait_dir(wait_dir: Path) -> list[Path]:
    """Return sorted list of CSV files in wait directory."""
    if not wait_dir.exists():
        return []
    return sorted(wait_dir.glob("*.csv"))


def launch_terminal_download(csv_path: Path, output_ds_dir: Path, delay: float) -> subprocess.Popen:
    """Launch a new terminal running the download for a single CSV."""
    cmd = [
        "gnome-terminal", "--", "bash", "-c",
        f'cd {Path.cwd()} && uv run python scripts/download_and_convert.py '
        f'--csv "{csv_path}" --output "{output_ds_dir}" --delay {delay}; '
        f'echo "=== Download finished for {csv_path.name} ==="; read -p "Press Enter to close..."',
    ]
    return subprocess.Popen(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Download Tenhou logs from CSV and convert to MJAI JSONL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单CSV模式
  python scripts/download_and_convert.py --csv dataset/wait/foo.csv --output artifacts/converted_mjai/ds99

  # 批处理模式（自动扫描 + 多terminal并行下载）
  python scripts/download_and_convert.py --batch --wait-dir dataset/wait --output artifacts/converted_mjai
        """,
    )
    parser.add_argument("--csv", nargs="+", help="CSV file(s) with tenhou log links (单CSV模式)")
    parser.add_argument("--output", help="Output directory for .mjson files (单CSV模式)")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between downloads in seconds (default: 1.5)")

    # 批处理模式参数
    parser.add_argument("--batch", action="store_true", help="启用批处理模式")
    parser.add_argument("--wait-dir", type=Path, default=Path("dataset/wait"), help="CSV文件所在目录 (批处理模式)")
    parser.add_argument("--output-base", type=Path, default=Path("artifacts/converted_mjai"), help="输出根目录 (批处理模式)")

    args = parser.parse_args()

    if args.batch:
        # ========== 批处理模式 ==========
        wait_dir: Path = args.wait_dir
        output_base: Path = args.output_base

        csv_files = scan_wait_dir(wait_dir)
        if not csv_files:
            print(f"ERROR: wait目录 {wait_dir} 中没有找到CSV文件")
            sys.exit(1)

        existing_ds = get_existing_ds_numbers(output_base)
        if existing_ds:
            next_ds = existing_ds[-1] + 1
            print(f"已发现 ds 目录: {existing_ds}")
            print(f"将从 ds{next_ds} 开始分配")
        else:
            next_ds = 1
            print(f"未发现已有 ds 目录，将从 ds1 开始分配")

        print(f"\n找到 {len(csv_files)} 个CSV文件:")
        for i, csv_path in enumerate(csv_files):
            print(f"  ds{next_ds + i}: {csv_path.name}")

        print(f"\n将在 {len(csv_files)} 个新 terminal 中并行启动下载任务...\n")

        processes = []
        for i, csv_path in enumerate(csv_files):
            ds_num = next_ds + i
            output_dir = output_base / f"ds{ds_num}"
            print(f"启动终端 {i+1}/{len(csv_files)}: {csv_path.name} -> {output_dir}")
            p = launch_terminal_download(csv_path, output_dir, args.delay)
            processes.append((csv_path.name, p))
            time.sleep(0.5)  # 错开启动时间

        print(f"\n已启动 {len(processes)} 个下载任务")
        print("各任务将在独立 terminal 中运行，完成后会显示结果")
        print("\n正在运行的进程:")
        for name, p in processes:
            print(f"  {name}: pid={p.pid}")

    else:
        # ========== 单CSV模式 ==========
        if not args.csv or not args.output:
            parser.print_help()
            print("\nERROR: 单CSV模式需要 --csv 和 --output 参数，或使用 --batch 启用批处理模式")
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
