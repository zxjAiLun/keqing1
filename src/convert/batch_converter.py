#!/usr/bin/env python3
"""
批量数据转换脚本

数据管道：
1. 雀魂 (Majsoul) 牌谱：
   - 需要使用 mjai-reviewer 浏览器插件导出 JSON
   - 放到 dataset/majsoul_raw/ 目录
   - 然后转换为 mjai JSONL

2. 天凤 (Tenhou) 牌谱：
   - 直接通过 API 下载
   - 然后转换为 mjai JSONL

3. 所有 JSONL 文件最终用于训练
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

from convert.link_converter import (
    convert_url_to_tenhou6,
    download_tenhou6_from_tenhou,
    parse_log_url,
)
from convert.libriichi_bridge import convert_raw_to_mjai


def ensure_libriichi() -> Optional[str]:
    """检查并返回 libriichi CLI 路径"""
    for cmd in ["libriichi", "/usr/local/bin/libriichi", "libriichi.exe"]:
        try:
            subprocess.run([cmd, "--version"], capture_output=True, check=True)
            return cmd
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    return None


def process_majsoul_raw_files(raw_dir: Path, out_dir: Path, libriichi: Optional[str], dry_run: bool = False):
    """处理雀魂原始 JSON 文件"""
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(raw_dir.glob("*.json"))
    if not json_files:
        print(f"[WARN] 没有找到 JSON 文件 in {raw_dir}")
        return 0, 0

    success = 0
    failed = 0

    for json_file in json_files:
        out_file = out_dir / f"{json_file.stem}.jsonl"
        if dry_run:
            print(f"[DRYRUN] 转换 {json_file.name} -> {out_file.name}")
            success += 1
            continue

        try:
            result = convert_raw_to_mjai(str(json_file), str(out_file), libriichi)
            if result.get("used_libriichi"):
                print(f"[OK] {json_file.name} -> {out_file.name} (via {result['engine']})")
                success += 1
            else:
                print(f"[FAIL] {json_file.name} 转换失败")
                failed += 1
        except Exception as e:
            print(f"[ERROR] {json_file.name}: {e}")
            failed += 1

    return success, failed


def process_tenhou_links(links_file: Path, out_raw_dir: Path, libriichi: Optional[str], dry_run: bool = False):
    """处理天凤链接列表"""
    out_raw_dir = Path(out_raw_dir)
    out_mjai_dir = out_raw_dir.parent / "mjai"
    out_raw_dir.mkdir(parents=True, exist_ok=True)
    out_mjai_dir.mkdir(parents=True, exist_ok=True)

    if not links_file.exists():
        print(f"[ERROR] 链接文件不存在: {links_file}")
        return 0, 0

    links = [line.strip() for line in links_file.open() if line.strip() and not line.startswith("#")]
    print(f"找到 {len(links)} 个天凤链接")

    success = 0
    failed = 0

    for i, url in enumerate(links):
        if dry_run:
            print(f"[DRYRUN] [{i+1}/{len(links)}] {url}")
            success += 1
            continue

        try:
            info = parse_log_url(url)
            log_id = info.get("log_id", f"log_{i}")
            raw_out = out_raw_dir / f"{log_id}.json"
            mjai_out = out_mjai_dir / f"{log_id}.jsonl"

            result = convert_url_to_tenhou6(url, str(raw_out))
            if result.get("ok"):
                convert_raw_to_mjai(str(raw_out), str(mjai_out), libriichi)
                print(f"[OK] [{i+1}/{len(links)}] {log_id}")
                success += 1
            else:
                print(f"[FAIL] [{i+1}/{len(links)}] {url}: {result.get('message', 'unknown error')}")
                failed += 1
        except Exception as e:
            print(f"[ERROR] [{i+1}/{len(links)}] {url}: {e}")
            failed += 1

    return success, failed


def main():
    parser = argparse.ArgumentParser(description="批量数据转换工具")
    parser.add_argument("--mode", choices=["majsoul", "tenhou", "all"], default="all",
                        help="转换模式: majsoul(雀魂), tenhou(天凤), all(全部)")
    parser.add_argument("--majsoul-raw", type=str, default="dataset/majsoul_raw",
                        help="雀魂原始 JSON 文件目录")
    parser.add_argument("--tenhou-links", type=str, default="dataset/links/tenhou_links.txt",
                        help="天凤链接列表文件")
    parser.add_argument("--out-dir", type=str, default="artifacts/converted",
                        help="输出目录")
    parser.add_argument("--libriichi", type=str, default=None,
                        help="libriichi CLI 路径")
    parser.add_argument("--dry-run", action="store_true",
                        help="仅打印计划，不执行转换")
    args = parser.parse_args()

    # 检查 libriichi
    libriichi = args.libriichi or ensure_libriichi()
    if not libriichi:
        print("[WARN] 未找到 libriichi CLI，将使用 fallback 转换器")
        print("[INFO] 安装 libriichi: cargo install libriichi")
    else:
        print(f"[OK] 找到 libriichi: {libriichi}")

    out_dir = Path(args.out_dir)
    total_success = 0
    total_failed = 0

    if args.mode in ["majsoul", "all"]:
        print("\n=== 处理雀魂牌谱 ===")
        s, f = process_majsoul_raw_files(
            Path(args.majsoul_raw),
            out_dir / "train",
            libriichi,
            args.dry_run
        )
        total_success += s
        total_failed += f

    if args.mode in ["tenhou", "all"]:
        print("\n=== 处理天凤牌谱 ===")
        s, f = process_tenhou_links(
            Path(args.tenhou_links),
            out_dir / "tenhou_raw",
            libriichi,
            args.dry_run
        )
        total_success += s
        total_failed += f

    print(f"\n=== 完成 ===")
    print(f"成功: {total_success}, 失败: {total_failed}")

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
