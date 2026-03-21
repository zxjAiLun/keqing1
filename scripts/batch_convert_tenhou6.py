#!/usr/bin/env python3
"""Batch convert tenhou6 directories to mjai jsonl format.

Usage:
    python scripts/batch_convert_tenhou6.py --input dataset/tenhou6 --output artifacts/converted
    python scripts/batch_convert_tenhou6.py --ds ds1 ds2 ds3  # specific directories
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from convert.tenhou6_to_mjai import batch_convert_all_tenhou6


def main():
    parser = argparse.ArgumentParser(description="批量转换 tenhou6 目录到 mjai jsonl")
    parser.add_argument("--input", "-i", type=str, default="dataset/tenhou6",
                        help="输入目录 (默认: dataset/tenhou6)")
    parser.add_argument("--output", "-o", type=str, default="artifacts/converted",
                        help="输出目录 (默认: artifacts/converted)")
    parser.add_argument("--ds", nargs="+", default=None,
                        help="要处理的 ds 目录列表 (默认: 所有 ds)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="跳过已存在的文件")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        print(f"[ERROR] 输入目录不存在: {input_dir}")
        sys.exit(1)

    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    if args.ds:
        print(f"处理目录: {args.ds}")
    else:
        print("处理目录: 所有 ds")
    print()

    total_success, total_failed = batch_convert_all_tenhou6(
        input_dir,
        output_dir,
        ds_list=args.ds,
        skip_existing=args.skip_existing,
    )

    print()
    print("=" * 50)
    print(f"总计: {total_success} 成功, {total_failed} 失败")
    print("=" * 50)

    if total_failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
