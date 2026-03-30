#!/usr/bin/env python3
"""
ds4数据集预处理脚本
预先将天凤6格式数据转换为MJAI格式，避免训练时等待数据转换

使用方法:
    # 从项目根目录运行
    ./preprocess_ds4.sh ds4
    
    # 或者手动运行
    cd src
    ../.venv/bin/python data/preprocess_ds4.py \
        --raw-dir ../dataset/tenhou6/ds4/ \
        --out-dir ../artifacts/converted/ds4 \
        --libriichi-bin ../third_party/Mortal/target/release/deps/libriichi.so
"""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
SRC_DIR = SCRIPT_DIR.parent
os.chdir(SRC_DIR)
sys.path.insert(0, str(SRC_DIR))

import argparse
import json
from typing import Optional
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from convert import libriichi_bridge
from mahjong_env import replay

convert_raw_to_mjai = libriichi_bridge.convert_raw_to_mjai
read_mjai_jsonl = replay.read_mjai_jsonl
build_supervised_samples = replay.build_supervised_samples


def preprocess_single_file(args):
    """预处理单个文件"""
    raw_path, out_dir, libriichi_bin = args
    try:
        stem = Path(raw_path).stem
        out_path = out_dir / f"{stem}.jsonl"

        result = convert_raw_to_mjai(raw_path, str(out_path), libriichi_bin)
        return {"status": "success", "input": raw_path, "output": str(out_path), **result}
    except Exception as e:
        return {"status": "error", "input": raw_path, "error": str(e)}


def validate_and_build_samples(jsonl_path: Path):
    """验证并构建训练样本"""
    try:
        events = read_mjai_jsonl(str(jsonl_path))
        samples = build_supervised_samples(events, actor_name_filter=None)
        return {
            "status": "success",
            "file": str(jsonl_path),
            "num_samples": len(samples)
        }
    except Exception as e:
        return {
            "status": "error",
            "file": str(jsonl_path),
            "error": str(e)
        }


def preprocess_dataset(
    raw_dir: str,
    out_dir: str,
    libriichi_bin: Optional[str] = None,
    num_workers: int = 8,
    skip_existing: bool = True
):
    """
    预处理数据集：将天凤格式转换为MJAI格式并验证

    Args:
        raw_dir: 原始数据目录
        out_dir: 输出目录
        libriichi_bin: libriichi二进制文件路径
        num_workers: 并行转换的线程数
        skip_existing: 跳过已存在的文件
    """
    raw_path = Path(raw_dir).resolve()
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    json_files = sorted(raw_path.glob("*.json"))
    if not json_files:
        raise ValueError(f"未找到JSON文件: {raw_dir}")

    print(f"找到 {len(json_files)} 个文件待处理")

    conversion_tasks = []
    for json_file in json_files:
        stem = json_file.stem
        potential_out = out_path / f"{stem}.jsonl"

        if skip_existing and potential_out.exists():
            continue

        conversion_tasks.append((str(json_file), out_path, libriichi_bin))

    if not conversion_tasks:
        print("所有文件已转换完成，跳过转换步骤")
    else:
        print(f"开始转换 {len(conversion_tasks)} 个文件 (并行度: {num_workers})")

        success_count = 0
        error_count = 0

        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(preprocess_single_file, task): task for task in conversion_tasks}

            for i, future in enumerate(as_completed(futures), 1):
                result = future.result()
                if result["status"] == "success":
                    success_count += 1
                    if i % 100 == 0 or i == len(conversion_tasks):
                        print(f"转换进度: {i}/{len(conversion_tasks)} ({100*i/len(conversion_tasks):.1f}%)")
                else:
                    error_count += 1
                    print(f"转换错误: {result['input']} - {result.get('error', 'Unknown')}")

        print(f"\n转换完成: 成功 {success_count}, 错误 {error_count}")

    jsonl_files = sorted(out_path.glob("*.jsonl"))
    print(f"\n验证并构建训练样本...")

    valid_count = 0
    total_samples = 0
    error_files = []

    for i, jsonl_file in enumerate(jsonl_files, 1):
        result = validate_and_build_samples(jsonl_file)
        if result["status"] == "success":
            valid_count += 1
            total_samples += result["num_samples"]
        else:
            error_files.append(result)

        if i % 500 == 0 or i == len(jsonl_files):
            print(f"验证进度: {i}/{len(jsonl_files)} - 有效文件: {valid_count}, 总样本数: {total_samples}")

    print(f"\n数据预处理完成:")
    print(f"  - 总文件数: {len(jsonl_files)}")
    print(f"  - 有效文件: {valid_count}")
    print(f"  - 总样本数: {total_samples}")
    print(f"  - 错误文件: {len(error_files)}")

    if error_files:
        print("\n错误文件列表:")
        for err in error_files[:10]:
            print(f"  - {err['file']}: {err.get('error', 'Unknown')}")
        if len(error_files) > 10:
            print(f"  ... 还有 {len(error_files) - 10} 个错误文件")

    return {
        "total_files": len(jsonl_files),
        "valid_files": valid_count,
        "total_samples": total_samples,
        "errors": error_files
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="预处理ds4数据集")
    parser.add_argument("--raw-dir", required=True, help="原始数据目录")
    parser.add_argument("--out-dir", required=True, help="输出目录")
    parser.add_argument("--libriichi-bin", help="libriichi二进制文件路径")
    parser.add_argument("--num-workers", type=int, default=8, help="并行线程数")
    parser.add_argument("--force", action="store_true", help="重新处理所有文件")

    args = parser.parse_args()

    preprocess_dataset(
        raw_dir=args.raw_dir,
        out_dir=args.out_dir,
        libriichi_bin=args.libriichi_bin,
        num_workers=args.num_workers,
        skip_existing=not args.force
    )
