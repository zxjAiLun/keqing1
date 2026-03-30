#!/usr/bin/env python3
import json
import sys
import os
from pathlib import Path
from collections import Counter

def verify_kyoku_detailed(orig_k, conv_k):
    """详细验证单局数据，返回差异列表"""
    differences = []

    if orig_k[0] != conv_k[0]:
        differences.append(f"KyokuMeta: {orig_k[0]} vs {conv_k[0]}")
    if orig_k[1] != conv_k[1]:
        differences.append(f"Scores: {orig_k[1]} vs {conv_k[1]}")
    if orig_k[2] != conv_k[2]:
        differences.append(f"Dora: {orig_k[2]} vs {conv_k[2]}")
    if orig_k[3] != conv_k[3]:
        differences.append(f"Ura: {orig_k[3]} vs {conv_k[3]}")

    for i in range(4, len(orig_k) - 1):
        if i >= len(conv_k):
            differences.append(f"Player {i-4} 数据: 缺少")
        elif orig_k[i] != conv_k[i]:
            if isinstance(orig_k[i], list) and len(orig_k[i]) > 0 and isinstance(orig_k[i][0], str):
                differences.append(f"Player {i-4} Takes: 含特殊标记")
            else:
                differences.append(f"Player {i-4} 数据: {orig_k[i][:5]}... vs {conv_k[i][:5]}...")

    return differences

def verify_file(jsonl_path, tenhou6_path):
    """验证单个文件"""
    try:
        with open(jsonl_path, 'r') as f:
            events = [json.loads(l) for l in f if l.strip()]
    except Exception as e:
        return False, f"读取jsonl失败: {e}", []

    try:
        with open(tenhou6_path, 'r') as f:
            original = json.load(f)
    except Exception as e:
        return False, f"读取tenhou6失败: {e}", []

    from tools.mjai_jsonl_to_tenhou6 import convert_mjai_jsonl_to_tenhou6
    converted = convert_mjai_jsonl_to_tenhou6(events)

    if len(original['log']) != len(converted['log']):
        return False, f"局数: {len(original['log'])} vs {len(converted['log'])}", []

    all_diffs = []
    for i, (orig_k, conv_k) in enumerate(zip(original['log'], converted['log'])):
        diffs = verify_kyoku_detailed(orig_k, conv_k)
        if diffs:
            all_diffs.append((i+1, diffs))

    if all_diffs:
        return False, f"第{all_diffs[0][0]}局有问题", all_diffs

    return True, "OK", []

def main():
    converted_dir = Path("artifacts/converted/train_ds3")
    dataset_dir = Path("dataset/tenhou6/ds3")

    converted_files = sorted(converted_dir.glob("*.jsonl"))
    total = len(converted_files)
    matched = 0
    failed = []

    error_types = Counter()

    print(f"开始验证 {total} 个文件...")
    print()

    for i, jsonl_file in enumerate(converted_files):
        tenhou6_file = dataset_dir / (jsonl_file.stem + ".json")

        if not tenhou6_file.exists():
            failed.append((jsonl_file.name, "对应文件不存在"))
            error_types["文件不存在"] += 1
            continue

        success, msg, diffs = verify_file(jsonl_file, tenhou6_file)

        if success:
            matched += 1
        else:
            failed.append((jsonl_file.name, msg, diffs))
            for kyoku_num, kyoku_diffs in diffs:
                for diff in kyoku_diffs:
                    if "Dora" in diff:
                        error_types["Dora不匹配"] += 1
                    elif "Ura" in diff:
                        error_types["Ura不匹配"] += 1
                    elif "Scores" in diff:
                        error_types["Scores不匹配"] += 1
                    elif "KyokuMeta" in diff:
                        error_types["KyokuMeta不匹配"] += 1
                    else:
                        error_types["Player数据不匹配"] += 1

        if (i + 1) % 200 == 0:
            print(f"进度: {i+1}/{total} ({matched} 匹配, {len(failed)} 失败)")

    print()
    print("=" * 60)
    print(f"总计: {total}")
    print(f"匹配: {matched} ({100*matched/total:.1f}%)")
    print(f"失败: {len(failed)} ({100*len(failed)/total:.1f}%)")

    print()
    print("=== 错误类型分布 ===")
    for error_type, count in error_types.most_common():
        print(f"  {error_type}: {count} ({100*count/len(failed):.1f}%)")

    if failed:
        print()
        print("=== 失败文件示例 ===")
        for name, msg, diffs in failed[:10]:
            print(f"  {name}: {msg}")

    print()
    print("=" * 60)
    print("根本原因分析:")
    print("  - converted jsonl 来自 tenhou6 转 mjai 格式")
    print("  - 转换过程中丢失了 dora_markers 和 ura_markers")
    print("  - hora 事件缺少 dora/ura 字段")

    return 0 if len(failed) == 0 else 1

if __name__ == "__main__":
    sys.exit(main())
