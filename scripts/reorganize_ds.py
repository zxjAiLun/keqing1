#!/usr/bin/env python3
"""
重新整理 dataset/tenhou6/ds* 目录下的数据文件，确保每个目录不超过指定数量。

用法:
    python scripts/reorganize_ds.py
    python scripts/reorganize_ds.py --max-per-dir 1000
    python scripts/reorganize_ds.py --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path


MAX_PER_DIR_DEFAULT = 1000
DS_BASE_DIR = Path(__file__).parent.parent / "dataset" / "tenhou6"


def get_ds_dirs(base_dir: Path) -> list[Path]:
    """获取所有ds开头的目录，按数字排序"""
    if not base_dir.exists():
        return []
    dirs = [p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("ds")]
    dirs.sort(key=lambda p: int(p.name[2:]) if p.name[2:].isdigit() else float("inf"))
    return dirs


def get_file_count(d: Path) -> int:
    """获取目录下的文件数量（仅计算普通文件）"""
    if not d.exists():
        return 0
    return sum(1 for f in d.iterdir() if f.is_file())


def collect_all_files(base_dir: Path) -> list[Path]:
    """收集所有ds目录下的文件"""
    all_files = []
    for d in get_ds_dirs(base_dir):
        for f in d.iterdir():
            if f.is_file():
                all_files.append(f)
    return all_files


def get_next_ds_name(existing_dirs: list[Path]) -> str:
    """获取下一个ds目录的名称（如ds5, ds6等）"""
    existing_nums = []
    for d in existing_dirs:
        name = d.name
        if name.startswith("ds") and name[2:].isdigit():
            existing_nums.append(int(name[2:]))
    
    if not existing_nums:
        return "ds1"
    
    next_num = max(existing_nums) + 1
    return f"ds{next_num}"


def reorganize_ds(base_dir: Path, max_per_dir: int, dry_run: bool = False) -> None:
    """重新整理数据集，确保每个目录不超过max_per_dir个文件"""
    import random
    
    print(f"数据集整理脚本")
    print(f"目标目录: {base_dir}")
    print(f"每个目录最大文件数: {max_per_dir}")
    print(f"模拟运行: {dry_run}")
    print("-" * 50)
    
    # 1. 收集所有ds目录及其文件
    existing_dirs = get_ds_dirs(base_dir)
    
    # 统计当前各目录文件数
    print(f"\n当前目录状态:")
    dir_files: dict[Path, list[Path]] = {}
    for d in existing_dirs:
        files = [f for f in d.iterdir() if f.is_file()]
        dir_files[d] = files
        print(f"  {d.name}: {len(files)} 个文件")
    
    # 2. 收集所有需要重新分配的文件
    all_files: list[Path] = []
    for d, files in dir_files.items():
        all_files.extend(files)
    
    total_files = len(all_files)
    print(f"\n总共找到 {total_files} 个文件")
    
    if total_files == 0:
        print("没有找到任何文件，无需整理")
        return
    
    # 打乱文件顺序
    random.seed(42)
    random.shuffle(all_files)
    
    # 3. 准备目标目录列表
    # 策略：优先使用现有目录（按文件数升序），然后创建新目录
    target_dirs: list[Path] = []
    
    # 现有目录按文件数升序排列（优先填满文件少的）
    dirs_with_count = [(d, len(files)) for d, files in dir_files.items()]
    dirs_with_count.sort(key=lambda x: x[1])
    
    # 计算需要多少个目录
    num_target_dirs = (total_files + max_per_dir - 1) // max_per_dir
    
    # 添加现有目录（但超过max_per_dir的目录只保留max_per_dir个文件在原位）
    for d, count in dirs_with_count:
        target_dirs.append(d)
    
    # 创建新目录直到数量足够
    existing_nums = [int(d.name[2:]) for d in existing_dirs if d.name[2:].isdigit()]
    
    while len(target_dirs) < num_target_dirs:
        next_num = max(existing_nums) + 1 if existing_nums else 1
        existing_nums.append(next_num)
        new_dir = base_dir / f"ds{next_num}"
        target_dirs.append(new_dir)
        if not dry_run:
            new_dir.mkdir(parents=True, exist_ok=True)
        print(f"  {'[DRY-RUN] 创建' if dry_run else '创建'}目录: {new_dir.name}")
    
    print(f"\n需要 {num_target_dirs} 个目录")
    
    # 4. 重新分配文件
    # 每个目录先保留最多max_per_dir个文件，多余的放入待分配列表
    excess_files: list[Path] = []
    
    for d in target_dirs:
        current_files = dir_files.get(d, [])
        if len(current_files) > max_per_dir:
            # 保留前max_per_dir个，多余的放入excess_files
            excess_files.extend(current_files[max_per_dir:])
            dir_files[d] = current_files[:max_per_dir]
            print(f"  {d.name}: 超过限制，移出 {len(current_files) - max_per_dir} 个文件到待分配")
        else:
            dir_files[d] = current_files
    
    # 5. 分配剩余文件到还有空位的目录
    print(f"\n开始填充目录（已分配 {total_files - len(excess_files)} 个文件到原目录）...")
    print(f"需要分配 {len(excess_files)} 个多余文件")
    
    # 按当前文件数升序排列目标目录（优先填充文件少的）
    target_dirs.sort(key=lambda d: len(dir_files.get(d, [])))
    
    moves = []
    file_iter = iter(excess_files)
    
    # 用于dry-run模式下追踪最终状态
    final_counts: dict[Path, int] = {}
    for d in target_dirs:
        final_counts[d] = len(dir_files.get(d, []))
    
    for target_dir in target_dirs:
        current_count = len(dir_files.get(target_dir, []))
        need = max_per_dir - current_count
        
        if need <= 0:
            continue
        
        print(f"  填充 {target_dir.name}: 当前{current_count}个，需要{need}个")
        
        for _ in range(need):
            try:
                src = next(file_iter)
            except StopIteration:
                break
            
            dst = target_dir / src.name
            
            # 处理文件名冲突
            if dst.exists():
                base, ext = src.stem, src.suffix
                counter = 1
                while dst.exists():
                    dst = target_dir / f"{base}_{counter}{ext}"
                    counter += 1
            
            moves.append((src, dst))
            if not dry_run:
                shutil.move(str(src), str(dst))
                dir_files.setdefault(target_dir, []).append(dst)
            final_counts[target_dir] += 1
        
    
    # 处理还有剩余的文件（理论上不应该发生，因为我们已经计算好了目录数量）
    remaining = list(file_iter)
    if remaining:
        print(f"\n警告: 还有 {len(remaining)} 个文件未分配，创建额外目录...")
        existing_nums = [int(d.name[2:]) for d in target_dirs if d.name[2:].isdigit()]
        
        while remaining:
            next_num = max(existing_nums) + 1 if existing_nums else 1
            existing_nums.append(next_num)
            new_dir = base_dir / f"ds{next_num}"
            target_dirs.append(new_dir)
            final_counts[new_dir] = 0
            if not dry_run:
                new_dir.mkdir(parents=True, exist_ok=True)
            
            print(f"  {'[DRY-RUN] 创建' if dry_run else '创建'}额外目录: {new_dir.name}")
            
            for _ in range(min(max_per_dir, len(remaining))):
                if not remaining:
                    break
                src = remaining.pop(0)
                dst = new_dir / src.name
                
                if dst.exists():
                    base, ext = src.stem, src.suffix
                    counter = 1
                    while dst.exists():
                        dst = new_dir / f"{base}_{counter}{ext}"
                        counter += 1
                
                moves.append((src, dst))
                if not dry_run:
                    shutil.move(str(src), str(dst))
                final_counts[new_dir] += 1
    
    # 6. 输出结果摘要
    print(f"\n移动了 {len(moves)} 个文件:")
    
    from collections import defaultdict
    by_source = defaultdict(int)
    for src, dst in moves:
        by_source[src.parent.name] += 1
    
    for src_dir_name, count in sorted(by_source.items(), key=lambda x: int(x[0][2:]) if x[0][2:].isdigit() else 0):
        print(f"  从 {src_dir_name}: {count} 个文件")
    
    # 显示每个目录的最终状态
    print(f"\n整理后目录状态:")
    for d in sorted(target_dirs, key=lambda x: int(x.name[2:]) if x.name[2:].isdigit() else 0):
        count = final_counts.get(d, get_file_count(d))
        status = "✓" if count <= max_per_dir else "⚠超过限制"
        print(f"  {d.name}: {count} 个文件 {status}")
    
    total_after = sum(final_counts.values())
    print(f"\n总计: 整理前 {total_files} 个文件, 整理后 {total_after} 个文件")
    
    if dry_run:
        print("\n[DRY-RUN 模式] 以上只是模拟，未实际移动任何文件")
    else:
        print("\n整理完成!")


def main() -> None:
    parser = argparse.ArgumentParser(description="重新整理ds数据集，每目录不超过指定数量")
    parser.add_argument(
        "--max-per-dir",
        type=int,
        default=MAX_PER_DIR_DEFAULT,
        help=f"每个目录最大文件数 (默认: {MAX_PER_DIR_DEFAULT})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="模拟运行，不实际移动文件"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help=f"数据集根目录 (默认: {DS_BASE_DIR})"
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir) if args.base_dir else DS_BASE_DIR
    
    if not base_dir.exists():
        print(f"错误: 目录不存在: {base_dir}")
        sys.exit(1)
    
    reorganize_ds(base_dir, args.max_per_dir, args.dry_run)


if __name__ == "__main__":
    main()
