#!/usr/bin/env python3
"""
批量训练脚本 - 按顺序用多个数据集训练模型

用法:
    python scripts/batch_train.py
    python scripts/batch_train.py --ds-list ds1,ds2,ds3
    python scripts/batch_train.py --init-checkpoint checkpoints/model.pth
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

# 添加src到path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from train.train_v2 import train


def main() -> None:
    parser = argparse.ArgumentParser(description="批量训练 - 按顺序用多个数据集训练模型")
    parser.add_argument(
        "--ds-list",
        type=str,
        default=None,
        help="数据集列表，如 ds1,ds2,ds3（默认使用ds1-ds13中所有有数据的）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v2.yaml",
        help="配置文件路径"
    )
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="初始模型路径（用于增量训练）"
    )
    parser.add_argument(
        "--base-out-dir",
        type=str,
        default="artifacts/models",
        help="模型输出根目录"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="如果某个数据集已训练则跳过"
    )
    args = parser.parse_args()
    
    # 确定数据集列表
    if args.ds_list:
        ds_names = args.ds_list.split(",")
    else:
        # 自动检测所有ds目录
        ds_base = Path("dataset/tenhou6")
        ds_dirs = sorted(ds_base.glob("ds*"), key=lambda p: int(p.name[2:]) if p.name[2:].isdigit() else 0)
        ds_names = [d.name for d in ds_dirs if d.is_dir()]
    
    print("=" * 60)
    print("批量训练开始")
    print("=" * 60)
    print(f"数据集: {ds_names}")
    print(f"配置: {args.config}")
    print(f"初始模型: {args.init_checkpoint or '无'}")
    print(f"输出目录: {args.base_out_dir}")
    print("=" * 60)
    
    current_checkpoint = args.init_checkpoint
    trained_count = 0
    skipped_count = 0
    
    for i, ds_name in enumerate(ds_names):
        print(f"\n{'=' * 60}")
        print(f"[{i+1}/{len(ds_names)}] 训练数据集: {ds_name}")
        print("=" * 60)
        
        # 检查是否已训练
        output_dir = Path(args.base_out_dir) / f"{ds_name}"
        if args.skip_existing and output_dir.exists():
            # 检查是否有best.npz
            if (output_dir / "best.npz").exists():
                print(f"  跳过（已存在）: {output_dir / 'best.npz'}")
                # 更新checkpoint指向这个
                current_checkpoint = str(output_dir / "best.npz")
                skipped_count += 1
                continue
        
        # 准备输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_out_dir = f"{args.base_out_dir}/{ds_name}_{timestamp}"
        
        # 构建命令行参数
        train_args = [
            "--config", args.config,
            "--pre-converted-dir", f"artifacts/converted/{ds_name}",
            "--out-dir", run_out_dir,
        ]
        
        if current_checkpoint:
            train_args.extend(["--init-checkpoint", current_checkpoint])
        
        # 解析参数
        import argparse as argparse_module
        parser_train = argparse_module.ArgumentParser()
        parser_train.add_argument("--config", default="configs/v2.yaml")
        parser_train.add_argument("--pre-converted-dir", default=None)
        parser_train.add_argument("--out-dir", default="artifacts/models")
        parser_train.add_argument("--init-checkpoint", default=None)
        parser_train.add_argument("--device", default="auto")
        parser_train.add_argument("--backend", default="torch")
        train_namespace = parser_train.parse_args(train_args)
        
        try:
            metrics = train(train_namespace)
            print(f"\n  数据集 {ds_name} 训练完成!")
            print(f"  样本数: {metrics.get('num_samples', 'N/A')}")
            print(f"  最佳验证损失: {metrics.get('best_val_loss', 'N/A')}")
            
            # 更新checkpoint指向最新的best.npz
            current_checkpoint = f"{run_out_dir}/best.npz"
            trained_count += 1
            
        except Exception as e:
            print(f"\n  数据集 {ds_name} 训练失败: {e}")
            # 如果失败，尝试继续用上一个checkpoint
            if current_checkpoint is None:
                print("  没有可用的checkpoint，停止训练")
                break
    
    print("\n" + "=" * 60)
    print("批量训练完成!")
    print("=" * 60)
    print(f"成功训练: {trained_count} 个数据集")
    print(f"跳过: {skipped_count} 个数据集")
    if current_checkpoint:
        print(f"最终模型: {current_checkpoint}")
    print("=" * 60)


if __name__ == "__main__":
    main()
