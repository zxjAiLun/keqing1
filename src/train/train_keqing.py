"""keqingv1 训练入口。

用法：
  python -m train.train_keqing --data_dirs artifacts/converted_mjai/ds1 artifacts/converted_mjai/ds2 --output_dir artifacts/models/keqingv1
  python -m train.train_keqing --config configs/keqing_default.yaml --resume artifacts/models/keqingv1/latest.pth
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from keqingv1.dataset import MjaiIterableDataset, split_files
from keqingv1.cached_dataset import CachedMjaiDataset, split_cached_files
from keqingv1.model import MahjongModel
from keqingv1.trainer import train


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train MahjongModel keqingv1")
    parser.add_argument("--config", type=str, default="configs/keqing_default.yaml")
    parser.add_argument("--data_dirs", nargs="+", type=str, default=None,
                        help="覆盖 config 中的 data_dirs")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="覆盖 config 中的 output_dir")
    parser.add_argument("--resume", type=str, default=None,
                        help="checkpoint 路径，用于断点续训")
    parser.add_argument("--weights-only", action="store_true",
                        help="只加载模型权重，optimizer/scheduler/epoch 全部重置")
    parser.add_argument("--cached", action="store_true",
                        help="使用预处理缓存数据集（.npz），需先运行 scripts/preprocess_cached.py")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg: dict = {}
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f) or {}
    else:
        print(f"Config not found: {cfg_path}, using defaults.")

    if args.data_dirs:
        cfg["data_dirs"] = args.data_dirs
    if args.output_dir:
        cfg["output_dir"] = args.output_dir

    set_seed(args.seed)

    data_dirs = [Path(d) for d in cfg.get("data_dirs", ["artifacts/converted_mjai/ds1"])]
    batch_size = cfg.get("batch_size", 1024)
    num_workers = cfg.get("num_workers", 4)
    use_cached = args.cached or cfg.get("use_cached", False)

    if use_cached:
        train_files, val_files = split_cached_files(
            data_dirs,
            val_ratio=cfg.get("val_ratio", 0.05),
            seed=args.seed,
        )
        print(f"[cached] Train files: {len(train_files)}, Val files: {len(val_files)}")
        aug_perms = cfg.get("aug_perms", 2)
        train_ds = CachedMjaiDataset(train_files, shuffle=True, seed=args.seed, aug_perms=aug_perms)
        val_ds = CachedMjaiDataset(val_files, shuffle=False, aug_perms=0)
        collate_fn = CachedMjaiDataset.collate
    else:
        train_files, val_files = split_files(
            data_dirs,
            val_ratio=cfg.get("val_ratio", 0.05),
            seed=args.seed,
        )
        print(f"Train files: {len(train_files)}, Val files: {len(val_files)}")
        actor_name = cfg.get("actor_name", None)
        augment = cfg.get("augment", False)
        train_ds = MjaiIterableDataset(train_files, shuffle=True, augment=augment, actor_name=actor_name)
        val_ds = MjaiIterableDataset(val_files, shuffle=False, augment=False, actor_name=actor_name)
        collate_fn = MjaiIterableDataset.collate

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(args.device == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=(args.device == "cuda"),
    )

    model = MahjongModel(
        hidden_dim=cfg.get("hidden_dim", 256),
        num_res_blocks=cfg.get("num_res_blocks", 4),
        dropout=cfg.get("dropout", 0.1),
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    output_dir = Path(cfg.get("output_dir", "artifacts/models/keqingv1"))
    resume_path = Path(args.resume) if args.resume else None

    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=output_dir,
        resume_path=resume_path,
        weights_only=args.weights_only,
        device_str=args.device,
    )


if __name__ == "__main__":
    main()
