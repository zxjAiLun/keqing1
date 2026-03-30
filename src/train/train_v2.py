"""keqingv2 训练入口（Meld Value Ranking Loss）。

用法：
  uv run python src/train/train_v2.py --config configs/keqingv2_default.yaml
  uv run python src/train/train_v2.py --config configs/keqingv2_default.yaml --resume artifacts/models/keqingv2/latest.pth
  uv run python src/train/train_v2.py --config configs/keqingv2_default.yaml --resume artifacts/models/keqingv1/best.pth --weights-only
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from keqingv1.model import MahjongModel
from keqingv1.cached_dataset import split_cached_files
from keqingv2.cached_dataset import CachedMjaiDatasetV2
from keqingv2.trainer import train


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train MahjongModel keqingv2 (Meld Ranking Loss)")
    parser.add_argument("--config", type=str, default="configs/keqingv2_default.yaml")
    parser.add_argument("--data_dirs", nargs="+", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--weights-only", action="store_true",
                        help="只加载模型权重，optimizer/scheduler/epoch 全部重置")
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

    data_dirs = [Path(d) for d in cfg.get("data_dirs", ["processed_v2/ds1"])]
    batch_size = cfg.get("batch_size", 1024)
    num_workers = cfg.get("num_workers", 4)
    aug_perms = cfg.get("aug_perms", 2)

    files_per_epoch_ratio = cfg.get("files_per_epoch_ratio", 1.0)

    train_files, val_files = split_cached_files(
        data_dirs,
        val_ratio=cfg.get("val_ratio", 0.05),
        seed=args.seed,
    )
    print(f"[keqingv2] Train files: {len(train_files)}, Val files: {len(val_files)}")

    use_cuda = args.device == "cuda"

    # val loader 固定不变
    val_ds = CachedMjaiDatasetV2(val_files, shuffle=False, aug_perms=0)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=CachedMjaiDatasetV2.collate,
        num_workers=num_workers,
        pin_memory=use_cuda,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )

    model = MahjongModel(
        hidden_dim=cfg.get("hidden_dim", 256),
        num_res_blocks=cfg.get("num_res_blocks", 4),
        dropout=cfg.get("dropout", 0.1),
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    output_dir = Path(cfg.get("output_dir", "artifacts/models/keqingv2"))
    resume_path = Path(args.resume) if args.resume else None

    train(
        model=model,
        train_files=train_files,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=output_dir,
        resume_path=resume_path,
        weights_only=args.weights_only,
        device_str=args.device,
        seed=args.seed,
        use_cuda=use_cuda,
        aug_perms=aug_perms,
        batch_size=batch_size,
        num_workers=num_workers,
        files_per_epoch_ratio=files_per_epoch_ratio,
    )


if __name__ == "__main__":
    main()
