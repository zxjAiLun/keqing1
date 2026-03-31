"""统一缓存训练入口：按 task 选择数据集与训练包装。"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from keqingv1.cached_dataset import CachedMjaiDataset, split_cached_files
from keqingv1.model import MahjongModel as MahjongModelV1
from keqingv1.trainer import train as train_base
from keqingv2.cached_dataset import CachedMjaiDatasetV2
from keqingv2.trainer import train as train_meld_rank
from keqingv3.cached_dataset import CachedMjaiDatasetV3
from keqingv3.model import MahjongModel as MahjongModelV3
from keqingv3.trainer import train as train_v3


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    parser = argparse.ArgumentParser(description="Train MahjongModel with shared cached task entrypoint")
    parser.add_argument("--task", choices=["base", "meld_rank", "v3_base"], default="base")
    parser.add_argument("--config", type=str, default="configs/keqing_default.yaml")
    parser.add_argument("--data_dirs", nargs="+", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--weights-only", action="store_true")
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

    data_dirs = [Path(d) for d in cfg.get("data_dirs", ["processed/ds1"])]
    batch_size = cfg.get("batch_size", 1024)
    num_workers = cfg.get("num_workers", 4)
    aug_perms = cfg.get("aug_perms", 2)
    use_cuda = args.device == "cuda"

    train_files, val_files = split_cached_files(
        data_dirs,
        val_ratio=cfg.get("val_ratio", 0.05),
        seed=args.seed,
    )
    print(f"[{args.task}] Train files: {len(train_files)}, Val files: {len(val_files)}")

    model_cls = MahjongModelV3 if args.task == "v3_base" else MahjongModelV1
    model = model_cls(
        hidden_dim=cfg.get("hidden_dim", 256),
        num_res_blocks=cfg.get("num_res_blocks", 4),
        dropout=cfg.get("dropout", 0.1),
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    output_dir = Path(
        cfg.get(
            "output_dir",
            (
                "artifacts/models/keqingv3"
                if args.task == "v3_base"
                else ("artifacts/models/keqingv2" if args.task == "meld_rank" else "artifacts/models/keqingv1")
            ),
        )
    )
    resume_path = Path(args.resume) if args.resume else None

    if args.task == "meld_rank":
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
        train_meld_rank(
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
            files_per_epoch_ratio=cfg.get("files_per_epoch_ratio", 1.0),
        )
        return

    if args.task == "v3_base":
        val_ds = CachedMjaiDatasetV3(val_files, shuffle=False, aug_perms=0)
        val_loader = DataLoader(
            val_ds,
            batch_size=batch_size,
            collate_fn=CachedMjaiDatasetV3.collate,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
        )
        train_v3(
            model=model,
            train_loader=None,
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
            files_per_epoch_ratio=cfg.get("files_per_epoch_ratio", 1.0),
        )
        return

    train_ds = CachedMjaiDataset(train_files, shuffle=True, seed=args.seed, aug_perms=aug_perms)
    val_ds = CachedMjaiDataset(val_files, shuffle=False, aug_perms=0)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        collate_fn=CachedMjaiDataset.collate,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=CachedMjaiDataset.collate,
        num_workers=num_workers,
        pin_memory=use_cuda,
    )
    train_base(
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
