#!/usr/bin/env python3
"""keqingv3 训练入口。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from torch.utils.data import DataLoader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train keqingv3 from cached npz files")
    parser.add_argument("--config", default="configs/keqingv3_default.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--weights-only", action="store_true")
    return parser.parse_args()


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))

    from keqingv3.cached_dataset import CachedMjaiDatasetV3, split_cached_files
    from keqingv3.model import MahjongModel
    from keqingv3.trainer import train

    args = _parse_args()
    cfg = _load_cfg(root / args.config)

    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    device = args.device or cfg.get("device", "cuda")

    if args.smoke:
        cfg["output_dir"] = cfg.get("output_dir", "artifacts/models/keqingv3") + "_smoke"
        cfg["num_epochs"] = 1
        cfg["batch_size"] = 64
        cfg["num_workers"] = 0
        cfg["buffer_size"] = 64
        cfg["prefetch_factor"] = 2
        cfg["pin_memory"] = False
        cfg["persistent_workers"] = False
        cfg["warmup_steps"] = 1
        cfg["steps_per_epoch"] = 2
        cfg["files_per_epoch_ratio"] = 1.0
        cfg["log_interval"] = 1

    data_dirs = [root / p for p in cfg.get("data_dirs", [])]
    if not data_dirs:
        raise ValueError("config.data_dirs is required")

    train_files, val_files = split_cached_files(
        data_dirs,
        val_ratio=float(cfg.get("val_ratio", 0.05)),
        seed=args.seed,
    )
    if not train_files or not val_files:
        raise RuntimeError(f"insufficient cached files: train={len(train_files)} val={len(val_files)}")

    if args.smoke:
        train_files = train_files[:2]
        val_files = val_files[:1]

    batch_size = int(cfg.get("batch_size", 1024))
    num_workers = int(cfg.get("num_workers", 4))
    aug_perms = int(cfg.get("aug_perms", 1))
    pin_memory = bool(cfg.get("pin_memory", args.device == "cuda"))
    persistent_workers = bool(cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(cfg.get("prefetch_factor", 2))
    buffer_size = int(cfg.get("buffer_size", 512))

    val_ds = CachedMjaiDatasetV3(
        val_files,
        shuffle=False,
        seed=args.seed,
        aug_perms=aug_perms,
        buffer_size=buffer_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=CachedMjaiDatasetV3.collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    model = MahjongModel(
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_res_blocks=int(cfg.get("num_res_blocks", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
    )

    output_dir = root / cfg.get("output_dir", "artifacts/models/keqingv3")
    resume_path = root / args.resume if args.resume else None

    print(
        f"keqingv3 train: train_files={len(train_files)} val_files={len(val_files)} "
        f"batch={batch_size} epochs={cfg.get('num_epochs')} device={device} output={output_dir}"
    )

    train(
        model=model,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=output_dir,
        resume_path=resume_path,
        weights_only=args.weights_only,
        device_str=device,
        train_files=train_files,
        seed=args.seed,
        use_cuda=(device == "cuda"),
        aug_perms=aug_perms,
        batch_size=batch_size,
        num_workers=num_workers,
        files_per_epoch_ratio=float(cfg.get("files_per_epoch_ratio", 1.0)),
    )


if __name__ == "__main__":
    main()
