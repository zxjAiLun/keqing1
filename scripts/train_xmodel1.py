#!/usr/bin/env python3
"""Train Xmodel1 from candidate-centric cached npz files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Xmodel1 from cached npz files")
    parser.add_argument("--config", default="configs/xmodel1_default.yaml")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))

    from xmodel1.cached_dataset import Xmodel1DiscardDataset, split_cached_files
    from xmodel1.model import Xmodel1Model
    from xmodel1.schema import (
        XMODEL1_CANDIDATE_FEATURE_DIM,
        XMODEL1_CANDIDATE_FLAG_DIM,
    )
    from xmodel1.trainer import train

    args = _parse_args()
    cfg = _load_cfg(root / args.config)
    cfg["model_name"] = "xmodel1"
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    device = args.device or cfg.get("device", "cuda")
    synthetic_smoke = False
    if args.smoke:
        cfg["output_dir"] = cfg.get("output_dir", "artifacts/models/xmodel1") + "_smoke"
        cfg["num_epochs"] = 1
        cfg["batch_size"] = 8
        cfg["num_workers"] = 0
        cfg["buffer_size"] = 16
        cfg["log_interval"] = 1

    data_dirs = []
    for p in cfg.get("data_dirs", []):
        path = Path(p)
        data_dirs.append(path if path.is_absolute() else root / path)
    batch_size = int(cfg.get("batch_size", 256))
    num_workers = int(cfg.get("num_workers", 2))
    buffer_size = int(cfg.get("buffer_size", 512))
    pin_memory = bool(cfg.get("pin_memory", device == "cuda"))
    persistent_workers = bool(cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(cfg.get("prefetch_factor", 2))
    train_loader = None
    val_loader = None
    train_files = []
    val_files = []
    if data_dirs and all(p.exists() for p in data_dirs):
        train_files, val_files = split_cached_files(
            data_dirs,
            val_ratio=float(cfg.get("val_ratio", 0.05)),
            seed=args.seed,
        )
        if args.smoke:
            all_files = train_files + [p for p in val_files if p not in train_files]
            if not all_files:
                raise RuntimeError("no cached files available for Xmodel1 smoke training")
            train_files = (train_files or all_files)[:2]
            val_files = (val_files or all_files[:1])[:1]
        elif not train_files or not val_files:
            raise RuntimeError(f"insufficient cached files: train={len(train_files)} val={len(val_files)}")
        train_loader = DataLoader(
            Xmodel1DiscardDataset(train_files, shuffle=True, buffer_size=buffer_size, seed=args.seed),
            batch_size=batch_size,
            collate_fn=Xmodel1DiscardDataset.collate,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
        val_loader = DataLoader(
            Xmodel1DiscardDataset(val_files, shuffle=False, buffer_size=buffer_size, seed=args.seed),
            batch_size=batch_size,
            collate_fn=Xmodel1DiscardDataset.collate,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
    elif args.smoke:
        synthetic_smoke = True

        class _SyntheticDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 32

            def __getitem__(self, idx):
                rng = np.random.default_rng(seed=args.seed + idx)
                candidate_mask = np.zeros((14,), dtype=np.uint8)
                candidate_mask[:5] = 1
                chosen = int(idx % 5)
                quality = np.zeros((14,), dtype=np.float32)
                quality[chosen] = 1.0
                rank_bucket = np.zeros((14,), dtype=np.int64)
                rank_bucket[chosen] = 3
                return {
                    "state_tile_feat": rng.random((57, 34), dtype=np.float32),
                    "state_scalar": rng.random((int(cfg.get("state_scalar_dim", 56)),), dtype=np.float32),
                    "candidate_feat": rng.random((14, XMODEL1_CANDIDATE_FEATURE_DIM), dtype=np.float32),
                    "candidate_tile_id": np.array([0, 1, 2, 3, 4] + [-1] * 9, dtype=np.int16),
                    "candidate_mask": candidate_mask,
                    "candidate_flags": np.zeros((14, XMODEL1_CANDIDATE_FLAG_DIM), dtype=np.uint8),
                    "chosen_candidate_idx": chosen,
                    "candidate_quality_score": quality,
                    "candidate_rank_bucket": rank_bucket,
                    "candidate_hard_bad_flag": np.zeros((14,), dtype=np.float32),
                    "global_value_target": np.float32(0.0),
                    "score_delta_target": np.float32(0.0),
                    "win_target": np.float32(0.0),
                    "dealin_target": np.float32(0.0),
                    "offense_quality_target": np.float32(1.0),
                }

        def _collate(batch):
            out = {}
            for key in batch[0].keys():
                values = [item[key] for item in batch]
                if key == "chosen_candidate_idx":
                    out[key] = torch.tensor(values, dtype=torch.long)
                else:
                    out[key] = torch.from_numpy(np.stack(values))
            return out

        train_loader = DataLoader(_SyntheticDataset(), batch_size=8, shuffle=False, collate_fn=_collate)
        val_loader = DataLoader(_SyntheticDataset(), batch_size=8, shuffle=False, collate_fn=_collate)
    else:
        raise ValueError("config.data_dirs is required and must exist unless --smoke is used")

    model = Xmodel1Model(
        state_tile_channels=int(cfg.get("state_tile_channels", 57)),
        state_scalar_dim=int(cfg.get("state_scalar_dim", 64)),
        candidate_feature_dim=int(cfg.get("candidate_feature_dim", XMODEL1_CANDIDATE_FEATURE_DIM)),
        candidate_flag_dim=int(cfg.get("candidate_flag_dim", XMODEL1_CANDIDATE_FLAG_DIM)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_res_blocks=int(cfg.get("num_res_blocks", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
    )

    output_dir = root / cfg.get("output_dir", "artifacts/models/xmodel1")
    resume_path = root / args.resume if args.resume else None
    print(
        f"xmodel1 train: synthetic_smoke={synthetic_smoke} train_files={len(train_files)} val_files={len(val_files)} "
        f"batch={batch_size} epochs={cfg.get('num_epochs')} device={device} output={output_dir}"
    )
    train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=output_dir,
        resume_path=resume_path,
        device_str=device,
    )


if __name__ == "__main__":
    main()
