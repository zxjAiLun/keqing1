"""keqingv2 训练包装：在共享 trainer 上注入 Meld Value Ranking Loss。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from keqingv1.model import MahjongModel
from training import TaskSpec, train_model


def _meld_rank_loss(
    model: nn.Module,
    device: torch.device,
    tf_none: torch.Tensor,
    sc_none: torch.Tensor,
    tf_meld: torch.Tensor,
    sc_meld: torch.Tensor,
    rank_signs: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """对预编码的 none/meld 特征对计算 margin ranking loss（GPU 并行）。"""
    valid = rank_signs != 0
    if not valid.any():
        return torch.tensor(0.0, device=device)

    tf_n = tf_none[valid].to(device)
    sc_n = sc_none[valid].to(device)
    tf_m = tf_meld[valid].to(device)
    sc_m = sc_meld[valid].to(device)
    signs_t = rank_signs[valid].to(device)

    tf_both = torch.cat([tf_n, tf_m], dim=0)
    sc_both = torch.cat([sc_n, sc_m], dim=0)
    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        with torch.enable_grad():
            _, v_both = model(tf_both, sc_both)
    v_both = v_both.squeeze(1)
    n = tf_n.shape[0]
    v_none_v = v_both[:n]
    v_meld_v = v_both[n:]

    raw = margin + signs_t * (v_none_v - v_meld_v)
    return F.relu(raw).mean()


def _unpack_v2_batch(batch, device: torch.device) -> Dict:
    tile_feat, scalar, mask, action_idx, value_target, tf_none, sc_none, tf_meld, sc_meld, rank_signs = batch
    return {
        "tile_feat": tile_feat.to(device),
        "scalar": scalar.to(device),
        "mask": mask.to(device),
        "action_idx": action_idx.to(device),
        "value_target": value_target.to(device),
        "tf_none": tf_none,
        "sc_none": sc_none,
        "tf_meld": tf_meld,
        "sc_meld": sc_meld,
        "rank_signs": rank_signs,
    }


def _make_v2_task(cfg: Dict) -> TaskSpec:
    rank_loss_weight = cfg.get("rank_loss_weight", 0.1)
    rank_margin = cfg.get("rank_margin", 0.05)
    rank_loss_every_n = cfg.get("rank_loss_every_n", 4)

    def compute_extra_loss(model, device: torch.device, batch_data: Dict, is_train: bool, batch_idx: int):
        if not is_train or rank_loss_weight <= 0 or (batch_idx % rank_loss_every_n != 0):
            return torch.tensor(0.0, device=device), {"rank_loss": 0.0}
        rank_signs = batch_data["rank_signs"]
        if not (rank_signs != 0).any():
            return torch.tensor(0.0, device=device), {"rank_loss": 0.0}

        rl = _meld_rank_loss(
            model,
            device,
            batch_data["tf_none"],
            batch_data["sc_none"],
            batch_data["tf_meld"],
            batch_data["sc_meld"],
            rank_signs,
            rank_margin,
        )
        return rank_loss_weight * rl, {"rank_loss": float(rl.item())}

    return TaskSpec(
        name="keqingv2_meld_rank",
        unpack_batch=_unpack_v2_batch,
        compute_extra_loss=compute_extra_loss,
        log_metric_keys=("rank_loss",),
    )


def train(
    model: MahjongModel,
    train_files: List,
    val_loader: DataLoader,
    cfg: Dict,
    output_dir: Path,
    resume_path: Optional[Path] = None,
    weights_only: bool = False,
    device_str: str = "cuda",
    seed: int = 42,
    use_cuda: bool = True,
    aug_perms: int = 1,
    batch_size: int = 2048,
    num_workers: int = 8,
    files_per_epoch_ratio: float = 1.0,
):
    from keqingv2.cached_dataset import CachedMjaiDatasetV2
    import random as _random

    task = _make_v2_task(cfg)

    def train_loader_factory(epoch: int):
        if files_per_epoch_ratio < 1.0:
            n = max(1, int(len(train_files) * files_per_epoch_ratio))
            epoch_files = _random.Random(seed + epoch).sample(train_files, n)
        else:
            epoch_files = train_files
        train_ds = CachedMjaiDatasetV2(
            epoch_files,
            shuffle=True,
            seed=seed + epoch,
            aug_perms=aug_perms,
        )
        return DataLoader(
            train_ds,
            batch_size=batch_size,
            collate_fn=CachedMjaiDatasetV2.collate,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
        )

    return train_model(
        model=model,
        train_loader=None,
        train_loader_factory=train_loader_factory,
        val_loader=val_loader,
        task=task,
        cfg=cfg,
        output_dir=output_dir,
        resume_path=resume_path,
        weights_only=weights_only,
        device_str=device_str,
    )
