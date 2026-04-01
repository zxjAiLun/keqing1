"""keqingv3 训练包装：基础 policy/value + score/win/dealin 多头。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from keqingv3.cached_dataset import CachedMjaiDatasetV3
from keqingv3.model import MahjongModel
from training import TaskSpec, train_model


def _unpack_v3_batch(batch, device: torch.device) -> Dict:
    (
        tile_feat,
        scalar,
        mask,
        action_idx,
        value_target,
        score_delta_target,
        win_target,
        dealin_target,
    ) = batch
    return {
        "tile_feat": tile_feat.to(device, non_blocking=True).float() if device.type != "cuda" else tile_feat.to(device, non_blocking=True),
        "scalar": scalar.to(device, non_blocking=True).float() if device.type != "cuda" else scalar.to(device, non_blocking=True),
        "mask": mask.to(device, non_blocking=True),
        "action_idx": action_idx.to(device),
        "value_target": value_target.to(device, non_blocking=True).float(),
        "score_delta_target": score_delta_target.to(device, non_blocking=True).float(),
        "win_target": win_target.to(device, non_blocking=True).float(),
        "dealin_target": dealin_target.to(device, non_blocking=True).float(),
    }


def _make_v3_task(cfg: Dict) -> TaskSpec:
    score_loss_weight = float(cfg.get("score_loss_weight", 0.5))
    win_loss_weight = float(cfg.get("win_loss_weight", 0.3))
    dealin_loss_weight = float(cfg.get("dealin_loss_weight", 0.3))
    win_pos_weight = float(cfg.get("win_pos_weight", 2.0))
    dealin_pos_weight = float(cfg.get("dealin_pos_weight", 6.0))

    def compute_extra_loss(model, device: torch.device, batch_data: Dict, is_train: bool, batch_idx: int):
        del is_train, batch_idx
        aux = model.get_last_aux_outputs()
        score_pred = aux["score_delta"].squeeze(-1)
        win_logits = aux["win_prob"].squeeze(-1)
        dealin_logits = aux["dealin_prob"].squeeze(-1)

        win_pos_weight_t = torch.tensor(win_pos_weight, device=device)
        dealin_pos_weight_t = torch.tensor(dealin_pos_weight, device=device)

        score_loss = F.smooth_l1_loss(score_pred, batch_data["score_delta_target"])
        win_loss = F.binary_cross_entropy_with_logits(
            win_logits,
            batch_data["win_target"],
            pos_weight=win_pos_weight_t,
        )
        dealin_loss = F.binary_cross_entropy_with_logits(
            dealin_logits,
            batch_data["dealin_target"],
            pos_weight=dealin_pos_weight_t,
        )

        loss = (
            score_loss_weight * score_loss
            + win_loss_weight * win_loss
            + dealin_loss_weight * dealin_loss
        )
        return loss, {
            "score_loss": float(score_loss.item()),
            "win_loss": float(win_loss.item()),
            "dealin_loss": float(dealin_loss.item()),
            "score_pred_mean": float(score_pred.mean().item()),
            "win_prob_mean": float(torch.sigmoid(win_logits).mean().item()),
            "dealin_prob_mean": float(torch.sigmoid(dealin_logits).mean().item()),
            "win_target_rate": float(batch_data["win_target"].mean().item()),
            "dealin_target_rate": float(batch_data["dealin_target"].mean().item()),
        }

    return TaskSpec(
        name="keqingv3_base",
        unpack_batch=_unpack_v3_batch,
        compute_extra_loss=compute_extra_loss,
        log_metric_keys=(
            "score_loss",
            "win_loss",
            "dealin_loss",
            "score_pred_mean",
            "win_prob_mean",
            "dealin_prob_mean",
            "win_target_rate",
            "dealin_target_rate",
        ),
        best_metric_name="objective",
        best_metric_mode="min",
    )


def train(
    model: MahjongModel,
    val_loader,
    cfg: Dict,
    output_dir: Path,
    train_loader=None,
    resume_path: Optional[Path] = None,
    weights_only: bool = False,
    device_str: str = "cuda",
    train_files: Optional[List] = None,
    seed: int = 42,
    use_cuda: bool = True,
    aug_perms: int = 2,
    batch_size: int = 1024,
    num_workers: int = 4,
    files_per_epoch_ratio: float = 1.0,
):
    import random as _random

    if train_loader is None and train_files is None:
        raise ValueError("train_loader or train_files is required for keqingv3 training")

    train_loader_factory = None
    if train_files is not None:
        buffer_size = int(cfg.get("buffer_size", 512))
        prefetch_factor = int(cfg.get("prefetch_factor", 2))
        pin_memory = bool(cfg.get("pin_memory", use_cuda))
        persistent_workers = bool(cfg.get("persistent_workers", num_workers > 0))

        def train_loader_factory(epoch: int):
            if files_per_epoch_ratio < 1.0:
                n = max(1, int(len(train_files) * files_per_epoch_ratio))
                epoch_files = _random.Random(seed + epoch).sample(train_files, n)
            else:
                epoch_files = train_files
            train_ds = CachedMjaiDatasetV3(
                epoch_files,
                shuffle=True,
                seed=seed + epoch,
                aug_perms=aug_perms,
                buffer_size=buffer_size,
            )
            return DataLoader(
                train_ds,
                batch_size=batch_size,
                collate_fn=CachedMjaiDatasetV3.collate,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(persistent_workers and num_workers > 0),
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            )

    return train_model(
        model=model,
        train_loader=train_loader,
        train_loader_factory=train_loader_factory,
        val_loader=val_loader,
        task=_make_v3_task(cfg),
        cfg=cfg,
        output_dir=output_dir,
        resume_path=resume_path,
        weights_only=weights_only,
        device_str=device_str,
    )
