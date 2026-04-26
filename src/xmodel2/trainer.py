"""xmodel2 trainer: decomposed EV heads + placement auxiliary supervision."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from training import TaskSpec, train_model
from xmodel2.cached_dataset import CachedMjaiDatasetXmodel2
from xmodel2.losses import masked_mean
from xmodel2.model import Xmodel2Model


def _unpack_xmodel2_batch(batch, device: torch.device) -> Dict:
    (
        tile_feat,
        scalar,
        mask,
        action_idx,
        value_target,
        score_delta_target,
        win_target,
        dealin_target,
        pts_given_win_target,
        pts_given_dealin_target,
        opp_tenpai_target,
        final_rank_target,
        final_score_delta_points_target,
    ) = batch
    to_float = lambda tensor: tensor.to(device, non_blocking=True).float()
    to_device = lambda tensor: tensor.to(device, non_blocking=True)
    return {
        "tile_feat": to_float(tile_feat) if device.type != "cuda" else to_device(tile_feat),
        "scalar": to_float(scalar) if device.type != "cuda" else to_device(scalar),
        "mask": to_device(mask),
        "action_idx": action_idx.to(device),
        "value_target": to_float(value_target),
        "score_delta_target": to_float(score_delta_target),
        "win_target": to_float(win_target),
        "dealin_target": to_float(dealin_target),
        "pts_given_win_target": to_float(pts_given_win_target),
        "pts_given_dealin_target": to_float(pts_given_dealin_target),
        "opp_tenpai_target": to_float(opp_tenpai_target),
        "final_rank_target": final_rank_target.to(device, non_blocking=True).long(),
        "final_score_delta_points_target": final_score_delta_points_target.to(device, non_blocking=True),
        "model_kwargs": {},
    }


def _build_loaders(
    *,
    train_files: List[Path],
    val_files: List[Path],
    batch_size: int,
    num_workers: int,
    buffer_size: int,
    seed: int,
    aug_perms: int,
) -> tuple[DataLoader, DataLoader]:
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    train_loader = DataLoader(
        CachedMjaiDatasetXmodel2(
            train_files,
            shuffle=True,
            buffer_size=buffer_size,
            seed=seed,
            aug_perms=aug_perms,
        ),
        collate_fn=CachedMjaiDatasetXmodel2.collate,
        **common,
    )
    val_loader = DataLoader(
        CachedMjaiDatasetXmodel2(
            val_files,
            shuffle=False,
            buffer_size=buffer_size,
            seed=seed,
            aug_perms=max(0, min(1, aug_perms)),
        ),
        collate_fn=CachedMjaiDatasetXmodel2.collate,
        **common,
    )
    return train_loader, val_loader


def _make_xmodel2_task(cfg: Dict) -> TaskSpec:
    win_loss_weight = float(cfg.get("win_loss_weight", 0.25))
    dealin_loss_weight = float(cfg.get("dealin_loss_weight", 0.25))
    pts_given_win_loss_weight = float(cfg.get("pts_given_win_loss_weight", 0.20))
    pts_given_dealin_loss_weight = float(cfg.get("pts_given_dealin_loss_weight", 0.20))
    opp_tenpai_loss_weight = float(cfg.get("opp_tenpai_loss_weight", 0.15))
    win_pos_weight = float(cfg.get("win_pos_weight", 2.0))
    dealin_pos_weight = float(cfg.get("dealin_pos_weight", 4.0))
    placement_cfg = cfg.get("placement", {})
    final_rank_loss_weight = float(placement_cfg.get("rank_loss_weight", cfg.get("final_rank_loss_weight", 0.35)))
    final_score_delta_loss_weight = float(
        placement_cfg.get("final_score_delta_loss_weight", cfg.get("final_score_delta_loss_weight", 0.05))
    )
    rank_pt_loss_weight = float(
        placement_cfg.get("rank_pt_value_loss_weight", cfg.get("rank_pt_value_loss_weight", 0.0))
    )
    rank_bonus = placement_cfg.get("rank_bonus", cfg.get("rank_bonus", [90.0, 45.0, 0.0, -135.0]))
    rank_bonus_norm = float(placement_cfg.get("rank_bonus_norm", cfg.get("rank_bonus_norm", 90.0)))
    rank_score_scale = float(placement_cfg.get("rank_score_scale", cfg.get("rank_score_scale", 0.0)))
    score_norm = float(placement_cfg.get("score_norm", cfg.get("score_norm", 30000.0)))

    def compute_extra_loss(model, device: torch.device, batch_data: Dict, is_train: bool, batch_idx: int):
        del is_train, batch_idx
        aux = model.get_last_aux_outputs()
        win_logit = aux["win_logit"].squeeze(-1)
        dealin_logit = aux["dealin_logit"].squeeze(-1)
        pts_given_win = aux["pts_given_win"].squeeze(-1)
        pts_given_dealin = aux["pts_given_dealin"].squeeze(-1)
        opp_tenpai_logits = aux["opp_tenpai_logits"]
        rank_logits = aux["rank_logits"]
        final_score_delta_pred = aux["final_score_delta"].squeeze(-1)
        composed_ev = aux["composed_ev"].squeeze(-1)
        final_rank_target = batch_data["final_rank_target"]
        final_score_delta_target = batch_data["final_score_delta_points_target"].float() / score_norm
        rank_bonus_t = torch.tensor(rank_bonus, dtype=torch.float32, device=device) / max(rank_bonus_norm, 1e-6)

        win_loss = F.binary_cross_entropy_with_logits(
            win_logit,
            batch_data["win_target"],
            pos_weight=torch.tensor(win_pos_weight, device=device),
        )
        dealin_loss = F.binary_cross_entropy_with_logits(
            dealin_logit,
            batch_data["dealin_target"],
            pos_weight=torch.tensor(dealin_pos_weight, device=device),
        )
        win_mask = batch_data["win_target"] > 0.5
        dealin_mask = batch_data["dealin_target"] > 0.5
        pts_win_loss = masked_mean(
            F.smooth_l1_loss(
                pts_given_win,
                batch_data["pts_given_win_target"],
                reduction="none",
            ),
            win_mask,
        )
        pts_dealin_loss = masked_mean(
            F.smooth_l1_loss(
                pts_given_dealin,
                batch_data["pts_given_dealin_target"],
                reduction="none",
            ),
            dealin_mask,
        )
        opp_tenpai_loss = F.binary_cross_entropy_with_logits(
            opp_tenpai_logits,
            batch_data["opp_tenpai_target"],
        )
        final_rank_loss = F.cross_entropy(
            rank_logits.float(),
            final_rank_target,
        )
        final_score_delta_loss = F.smooth_l1_loss(
            final_score_delta_pred.float(),
            final_score_delta_target,
        )
        rank_probs = F.softmax(rank_logits.float(), dim=-1)
        rank_pt_pred = (rank_probs * rank_bonus_t.unsqueeze(0)).sum(dim=-1) + rank_score_scale * final_score_delta_pred.float()
        rank_pt_target = rank_bonus_t[final_rank_target] + rank_score_scale * final_score_delta_target
        rank_pt_loss = F.smooth_l1_loss(rank_pt_pred, rank_pt_target)
        final_rank_acc = (rank_logits.argmax(dim=-1) == final_rank_target).float().mean()

        extra_loss = (
            win_loss_weight * win_loss
            + dealin_loss_weight * dealin_loss
            + pts_given_win_loss_weight * pts_win_loss
            + pts_given_dealin_loss_weight * pts_dealin_loss
            + opp_tenpai_loss_weight * opp_tenpai_loss
            + final_rank_loss_weight * final_rank_loss
            + final_score_delta_loss_weight * final_score_delta_loss
            + rank_pt_loss_weight * rank_pt_loss
        )
        return extra_loss, {
            "win": float(win_loss.item()),
            "dealin": float(dealin_loss.item()),
            "pts_win": float(pts_win_loss.item()),
            "pts_dealin": float(pts_dealin_loss.item()),
            "opp_tenpai": float(opp_tenpai_loss.item()),
            "final_rank_loss": float(final_rank_loss.item()),
            "final_score_delta_loss": float(final_score_delta_loss.item()),
            "rank_pt_loss": float(rank_pt_loss.item()),
            "final_rank_acc": float(final_rank_acc.item()),
            "rank1_prob_mean": float(rank_probs[:, 0].mean().item()),
            "final_score_delta_mean": float(final_score_delta_pred.float().mean().item()),
            "rank_pt_value_mean": float(rank_pt_pred.mean().item()),
            "composed_ev_mean": float(composed_ev.mean().item()),
        }

    return TaskSpec(
        name="xmodel2",
        unpack_batch=_unpack_xmodel2_batch,
        compute_extra_loss=compute_extra_loss,
        log_metric_keys=(
            "win",
            "dealin",
            "pts_win",
            "pts_dealin",
            "opp_tenpai",
            "final_rank_loss",
            "final_score_delta_loss",
            "rank_pt_loss",
            "final_rank_acc",
            "rank1_prob_mean",
            "final_score_delta_mean",
            "rank_pt_value_mean",
            "composed_ev_mean",
        ),
        best_metric_name="objective",
        best_metric_mode="min",
    )


def train(
    *,
    model: Xmodel2Model,
    cfg: Dict,
    output_dir: Path,
    train_files: Optional[List[Path]] = None,
    val_files: Optional[List[Path]] = None,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 0,
    buffer_size: int = 512,
    aug_perms: int = 2,
    device_str: str = "cuda",
    resume_path: Optional[Path] = None,
    weights_only: bool = False,
):
    if train_loader is None or val_loader is None:
        if train_files is None or val_files is None:
            raise ValueError("train_files/val_files or train_loader/val_loader are required for xmodel2 training")
        train_loader, val_loader = _build_loaders(
            train_files=train_files,
            val_files=val_files,
            batch_size=batch_size,
            num_workers=num_workers,
            buffer_size=buffer_size,
            seed=seed,
            aug_perms=aug_perms,
        )

    return train_model(
        model,
        train_loader=train_loader,
        train_loader_factory=None,
        val_loader=val_loader,
        task=_make_xmodel2_task(cfg),
        cfg=cfg,
        output_dir=output_dir,
        resume_path=resume_path,
        weights_only=weights_only,
        device_str=device_str,
    )


__all__ = ["train"]
