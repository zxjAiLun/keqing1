"""keqingv4 trainer, v2 phase 1."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

from keqingv4.checkpoint import (
    build_keqingv4_checkpoint_payload,
    restore_keqingv4_checkpoint,
)
from mahjong_env.action_space import HORA_IDX, NONE_IDX, REACH_IDX
from keqingv4.cached_dataset import CachedMjaiDatasetV4
from keqingv4.model import KeqingV4Model
from training import TaskSpec, train_model


def _unpack_v4_batch(batch, device: torch.device) -> Dict:
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
        pts_given_win_available,
        pts_given_dealin_available,
        opp_tenpai_available,
        event_history,
        event_history_available,
        v4_opportunity,
        discard_summary,
        call_summary,
        special_summary,
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
        "pts_given_win_available": to_device(pts_given_win_available),
        "pts_given_dealin_available": to_device(pts_given_dealin_available),
        "opp_tenpai_available": to_device(opp_tenpai_available),
        "v4_opportunity": to_device(v4_opportunity),
        "final_rank_target": final_rank_target.to(device, non_blocking=True).long(),
        "final_score_delta_points_target": final_score_delta_points_target.to(device, non_blocking=True),
        "model_kwargs": {
            "event_history": to_device(event_history),
            "discard_summary": to_float(discard_summary),
            "call_summary": to_float(call_summary),
            "special_summary": to_float(special_summary),
        },
    }


def _make_v4_task(cfg: Dict) -> TaskSpec:
    win_loss_weight = float(cfg.get("win_loss_weight", 0.5))
    dealin_loss_weight = float(cfg.get("dealin_loss_weight", 0.5))
    pts_given_win_loss_weight = float(cfg.get("pts_given_win_loss_weight", 0.0))
    pts_given_dealin_loss_weight = float(cfg.get("pts_given_dealin_loss_weight", 0.0))
    opp_tenpai_loss_weight = float(cfg.get("opp_tenpai_loss_weight", 0.0))
    mc_reg_loss_weight = float(cfg.get("mc_reg_loss_weight", 0.05))
    placement_cfg = cfg.get("placement", {})
    final_rank_loss_weight = float(
        placement_cfg.get("rank_loss_weight", cfg.get("final_rank_loss_weight", 0.05))
    )
    final_score_delta_loss_weight = float(
        placement_cfg.get(
            "final_score_delta_loss_weight",
            cfg.get("final_score_delta_loss_weight", 0.05),
        )
    )
    rank_pt_loss_weight = float(
        placement_cfg.get(
            "rank_pt_value_loss_weight",
            cfg.get("rank_pt_value_loss_weight", 0.0),
        )
    )
    rank_bonus = placement_cfg.get(
        "rank_bonus",
        cfg.get("rank_bonus", [90.0, 45.0, 0.0, -135.0]),
    )
    rank_bonus_norm = float(
        placement_cfg.get("rank_bonus_norm", cfg.get("rank_bonus_norm", 90.0))
    )
    rank_score_scale = float(
        placement_cfg.get("rank_score_scale", cfg.get("rank_score_scale", 0.0))
    )
    score_norm = float(
        placement_cfg.get("score_norm", cfg.get("score_norm", 30000.0))
    )
    typed_rank_loss_weight = float(cfg.get("typed_rank_loss_weight", 0.15))
    typed_rank_margin = float(cfg.get("typed_rank_margin", 0.20))
    reach_opportunity_weight = float(cfg.get("reach_opportunity_weight", 1.5))
    call_opportunity_weight = float(cfg.get("call_opportunity_weight", 1.5))
    hora_opportunity_weight = float(cfg.get("hora_opportunity_weight", 2.0))
    chosen_reach_weight = float(cfg.get("chosen_reach_weight", 1.0))
    chosen_dama_weight = float(cfg.get("chosen_dama_weight", 1.25))
    chosen_call_weight = float(cfg.get("chosen_call_weight", 1.0))
    chosen_none_weight = float(cfg.get("chosen_none_weight", 1.25))
    chosen_hora_weight = float(cfg.get("chosen_hora_weight", 1.0))
    chosen_continue_weight = float(cfg.get("chosen_continue_weight", 1.35))
    win_pos_weight = float(cfg.get("win_pos_weight", 2.0))
    dealin_pos_weight = float(cfg.get("dealin_pos_weight", 4.0))

    def _margin_pairwise_loss(
        pos: torch.Tensor,
        neg: torch.Tensor,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        raw = F.relu(typed_rank_margin - (pos - neg))
        if sample_weight is None:
            return raw.mean()
        weight = sample_weight.to(raw.device).float()
        return (raw * weight).sum() / weight.sum().clamp_min(1.0)

    def _masked_best_logit(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        neg_fill = torch.finfo(logits.dtype).min
        masked = logits.masked_fill(~mask, neg_fill)
        return masked.max(dim=-1).values

    def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_f = mask.to(values.device).float()
        if mask_f.ndim < values.ndim:
            while mask_f.ndim < values.ndim:
                mask_f = mask_f.unsqueeze(-1)
        return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)

    def compute_extra_loss(model, device: torch.device, batch_data: Dict, is_train: bool, batch_idx: int):
        del is_train, batch_idx
        aux = model.get_last_aux_outputs()
        policy_logits = model.get_last_policy_logits()
        win_logits = aux["win_prob"].squeeze(-1)
        dealin_logits = aux["dealin_prob"].squeeze(-1)
        pts_given_win_pred = aux["pts_given_win"].squeeze(-1)
        pts_given_dealin_pred = aux["pts_given_dealin"].squeeze(-1)
        opp_tenpai_logits = aux["opp_tenpai_logits"]
        rank_logits = aux["rank_logits"]
        final_score_delta_pred = aux["final_score_delta"].squeeze(-1)
        composed_ev = aux["composed_ev"].squeeze(-1)
        action_idx = batch_data["action_idx"]
        action_mask = batch_data["mask"] > 0
        v4_opportunity = batch_data["v4_opportunity"].bool()
        final_rank_target = batch_data["final_rank_target"]
        final_score_delta_target = batch_data["final_score_delta_points_target"].float() / score_norm

        win_pos_weight_t = torch.tensor(win_pos_weight, device=device)
        dealin_pos_weight_t = torch.tensor(dealin_pos_weight, device=device)
        rank_bonus_t = (
            torch.tensor(rank_bonus, dtype=torch.float32, device=device)
            / max(rank_bonus_norm, 1e-6)
        )
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

        pts_given_win_mask = batch_data["pts_given_win_available"] & (batch_data["win_target"] > 0.5)
        pts_given_win_loss = _masked_mean(
            F.smooth_l1_loss(
                pts_given_win_pred,
                batch_data["pts_given_win_target"],
                reduction="none",
            ),
            pts_given_win_mask,
        )
        pts_given_dealin_mask = batch_data["pts_given_dealin_available"] & (batch_data["dealin_target"] > 0.5)
        pts_given_dealin_loss = _masked_mean(
            F.smooth_l1_loss(
                pts_given_dealin_pred,
                batch_data["pts_given_dealin_target"],
                reduction="none",
            ),
            pts_given_dealin_mask,
        )
        opp_tenpai_mask = batch_data["opp_tenpai_available"]
        opp_tenpai_loss = _masked_mean(
            F.binary_cross_entropy_with_logits(
                opp_tenpai_logits,
                batch_data["opp_tenpai_target"],
                reduction="none",
            ).mean(dim=-1),
            opp_tenpai_mask,
        )
        final_rank_loss = F.cross_entropy(rank_logits.float(), final_rank_target)
        final_score_delta_loss = F.smooth_l1_loss(
            final_score_delta_pred.float(),
            final_score_delta_target,
        )
        rank_probs = F.softmax(rank_logits.float(), dim=-1)
        rank_pt_pred = (
            (rank_probs * rank_bonus_t.unsqueeze(0)).sum(dim=-1)
            + rank_score_scale * final_score_delta_pred.float()
        )
        rank_pt_target = (
            rank_bonus_t[final_rank_target]
            + rank_score_scale * final_score_delta_target
        )
        rank_pt_loss = F.smooth_l1_loss(rank_pt_pred, rank_pt_target)
        final_rank_acc = (rank_logits.argmax(dim=-1) == final_rank_target).float().mean()
        mc_reg_loss = F.smooth_l1_loss(composed_ev, batch_data["value_target"])

        selected_logits = policy_logits.gather(1, action_idx.unsqueeze(1)).squeeze(1)

        reach_opp_mask = v4_opportunity[:, 0]
        hora_opp_mask = v4_opportunity[:, 1]
        call_opp_mask = v4_opportunity[:, 2]

        discard_mask = action_mask[:, :34]
        best_discard_logits = _masked_best_logit(policy_logits[:, :34], discard_mask)

        call_family_mask = torch.zeros_like(policy_logits, dtype=torch.bool)
        call_family_mask[:, 35:42] = action_mask[:, 35:42]
        best_call_logits = _masked_best_logit(policy_logits, call_family_mask)
        none_logits = policy_logits[:, NONE_IDX]
        reach_logits = policy_logits[:, REACH_IDX]
        hora_logits = policy_logits[:, HORA_IDX]
        non_hora_mask = action_mask.clone()
        non_hora_mask[:, HORA_IDX] = False
        best_non_hora_logits = _masked_best_logit(policy_logits, non_hora_mask)

        zero = torch.tensor(0.0, device=device)
        reach_rank_loss = zero
        if typed_rank_loss_weight > 0:
            terms: list[torch.Tensor] = []
            chosen_reach = reach_opp_mask & (action_idx == REACH_IDX)
            chosen_dama = reach_opp_mask & (action_idx < 34)
            if chosen_reach.any():
                terms.append(
                    _margin_pairwise_loss(
                        selected_logits[chosen_reach],
                        best_discard_logits[chosen_reach],
                        torch.full_like(selected_logits[chosen_reach], reach_opportunity_weight * chosen_reach_weight),
                    )
                )
            if chosen_dama.any():
                terms.append(
                    _margin_pairwise_loss(
                        selected_logits[chosen_dama],
                        reach_logits[chosen_dama],
                        torch.full_like(selected_logits[chosen_dama], reach_opportunity_weight * chosen_dama_weight),
                    )
                )
            if terms:
                reach_rank_loss = torch.stack(terms).mean()

        call_rank_loss = zero
        if typed_rank_loss_weight > 0:
            terms = []
            chosen_none = call_opp_mask & (action_idx == NONE_IDX)
            chosen_call = call_opp_mask & (action_idx >= 35) & (action_idx <= 41)
            if chosen_none.any():
                terms.append(
                    _margin_pairwise_loss(
                        none_logits[chosen_none],
                        best_call_logits[chosen_none],
                        torch.full_like(none_logits[chosen_none], call_opportunity_weight * chosen_none_weight),
                    )
                )
            if chosen_call.any():
                terms.append(
                    _margin_pairwise_loss(
                        selected_logits[chosen_call],
                        none_logits[chosen_call],
                        torch.full_like(selected_logits[chosen_call], call_opportunity_weight * chosen_call_weight),
                    )
                )
            if terms:
                call_rank_loss = torch.stack(terms).mean()

        hora_rank_loss = zero
        if typed_rank_loss_weight > 0:
            terms = []
            chosen_hora = hora_opp_mask & (action_idx == HORA_IDX)
            chosen_continue = hora_opp_mask & (action_idx != HORA_IDX)
            if chosen_hora.any():
                terms.append(
                    _margin_pairwise_loss(
                        hora_logits[chosen_hora],
                        best_non_hora_logits[chosen_hora],
                        torch.full_like(hora_logits[chosen_hora], hora_opportunity_weight * chosen_hora_weight),
                    )
                )
            if chosen_continue.any():
                terms.append(
                    _margin_pairwise_loss(
                        selected_logits[chosen_continue],
                        hora_logits[chosen_continue],
                        torch.full_like(selected_logits[chosen_continue], hora_opportunity_weight * chosen_continue_weight),
                    )
                )
            if terms:
                hora_rank_loss = torch.stack(terms).mean()

        typed_rank_loss = (reach_rank_loss + call_rank_loss + hora_rank_loss) / 3.0

        legal_logits = policy_logits.masked_fill(~action_mask, torch.finfo(policy_logits.dtype).min)
        pred_idx = legal_logits.argmax(dim=-1)
        reach_slice_acc = ((pred_idx == action_idx) & reach_opp_mask).float().sum() / reach_opp_mask.float().sum().clamp_min(1.0)
        meld_slice_acc = ((pred_idx == action_idx) & call_opp_mask).float().sum() / call_opp_mask.float().sum().clamp_min(1.0)
        special_slice_mask = reach_opp_mask | hora_opp_mask | call_opp_mask
        special_slice_acc = ((pred_idx == action_idx) & special_slice_mask).float().sum() / special_slice_mask.float().sum().clamp_min(1.0)
        reach_chosen_rate = ((action_idx == REACH_IDX) & reach_opp_mask).float().sum() / reach_opp_mask.float().sum().clamp_min(1.0)
        meld_chosen_rate = (((action_idx >= 35) & (action_idx <= 41)) & call_opp_mask).float().sum() / call_opp_mask.float().sum().clamp_min(1.0)

        loss = (
            win_loss_weight * win_loss
            + dealin_loss_weight * dealin_loss
            + pts_given_win_loss_weight * pts_given_win_loss
            + pts_given_dealin_loss_weight * pts_given_dealin_loss
            + opp_tenpai_loss_weight * opp_tenpai_loss
            + final_rank_loss_weight * final_rank_loss
            + final_score_delta_loss_weight * final_score_delta_loss
            + rank_pt_loss_weight * rank_pt_loss
            + mc_reg_loss_weight * mc_reg_loss
            + typed_rank_loss_weight * typed_rank_loss
        )
        return loss, {
            "win_loss": float(win_loss.item()),
            "dealin_loss": float(dealin_loss.item()),
            "pts_given_win_loss": float(pts_given_win_loss.item()),
            "pts_given_dealin_loss": float(pts_given_dealin_loss.item()),
            "opp_tenpai_loss": float(opp_tenpai_loss.item()),
            "final_rank_loss": float(final_rank_loss.item()),
            "final_score_delta_loss": float(final_score_delta_loss.item()),
            "rank_pt_loss": float(rank_pt_loss.item()),
            "mc_reg_loss": float(mc_reg_loss.item()),
            "reach_rank_loss": float(reach_rank_loss.item()),
            "call_rank_loss": float(call_rank_loss.item()),
            "hora_rank_loss": float(hora_rank_loss.item()),
            "typed_rank_loss": float(typed_rank_loss.item()),
            "composed_ev_mean": float(composed_ev.mean().item()),
            "final_rank_acc": float(final_rank_acc.item()),
            "rank1_prob_mean": float(rank_probs[:, 0].mean().item()),
            "final_score_delta_mean": float(final_score_delta_pred.float().mean().item()),
            "rank_pt_value_mean": float(rank_pt_pred.mean().item()),
            "win_prob_mean": float(torch.sigmoid(win_logits).mean().item()),
            "dealin_prob_mean": float(torch.sigmoid(dealin_logits).mean().item()),
            "pts_given_win_mean": float(pts_given_win_pred.mean().item()),
            "pts_given_dealin_mean": float(pts_given_dealin_pred.mean().item()),
            "opp_tenpai_prob_mean": float(torch.sigmoid(opp_tenpai_logits).mean().item()),
            "pts_given_win_mask_rate": float(pts_given_win_mask.float().mean().item()),
            "pts_given_dealin_mask_rate": float(pts_given_dealin_mask.float().mean().item()),
            "opp_tenpai_mask_rate": float(opp_tenpai_mask.float().mean().item()),
            "reach_opp_rate": float(reach_opp_mask.float().mean().item()),
            "meld_opp_rate": float(call_opp_mask.float().mean().item()),
            "hora_opp_rate": float(hora_opp_mask.float().mean().item()),
            "reach_chosen_rate": float(reach_chosen_rate.item()),
            "meld_chosen_rate": float(meld_chosen_rate.item()),
            "reach_slice_acc": float(reach_slice_acc.item()),
            "meld_slice_acc": float(meld_slice_acc.item()),
            "special_slice_acc": float(special_slice_acc.item()),
        }

    return TaskSpec(
        name="keqingv4",
        unpack_batch=_unpack_v4_batch,
        compute_extra_loss=compute_extra_loss,
        log_metric_keys=(
            "win_loss",
            "dealin_loss",
            "pts_given_win_loss",
            "pts_given_dealin_loss",
            "opp_tenpai_loss",
            "final_rank_loss",
            "final_score_delta_loss",
            "rank_pt_loss",
            "mc_reg_loss",
            "reach_rank_loss",
            "call_rank_loss",
            "hora_rank_loss",
            "typed_rank_loss",
            "composed_ev_mean",
            "final_rank_acc",
            "rank1_prob_mean",
            "final_score_delta_mean",
            "rank_pt_value_mean",
            "win_prob_mean",
            "dealin_prob_mean",
            "pts_given_win_mean",
            "pts_given_dealin_mean",
            "opp_tenpai_prob_mean",
            "pts_given_win_mask_rate",
            "pts_given_dealin_mask_rate",
            "opp_tenpai_mask_rate",
            "reach_opp_rate",
            "meld_opp_rate",
            "hora_opp_rate",
            "reach_chosen_rate",
            "meld_chosen_rate",
            "reach_slice_acc",
            "meld_slice_acc",
            "special_slice_acc",
        ),
        best_metric_name="objective",
        best_metric_mode="min",
    )


def train(
    model: KeqingV4Model,
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
    aug_perms: int = 1,
    batch_size: int = 1024,
    num_workers: int = 4,
    files_per_epoch_ratio: float = 1.0,
):
    import random as _random
    from torch.utils.data import DataLoader

    if train_loader is None and train_files is None:
        raise ValueError("train_loader or train_files is required for keqingv4 training")

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
            train_ds = CachedMjaiDatasetV4(
                epoch_files,
                shuffle=True,
                seed=seed + epoch,
                aug_perms=aug_perms,
                buffer_size=buffer_size,
            )
            return DataLoader(
                train_ds,
                batch_size=batch_size,
                collate_fn=CachedMjaiDatasetV4.collate,
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
        task=_make_v4_task(cfg),
        cfg=cfg,
        output_dir=output_dir,
        resume_path=resume_path,
        weights_only=weights_only,
        device_str=device_str,
        checkpoint_payload_builder=build_keqingv4_checkpoint_payload,
        checkpoint_loader=restore_keqingv4_checkpoint,
    )
