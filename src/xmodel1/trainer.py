"""Xmodel1 trainer on top of shared training.train_model."""

from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

from training import TaskSpec, train_model
from xmodel1.cached_dataset import Xmodel1DiscardDataset
from xmodel1.checkpoint import (
    build_xmodel1_checkpoint_metadata,
    validate_xmodel1_checkpoint_metadata,
)
from xmodel1.model import Xmodel1Model


def _autocast_enabled(*, device: torch.device, is_train: bool, scaler) -> bool:
    if device.type != "cuda":
        return False
    if not is_train:
        return True
    return bool(scaler is not None and scaler.is_enabled())


def build_dataloaders(
    *,
    train_files,
    val_files,
    batch_size: int,
    num_workers: int,
    buffer_size: int,
    seed: int,
):
    common = dict(
        batch_size=batch_size,
        collate_fn=Xmodel1DiscardDataset.collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    train_ds = Xmodel1DiscardDataset(
        train_files,
        shuffle=True,
        buffer_size=buffer_size,
        seed=seed,
    )
    val_ds = Xmodel1DiscardDataset(
        val_files,
        shuffle=False,
        buffer_size=buffer_size,
        seed=seed,
    )
    return DataLoader(train_ds, **common), DataLoader(val_ds, **common)


def _unpack_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        moved = value.to(device, non_blocking=True)
        if moved.dtype == torch.float16 and device.type != "cuda":
            moved = moved.float()
        if key in {
            "candidate_tile_id",
            "chosen_candidate_idx",
            "response_action_idx",
            "chosen_response_action_idx",
            "response_post_candidate_tile_id",
            "action_idx_target",
            "final_rank_target",
        }:
            moved = moved.long()
        out[key] = moved
    return out


def _chosen_action_targets(
    candidate_tile_id: torch.Tensor,
    chosen_idx: torch.Tensor,
) -> torch.Tensor:
    targets = candidate_tile_id.gather(1, chosen_idx.unsqueeze(-1)).squeeze(-1).long()
    if torch.any((targets < 0) | (targets >= 34)):
        raise ValueError("chosen Xmodel1 discard target must map to a valid dahai action index")
    return targets


def _masked_mean(loss_values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    if loss_values.ndim == 0:
        return loss_values if torch.any(valid_mask) else loss_values.new_tensor(0.0)
    valid = valid_mask.to(loss_values.device).float()
    if valid.sum() <= 0:
        return loss_values.new_tensor(0.0)
    return (loss_values * valid).sum() / valid.sum().clamp_min(1.0)


def _resolve_pts_given_targets(batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    pts_given_win_target = batch.get("pts_given_win_target")
    pts_given_dealin_target = batch.get("pts_given_dealin_target")
    if pts_given_win_target is None or pts_given_dealin_target is None:
        raise ValueError("Xmodel1 batches require pts_given_win_target and pts_given_dealin_target")
    return pts_given_win_target.float(), pts_given_dealin_target.float()


def _candidate_ranking_loss(
    logits: torch.Tensor,
    teacher_scores: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    masked_logits = logits.float().masked_fill(mask <= 0, -1e4)
    teacher = teacher_scores.float().masked_fill(mask <= 0, -1e4)
    teacher_probs = torch.softmax(teacher, dim=-1)
    model_log_probs = F.log_softmax(masked_logits, dim=-1)
    return -(teacher_probs * model_log_probs).sum(dim=-1).mean()


def _hard_bad_penalty(
    logits: torch.Tensor,
    hard_bad_flag: torch.Tensor,
    chosen_idx: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    chosen_logits = logits.gather(1, chosen_idx.unsqueeze(-1)).squeeze(-1)
    chosen_hard_bad = hard_bad_flag.gather(1, chosen_idx.unsqueeze(-1)).squeeze(-1)
    valid = (mask.sum(dim=-1) > 0).float()
    return (chosen_logits.relu() * chosen_hard_bad * valid).mean()


def _response_candidate_ce_loss(
    logits: torch.Tensor,
    chosen_idx: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid_rows = (chosen_idx >= 0) & (mask.sum(dim=-1) > 0)
    if not torch.any(valid_rows):
        return logits.new_tensor(0.0)
    masked_logits = logits[valid_rows].float().masked_fill(mask[valid_rows] <= 0, -1e4)
    return F.cross_entropy(masked_logits, chosen_idx[valid_rows].long())


def _response_post_teacher_ce_loss(
    response_post_logits: torch.Tensor,
    chosen_response_idx: torch.Tensor,
    response_teacher_discard_idx: torch.Tensor,
    response_action_mask: torch.Tensor,
    response_post_candidate_mask: torch.Tensor,
) -> torch.Tensor:
    valid_rows = (chosen_response_idx >= 0) & (response_action_mask.sum(dim=-1) > 0)
    if not torch.any(valid_rows):
        return response_post_logits.new_tensor(0.0)
    row_idx = torch.nonzero(valid_rows, as_tuple=False).flatten()
    chosen_slots = chosen_response_idx[valid_rows].long()
    slot_logits = response_post_logits[row_idx, chosen_slots]
    slot_mask = response_post_candidate_mask[row_idx, chosen_slots]
    slot_teacher = response_teacher_discard_idx[row_idx, chosen_slots]
    valid_teacher = (slot_teacher >= 0) & (slot_mask.sum(dim=-1) > 0)
    if not torch.any(valid_teacher):
        return response_post_logits.new_tensor(0.0)
    masked_logits = slot_logits[valid_teacher].float().masked_fill(slot_mask[valid_teacher] <= 0, -1e4)
    return F.cross_entropy(masked_logits, slot_teacher[valid_teacher].long())


def _response_post_ranking_loss(
    response_post_logits: torch.Tensor,
    response_post_teacher_scores: torch.Tensor,
    chosen_response_idx: torch.Tensor,
    response_action_mask: torch.Tensor,
    response_post_candidate_mask: torch.Tensor,
) -> torch.Tensor:
    valid_rows = (chosen_response_idx >= 0) & (response_action_mask.sum(dim=-1) > 0)
    if not torch.any(valid_rows):
        return response_post_logits.new_tensor(0.0)
    row_idx = torch.nonzero(valid_rows, as_tuple=False).flatten()
    chosen_slots = chosen_response_idx[valid_rows].long()
    slot_logits = response_post_logits[row_idx, chosen_slots]
    slot_teacher_scores = response_post_teacher_scores[row_idx, chosen_slots]
    slot_mask = response_post_candidate_mask[row_idx, chosen_slots]
    valid_teacher = slot_mask.sum(dim=-1) > 0
    if not torch.any(valid_teacher):
        return response_post_logits.new_tensor(0.0)
    return _candidate_ranking_loss(
        slot_logits[valid_teacher],
        slot_teacher_scores[valid_teacher],
        slot_mask[valid_teacher],
    )


def _build_action_mask(
    candidate_tile_id: torch.Tensor,
    candidate_mask: torch.Tensor,
    response_action_idx: torch.Tensor,
    response_action_mask: torch.Tensor,
) -> torch.Tensor:
    batch = candidate_tile_id.shape[0]
    action_mask = torch.zeros((batch, 45), dtype=torch.uint8, device=candidate_tile_id.device)
    valid_discard = (candidate_mask > 0) & (candidate_tile_id >= 0) & (candidate_tile_id < 34)
    for tile_id in range(34):
        action_mask[:, tile_id] = (valid_discard & (candidate_tile_id == tile_id)).any(dim=1).to(torch.uint8)
    valid_response = (response_action_mask > 0) & (response_action_idx >= 34) & (response_action_idx < 45)
    for action_idx in range(34, 45):
        action_mask[:, action_idx] = (valid_response & (response_action_idx == action_idx)).any(dim=1).to(torch.uint8)
    return action_mask


class _Xmodel1TrainWrapper(nn.Module):
    def __init__(self, model: Xmodel1Model) -> None:
        super().__init__()
        self.model = model
        self._last_output = None

    def forward(self, tile_feat, scalar, **model_kwargs):
        out = self.model(
            tile_feat,
            scalar,
            model_kwargs["candidate_feat"],
            model_kwargs["candidate_tile_id"],
            model_kwargs["candidate_flags"],
            model_kwargs["candidate_mask"],
            model_kwargs.get("response_action_idx"),
            model_kwargs.get("response_action_mask"),
            model_kwargs.get("response_post_candidate_feat"),
            model_kwargs.get("response_post_candidate_tile_id"),
            model_kwargs.get("response_post_candidate_mask"),
            model_kwargs.get("response_post_candidate_flags"),
            history_summary=model_kwargs.get("history_summary"),
        )
        self._last_output = out
        composed_ev = self.model.get_last_aux_outputs()["composed_ev"]
        return out.action_logits, composed_ev

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict: bool = True):
        return self.model.load_state_dict(state_dict, strict=strict)

    def get_last_xmodel1_output(self):
        if self._last_output is None:
            raise RuntimeError("Xmodel1 training wrapper has no cached forward output")
        return self._last_output

    def get_last_aux_outputs(self):
        return self.model.get_last_aux_outputs()


def _make_xmodel1_task(cfg: dict[str, Any]) -> TaskSpec:
    ce_loss_weight = float(cfg.get("ce_loss_weight", 1.0))
    response_ce_loss_weight = float(cfg.get("response_ce_loss_weight", cfg.get("special_ce_loss_weight", 0.25)))
    response_post_ce_loss_weight = float(
        cfg.get("response_post_ce_loss_weight", cfg.get("special_rank_loss_weight", 0.25))
    )
    response_post_rank_loss_weight = float(cfg.get("response_post_rank_loss_weight", 0.15))
    rank_loss_weight = float(cfg.get("rank_loss_weight", 0.5))
    hard_bad_loss_weight = float(cfg.get("hard_bad_loss_weight", 0.25))
    win_loss_weight = float(cfg.get("win_loss_weight", 0.5))
    dealin_loss_weight = float(cfg.get("dealin_loss_weight", 0.5))
    pts_win_loss_weight = float(cfg.get("pts_win_loss_weight", 0.3))
    pts_dealin_loss_weight = float(cfg.get("pts_dealin_loss_weight", 0.3))
    opp_tenpai_loss_weight = float(cfg.get("opp_tenpai_loss_weight", 0.25))
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
            cfg.get("rank_pt_value_loss_weight", 0.01),
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

    def unpack_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, Any]:
        moved = _unpack_batch(batch, device)
        candidate_tile_id = moved["candidate_tile_id"]
        candidate_mask = moved["candidate_mask"]
        response_action_idx = moved["response_action_idx"]
        response_action_mask = moved["response_action_mask"]
        moved["tile_feat"] = moved["state_tile_feat"]
        moved["scalar"] = moved["state_scalar"]
        moved["mask"] = _build_action_mask(
            candidate_tile_id,
            candidate_mask,
            response_action_idx,
            response_action_mask,
        )
        moved["action_idx"] = moved["action_idx_target"].long()
        moved["value_target"] = (
            moved["win_target"].float() * moved["pts_given_win_target"].float()
            - moved["dealin_target"].float() * moved["pts_given_dealin_target"].float()
        )
        moved["model_kwargs"] = {
            "candidate_feat": moved["candidate_feat"],
            "candidate_tile_id": moved["candidate_tile_id"],
            "candidate_flags": moved["candidate_flags"],
            "candidate_mask": moved["candidate_mask"],
            "response_action_idx": moved.get("response_action_idx"),
            "response_action_mask": moved.get("response_action_mask"),
            "response_post_candidate_feat": moved.get("response_post_candidate_feat"),
            "response_post_candidate_tile_id": moved.get("response_post_candidate_tile_id"),
            "response_post_candidate_mask": moved.get("response_post_candidate_mask"),
            "response_post_candidate_flags": moved.get("response_post_candidate_flags"),
            "history_summary": moved.get("history_summary"),
        }
        return moved

    def compute_extra_loss(model, device: torch.device, batch_data: dict[str, Any], is_train: bool, batch_idx: int):
        del device, is_train, batch_idx
        out = model.get_last_xmodel1_output()
        sample_type = batch_data.get("sample_type")
        if sample_type is None:
            sample_type = torch.zeros_like(batch_data["action_idx"])
        sample_type = sample_type.long()
        discard_rows = sample_type == 0
        discard_target = batch_data["chosen_candidate_idx"].clamp_min(0)
        ce_raw = F.cross_entropy(out.discard_logits.float(), discard_target, reduction="none")
        ce_loss = _masked_mean(ce_raw, discard_rows)
        action_ce_loss = F.cross_entropy(
            out.action_logits.float().masked_fill(batch_data["mask"] <= 0, -1e4),
            batch_data["action_idx"],
        )
        response_ce_loss = _response_candidate_ce_loss(
            out.response_logits,
            batch_data["chosen_response_action_idx"],
            batch_data["response_action_mask"],
        )
        rank_loss = (
            _candidate_ranking_loss(
                out.discard_logits[discard_rows],
                batch_data["candidate_quality_score"][discard_rows],
                batch_data["candidate_mask"][discard_rows],
            )
            if torch.any(discard_rows)
            else out.discard_logits.new_tensor(0.0)
        )
        response_post_ce_loss = _response_post_teacher_ce_loss(
            out.response_post_logits,
            batch_data["chosen_response_action_idx"],
            batch_data["response_teacher_discard_idx"],
            batch_data["response_action_mask"],
            batch_data["response_post_candidate_mask"],
        )
        response_post_rank_loss = _response_post_ranking_loss(
            out.response_post_logits,
            batch_data["response_post_candidate_quality_score"],
            batch_data["chosen_response_action_idx"],
            batch_data["response_action_mask"],
            batch_data["response_post_candidate_mask"],
        )
        hard_bad_loss = (
            _hard_bad_penalty(
                out.discard_logits[discard_rows].float(),
                batch_data["candidate_hard_bad_flag"][discard_rows].float(),
                batch_data["chosen_candidate_idx"][discard_rows].clamp_min(0),
                batch_data["candidate_mask"][discard_rows],
            )
            if torch.any(discard_rows)
            else out.discard_logits.new_tensor(0.0)
        )
        win_target = batch_data["win_target"].float()
        dealin_target = batch_data["dealin_target"].float()
        win_loss = F.binary_cross_entropy_with_logits(out.win_logit.squeeze(-1).float(), win_target)
        dealin_loss = F.binary_cross_entropy_with_logits(out.dealin_logit.squeeze(-1).float(), dealin_target)
        pts_win_target, pts_dealin_target = _resolve_pts_given_targets(batch_data)
        pts_win_loss = _masked_mean(
            F.smooth_l1_loss(out.pts_given_win.squeeze(-1).float(), pts_win_target, reduction="none"),
            win_target > 0.5,
        )
        pts_dealin_loss = _masked_mean(
            F.smooth_l1_loss(out.pts_given_dealin.squeeze(-1).float(), pts_dealin_target, reduction="none"),
            dealin_target > 0.5,
        )
        opp_tenpai_target = batch_data.get("opp_tenpai_target")
        if opp_tenpai_target is None:
            opp_tenpai_loss = out.opp_tenpai_logits.new_tensor(0.0)
        else:
            opp_tenpai_loss = F.binary_cross_entropy_with_logits(
                out.opp_tenpai_logits.float(),
                opp_tenpai_target.float(),
            )
        final_rank_target = batch_data["final_rank_target"].long()
        final_score_delta_target = (
            batch_data["final_score_delta_points_target"].float() / score_norm
        )
        final_rank_loss = F.cross_entropy(out.rank_logits.float(), final_rank_target)
        final_score_delta_loss = F.smooth_l1_loss(
            out.final_score_delta.squeeze(-1).float(),
            final_score_delta_target,
        )
        rank_bonus_t = (
            torch.tensor(rank_bonus, dtype=torch.float32, device=out.rank_logits.device)
            / max(rank_bonus_norm, 1e-6)
        )
        rank_probs = F.softmax(out.rank_logits.float(), dim=-1)
        rank_pt_pred = (
            (rank_probs * rank_bonus_t.unsqueeze(0)).sum(dim=-1)
            + rank_score_scale * out.final_score_delta.squeeze(-1).float()
        )
        rank_pt_target = (
            rank_bonus_t[final_rank_target] + rank_score_scale * final_score_delta_target
        )
        rank_pt_loss = F.smooth_l1_loss(rank_pt_pred, rank_pt_target)
        extra_loss = (
            ce_loss_weight * ce_loss
            + response_ce_loss_weight * response_ce_loss
            + response_post_ce_loss_weight * response_post_ce_loss
            + response_post_rank_loss_weight * response_post_rank_loss
            + rank_loss_weight * rank_loss
            + hard_bad_loss_weight * hard_bad_loss
            + win_loss_weight * win_loss
            + dealin_loss_weight * dealin_loss
            + pts_win_loss_weight * pts_win_loss
            + pts_dealin_loss_weight * pts_dealin_loss
            + opp_tenpai_loss_weight * opp_tenpai_loss
            + final_rank_loss_weight * final_rank_loss
            + final_score_delta_loss_weight * final_score_delta_loss
            + rank_pt_loss_weight * rank_pt_loss
        )
        discard_acc = (
            (out.discard_logits.argmax(dim=-1)[discard_rows] == batch_data["chosen_candidate_idx"][discard_rows]).float().mean().item()
            if torch.any(discard_rows)
            else 0.0
        )
        action_acc = (
            (
                out.action_logits.float().masked_fill(batch_data["mask"] <= 0, -1e4).argmax(dim=-1)
                == batch_data["action_idx"]
            ).float().mean().item()
        )
        final_rank_acc = (
            (out.rank_logits.argmax(dim=-1) == final_rank_target).float().mean().item()
        )
        return extra_loss, {
            "loss": float((extra_loss + action_ce_loss).item()),
            "action_ce": float(action_ce_loss.item()),
            "response_ce": float(response_ce_loss.item()),
            "response_post_ce": float(response_post_ce_loss.item()),
            "response_post_rank": float(response_post_rank_loss.item()),
            "ce": float(ce_loss.item()),
            "rank": float(rank_loss.item()),
            "hard_bad": float(hard_bad_loss.item()),
            "win": float(win_loss.item()),
            "dealin": float(dealin_loss.item()),
            "pts_win": float(pts_win_loss.item()),
            "pts_dealin": float(pts_dealin_loss.item()),
            "opp_tenpai": float(opp_tenpai_loss.item()),
            "final_rank": float(final_rank_loss.item()),
            "final_score_delta": float(final_score_delta_loss.item()),
            "rank_pt": float(rank_pt_loss.item()),
            "acc": float(discard_acc),
            "action_acc": float(action_acc),
            "final_rank_acc": float(final_rank_acc),
            "rank_pt_value_mean": float(rank_pt_pred.mean().item()),
        }

    return TaskSpec(
        name="xmodel1",
        unpack_batch=unpack_batch,
        compute_extra_loss=compute_extra_loss,
        log_metric_keys=(
            "loss",
            "action_ce",
            "response_ce",
            "response_post_ce",
            "response_post_rank",
            "ce",
            "rank",
            "hard_bad",
            "win",
            "dealin",
            "pts_win",
            "pts_dealin",
            "opp_tenpai",
            "final_rank",
            "final_score_delta",
            "rank_pt",
            "acc",
            "action_acc",
            "final_rank_acc",
            "rank_pt_value_mean",
        ),
        best_metric_name="objective",
        best_metric_mode="min",
    )


def _rewrite_checkpoint(path: Path, *, cfg: dict[str, Any], model: Xmodel1Model) -> None:
    if not path.exists():
        return
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    checkpoint.update(build_xmodel1_checkpoint_metadata(cfg=cfg, model=model))
    torch.save(checkpoint, path)


def _rewrite_train_log(path: Path, *, cfg: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    rewritten: list[dict[str, Any]] = []
    last_row: dict[str, Any] | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        if "train" in row and "val" in row:
            last_row = row
            rewritten.append(row)
            continue
        train = {
            "loss": row["train_objective"],
            "ce": row["train_ce"],
            "action_ce": row.get("train_action_ce"),
            "response_ce": row.get("train_response_ce"),
            "response_post_ce": row.get("train_response_post_ce"),
            "response_post_rank": row.get("train_response_post_rank"),
            "acc": row.get("train_acc"),
            "action_acc": row.get("train_action_acc", row.get("train_acc")),
            "final_rank_acc": row.get("train_final_rank_acc"),
            "rank_pt_value_mean": row.get("train_rank_pt_value_mean"),
        }
        val = {
            "loss": row["val_objective"],
            "ce": row["val_ce"],
            "action_ce": row.get("val_action_ce"),
            "response_ce": row.get("val_response_ce"),
            "response_post_ce": row.get("val_response_post_ce"),
            "response_post_rank": row.get("val_response_post_rank"),
            "acc": row.get("val_acc"),
            "action_acc": row.get("val_action_acc", row.get("val_acc")),
            "final_rank_acc": row.get("val_final_rank_acc"),
            "rank_pt_value_mean": row.get("val_rank_pt_value_mean"),
        }
        for key in (
            "ce",
            "rank",
            "hard_bad",
            "win",
            "dealin",
            "pts_win",
            "pts_dealin",
            "opp_tenpai",
            "final_rank",
            "final_score_delta",
            "rank_pt",
        ):
            train[key] = row.get(f"train_{key}")
            val[key] = row.get(f"val_{key}")
        train["num_batches"] = int(row.get("train_num_batches", cfg.get("steps_per_epoch", 0) or 0) or 0)
        val["num_batches"] = int(row.get("val_num_batches", cfg.get("val_steps_per_epoch", 0) or 0) or 0)
        last_row = {
            "epoch": row["epoch"],
            "step": row["step"],
            "elapsed_s": 0.0,
            "train": train,
            "val": val,
        }
        rewritten.append(last_row)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rewritten),
        encoding="utf-8",
    )
    return last_row


def _write_training_summary(
    *,
    output_dir: Path,
    cfg: dict[str, Any],
    resume_path: Path | None,
    dataset_summary: dict[str, Any] | None,
) -> None:
    last_path = output_dir / "last.pth"
    best_path = output_dir / "best.pth"
    checkpoint = torch.load(last_path, map_location="cpu", weights_only=False)
    best_checkpoint = (
        torch.load(best_path, map_location="cpu", weights_only=False)
        if best_path.exists()
        else checkpoint
    )
    best_metric_value = float(
        best_checkpoint.get("best_val_loss", best_checkpoint.get("best_metric", 0.0))
    )
    summary = {
        "model_version": "xmodel1",
        "schema_name": checkpoint.get("schema_name"),
        "schema_version": checkpoint.get("schema_version"),
        "completed_epochs": int(checkpoint.get("epoch", cfg.get("num_epochs", 0))),
        "best_metric_name": "val_objective",
        "best_metric_mode": "min",
        "best_metric_value": best_metric_value,
        "best_val_loss": best_metric_value,
        "resume_path": str(resume_path) if resume_path is not None else None,
        "log_path": str(output_dir / "train_log.jsonl"),
        "last_checkpoint": str(last_path),
        "best_checkpoint": str(best_path) if best_path.exists() else None,
        "dataset_summary": dataset_summary,
        "cfg": cfg,
    }
    (output_dir / "training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def train(
    model: Xmodel1Model,
    *,
    train_loader,
    train_loader_factory=None,
    train_loader_summary_factory=None,
    val_loader,
    cfg: dict[str, Any],
    output_dir: Path,
    resume_path: Path | None = None,
    device_str: str = "cuda",
    dataset_summary: dict[str, Any] | None = None,
):
    if train_loader is None and train_loader_factory is None:
        raise ValueError("train_loader or train_loader_factory is required")

    if resume_path is not None:
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        validate_xmodel1_checkpoint_metadata(
            checkpoint,
            checkpoint_label=f"Xmodel1 checkpoint {resume_path}",
            allow_legacy_inference=False,
            require_complete_metadata=True,
        )
        print(f"xmodel1 train: resumed checkpoint={resume_path}", flush=True)

    cfg = dict(cfg)
    cfg.setdefault("policy_loss_weight", float(cfg.get("action_ce_loss_weight", 0.25)))
    cfg.setdefault("value_loss_weight", 0.0)
    output_dir.mkdir(parents=True, exist_ok=True)

    if train_loader_summary_factory is not None:
        summary = train_loader_summary_factory(0)
        if summary:
            print(
                f"  [epoch-plan] sampled_files={summary.get('sampled_files')} "
                f"sampled_samples={summary.get('sampled_samples')} "
                f"sampled_ratio={float(summary.get('sampled_ratio', 0.0) or 0.0):.3f} "
                f"global_step=0/{cfg.get('steps_per_epoch', 0)}",
                flush=True,
            )
    if cfg.get("steps_per_epoch"):
        print(f"  [train] batch=1/{cfg.get('steps_per_epoch')}", flush=True)
    if cfg.get("val_steps_per_epoch"):
        print(f"  [val] batch=1/{cfg.get('val_steps_per_epoch')}", flush=True)

    wrapped = _Xmodel1TrainWrapper(model)
    train_model(
        model=wrapped,
        train_loader=train_loader,
        train_loader_factory=train_loader_factory,
        val_loader=val_loader,
        task=_make_xmodel1_task(cfg),
        cfg=cfg,
        output_dir=output_dir,
        resume_path=resume_path,
        weights_only=False,
        device_str=device_str,
    )

    for checkpoint_name in ("last.pth", "best.pth", "best_meld.pth"):
        _rewrite_checkpoint(output_dir / checkpoint_name, cfg=cfg, model=model)
    last_row = _rewrite_train_log(output_dir / "train_log.jsonl", cfg=cfg)
    if last_row is not None:
        print(
            f"  [epoch-summary] train_batches={int(cfg.get('steps_per_epoch', 0) or 0)} "
            f"val_batches={int(cfg.get('val_steps_per_epoch', 0) or 0)} best_updated=true "
            f"global_step={last_row['step']}/{cfg.get('steps_per_epoch', 0)} epoch_eta=0s",
            flush=True,
        )
    _write_training_summary(
        output_dir=output_dir,
        cfg=cfg,
        resume_path=resume_path,
        dataset_summary=dataset_summary,
    )
    del train_loader
    del train_loader_factory
    del val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model


__all__ = ["build_dataloaders", "train", "_chosen_action_targets", "_unpack_batch", "_autocast_enabled"]
