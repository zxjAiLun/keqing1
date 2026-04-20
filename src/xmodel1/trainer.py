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
from xmodel1.schema import (
    XMODEL1_CHI_SPECIAL_TYPES,
    XMODEL1_KAN_SPECIAL_TYPES,
    XMODEL1_SPECIAL_TYPE_DAMA,
    XMODEL1_SPECIAL_TYPE_NONE,
    XMODEL1_SPECIAL_TYPE_PON,
    XMODEL1_SPECIAL_TYPE_REACH,
)


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
            "special_candidate_type_id",
            "chosen_special_candidate_idx",
            "action_idx_target",
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
        raise ValueError("Xmodel1 v3 batches require pts_given_win_target and pts_given_dealin_target")
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


def _special_candidate_ce_loss(
    logits: torch.Tensor,
    chosen_idx: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    valid_rows = (chosen_idx >= 0) & (mask.sum(dim=-1) > 0)
    if not torch.any(valid_rows):
        return logits.new_tensor(0.0)
    masked_logits = logits[valid_rows].float().masked_fill(mask[valid_rows] <= 0, -1e4)
    return F.cross_entropy(masked_logits, chosen_idx[valid_rows].long())


def _pairwise_margin_loss(
    pos_logits: torch.Tensor,
    neg_logits: torch.Tensor,
    *,
    margin: float,
) -> torch.Tensor:
    if pos_logits.numel() == 0:
        return pos_logits.new_tensor(0.0)
    return F.relu(margin - (pos_logits - neg_logits)).mean()


def _special_comparison_losses(
    special_logits: torch.Tensor,
    special_type_id: torch.Tensor,
    special_mask: torch.Tensor,
    chosen_special_idx: torch.Tensor,
    special_hard_bad_flag: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = special_logits.device
    valid = (special_mask > 0) & (special_type_id >= 0)
    chosen_valid = (chosen_special_idx >= 0) & (valid.sum(dim=-1) > 0)
    zero = torch.tensor(0.0, device=device)
    if not torch.any(chosen_valid):
        return zero, zero

    reach_losses: list[torch.Tensor] = []
    call_losses: list[torch.Tensor] = []
    call_family_mask = (
        torch.isin(special_type_id, torch.tensor(XMODEL1_CHI_SPECIAL_TYPES, device=device))
        | (special_type_id == XMODEL1_SPECIAL_TYPE_PON)
        | torch.isin(special_type_id, torch.tensor(XMODEL1_KAN_SPECIAL_TYPES, device=device))
    ) & valid
    none_mask = (special_type_id == XMODEL1_SPECIAL_TYPE_NONE) & valid
    reach_mask = (special_type_id == XMODEL1_SPECIAL_TYPE_REACH) & valid
    dama_mask = (special_type_id == XMODEL1_SPECIAL_TYPE_DAMA) & valid

    for row in range(special_logits.shape[0]):
        if not bool(chosen_valid[row]):
            continue
        chosen = int(chosen_special_idx[row].item())
        reach_slots = torch.nonzero(reach_mask[row], as_tuple=False).flatten()
        dama_slots = torch.nonzero(dama_mask[row], as_tuple=False).flatten()
        if reach_slots.numel() > 0 and dama_slots.numel() > 0:
            reach_slot = int(reach_slots[0].item())
            dama_slot = int(dama_slots[0].item())
            if chosen == reach_slot:
                reach_losses.append(
                    _pairwise_margin_loss(
                        special_logits[row, reach_slot].unsqueeze(0),
                        special_logits[row, dama_slot].unsqueeze(0),
                        margin=margin,
                    )
                )
            elif chosen == dama_slot:
                reach_losses.append(
                    _pairwise_margin_loss(
                        special_logits[row, dama_slot].unsqueeze(0),
                        special_logits[row, reach_slot].unsqueeze(0),
                        margin=margin,
                    )
                )

        none_slots = torch.nonzero(none_mask[row], as_tuple=False).flatten()
        call_slots = torch.nonzero(call_family_mask[row], as_tuple=False).flatten()
        if none_slots.numel() > 0 and call_slots.numel() > 0:
            none_slot = int(none_slots[0].item())
            chosen_type = int(special_type_id[row, chosen].item()) if chosen < special_type_id.shape[1] else -1
            if chosen_type in {*XMODEL1_CHI_SPECIAL_TYPES, XMODEL1_SPECIAL_TYPE_PON, *XMODEL1_KAN_SPECIAL_TYPES}:
                call_losses.append(
                    _pairwise_margin_loss(
                        special_logits[row, chosen].unsqueeze(0),
                        special_logits[row, none_slot].unsqueeze(0),
                        margin=margin,
                    )
                )
            elif chosen == none_slot:
                call_scores = special_logits[row].masked_fill(~call_family_mask[row], -1e4)
                best_call_idx = int(call_scores.argmax().item())
                best_call_hard_bad = float(special_hard_bad_flag[row, best_call_idx].item())
                if best_call_hard_bad > 0.5:
                    call_losses.append(
                        _pairwise_margin_loss(
                            special_logits[row, none_slot].unsqueeze(0),
                            special_logits[row, best_call_idx].unsqueeze(0),
                            margin=margin,
                        )
                    )

    reach_loss = torch.stack(reach_losses).mean() if reach_losses else zero
    call_loss = torch.stack(call_losses).mean() if call_losses else zero
    return reach_loss, call_loss


def _build_action_mask(
    candidate_tile_id: torch.Tensor,
    candidate_mask: torch.Tensor,
    special_type_id: torch.Tensor,
    special_mask: torch.Tensor,
) -> torch.Tensor:
    batch = candidate_tile_id.shape[0]
    action_mask = torch.zeros((batch, 45), dtype=torch.uint8, device=candidate_tile_id.device)
    valid_discard = (candidate_mask > 0) & (candidate_tile_id >= 0) & (candidate_tile_id < 34)
    for tile_id in range(34):
        action_mask[:, tile_id] = (valid_discard & (candidate_tile_id == tile_id)).any(dim=1).to(torch.uint8)
    for special_type, action_idx in (
        (0, 34),
        (3, 35),
        (4, 36),
        (5, 37),
        (6, 38),
        (7, 39),
        (8, 40),
        (9, 41),
        (2, 42),
        (10, 43),
        (11, 44),
    ):
        action_mask[:, action_idx] = ((special_mask > 0) & (special_type_id == special_type)).any(dim=1).to(torch.uint8)
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
            model_kwargs.get("special_candidate_feat"),
            model_kwargs.get("special_candidate_type_id"),
            model_kwargs.get("special_candidate_mask"),
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
    special_ce_loss_weight = float(cfg.get("special_ce_loss_weight", 0.25))
    special_rank_loss_weight = float(cfg.get("special_rank_loss_weight", 0.25))
    reach_pair_loss_weight = float(cfg.get("reach_pair_loss_weight", 0.2))
    call_pair_loss_weight = float(cfg.get("call_pair_loss_weight", 0.2))
    rank_loss_weight = float(cfg.get("rank_loss_weight", 0.5))
    hard_bad_loss_weight = float(cfg.get("hard_bad_loss_weight", 0.25))
    win_loss_weight = float(cfg.get("win_loss_weight", 0.5))
    dealin_loss_weight = float(cfg.get("dealin_loss_weight", 0.5))
    pts_win_loss_weight = float(cfg.get("pts_win_loss_weight", 0.3))
    pts_dealin_loss_weight = float(cfg.get("pts_dealin_loss_weight", 0.3))
    opp_tenpai_loss_weight = float(cfg.get("opp_tenpai_loss_weight", 0.25))
    special_margin = float(cfg.get("special_pair_margin", 0.25))

    def unpack_batch(batch: dict[str, torch.Tensor], device: torch.device) -> dict[str, Any]:
        moved = _unpack_batch(batch, device)
        candidate_tile_id = moved["candidate_tile_id"]
        candidate_mask = moved["candidate_mask"]
        special_type_id = moved["special_candidate_type_id"]
        special_mask = moved["special_candidate_mask"]
        moved["tile_feat"] = moved["state_tile_feat"]
        moved["scalar"] = moved["state_scalar"]
        moved["mask"] = _build_action_mask(
            candidate_tile_id,
            candidate_mask,
            special_type_id,
            special_mask,
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
            "special_candidate_feat": moved.get("special_candidate_feat"),
            "special_candidate_type_id": moved.get("special_candidate_type_id"),
            "special_candidate_mask": moved.get("special_candidate_mask"),
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
        special_ce_loss = _special_candidate_ce_loss(
            out.special_logits,
            batch_data["chosen_special_candidate_idx"],
            batch_data["special_candidate_mask"],
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
        special_rank_loss = _candidate_ranking_loss(
            out.special_logits,
            batch_data["special_candidate_quality_score"],
            batch_data["special_candidate_mask"],
        )
        reach_pair_loss, call_pair_loss = _special_comparison_losses(
            out.special_logits.float(),
            batch_data["special_candidate_type_id"],
            batch_data["special_candidate_mask"],
            batch_data["chosen_special_candidate_idx"],
            batch_data["special_candidate_hard_bad_flag"].float(),
            margin=special_margin,
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
        extra_loss = (
            ce_loss_weight * ce_loss
            + special_ce_loss_weight * special_ce_loss
            + special_rank_loss_weight * special_rank_loss
            + reach_pair_loss_weight * reach_pair_loss
            + call_pair_loss_weight * call_pair_loss
            + rank_loss_weight * rank_loss
            + hard_bad_loss_weight * hard_bad_loss
            + win_loss_weight * win_loss
            + dealin_loss_weight * dealin_loss
            + pts_win_loss_weight * pts_win_loss
            + pts_dealin_loss_weight * pts_dealin_loss
            + opp_tenpai_loss_weight * opp_tenpai_loss
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
        return extra_loss, {
            "loss": float((extra_loss + action_ce_loss).item()),
            "action_ce": float(action_ce_loss.item()),
            "special_ce": float(special_ce_loss.item()),
            "special_rank": float(special_rank_loss.item()),
            "reach_pair": float(reach_pair_loss.item()),
            "call_pair": float(call_pair_loss.item()),
            "ce": float(ce_loss.item()),
            "rank": float(rank_loss.item()),
            "hard_bad": float(hard_bad_loss.item()),
            "win": float(win_loss.item()),
            "dealin": float(dealin_loss.item()),
            "pts_win": float(pts_win_loss.item()),
            "pts_dealin": float(pts_dealin_loss.item()),
            "opp_tenpai": float(opp_tenpai_loss.item()),
            "acc": float(discard_acc),
            "action_acc": float(action_acc),
        }

    return TaskSpec(
        name="xmodel1",
        unpack_batch=unpack_batch,
        compute_extra_loss=compute_extra_loss,
        log_metric_keys=(
            "loss",
            "action_ce",
            "special_ce",
            "special_rank",
            "reach_pair",
            "call_pair",
            "ce",
            "rank",
            "hard_bad",
            "win",
            "dealin",
            "pts_win",
            "pts_dealin",
            "opp_tenpai",
            "acc",
            "action_acc",
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
            "special_rank": row.get("train_special_rank"),
            "reach_pair": row.get("train_reach_pair"),
            "call_pair": row.get("train_call_pair"),
            "acc": row.get("train_acc"),
            "action_acc": row.get("train_action_acc", row.get("train_acc")),
        }
        val = {
            "loss": row["val_objective"],
            "ce": row["val_ce"],
            "action_ce": row.get("val_action_ce"),
            "special_rank": row.get("val_special_rank"),
            "reach_pair": row.get("val_reach_pair"),
            "call_pair": row.get("val_call_pair"),
            "acc": row.get("val_acc"),
            "action_acc": row.get("val_action_acc", row.get("val_acc")),
        }
        for key in ("special_ce", "ce", "rank", "hard_bad", "win", "dealin", "pts_win", "pts_dealin", "opp_tenpai"):
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
    summary = {
        "model_version": "xmodel1",
        "schema_name": checkpoint.get("schema_name"),
        "schema_version": checkpoint.get("schema_version"),
        "completed_epochs": int(checkpoint.get("epoch", cfg.get("num_epochs", 0))),
        "best_val_loss": float(checkpoint.get("best_val_loss", checkpoint.get("best_metric", 0.0))),
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
