"""Xmodel1 discard-first trainer."""

from __future__ import annotations

import json
import gc
import math
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from xmodel1.cached_dataset import Xmodel1DiscardDataset
from xmodel1.model import Xmodel1Model
from xmodel1.schema import (
    XMODEL1_CHI_SPECIAL_TYPES,
    XMODEL1_SPECIAL_TYPE_DAMA,
    XMODEL1_KAN_SPECIAL_TYPES,
    XMODEL1_SPECIAL_TYPE_NONE,
    XMODEL1_SPECIAL_TYPE_PON,
    XMODEL1_SPECIAL_TYPE_REACH,
)


def _format_duration(seconds: float) -> str:
    if seconds >= 3600:
        return f"{seconds / 3600:.1f}h"
    if seconds >= 60:
        return f"{seconds / 60:.1f}m"
    return f"{seconds:.0f}s"


def _build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(max(progress, 0.0), 1.0)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

def build_dataloaders(
    *,
    train_files,
    val_files,
    batch_size: int,
    num_workers: int,
    buffer_size: int,
    seed: int,
):
    from torch.utils.data import DataLoader

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
    common = dict(
        batch_size=batch_size,
        collate_fn=Xmodel1DiscardDataset.collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )
    return DataLoader(train_ds, **common), DataLoader(val_ds, **common)

def _unpack_batch(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    out = {}
    for key, value in batch.items():
        if not torch.is_tensor(value):
            out[key] = value
            continue
        moved = value.to(device, non_blocking=True)
        if moved.dtype == torch.float16:
            moved = moved.float()
        if key in {
            "candidate_tile_id",
            "chosen_candidate_idx",
            "candidate_rank_bucket",
            "special_candidate_type_id",
            "chosen_special_candidate_idx",
            "special_candidate_rank_bucket",
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


def _resolve_pts_given_targets(batch: Dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    """Xmodel1 v2 requires true conditional point labels."""
    pts_given_win_target = batch.get("pts_given_win_target")
    pts_given_dealin_target = batch.get("pts_given_dealin_target")
    if pts_given_win_target is None or pts_given_dealin_target is None:
        raise ValueError("Xmodel1 v2 batches require pts_given_win_target and pts_given_dealin_target")
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

    batch_size = special_logits.shape[0]
    for row in range(batch_size):
        if not bool(chosen_valid[row]):
            continue
        chosen = int(chosen_special_idx[row].item())
        if chosen < 0:
            continue

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
                # Focus the "none beats bad chi/call" relation on clearly bad call candidates.
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

def _run_epoch(
    model: Xmodel1Model,
    loader,
    *,
    optimizer,
    scheduler,
    device: torch.device,
    is_train: bool,
    loss_weights: Dict[str, float],
    log_interval: int,
    total_batches_override: int | None = None,
    max_batches: int | None = None,
    global_step: int = 0,
) -> Dict[str, float]:
    model.train(is_train)
    total = {
        "loss": 0.0,
        "ce": 0.0,
        "action_ce": 0.0,
        "special_ce": 0.0,
        "special_rank": 0.0,
        "reach_pair": 0.0,
        "call_pair": 0.0,
        "rank": 0.0,
        "hard_bad": 0.0,
        "win": 0.0,
        "dealin": 0.0,
        "pts_win": 0.0,
        "pts_dealin": 0.0,
        "opp_tenpai": 0.0,
        "acc": 0.0,
        "action_acc": 0.0,
    }
    n_batches = 0
    epoch_t0 = time.time()
    try:
        total_batches = len(loader)
    except (TypeError, NotImplementedError):
        total_batches = None
    if total_batches_override is not None and total_batches_override > 0:
        total_batches = total_batches_override
    if max_batches is not None and max_batches > 0:
        total_batches = max_batches if total_batches is None else min(total_batches, max_batches)
    tag = "train" if is_train else "val"
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            if max_batches is not None and max_batches > 0 and n_batches >= max_batches:
                break
            batch = _unpack_batch(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            out = model(
                batch["state_tile_feat"],
                batch["state_scalar"],
                batch["candidate_feat"],
                batch["candidate_tile_id"],
                batch["candidate_flags"],
                batch["candidate_mask"],
                batch.get("special_candidate_feat"),
                batch.get("special_candidate_type_id"),
                batch.get("special_candidate_mask"),
                event_history=batch.get("event_history"),
            )
            action_targets = batch["action_idx_target"].long()
            sample_type = batch.get("sample_type")
            if sample_type is None:
                sample_type = torch.zeros_like(action_targets)
            sample_type = sample_type.long()
            discard_rows = sample_type == 0
            discard_target = batch["chosen_candidate_idx"].clamp_min(0)
            ce_raw = F.cross_entropy(
                out.discard_logits.float(),
                discard_target,
                reduction="none",
            )
            ce_loss = _masked_mean(ce_raw, discard_rows)
            action_ce_loss = F.cross_entropy(out.action_logits.float(), action_targets)
            special_ce_loss = _special_candidate_ce_loss(
                out.special_logits,
                batch["chosen_special_candidate_idx"],
                batch["special_candidate_mask"],
            )
            rank_loss = (
                _candidate_ranking_loss(
                    out.discard_logits[discard_rows],
                    batch["candidate_quality_score"][discard_rows],
                    batch["candidate_mask"][discard_rows],
                )
                if torch.any(discard_rows)
                else out.discard_logits.new_tensor(0.0)
            )
            special_rank_loss = _candidate_ranking_loss(
                out.special_logits,
                batch["special_candidate_quality_score"],
                batch["special_candidate_mask"],
            )
            reach_pair_loss, call_pair_loss = _special_comparison_losses(
                out.special_logits.float(),
                batch["special_candidate_type_id"],
                batch["special_candidate_mask"],
                batch["chosen_special_candidate_idx"],
                batch["special_candidate_hard_bad_flag"].float(),
                margin=float(loss_weights["special_margin"]),
            )
            hard_bad_loss = (
                _hard_bad_penalty(
                    out.discard_logits[discard_rows].float(),
                    batch["candidate_hard_bad_flag"][discard_rows].float(),
                    batch["chosen_candidate_idx"][discard_rows].clamp_min(0),
                    batch["candidate_mask"][discard_rows],
                )
                if torch.any(discard_rows)
                else out.discard_logits.new_tensor(0.0)
            )
            win_target = batch["win_target"].float()
            dealin_target = batch["dealin_target"].float()
            win_loss = F.binary_cross_entropy_with_logits(
                out.win_logit.squeeze(-1).float(),
                win_target,
            )
            dealin_loss = F.binary_cross_entropy_with_logits(
                out.dealin_logit.squeeze(-1).float(),
                dealin_target,
            )
            # 分解 EV 条件回归:
            # 优先消费 preprocess 输出的真标签 pts_given_win_target /
            # pts_given_dealin_target。老 cache 不含该字段时回退到旧近似
            # (|score_delta_target| × win/dealin mask),保证向后兼容。
            pts_win_target, pts_dealin_target = _resolve_pts_given_targets(batch)
            win_mask = win_target > 0.5
            dealin_mask = dealin_target > 0.5
            pts_given_win_raw = F.smooth_l1_loss(
                out.pts_given_win.squeeze(-1).float(),
                pts_win_target,
                reduction="none",
            )
            pts_given_dealin_raw = F.smooth_l1_loss(
                out.pts_given_dealin.squeeze(-1).float(),
                pts_dealin_target,
                reduction="none",
            )
            pts_win_loss = _masked_mean(pts_given_win_raw, win_mask)
            pts_dealin_loss = _masked_mean(pts_given_dealin_raw, dealin_mask)
            # xmodel1_discard_v2 requires opp_tenpai_target in cache; this guard is only
            # for synthetic/unit-test batches that bypass the dataset contract.
            opp_tenpai_target = batch.get("opp_tenpai_target")
            if opp_tenpai_target is None:
                opp_tenpai_loss = out.opp_tenpai_logits.new_tensor(0.0)
            else:
                opp_tenpai_loss = F.binary_cross_entropy_with_logits(
                    out.opp_tenpai_logits.float(),
                    opp_tenpai_target.float(),
                )
            loss = (
                loss_weights["ce"] * ce_loss
                + loss_weights["action_ce"] * action_ce_loss
                + loss_weights["special_ce"] * special_ce_loss
                + loss_weights["special_rank"] * special_rank_loss
                + loss_weights["reach_pair"] * reach_pair_loss
                + loss_weights["call_pair"] * call_pair_loss
                + loss_weights["rank"] * rank_loss
                + loss_weights["hard_bad"] * hard_bad_loss
                + loss_weights["win"] * win_loss
                + loss_weights["dealin"] * dealin_loss
                + loss_weights["pts_win"] * pts_win_loss
                + loss_weights["pts_dealin"] * pts_dealin_loss
                + loss_weights["opp_tenpai"] * opp_tenpai_loss
            )

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                global_step += 1

            pred = out.discard_logits.argmax(dim=-1)
            action_pred = out.action_logits.argmax(dim=-1)
            if torch.any(discard_rows):
                total["acc"] += (
                    pred[discard_rows] == batch["chosen_candidate_idx"][discard_rows]
                ).float().mean().item()
            total["action_acc"] += (action_pred == action_targets).float().mean().item()
            total["loss"] += float(loss.item())
            total["ce"] += float(ce_loss.item())
            total["action_ce"] += float(action_ce_loss.item())
            total["special_ce"] += float(special_ce_loss.item())
            total["special_rank"] += float(special_rank_loss.item())
            total["reach_pair"] += float(reach_pair_loss.item())
            total["call_pair"] += float(call_pair_loss.item())
            total["rank"] += float(rank_loss.item())
            total["hard_bad"] += float(hard_bad_loss.item())
            total["win"] += float(win_loss.item())
            total["dealin"] += float(dealin_loss.item())
            total["pts_win"] += float(pts_win_loss.item())
            total["pts_dealin"] += float(pts_dealin_loss.item())
            total["opp_tenpai"] += float(opp_tenpai_loss.item())
            n_batches += 1

            should_log = n_batches == 1
            if log_interval > 0 and n_batches % log_interval == 0:
                should_log = True
            if total_batches is not None and n_batches == total_batches:
                should_log = True
            if should_log:
                elapsed_s = time.time() - epoch_t0
                avg_batch_s = elapsed_s / max(1, n_batches)
                eta_s = 0.0
                batch_progress = f"{n_batches}"
                if total_batches is not None:
                    eta_s = avg_batch_s * max(0, total_batches - n_batches)
                    batch_progress = f"{n_batches}/{total_batches}"
                lr_str = f" lr={optimizer.param_groups[0]['lr']:.2e}" if is_train else ""
                print(
                    f"  [{tag}] batch={batch_progress} "
                    f"loss={total['loss']/n_batches:.4f} "
                    f"ce={total['ce']/n_batches:.4f} "
                    f"action_ce={total['action_ce']/n_batches:.4f} "
                    f"special_ce={total['special_ce']/n_batches:.4f} "
                    f"special_rank={total['special_rank']/n_batches:.4f} "
                    f"reach_pair={total['reach_pair']/n_batches:.4f} "
                    f"call_pair={total['call_pair']/n_batches:.4f} "
                    f"acc={total['acc']/n_batches:.3f} "
                    f"action_acc={total['action_acc']/n_batches:.3f} "
                    f"| elapsed={_format_duration(elapsed_s)} eta={_format_duration(eta_s)}"
                    f"{lr_str}",
                    flush=True,
                )

    if n_batches == 0:
        raise RuntimeError("no batches were produced by the Xmodel1 loader")
    stats = {k: v / n_batches for k, v in total.items()}
    stats["step"] = global_step
    stats["num_batches"] = n_batches
    return stats

def _load_checkpoint(path: Path) -> Dict:
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model" not in checkpoint:
        raise ValueError(f"invalid Xmodel1 checkpoint: {path}")
    return checkpoint


def _write_training_summary(
    *,
    output_dir: Path,
    cfg: Dict,
    best_val_loss: float,
    completed_epochs: int,
    resume_path: Path | None,
    log_path: Path,
    dataset_summary: Dict | None = None,
) -> None:
    summary = {
        "model_version": "xmodel1",
        "completed_epochs": int(completed_epochs),
        "best_val_loss": float(best_val_loss),
        "resume_path": str(resume_path) if resume_path is not None else None,
        "log_path": str(log_path),
        "last_checkpoint": str(output_dir / "last.pth"),
        "best_checkpoint": str(output_dir / "best.pth") if (output_dir / "best.pth").exists() else None,
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
    cfg: Dict,
    output_dir: Path,
    resume_path: Path | None = None,
    device_str: str = "cuda",
    dataset_summary: Dict | None = None,
):
    if train_loader is None and train_loader_factory is None:
        raise ValueError("train_loader or train_loader_factory is required")
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    num_epochs = int(cfg.get("num_epochs", 1))
    log_interval = int(cfg.get("log_interval", 100))
    train_steps_per_epoch_cfg = cfg.get("steps_per_epoch", None)
    train_steps_per_epoch = (
        int(train_steps_per_epoch_cfg)
        if train_steps_per_epoch_cfg is not None and int(train_steps_per_epoch_cfg) > 0
        else None
    )
    val_steps_per_epoch_cfg = cfg.get("val_steps_per_epoch", None)
    val_steps_per_epoch = (
        int(val_steps_per_epoch_cfg)
        if val_steps_per_epoch_cfg is not None and int(val_steps_per_epoch_cfg) > 0
        else None
    )
    effective_train_steps = train_steps_per_epoch or int(cfg.get("steps_per_epoch_fallback", 1000))
    warmup_steps = int(cfg.get("warmup_steps", 1000))
    total_steps = max(1, effective_train_steps * num_epochs)
    scheduler = _build_scheduler(optimizer, warmup_steps, total_steps)
    loss_weights = {
        "ce": float(cfg.get("ce_loss_weight", 1.0)),
        "action_ce": float(cfg.get("action_ce_loss_weight", 0.25)),
        "special_ce": float(cfg.get("special_ce_loss_weight", 0.25)),
        "special_rank": float(cfg.get("special_rank_loss_weight", 0.25)),
        "reach_pair": float(cfg.get("reach_pair_loss_weight", 0.2)),
        "call_pair": float(cfg.get("call_pair_loss_weight", 0.2)),
        "special_margin": float(cfg.get("special_pair_margin", 0.25)),
        "rank": float(cfg.get("rank_loss_weight", 0.5)),
        "hard_bad": float(cfg.get("hard_bad_loss_weight", 0.25)),
        # 分解 EV 权重(Stage 1):
        #   win / dealin BCE:0.5
        #   pts_given_{win,dealin} SmoothL1:0.3 (仅对应样本监督)
        # 已弃用的 value/score_delta/offense_quality 权重仍接受读取
        # 但不再参与 loss,便于旧 checkpoint 的 cfg 元数据兼容。
        "win": float(cfg.get("win_loss_weight", 0.5)),
        "dealin": float(cfg.get("dealin_loss_weight", 0.5)),
        "pts_win": float(cfg.get("pts_win_loss_weight", 0.3)),
        "pts_dealin": float(cfg.get("pts_dealin_loss_weight", 0.3)),
        "opp_tenpai": float(cfg.get("opp_tenpai_loss_weight", 0.25)),
    }
    best_val = math.inf
    start_epoch = 0
    global_step = 0
    train_total_batches = None
    val_total_batches = None
    if dataset_summary:
        batch_size = int(cfg.get("batch_size", 1))
        train_samples = int((dataset_summary.get("train") or {}).get("num_samples", 0) or 0)
        val_samples = int((dataset_summary.get("val") or {}).get("num_samples", 0) or 0)
        if train_samples > 0:
            train_total_batches = math.ceil(train_samples / max(1, batch_size))
        if val_samples > 0:
            val_total_batches = math.ceil(val_samples / max(1, batch_size))
    if resume_path is not None:
        checkpoint = _load_checkpoint(resume_path)
        model.load_state_dict(checkpoint["model"])
        if checkpoint.get("optimizer"):
            optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scheduler"):
            scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = int(checkpoint.get("epoch", 0))
        global_step = int(checkpoint.get("step", start_epoch * effective_train_steps))
        best_val = float(checkpoint.get("best_val_loss", math.inf))
        print(
            f"xmodel1 train: resumed checkpoint={resume_path} start_epoch={start_epoch + 1} step={global_step} best_val={best_val:.4f}",
            flush=True,
        )
    log_path = output_dir / "train_log.jsonl"
    if start_epoch >= num_epochs:
        print(
            f"xmodel1 train: checkpoint epoch {start_epoch} already satisfies num_epochs={num_epochs}; skipping training",
            flush=True,
        )
        _write_training_summary(
            output_dir=output_dir,
            cfg=cfg,
            best_val_loss=best_val,
            completed_epochs=start_epoch,
            resume_path=resume_path,
            log_path=log_path,
            dataset_summary=dataset_summary,
        )
        return model

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        print(f"\n[Epoch {epoch+1}/{num_epochs}]", flush=True)
        total_train_steps_plan = effective_train_steps * num_epochs
        if train_loader_summary_factory is not None:
            summary = train_loader_summary_factory(epoch)
            if summary:
                sampled_files = summary.get("sampled_files")
                sampled_samples = summary.get("sampled_samples")
                sampled_ratio = summary.get("sampled_ratio")
                print(
                    f"  [epoch-plan] sampled_files={sampled_files} "
                    f"sampled_samples≈{sampled_samples} sampled_ratio≈{sampled_ratio:.3f} "
                    f"global_step={global_step}/{total_train_steps_plan}",
                    flush=True,
                )
        else:
            print(
                f"  [epoch-plan] global_step={global_step}/{total_train_steps_plan}",
                flush=True,
            )
        current_train_loader = train_loader_factory(epoch) if train_loader_factory is not None else train_loader
        train_dataset = getattr(current_train_loader, "dataset", None)
        if hasattr(train_dataset, "set_epoch"):
            train_dataset.set_epoch(epoch)
        val_dataset = getattr(val_loader, "dataset", None)
        if hasattr(val_dataset, "set_epoch"):
            val_dataset.set_epoch(epoch)
        train_stats = _run_epoch(
            model,
            current_train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            is_train=True,
            loss_weights=loss_weights,
            log_interval=log_interval,
            total_batches_override=train_total_batches,
            max_batches=train_steps_per_epoch,
            global_step=global_step,
        )
        global_step = int(train_stats["step"])
        val_stats = _run_epoch(
            model,
            val_loader,
            optimizer=optimizer,
            scheduler=None,
            device=device,
            is_train=False,
            loss_weights=loss_weights,
            log_interval=log_interval,
            total_batches_override=val_total_batches,
            max_batches=val_steps_per_epoch,
            global_step=global_step,
        )
        elapsed = time.time() - t0
        val_loss = float(val_stats["loss"])
        is_best = val_loss <= best_val
        if is_best:
            best_val = val_loss
        remaining_epochs = max(0, num_epochs - (epoch + 1))
        eta_epochs_s = elapsed * remaining_epochs
        row = {
            "epoch": epoch + 1,
            "step": global_step,
            "elapsed_s": elapsed,
            "train": train_stats,
            "val": val_stats,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        checkpoint = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "cfg": cfg,
            "model_version": "xmodel1",
            "epoch": epoch + 1,
            "step": global_step,
            "best_val_loss": best_val,
        }
        torch.save(checkpoint, output_dir / "last.pth")
        if is_best:
            torch.save(checkpoint, output_dir / "best.pth")
        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"train loss={train_stats['loss']:.4f} acc={train_stats['acc']:.3f} action_acc={train_stats['action_acc']:.3f} "
            f"| val loss={val_stats['loss']:.4f} acc={val_stats['acc']:.3f} action_acc={val_stats['action_acc']:.3f} "
            f"| {elapsed:.0f}s",
            flush=True,
        )
        print(
            f"  [epoch-summary] train_batches={train_stats['num_batches']} "
            f"val_batches={val_stats['num_batches']} "
            f"best_updated={str(is_best).lower()} "
            f"global_step={global_step}/{total_train_steps_plan} "
            f"epoch_eta={_format_duration(eta_epochs_s)}",
            flush=True,
        )

    _write_training_summary(
        output_dir=output_dir,
        cfg=cfg,
        best_val_loss=best_val,
        completed_epochs=num_epochs,
        resume_path=resume_path,
        log_path=log_path,
        dataset_summary=dataset_summary,
    )
    del train_loader
    del train_loader_factory
    del val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return model

__all__ = ["build_dataloaders", "train", "_chosen_action_targets"]
