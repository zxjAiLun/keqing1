"""Xmodel1 discard-first trainer."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from xmodel1.cached_dataset import Xmodel1DiscardDataset
from xmodel1.model import Xmodel1Model


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
        persistent_workers=(num_workers > 0),
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
        if key in {"candidate_tile_id", "chosen_candidate_idx", "candidate_rank_bucket"}:
            moved = moved.long()
        out[key] = moved
    return out


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


def _run_epoch(
    model: Xmodel1Model,
    loader,
    *,
    optimizer,
    device: torch.device,
    is_train: bool,
    loss_weights: Dict[str, float],
) -> Dict[str, float]:
    model.train(is_train)
    total = {
        "loss": 0.0,
        "ce": 0.0,
        "rank": 0.0,
        "hard_bad": 0.0,
        "value": 0.0,
        "score_delta": 0.0,
        "win": 0.0,
        "dealin": 0.0,
        "offense_quality": 0.0,
        "acc": 0.0,
    }
    n_batches = 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for batch in loader:
            batch = _unpack_batch(batch, device)
            if is_train:
                optimizer.zero_grad(set_to_none=True)

            out = model(
                batch["state_tile_feat"],
                batch["state_scalar"],
                batch["candidate_feat"],
                batch["candidate_flags"],
                batch["candidate_mask"],
                batch["candidate_tile_id"],
            )
            ce_loss = F.cross_entropy(out.discard_logits.float(), batch["chosen_candidate_idx"])
            rank_loss = _candidate_ranking_loss(
                out.discard_logits,
                batch["candidate_quality_score"],
                batch["candidate_mask"],
            )
            hard_bad_loss = _hard_bad_penalty(
                out.discard_logits.float(),
                batch["candidate_hard_bad_flag"].float(),
                batch["chosen_candidate_idx"],
                batch["candidate_mask"],
            )
            value_loss = F.mse_loss(
                out.global_value.squeeze(-1).float(),
                batch["global_value_target"].float(),
            )
            score_delta_loss = F.smooth_l1_loss(
                out.score_delta.squeeze(-1).float(),
                batch["score_delta_target"].float(),
            )
            win_loss = F.binary_cross_entropy_with_logits(
                out.win_logit.squeeze(-1).float(),
                batch["win_target"].float(),
            )
            dealin_loss = F.binary_cross_entropy_with_logits(
                out.dealin_logit.squeeze(-1).float(),
                batch["dealin_target"].float(),
            )
            offense_quality_loss = F.mse_loss(
                out.offense_quality.squeeze(-1).float(),
                batch["offense_quality_target"].float(),
            )
            loss = (
                loss_weights["ce"] * ce_loss
                + loss_weights["rank"] * rank_loss
                + loss_weights["hard_bad"] * hard_bad_loss
                + loss_weights["value"] * value_loss
                + loss_weights["score_delta"] * score_delta_loss
                + loss_weights["win"] * win_loss
                + loss_weights["dealin"] * dealin_loss
                + loss_weights["offense_quality"] * offense_quality_loss
            )

            if is_train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            pred = out.discard_logits.argmax(dim=-1)
            total["acc"] += (pred == batch["chosen_candidate_idx"]).float().mean().item()
            total["loss"] += float(loss.item())
            total["ce"] += float(ce_loss.item())
            total["rank"] += float(rank_loss.item())
            total["hard_bad"] += float(hard_bad_loss.item())
            total["value"] += float(value_loss.item())
            total["score_delta"] += float(score_delta_loss.item())
            total["win"] += float(win_loss.item())
            total["dealin"] += float(dealin_loss.item())
            total["offense_quality"] += float(offense_quality_loss.item())
            n_batches += 1

    if n_batches == 0:
        raise RuntimeError("no batches were produced by the Xmodel1 loader")
    return {k: v / n_batches for k, v in total.items()}


def train(
    model: Xmodel1Model,
    *,
    train_loader,
    val_loader,
    cfg: Dict,
    output_dir: Path,
    resume_path: Path | None = None,
    device_str: str = "cuda",
):
    del resume_path
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg.get("learning_rate", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    num_epochs = int(cfg.get("num_epochs", 1))
    loss_weights = {
        "ce": float(cfg.get("ce_loss_weight", 1.0)),
        "rank": float(cfg.get("rank_loss_weight", 0.5)),
        "hard_bad": float(cfg.get("hard_bad_loss_weight", 0.25)),
        "value": float(cfg.get("value_loss_weight", 0.5)),
        "score_delta": float(cfg.get("score_delta_loss_weight", 0.25)),
        "win": float(cfg.get("win_loss_weight", 0.2)),
        "dealin": float(cfg.get("dealin_loss_weight", 0.2)),
        "offense_quality": float(cfg.get("offense_quality_loss_weight", 0.3)),
    }
    best_val = math.inf
    log_path = output_dir / "train_log.jsonl"
    for epoch in range(num_epochs):
        t0 = time.time()
        train_stats = _run_epoch(
            model,
            train_loader,
            optimizer=optimizer,
            device=device,
            is_train=True,
            loss_weights=loss_weights,
        )
        val_stats = _run_epoch(
            model,
            val_loader,
            optimizer=optimizer,
            device=device,
            is_train=False,
            loss_weights=loss_weights,
        )
        elapsed = time.time() - t0
        row = {
            "epoch": epoch + 1,
            "elapsed_s": elapsed,
            "train": train_stats,
            "val": val_stats,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
        torch.save({"model": model.state_dict(), "cfg": cfg, "model_version": "xmodel1"}, output_dir / "last.pth")
        if val_stats["loss"] < best_val:
            best_val = val_stats["loss"]
            torch.save({"model": model.state_dict(), "cfg": cfg, "model_version": "xmodel1"}, output_dir / "best.pth")
        print(
            f"[Epoch {epoch+1}/{num_epochs}] "
            f"train loss={train_stats['loss']:.4f} acc={train_stats['acc']:.3f} "
            f"| val loss={val_stats['loss']:.4f} acc={val_stats['acc']:.3f} "
            f"| {elapsed:.0f}s"
        )

    return model


__all__ = ["build_dataloaders", "train"]
