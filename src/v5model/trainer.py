"""训练循环：AMP + 梯度累积 + 断点续训 + LR warmup + cosine decay。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from v5model.model import MahjongModel


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------

def save_checkpoint(path: Path, model: MahjongModel, optimizer, scheduler, epoch: int, step: int, best_val_loss: float):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": epoch,
        "step": step,
        "best_val_loss": best_val_loss,
    }, path)


def load_checkpoint(path: Path, model: MahjongModel, optimizer, scheduler):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model"], strict=False)
    optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    return ckpt["epoch"], ckpt["step"], ckpt.get("best_val_loss", float("inf"))


# ---------------------------------------------------------------------------
# LR Schedule
# ---------------------------------------------------------------------------

def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """CE loss，mask 为 0 的动作设为 -1e9 再计算 softmax。"""
    logits = logits.masked_fill(mask == 0, -1e4)
    return nn.functional.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Train / Eval
# ---------------------------------------------------------------------------

def _run_epoch(
    model: MahjongModel,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    accumulation_steps: int,
    value_loss_weight: float,
    is_train: bool,
    step: int,
    log_interval: int = 100,
) -> Dict:
    import sys
    model.train(is_train)
    total_ce = total_val_loss = total_acc = 0.0
    n_batches = 0
    tag = "train" if is_train else "val"

    if is_train:
        optimizer.zero_grad()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for i, (tile_feat, scalar, mask, action_idx, value_target) in enumerate(loader):
            tile_feat = tile_feat.to(device)
            scalar = scalar.to(device)
            mask = mask.to(device)
            action_idx = action_idx.to(device)
            value_target = value_target.to(device)

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                policy_logits, value_pred = model(tile_feat, scalar)
                ce = masked_ce_loss(policy_logits, action_idx, mask)
                val_loss = nn.functional.mse_loss(value_pred.squeeze(-1), value_target)
                loss = ce + value_loss_weight * val_loss

            if is_train:
                loss_scaled = loss / accumulation_steps
                scaler.scale(loss_scaled).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                    step += 1

            with torch.no_grad():
                masked_logits = policy_logits.masked_fill(mask == 0, -1e4)
                pred = masked_logits.argmax(dim=-1)
                total_acc += (pred == action_idx).float().mean().item()

            total_ce += ce.item()
            total_val_loss += val_loss.item()
            n_batches += 1

            # 进度条：每 batch 刷新一行
            lr_str = f" lr={optimizer.param_groups[0]['lr']:.2e}" if is_train else ""
            sys.stdout.write(
                f"\r  [{tag}] batch={n_batches:5d} | "
                f"ce={total_ce/n_batches:.4f} "
                f"val={total_val_loss/n_batches:.4f} "
                f"acc={total_acc/n_batches:.3f}"
                f"{lr_str}   "
            )
            sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    n = max(1, n_batches)
    return {
        "ce": total_ce / n,
        "val_loss": total_val_loss / n,
        "acc": total_acc / n,
        "step": step,
    }


def train(
    model: MahjongModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    cfg: Dict,
    output_dir: Path,
    resume_path: Optional[Path] = None,
    device_str: str = "cuda",
):
    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 3e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    # 粗略估计 total_steps（用 epoch 数 * 每 epoch 预估步数）
    steps_per_epoch = cfg.get("steps_per_epoch", 5000)
    num_epochs = cfg.get("num_epochs", 10)
    accumulation_steps = cfg.get("accumulation_steps", 4)
    warmup_steps = cfg.get("warmup_steps", 500)
    total_steps = steps_per_epoch * num_epochs

    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0
    global_step = 0
    best_val_loss = float("inf")

    if resume_path is not None and resume_path.exists():
        start_epoch, global_step, best_val_loss = load_checkpoint(resume_path, model, optimizer, scheduler)
        print(f"Resumed from {resume_path} (epoch={start_epoch}, step={global_step})")

    output_dir.mkdir(parents=True, exist_ok=True)
    value_loss_weight = cfg.get("value_loss_weight", 0.5)
    log_interval = cfg.get("log_interval", 100)

    for epoch in range(start_epoch, num_epochs):
        t0 = time.time()
        print(f"\n[Epoch {epoch+1}/{num_epochs}]")

        train_stats = _run_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, accumulation_steps, value_loss_weight,
            is_train=True, step=global_step, log_interval=log_interval,
        )
        global_step = train_stats["step"]

        val_stats = _run_epoch(
            model, val_loader, optimizer, scheduler, scaler,
            device, accumulation_steps, value_loss_weight,
            is_train=False, step=global_step, log_interval=log_interval,
        )

        elapsed = time.time() - t0
        print(
            f"  train ce={train_stats['ce']:.4f} acc={train_stats['acc']:.3f} "
            f"| val ce={val_stats['ce']:.4f} acc={val_stats['acc']:.3f} "
            f"| {elapsed:.0f}s"
        )

        # 保存最新 checkpoint
        save_checkpoint(
            output_dir / "latest.pth", model, optimizer, scheduler,
            epoch + 1, global_step, best_val_loss,
        )

        # 保存最佳 checkpoint
        if val_stats["ce"] < best_val_loss:
            best_val_loss = val_stats["ce"]
            save_checkpoint(
                output_dir / "best.pth", model, optimizer, scheduler,
                epoch + 1, global_step, best_val_loss,
            )
            print(f"  [best checkpoint saved, val_ce={best_val_loss:.4f}]")

        # epoch 统计写 jsonl
        with open(output_dir / "train_log.jsonl", "a") as f:
            f.write(json.dumps({
                "epoch": epoch + 1,
                "step": global_step,
                "train_ce": train_stats["ce"],
                "train_acc": train_stats["acc"],
                "val_ce": val_stats["ce"],
                "val_acc": val_stats["acc"],
            }) + "\n")

    print(f"\nTraining complete. Best val_ce={best_val_loss:.4f}")
    return model
