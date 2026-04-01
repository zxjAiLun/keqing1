"""共享训练循环：基础 policy/value 训练 + task 级扩展 loss。"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from keqingv1.model import MahjongModel
from training.specs import TaskSpec

# action index → 类型名（dahai 合并为一个大类，chi_*/kan 系列合并显示）
_ACTION_LABELS = (
    ["dahai"] * 34
    + ["reach", "chi_low", "chi_mid", "chi_high", "pon", "daiminkan", "ankan", "kakan", "hora", "ryukyoku", "none"]
)

# 合并显示时的分组映射
_MERGE_MAP = {
    "chi_low": "chi",
    "chi_mid": "chi",
    "chi_high": "chi",
    "daiminkan": "kan",
    "ankan": "kan",
    "kakan": "kan",
}


def _action_type_name(idx: int) -> str:
    if 0 <= idx < len(_ACTION_LABELS):
        return _ACTION_LABELS[idx]
    return f"unknown_{idx}"


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


def build_scheduler(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def masked_ce_loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """CE loss，mask 为 0 的动作设为 -1e4 再计算 softmax。"""
    logits = logits.masked_fill(mask == 0, -1e4)
    return nn.functional.cross_entropy(logits, labels)


def _run_epoch(
    model: MahjongModel,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    accumulation_steps: int,
    value_loss_weight: float,
    task: TaskSpec,
    is_train: bool,
    step: int,
    log_interval: int = 100,
) -> Dict:
    import sys

    del log_interval

    model.train(is_train)
    total_ce = total_val_loss = total_acc = 0.0
    total_extra_loss = 0.0
    total_grad_norm = 0.0
    grad_norm_steps = 0
    extra_totals = {k: 0.0 for k in task.log_metric_keys}
    n_batches = 0
    tag = "train" if is_train else "val"

    correct_by_type: Dict[str, int] = defaultdict(int)
    total_by_type: Dict[str, int] = defaultdict(int)

    if is_train:
        optimizer.zero_grad()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for i, batch in enumerate(loader):
            batch_data = task.unpack_batch(batch, device)
            tile_feat = batch_data["tile_feat"]
            scalar = batch_data["scalar"]
            mask = batch_data["mask"]
            action_idx = batch_data["action_idx"]
            value_target = batch_data["value_target"]

            with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                policy_logits, value_pred = model(tile_feat, scalar)
                ce = masked_ce_loss(policy_logits, action_idx, mask)
                val_loss = nn.functional.mse_loss(value_pred.squeeze(-1), value_target)
                extra_loss, extra_metrics = task.compute_extra_loss(
                    model, device, batch_data, is_train, i
                )
                loss = ce + value_loss_weight * val_loss + extra_loss

            if is_train:
                loss_scaled = loss / accumulation_steps
                scaler.scale(loss_scaled).backward()

                if (i + 1) % accumulation_steps == 0:
                    scaler.unscale_(optimizer)
                    grad_norm = float(nn.utils.clip_grad_norm_(model.parameters(), 1.0))
                    total_grad_norm += grad_norm
                    grad_norm_steps += 1
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if scheduler is not None:
                        scheduler.step()
                    step += 1

            with torch.no_grad():
                masked_logits = policy_logits.masked_fill(mask == 0, -1e4)
                pred = masked_logits.argmax(dim=-1)
                correct_mask = pred == action_idx
                total_acc += correct_mask.float().mean().item()

                for atype, correct in zip(action_idx, correct_mask):
                    name = _action_type_name(atype.item())
                    total_by_type[name] += 1
                    if correct:
                        correct_by_type[name] += 1

            total_ce += ce.item()
            total_val_loss += val_loss.item()
            total_extra_loss += float(extra_loss.item())
            for key in task.log_metric_keys:
                extra_totals[key] += float(extra_metrics.get(key, 0.0))
            n_batches += 1

            lr_str = f" lr={optimizer.param_groups[0]['lr']:.2e}" if is_train else ""
            grad_norm_str = ""
            if is_train and grad_norm_steps > 0:
                grad_norm_str = f" gnorm={total_grad_norm/grad_norm_steps:.3f}"
            extra_str = "".join(
                f" {key}={extra_totals[key]/n_batches:.4f}"
                for key in task.log_metric_keys
            )
            sys.stdout.write(
                f"\r  [{tag}] batch={n_batches:5d} | "
                f"ce={total_ce/n_batches:.4f} "
                f"val={total_val_loss/n_batches:.4f}"
                f"{extra_str} "
                f"acc={total_acc/n_batches:.3f}"
                f"{grad_norm_str}"
                f"{lr_str}   "
            )
            sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()

    merged_cor: Dict[str, int] = defaultdict(int)
    merged_tot: Dict[str, int] = defaultdict(int)
    for name in total_by_type:
        group = _MERGE_MAP.get(name, name)
        merged_cor[group] += correct_by_type[name]
        merged_tot[group] += total_by_type[name]
    acc_lines = []
    for name in sorted(merged_tot):
        tot = merged_tot[name]
        cor = merged_cor[name]
        acc = cor / tot if tot > 0 else 0.0
        acc_lines.append(f"    {name:>8s}: {cor:5d}/{tot:5d} = {acc:.3f}")
    if acc_lines:
        print("\n".join(acc_lines))

    n = max(1, n_batches)
    acc_by_type = {k: correct_by_type[k] / total_by_type[k] for k in total_by_type}
    stats = {
        "ce": total_ce / n,
        "val_loss": total_val_loss / n,
        "extra_loss": total_extra_loss / n,
        "objective": (total_ce / n) + value_loss_weight * (total_val_loss / n) + (total_extra_loss / n),
        "acc": total_acc / n,
        "grad_norm": (total_grad_norm / grad_norm_steps if grad_norm_steps > 0 else None),
        "step": step,
        "acc_by_type": acc_by_type,
        "total_by_type": dict(total_by_type),
    }
    for key in task.log_metric_keys:
        stats[key] = extra_totals[key] / n
    return stats


def _is_better_metric(candidate: float, best: float, mode: str) -> bool:
    if mode == "max":
        return candidate > best
    return candidate < best


def train_model(
    model: MahjongModel,
    *,
    train_loader: Optional[DataLoader],
    train_loader_factory: Optional[Callable[[int], DataLoader]],
    val_loader: DataLoader,
    task: TaskSpec,
    cfg: Dict,
    output_dir: Path,
    resume_path: Optional[Path] = None,
    weights_only: bool = False,
    device_str: str = "cuda",
) -> MahjongModel:
    if train_loader is None and train_loader_factory is None:
        raise ValueError("train_loader or train_loader_factory is required")

    device = torch.device(device_str if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.get("learning_rate", 3e-4),
        weight_decay=cfg.get("weight_decay", 1e-4),
    )

    num_epochs = cfg.get("num_epochs", 10)
    accumulation_steps = cfg.get("accumulation_steps", 4)
    warmup_steps = cfg.get("warmup_steps", 500)
    steps_per_epoch_cfg = cfg.get("steps_per_epoch", None)
    steps_per_epoch = steps_per_epoch_cfg if steps_per_epoch_cfg is not None else 5000
    total_steps = steps_per_epoch * num_epochs

    scheduler = build_scheduler(optimizer, warmup_steps, total_steps)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    start_epoch = 0
    global_step = 0
    best_metric = -float("inf") if task.best_metric_mode == "max" else float("inf")
    best_ce = float("inf")

    if resume_path is not None and resume_path.exists():
        if weights_only:
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"], strict=False)
            print(f"Loaded weights from {resume_path} (optimizer/scheduler/epoch reset)")
        else:
            start_epoch, global_step, best_metric = load_checkpoint(resume_path, model, optimizer, scheduler)
            print(f"Resumed from {resume_path} (epoch={start_epoch}, step={global_step})")

    output_dir.mkdir(parents=True, exist_ok=True)
    value_loss_weight = cfg.get("value_loss_weight", 0.5)
    log_interval = cfg.get("log_interval", 100)

    end_epoch = start_epoch + num_epochs
    for epoch in range(start_epoch, end_epoch):
        t0 = time.time()
        print(f"\n[Epoch {epoch+1}/{end_epoch}]")

        current_train_loader = train_loader_factory(epoch) if train_loader_factory is not None else train_loader
        train_stats = _run_epoch(
            model, current_train_loader, optimizer, scheduler, scaler,
            device, accumulation_steps, value_loss_weight,
            task, is_train=True, step=global_step, log_interval=log_interval,
        )
        if epoch == start_epoch and steps_per_epoch_cfg is None:
            actual_spe = train_stats["step"] - global_step
            print(f"  [auto] 实际 steps/epoch={actual_spe}，重建 scheduler（原估算={steps_per_epoch}）")
            remaining_epochs = end_epoch - start_epoch - 1
            scheduler = build_scheduler(optimizer, warmup_steps, actual_spe * remaining_epochs)

        global_step = train_stats["step"]

        val_stats = _run_epoch(
            model, val_loader, optimizer, scheduler, scaler,
            device, accumulation_steps, value_loss_weight,
            task, is_train=False, step=global_step, log_interval=log_interval,
        )

        elapsed = time.time() - t0
        train_extra = " ".join(
            f"{key}={train_stats[key]:.4f}" for key in task.log_metric_keys
        )
        val_extra = " ".join(
            f"{key}={val_stats[key]:.4f}" for key in task.log_metric_keys
        )
        train_extra = f" {train_extra}" if train_extra else ""
        val_extra = f" {val_extra}" if val_extra else ""
        grad_norm = train_stats.get("grad_norm")
        gnorm_str = f" gnorm={grad_norm:.3f}" if grad_norm is not None else ""
        print(
            f"  train ce={train_stats['ce']:.4f}{train_extra} acc={train_stats['acc']:.3f}{gnorm_str} "
            f"| val ce={val_stats['ce']:.4f}{val_extra} acc={val_stats['acc']:.3f} "
            f"| {elapsed:.0f}s"
        )

        save_checkpoint(
            output_dir / "latest.pth", model, optimizer, scheduler,
            epoch + 1, global_step, best_metric,
        )

        metric_value = val_stats[task.best_metric_name]
        if _is_better_metric(metric_value, best_metric, task.best_metric_mode):
            best_metric = metric_value
            save_checkpoint(
                output_dir / "best.pth", model, optimizer, scheduler,
                epoch + 1, global_step, best_metric,
            )
            print(f"  [best checkpoint saved, val_{task.best_metric_name}={best_metric:.4f}]")

        if val_stats["ce"] < best_ce:
            best_ce = val_stats["ce"]
            save_checkpoint(
                output_dir / "best_ce.pth", model, optimizer, scheduler,
                epoch + 1, global_step, best_ce,
            )
            print(f"  [best_ce checkpoint saved, val_ce={best_ce:.4f}]")

        log_row = {
            "epoch": epoch + 1,
            "step": global_step,
            "train_ce": train_stats["ce"],
            "train_value_loss": train_stats["val_loss"],
            "train_extra_loss": train_stats["extra_loss"],
            "train_objective": train_stats["objective"],
            "train_acc": train_stats["acc"],
            "train_grad_norm": train_stats.get("grad_norm"),
            "val_ce": val_stats["ce"],
            "val_value_loss": val_stats["val_loss"],
            "val_extra_loss": val_stats["extra_loss"],
            "val_objective": val_stats["objective"],
            "val_acc": val_stats["acc"],
            "lr": optimizer.param_groups[0]["lr"],
            "val_acc_by_type": val_stats.get("acc_by_type", {}),
            "val_total_by_type": val_stats.get("total_by_type", {}),
        }
        for key in task.log_metric_keys:
            log_row[f"train_{key}"] = train_stats.get(key)
            log_row[f"val_{key}"] = val_stats.get(key)
        with open(output_dir / "train_log.jsonl", "a") as f:
            f.write(json.dumps(log_row) + "\n")

    print(f"\nTraining complete. Best val_{task.best_metric_name}={best_metric:.4f}")
    return model
