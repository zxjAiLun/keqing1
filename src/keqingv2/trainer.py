"""keqingv2 训练循环：在 v1 基础上加入 Meld Value Ranking Loss。

复用 keqingv1.trainer 的 checkpoint/scheduler/logging 组件，
仅在 _run_epoch 内的 loss 计算部分扩展 rank loss。
"""

from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from keqingv1.model import MahjongModel
from keqingv1.trainer import (
    save_checkpoint,
    load_checkpoint,
    build_scheduler,
    masked_ce_loss,
    _ACTION_LABELS,
    _MERGE_MAP,
    _action_type_name,
)

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
    """对预编码的 none/meld 特征对计算 margin ranking loss（GPU 并行）。

    特征已在 DataLoader worker 内编码，主线程直接做 forward。
    rank_signs: +1=GT副露, -1=GT none, 0=非 meld/none 样本（跳过）
    """
    valid = rank_signs != 0
    if not valid.any():
        return torch.tensor(0.0, device=device)

    tf_n = tf_none[valid].to(device)
    sc_n = sc_none[valid].to(device)
    tf_m = tf_meld[valid].to(device)
    sc_m = sc_meld[valid].to(device)
    signs_t = rank_signs[valid].to(device)

    # 拼成一个大 batch，单次 forward
    tf_both = torch.cat([tf_n, tf_m], dim=0)
    sc_both = torch.cat([sc_n, sc_m], dim=0)
    with torch.amp.autocast(device_type=device.type, enabled=(device.type == "cuda")):
        with torch.enable_grad():
            _, v_both = model(tf_both, sc_both)
    v_both = v_both.squeeze(1)
    n = tf_n.shape[0]
    v_none_v = v_both[:n]
    v_meld_v = v_both[n:]

    # sign=+1(meld): relu(margin + v_none - v_meld)
    # sign=-1(none): relu(margin + v_meld - v_none)
    raw = margin + signs_t * (v_none_v - v_meld_v)
    return F.relu(raw).mean()


def _run_epoch(
    model: MahjongModel,
    loader: DataLoader,
    optimizer,
    scheduler,
    scaler,
    device: torch.device,
    accumulation_steps: int,
    value_loss_weight: float,
    rank_loss_weight: float,
    rank_margin: float,
    is_train: bool,
    step: int,
    log_interval: int = 100,
    rank_loss_every_n: int = 1,
) -> Dict:
    import sys
    model.train(is_train)
    total_ce = total_val_loss = total_acc = total_rank_loss = 0.0
    n_batches = 0
    tag = "train" if is_train else "val"

    correct_by_type: Dict[str, int] = defaultdict(int)
    total_by_type: Dict[str, int] = defaultdict(int)

    if is_train:
        optimizer.zero_grad()

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for i, batch in enumerate(loader):
            tile_feat, scalar, mask, action_idx, value_target, \
                tf_none, sc_none, tf_meld, sc_meld, rank_signs = batch
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

            # ranking loss 仅在训练时，每 rank_loss_every_n batch 计算一次
            rank_loss_val = 0.0
            if is_train and rank_loss_weight > 0 and (i % rank_loss_every_n == 0):
                if (rank_signs != 0).any():
                    rl = _meld_rank_loss(
                        model, device,
                        tf_none, sc_none, tf_meld, sc_meld, rank_signs,
                        rank_margin,
                    )
                    rank_loss_val = rl.item()
                    loss = loss + rank_loss_weight * rl

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
                correct_mask = pred == action_idx
                total_acc += correct_mask.float().mean().item()

                for atype, correct in zip(action_idx, correct_mask):
                    name = _action_type_name(atype.item())
                    total_by_type[name] += 1
                    if correct:
                        correct_by_type[name] += 1

            total_ce += ce.item()
            total_val_loss += val_loss.item()
            total_rank_loss += rank_loss_val
            n_batches += 1

            lr_str = f" lr={optimizer.param_groups[0]['lr']:.2e}" if is_train else ""
            sys.stdout.write(
                f"\r  [{tag}] batch={n_batches:5d} | "
                f"ce={total_ce/n_batches:.4f} "
                f"val={total_val_loss/n_batches:.4f} "
                f"rank={total_rank_loss/n_batches:.4f} "
                f"acc={total_acc/n_batches:.3f}"
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
    return {
        "ce": total_ce / n,
        "val_loss": total_val_loss / n,
        "rank_loss": total_rank_loss / n,
        "acc": total_acc / n,
        "step": step,
        "acc_by_type": acc_by_type,
        "total_by_type": dict(total_by_type),
    }


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
    best_val_loss = float("inf")

    if resume_path is not None and resume_path.exists():
        if weights_only:
            ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model"], strict=False)
            print(f"Loaded weights from {resume_path} (optimizer/scheduler/epoch reset)")
        else:
            start_epoch, global_step, best_val_loss = load_checkpoint(resume_path, model, optimizer, scheduler)
            print(f"Resumed from {resume_path} (epoch={start_epoch}, step={global_step})")

    output_dir.mkdir(parents=True, exist_ok=True)
    value_loss_weight = cfg.get("value_loss_weight", 0.5)
    rank_loss_weight = cfg.get("rank_loss_weight", 0.1)
    rank_margin = cfg.get("rank_margin", 0.05)
    rank_loss_every_n = cfg.get("rank_loss_every_n", 4)
    log_interval = cfg.get("log_interval", 100)

    from keqingv2.cached_dataset import CachedMjaiDatasetV2
    import random as _random

    end_epoch = start_epoch + num_epochs
    for epoch in range(start_epoch, end_epoch):
        t0 = time.time()
        print(f"\n[Epoch {epoch+1}/{end_epoch}]")

        # 每 epoch 随机抽取子集文件
        if files_per_epoch_ratio < 1.0:
            n = max(1, int(len(train_files) * files_per_epoch_ratio))
            epoch_files = _random.Random(seed + epoch).sample(train_files, n)
        else:
            epoch_files = train_files
        train_ds = CachedMjaiDatasetV2(epoch_files, shuffle=True, seed=seed + epoch, aug_perms=aug_perms)
        train_loader = DataLoader(
            train_ds,
            batch_size=batch_size,
            collate_fn=CachedMjaiDatasetV2.collate,
            num_workers=num_workers,
            pin_memory=use_cuda,
            persistent_workers=(num_workers > 0),
            prefetch_factor=4 if num_workers > 0 else None,
        )

        train_stats = _run_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            device, accumulation_steps, value_loss_weight,
            rank_loss_weight, rank_margin,
            is_train=True, step=global_step, log_interval=log_interval,
            rank_loss_every_n=rank_loss_every_n,
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
            rank_loss_weight=0.0, rank_margin=rank_margin,  # val 不计算 rank loss
            is_train=False, step=global_step, log_interval=log_interval,
        )

        elapsed = time.time() - t0
        print(
            f"  train ce={train_stats['ce']:.4f} rank={train_stats['rank_loss']:.4f} acc={train_stats['acc']:.3f} "
            f"| val ce={val_stats['ce']:.4f} acc={val_stats['acc']:.3f} "
            f"| {elapsed:.0f}s"
        )

        save_checkpoint(
            output_dir / "latest.pth", model, optimizer, scheduler,
            epoch + 1, global_step, best_val_loss,
        )

        if val_stats["ce"] < best_val_loss:
            best_val_loss = val_stats["ce"]
            save_checkpoint(
                output_dir / "best.pth", model, optimizer, scheduler,
                epoch + 1, global_step, best_val_loss,
            )
            print(f"  [best checkpoint saved, val_ce={best_val_loss:.4f}]")

        with open(output_dir / "train_log.jsonl", "a") as f:
            f.write(json.dumps({
                "epoch": epoch + 1,
                "step": global_step,
                "train_ce": train_stats["ce"],
                "train_rank_loss": train_stats["rank_loss"],
                "train_acc": train_stats["acc"],
                "val_ce": val_stats["ce"],
                "val_acc": val_stats["acc"],
                "val_acc_by_type": val_stats.get("acc_by_type", {}),
                "val_total_by_type": val_stats.get("total_by_type", {}),
            }) + "\n")

    print(f"\nTraining complete. Best val_ce={best_val_loss:.4f}")
    return model
