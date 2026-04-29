#!/usr/bin/env python3
"""Run finite-step Mortal GRP offline training without patching third_party/Mortal."""

from __future__ import annotations

import argparse
from datetime import datetime
from glob import glob
import logging
from os import path
from pathlib import Path
import random
import sys
from typing import Any

import torch
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run finite-step Mortal GRP offline training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--val-steps", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_to_target_steps(
        config_path=args.config,
        mortal_root=args.mortal_root,
        target_steps=int(args.target_steps),
        val_steps=args.val_steps,
        device_override=args.device,
        seed=int(args.seed),
        num_workers=int(args.num_workers),
        log_every=int(args.log_every),
    )


def train_to_target_steps(
    *,
    config_path: Path,
    mortal_root: Path,
    target_steps: int,
    val_steps: int | None = None,
    device_override: str | None = None,
    seed: int = 20260428,
    num_workers: int = 1,
    log_every: int = 50,
) -> dict[str, Any]:
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    if num_workers < 0:
        raise ValueError(f"num_workers must be non-negative, got {num_workers}")
    if log_every <= 0:
        raise ValueError(f"log_every must be positive, got {log_every}")
    random.seed(seed)
    torch.manual_seed(seed)
    mortal_python_dir = (mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))

    import os  # noqa: PLC0415

    os.environ["MORTAL_CFG"] = str(config_path.resolve())

    from config import config  # noqa: PLC0415
    from model import GRP  # noqa: PLC0415
    from train_grp import GrpFileDatasetsIter, collate  # noqa: PLC0415

    cfg = config["grp"]
    control = cfg["control"]
    dataset = cfg["dataset"]
    device_name = str(device_override) if device_override is not None else str(control["device"])
    device = torch.device(device_name)
    if device.type == "cuda":
        logging.info("device: %s (%s)", device, torch.cuda.get_device_name(device))
    else:
        logging.info("device: %s", device)

    grp = GRP(**cfg["network"]).to(device)
    optimizer = optim.AdamW(grp.parameters())
    optimizer.param_groups[0]["lr"] = float(cfg["optim"]["lr"])

    state_file = str(cfg["state_file"])
    steps = 0
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        logging.info("loaded GRP checkpoint from %s at steps=%s", timestamp, state["steps"])
        grp.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        steps = int(state["steps"])

    if steps >= target_steps:
        logging.info("GRP already at steps=%s, target_steps=%s; no training needed", steps, target_steps)
        return {"steps": steps, "trained_steps": 0, "state_file": state_file}

    train_file_list, val_file_list = _load_or_build_file_index(
        file_index=Path(dataset["file_index"]),
        train_globs=tuple(str(value) for value in dataset["train_globs"]),
        val_globs=tuple(str(value) for value in dataset["val_globs"]),
    )
    logging.info("train file list size: %s", f"{len(train_file_list):,}")
    logging.info("val file list size: %s", f"{len(val_file_list):,}")

    train_loader = iter(
        DataLoader(
            dataset=GrpFileDatasetsIter(
                file_list=train_file_list,
                file_batch_size=int(dataset["file_batch_size"]),
                cycle=True,
            ),
            batch_size=int(control["batch_size"]),
            drop_last=True,
            num_workers=num_workers,
            collate_fn=collate,
        )
    )
    writer = SummaryWriter(str(control["tensorboard_dir"]))
    trained_steps = 0
    running_loss = 0.0
    running_acc = 0.0
    window_loss = 0.0
    window_acc = 0.0
    window_count = 0
    grp.train()

    def log_train_metrics(*, prefix: str = "GRP train metrics") -> None:
        nonlocal window_loss, window_acc, window_count
        if window_count <= 0:
            return
        avg_loss = window_loss / window_count
        avg_acc = window_acc / window_count
        lr = float(optimizer.param_groups[0]["lr"])
        logging.info(
            "%s: steps=%s/%s window=%s loss=%.6f acc=%.4f lr=%.8g",
            prefix,
            steps,
            target_steps,
            window_count,
            avg_loss,
            avg_acc,
            lr,
        )
        writer.add_scalar("loss/train_window", avg_loss, steps)
        writer.add_scalar("acc/train_window", avg_acc, steps)
        writer.add_scalar("hparam/lr", lr, steps)
        writer.flush()
        window_loss = 0.0
        window_acc = 0.0
        window_count = 0

    while steps < target_steps:
        inputs, rank_by_players = next(train_loader)
        inputs = inputs.to(dtype=torch.float64, device=device)
        rank_by_players = rank_by_players.to(dtype=torch.int64, device=device)

        logits = grp.forward_packed(inputs)
        labels = grp.get_label(rank_by_players)
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        with torch.inference_mode():
            batch_loss = float(loss.detach().cpu())
            batch_acc = float((logits.argmax(-1) == labels).to(torch.float64).mean().detach().cpu())
            running_loss += batch_loss
            running_acc += batch_acc
            window_loss += batch_loss
            window_acc += batch_acc
            window_count += 1
        steps += 1
        trained_steps += 1
        if trained_steps == 1 or trained_steps % log_every == 0 or steps >= target_steps:
            log_train_metrics()

    log_train_metrics(prefix="GRP final train metrics")
    if trained_steps:
        writer.add_scalar("loss/train", running_loss / trained_steps, steps)
        writer.add_scalar("acc/train", running_acc / trained_steps, steps)

    resolved_val_steps = int(control["val_steps"] if val_steps is None else val_steps)
    val_summary = _validate(
        grp=grp,
        val_file_list=val_file_list,
        file_batch_size=int(dataset["file_batch_size"]),
        batch_size=int(control["batch_size"]),
        val_steps=resolved_val_steps,
        device=device,
        num_workers=num_workers,
    )
    if val_summary is not None:
        logging.info(
            "GRP val metrics: steps=%s val_steps=%s loss=%.6f acc=%.4f",
            steps,
            resolved_val_steps,
            val_summary["loss"],
            val_summary["acc"],
        )
        writer.add_scalar("loss/val", val_summary["loss"], steps)
        writer.add_scalar("acc/val", val_summary["acc"], steps)
    writer.flush()
    writer.close()

    Path(state_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": grp.state_dict(),
            "optimizer": optimizer.state_dict(),
            "steps": steps,
            "timestamp": datetime.now().timestamp(),
        },
        state_file,
    )
    logging.info("saved GRP checkpoint: %s steps=%s", state_file, steps)
    return {"steps": steps, "trained_steps": trained_steps, "state_file": state_file, "val": val_summary}


def _load_or_build_file_index(
    *,
    file_index: Path,
    train_globs: tuple[str, ...],
    val_globs: tuple[str, ...],
) -> tuple[list[str], list[str]]:
    if file_index.exists():
        index = torch.load(file_index, weights_only=True)
        return list(index["train_file_list"]), list(index["val_file_list"])
    logging.info("building GRP file index...")
    train_file_list: list[str] = []
    val_file_list: list[str] = []
    for pat in train_globs:
        train_file_list.extend(glob(pat, recursive=True))
    for pat in val_globs:
        val_file_list.extend(glob(pat, recursive=True))
    train_file_list.sort(reverse=True)
    val_file_list.sort(reverse=True)
    file_index.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"train_file_list": train_file_list, "val_file_list": val_file_list}, file_index)
    return train_file_list, val_file_list


def _validate(
    *,
    grp,
    val_file_list: list[str],
    file_batch_size: int,
    batch_size: int,
    val_steps: int,
    device: torch.device,
    num_workers: int,
) -> dict[str, float] | None:
    if val_steps <= 0 or not val_file_list:
        return None
    from train_grp import GrpFileDatasetsIter, collate  # noqa: PLC0415

    val_loader = iter(
        DataLoader(
            dataset=GrpFileDatasetsIter(
                file_list=val_file_list,
                file_batch_size=file_batch_size,
                cycle=True,
            ),
            batch_size=batch_size,
            drop_last=True,
            num_workers=num_workers,
            collate_fn=collate,
        )
    )
    grp.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.inference_mode():
        for idx in range(val_steps):
            inputs, rank_by_players = next(val_loader)
            inputs = inputs.to(dtype=torch.float64, device=device)
            rank_by_players = rank_by_players.to(dtype=torch.int64, device=device)
            logits = grp.forward_packed(inputs)
            labels = grp.get_label(rank_by_players)
            loss = F.cross_entropy(logits, labels)
            total_loss += float(loss.detach().cpu())
            total_acc += float((logits.argmax(-1) == labels).to(torch.float64).mean().detach().cpu())
            if idx == 0 or (idx + 1) % 50 == 0 or idx + 1 >= val_steps:
                logging.info("GRP val progress: %s/%s", idx + 1, val_steps)
    grp.train()
    return {"loss": total_loss / val_steps, "acc": total_acc / val_steps}


if __name__ == "__main__":
    main()
