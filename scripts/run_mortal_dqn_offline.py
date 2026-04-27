#!/usr/bin/env python3
"""Run finite-step Mortal Brain+DQN offline training without patching Mortal."""

from __future__ import annotations

import argparse
from datetime import datetime
from glob import glob
import gzip
import json
import logging
from os import path
from pathlib import Path
import random
import sys
from typing import Any

import torch
from torch import nn, optim
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run finite-step Mortal DQN offline training")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--target-steps", type=int, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=20260428)
    parser.add_argument("--num-workers", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    train_to_target_steps(
        config_path=args.config,
        mortal_root=args.mortal_root,
        target_steps=int(args.target_steps),
        device_override=args.device,
        seed=int(args.seed),
        num_workers=args.num_workers,
    )


def train_to_target_steps(
    *,
    config_path: Path,
    mortal_root: Path,
    target_steps: int,
    device_override: str | None = None,
    seed: int = 20260428,
    num_workers: int | None = None,
) -> dict[str, Any]:
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    random.seed(seed)
    torch.manual_seed(seed)
    mortal_python_dir = (mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))

    import os  # noqa: PLC0415

    os.environ["MORTAL_CFG"] = str(config_path.resolve())

    from config import config  # noqa: PLC0415
    from dataloader import FileDatasetsIter, worker_init_fn  # noqa: PLC0415
    from lr_scheduler import LinearWarmUpCosineAnnealingLR  # noqa: PLC0415
    from model import AuxNet, Brain, DQN  # noqa: PLC0415

    control = config["control"]
    version = int(control["version"])
    batch_size = int(control["batch_size"])
    opt_step_every = int(control["opt_step_every"])
    device_name = str(device_override) if device_override is not None else str(control["device"])
    device = torch.device(device_name)
    if device.type == "cuda":
        logging.info("device: %s (%s)", device, torch.cuda.get_device_name(device))
    else:
        logging.info("device: %s", device)

    mortal = Brain(version=version, **config["resnet"]).to(device)
    dqn = DQN(version=version).to(device)
    aux_net = AuxNet((4,)).to(device)
    all_models = (mortal, dqn, aux_net)
    mortal.freeze_bn(bool(config["freeze_bn"]["mortal"]))

    optimizer = optim.AdamW(
        _optimizer_param_groups(all_models, weight_decay=float(config["optim"]["weight_decay"])),
        lr=1,
        weight_decay=0,
        betas=tuple(float(value) for value in config["optim"]["betas"]),
        eps=float(config["optim"]["eps"]),
    )
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config["optim"]["scheduler"])
    scaler = GradScaler(device.type, enabled=bool(control["enable_amp"]))
    best_perf = {"avg_rank": 4.0, "avg_pt": -135.0}
    steps = 0

    state_file = str(control["state_file"])
    if path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        logging.info("loaded Mortal checkpoint from %s at steps=%s", timestamp, state["steps"])
        mortal.load_state_dict(state["mortal"])
        dqn.load_state_dict(state["current_dqn"])
        aux_net.load_state_dict(state["aux_net"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        scaler.load_state_dict(state["scaler"])
        best_perf = dict(state["best_perf"])
        steps = int(state["steps"])

    if steps >= target_steps:
        logging.info("Mortal already at steps=%s, target_steps=%s; no training needed", steps, target_steps)
        return {"steps": steps, "trained_steps": 0, "state_file": state_file}

    file_list = _load_or_build_file_index(config)
    logging.info("file list size: %s", f"{len(file_list):,}")
    dataset = config["dataset"]
    loader_workers = int(dataset["num_workers"] if num_workers is None else num_workers)
    data_loader = iter(
        DataLoader(
            dataset=FileDatasetsIter(
                version=version,
                file_list=file_list,
                pts=config["env"]["pts"],
                file_batch_size=int(dataset["file_batch_size"]),
                reserve_ratio=float(dataset["reserve_ratio"]),
                player_names=_load_player_names(config),
                num_epochs=int(dataset["num_epochs"]),
                enable_augmentation=bool(dataset["enable_augmentation"]),
                augmented_first=bool(dataset["augmented_first"]),
            ),
            batch_size=batch_size,
            drop_last=True,
            num_workers=loader_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn if loader_workers > 0 else None,
        )
    )

    writer = SummaryWriter(str(control["tensorboard_dir"]))
    stats = {"dqn_loss": 0.0, "cql_loss": 0.0, "next_rank_loss": 0.0}
    trained_steps = 0
    optimizer.zero_grad(set_to_none=True)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    mortal.train()
    dqn.train()
    aux_net.train()

    while steps < target_steps:
        try:
            obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = next(data_loader)
        except StopIteration as exc:
            raise RuntimeError(
                f"Mortal offline dataset ended at steps={steps} before target_steps={target_steps}"
            ) from exc
        if int(obs.shape[0]) != batch_size:
            continue
        obs = obs.to(dtype=torch.float32, device=device)
        actions = actions.to(dtype=torch.int64, device=device)
        masks = masks.to(dtype=torch.bool, device=device)
        steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
        kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device)
        player_ranks = player_ranks.to(dtype=torch.int64, device=device)
        if not bool(masks[range(batch_size), actions].all().item()):
            raise RuntimeError("Mortal dataset produced an action outside its legal mask")

        q_target_mc = (float(config["env"]["gamma"]) ** steps_to_done * kyoku_rewards).to(torch.float32)
        with torch.autocast(device.type, enabled=bool(control["enable_amp"])):
            phi = mortal(obs)
            q_out = dqn(phi, masks)
            q = q_out[range(batch_size), actions]
            dqn_loss = 0.5 * mse(q, q_target_mc)
            cql_loss = q_out.logsumexp(-1).mean() - q.mean()
            (next_rank_logits,) = aux_net(phi)
            next_rank_loss = ce(next_rank_logits, player_ranks)
            loss = dqn_loss + cql_loss * float(config["cql"]["min_q_weight"]) + next_rank_loss * float(config["aux"]["next_rank_weight"])

        scaler.scale(loss / opt_step_every).backward()
        with torch.inference_mode():
            stats["dqn_loss"] += float(dqn_loss.detach().cpu())
            stats["cql_loss"] += float(cql_loss.detach().cpu())
            stats["next_rank_loss"] += float(next_rank_loss.detach().cpu())

        steps += 1
        trained_steps += 1
        if steps % opt_step_every == 0:
            max_grad_norm = float(config["optim"]["max_grad_norm"])
            if max_grad_norm > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(
                    [param for group in optimizer.param_groups for param in group["params"]],
                    max_grad_norm,
                )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if trained_steps == 1 or trained_steps % 10 == 0 or steps >= target_steps:
            logging.info("Mortal train progress: steps=%s/%s", steps, target_steps)

    if trained_steps:
        for key, value in stats.items():
            writer.add_scalar(f"loss/{key}", value / trained_steps, steps)
        writer.add_scalar("hparam/lr", scheduler.get_last_lr()[0], steps)
        writer.flush()
    writer.close()

    Path(state_file).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mortal": mortal.state_dict(),
            "current_dqn": dqn.state_dict(),
            "aux_net": aux_net.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "steps": steps,
            "timestamp": datetime.now().timestamp(),
            "best_perf": best_perf,
            "config": config,
        },
        state_file,
    )
    logging.info("saved Mortal checkpoint: %s steps=%s", state_file, steps)
    return {"steps": steps, "trained_steps": trained_steps, "state_file": state_file}


def _optimizer_param_groups(all_models: tuple[nn.Module, ...], *, weight_decay: float) -> list[dict[str, Any]]:
    decay_params = []
    no_decay_params = []
    for model in all_models:
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith("weight"):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    return [
        {"params": decay_params, "weight_decay": float(weight_decay)},
        {"params": no_decay_params},
    ]


def _load_or_build_file_index(config: dict[str, Any]) -> list[str]:
    dataset = config["dataset"]
    file_index = Path(dataset["file_index"])
    if file_index.exists():
        return list(torch.load(file_index, weights_only=True)["file_list"])
    logging.info("building Mortal file index...")
    player_names = _load_player_names(config)
    player_names_set = set(player_names)
    file_list: list[str] = []
    for pat in dataset["globs"]:
        file_list.extend(glob(str(pat), recursive=True))
    if player_names_set:
        filtered = []
        for filename in file_list:
            with gzip.open(filename, "rt", encoding="utf-8") as handle:
                start = json.loads(next(handle))
            if not set(start["names"]).isdisjoint(player_names_set):
                filtered.append(filename)
        file_list = filtered
    file_list.sort(reverse=True)
    file_index.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"file_list": file_list}, file_index)
    return file_list


def _load_player_names(config: dict[str, Any]) -> list[str]:
    names: set[str] = set()
    for filename in config["dataset"]["player_names_files"]:
        with open(filename, encoding="utf-8") as handle:
            names.update(line.strip() for line in handle if line.strip() and not line.startswith("#"))
    return list(names)


if __name__ == "__main__":
    main()
