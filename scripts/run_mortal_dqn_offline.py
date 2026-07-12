#!/usr/bin/env python3
"""Run finite-step Mortal Brain+DQN offline training without patching Mortal."""

from __future__ import annotations

import argparse
from datetime import datetime
from glob import glob
import gzip
import hashlib
import json
import logging
import os
from os import path
from pathlib import Path
import random
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

if os.name == "nt" and not os.environ.get("TORCHINDUCTOR_CACHE_DIR"):
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = str((Path.cwd() / ".torchinductor_cache").resolve())

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
    parser.add_argument(
        "--data-seed",
        type=int,
        default=None,
        help="deterministic dataset stream seed; defaults to --seed for a new run and is restored from checkpoints",
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        default=None,
        help="initialize model weights from a parent checkpoint with fresh optimizer and data stream",
    )
    parser.add_argument(
        "--initial-steps",
        type=int,
        default=0,
        help="global step assigned to a fresh --initialize-from run",
    )
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument(
        "--archive-steps",
        default="",
        help="comma-separated global steps to archive without restarting the data stream",
    )
    parser.add_argument("--archive-dir", type=Path, default=None)
    parser.add_argument(
        "--allow-legacy-data-replay",
        action="store_true",
        help="allow old checkpoints without a data cursor to replay their dataset prefix; never use for clean experiments",
    )
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
        data_seed=args.data_seed,
        num_workers=args.num_workers,
        log_every=int(args.log_every),
        archive_steps=_parse_archive_steps(args.archive_steps),
        archive_dir=args.archive_dir,
        allow_legacy_data_replay=bool(args.allow_legacy_data_replay),
        initialize_from=args.initialize_from,
        initial_steps=int(args.initial_steps),
    )


def _parse_archive_steps(value: str) -> tuple[int, ...]:
    if not value.strip():
        return ()
    steps = tuple(sorted({int(part.strip()) for part in value.split(",") if part.strip()}))
    if any(step <= 0 for step in steps):
        raise ValueError("--archive-steps must contain only positive integers")
    return steps


def _sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def train_to_target_steps(
    *,
    config_path: Path,
    mortal_root: Path,
    target_steps: int,
    device_override: str | None = None,
    seed: int = 20260428,
    data_seed: int | None = None,
    num_workers: int | None = None,
    log_every: int = 50,
    archive_steps: tuple[int, ...] = (),
    archive_dir: Path | None = None,
    allow_legacy_data_replay: bool = False,
    initialize_from: Path | None = None,
    initial_steps: int = 0,
) -> dict[str, Any]:
    if target_steps <= 0:
        raise ValueError(f"target_steps must be positive, got {target_steps}")
    if log_every <= 0:
        raise ValueError(f"log_every must be positive, got {log_every}")
    if initial_steps < 0:
        raise ValueError(f"initial_steps must be non-negative, got {initial_steps}")
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
    loaded_data_stream: dict[str, Any] | None = None
    expected_python_rng_state: object | None = None
    restore_torch_rng_state: torch.Tensor | None = None
    restore_cuda_rng_states: list[torch.Tensor] | None = None
    initialization: dict[str, Any] | None = None

    state_file = str(control["state_file"])
    if path.exists(state_file):
        if initialize_from is not None:
            raise RuntimeError(
                f"state file already exists at {state_file}; --initialize-from is only valid for a fresh experiment"
            )
        state = torch.load(state_file, weights_only=True, map_location=device)
        mortal.load_state_dict(state["mortal"])
        dqn.load_state_dict(state["current_dqn"])
        if "steps" in state:
            timestamp = datetime.fromtimestamp(state["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
            logging.info("loaded Mortal checkpoint from %s at steps=%s", timestamp, state["steps"])
            aux_net.load_state_dict(state["aux_net"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            best_perf = dict(state["best_perf"])
            steps = int(state["steps"])
            loaded_data_stream = state.get("data_stream")
            expected_python_rng_state = state.get("python_rng_state")
            restore_torch_rng_state = state.get("torch_rng_state")
            restore_cuda_rng_states = state.get("cuda_rng_states")
            if loaded_data_stream is None:
                if not allow_legacy_data_replay:
                    raise RuntimeError(
                        "checkpoint has no resumable data_stream metadata; refusing to replay an unknown dataset prefix. "
                        "Create a fresh experiment or pass --allow-legacy-data-replay explicitly."
                    )
                logging.warning("resuming legacy checkpoint with a replayed dataset prefix by explicit request")
            else:
                saved_data_seed = int(loaded_data_stream["data_seed"])
                if data_seed is not None and int(data_seed) != saved_data_seed:
                    raise ValueError(
                        f"--data-seed={data_seed} conflicts with checkpoint data_seed={saved_data_seed}"
                    )
                data_seed = saved_data_seed
        else:
            logging.info(
                "loaded weights-only Mortal checkpoint from %s; optimizer/scheduler/aux are initialized fresh at steps=0",
                state_file,
            )
    elif initialize_from is not None:
        parent_path = initialize_from.resolve()
        if not parent_path.exists():
            raise FileNotFoundError(f"--initialize-from checkpoint does not exist: {parent_path}")
        parent = torch.load(parent_path, weights_only=True, map_location=device)
        try:
            mortal.load_state_dict(parent["mortal"])
            dqn.load_state_dict(parent["current_dqn"])
        except KeyError as exc:
            raise RuntimeError(f"parent checkpoint is missing required model weights: {parent_path}") from exc
        if "aux_net" in parent:
            aux_net.load_state_dict(parent["aux_net"])
        else:
            logging.warning("parent checkpoint has no aux_net; keeping fresh auxiliary weights")
        steps = int(initial_steps)
        initialization = {
            "mode": "weights_only_warm_start",
            "parent_checkpoint": str(parent_path),
            "parent_sha256": _sha256_file(parent_path),
            "parent_steps": int(parent.get("steps", 0)),
            "initial_steps": int(initial_steps),
            "loaded_aux_net": "aux_net" in parent,
            "optimizer": "fresh",
            "scheduler": "fresh",
            "data_stream": "fresh",
        }
        logging.info(
            "initialized fresh experiment from %s; assigned global steps=%s with fresh optimizer/data stream",
            parent_path,
            steps,
        )

    if steps >= target_steps:
        logging.info("Mortal already at steps=%s, target_steps=%s; no training needed", steps, target_steps)
        return {"steps": steps, "trained_steps": 0, "state_file": state_file}

    if data_seed is None:
        data_seed = int(seed)
    # Keep the data order independent from model initialization and reconstruct it exactly on resume.
    random.seed(int(data_seed))
    file_list = _load_or_build_file_index(config)
    logging.info("file list size: %s", f"{len(file_list):,}")
    dataset = config["dataset"]
    loader_workers = int(dataset["num_workers"] if num_workers is None else num_workers)
    dataset_iter = FileDatasetsIter(
        version=version,
        file_list=file_list,
        pts=config["env"]["pts"],
        file_batch_size=int(dataset["file_batch_size"]),
        reserve_ratio=float(dataset["reserve_ratio"]),
        player_names=_load_player_names(config),
        num_epochs=int(dataset["num_epochs"]),
        enable_augmentation=bool(dataset["enable_augmentation"]),
        augmented_first=bool(dataset["augmented_first"]),
    )
    data_loader = iter(
        DataLoader(
            dataset=dataset_iter,
            batch_size=batch_size,
            drop_last=True,
            num_workers=loader_workers,
            pin_memory=True,
            worker_init_fn=worker_init_fn if loader_workers > 0 else None,
        )
    )

    data_batches_consumed = int(loaded_data_stream.get("batches_consumed", 0)) if loaded_data_stream else 0
    resume_skipped_batches = 0
    if data_batches_consumed:
        logging.info("reconstructing dataset stream: skipping %s delivered batches", data_batches_consumed)
        for _ in range(data_batches_consumed):
            try:
                next(data_loader)
            except StopIteration as exc:
                raise RuntimeError(
                    "checkpoint data stream exceeds the available dataset; cannot resume safely"
                ) from exc
            resume_skipped_batches += 1
        if expected_python_rng_state is not None and random.getstate() != expected_python_rng_state:
            raise RuntimeError(
                "dataset RNG state mismatch while reconstructing resume cursor; refusing to repeat or skip data"
            )
    if restore_torch_rng_state is not None:
        torch.set_rng_state(restore_torch_rng_state.detach().cpu())
    if restore_cuda_rng_states is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([state.detach().cpu() for state in restore_cuda_rng_states])

    writer = SummaryWriter(str(control["tensorboard_dir"]))
    stats = {
        "dqn_loss": 0.0,
        "cql_loss": 0.0,
        "next_rank_loss": 0.0,
        "total_loss": 0.0,
        "next_rank_acc": 0.0,
        "q_mean": 0.0,
        "target_mean": 0.0,
        "q_abs_err": 0.0,
    }
    window_stats = {key: 0.0 for key in stats}
    window_count = 0
    trained_steps = 0
    archive_steps_set = set(int(step) for step in archive_steps)
    archived_steps_written: set[int] = set()
    resolved_archive_dir = Path(archive_dir) if archive_dir is not None else Path(state_file).parent / "checkpoints"
    exposure_path = Path(state_file).parent / "data_exposure.json"
    optimizer.zero_grad(set_to_none=True)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()
    mortal.train()
    dqn.train()
    aux_net.train()

    def save_checkpoint() -> None:
        Path(state_file).parent.mkdir(parents=True, exist_ok=True)
        data_stream = {
            "schema": "keqing.mortal.data_stream.v1",
            "data_seed": int(data_seed),
            "batches_consumed": int(data_batches_consumed),
            "samples_consumed": int(data_batches_consumed * batch_size),
            "dataset_file_count": int(len(file_list)),
            "num_workers": int(loader_workers),
            "resume_skipped_batches": int(resume_skipped_batches),
        }
        checkpoint = {
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
            "data_stream": data_stream,
            "python_rng_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "initialization": initialization,
        }
        torch.save(checkpoint, state_file)
        exposure_path.write_text(
            json.dumps(
                {
                    "steps": int(steps),
                    "trained_steps_this_invocation": int(trained_steps),
                    "data_stream": data_stream,
                    "archive_steps": sorted(archive_steps_set),
                    "initialization": initialization,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        if steps in archive_steps_set and steps not in archived_steps_written:
            resolved_archive_dir.mkdir(parents=True, exist_ok=True)
            archive_path = resolved_archive_dir / f"mortal_{steps}.pth"
            torch.save(checkpoint, archive_path)
            archived_steps_written.add(int(steps))
            logging.info("archived Mortal checkpoint: %s", archive_path)
        logging.info("saved Mortal checkpoint: %s steps=%s", state_file, steps)

    def log_train_metrics(*, prefix: str = "Mortal train metrics") -> None:
        nonlocal window_count, window_stats
        if window_count <= 0:
            return
        avg = {key: value / window_count for key, value in window_stats.items()}
        lr = float(scheduler.get_last_lr()[0])
        logging.info(
            "%s: steps=%s/%s window=%s "
            "loss_total=%.6f dqn_loss=%.6f cql_loss=%.6f next_rank_loss=%.6f "
            "next_rank_acc=%.4f q_mean=%.4f target_mean=%.4f q_abs_err=%.4f lr=%.8g",
            prefix,
            steps,
            target_steps,
            window_count,
            avg["total_loss"],
            avg["dqn_loss"],
            avg["cql_loss"],
            avg["next_rank_loss"],
            avg["next_rank_acc"],
            avg["q_mean"],
            avg["target_mean"],
            avg["q_abs_err"],
            lr,
        )
        writer.add_scalar("loss/total_window", avg["total_loss"], steps)
        writer.add_scalar("loss/dqn_window", avg["dqn_loss"], steps)
        writer.add_scalar("loss/cql_window", avg["cql_loss"], steps)
        writer.add_scalar("loss/next_rank_window", avg["next_rank_loss"], steps)
        writer.add_scalar("acc/next_rank_window", avg["next_rank_acc"], steps)
        writer.add_scalar("q/q_mean_window", avg["q_mean"], steps)
        writer.add_scalar("q/target_mean_window", avg["target_mean"], steps)
        writer.add_scalar("q/q_abs_err_window", avg["q_abs_err"], steps)
        writer.add_scalar("hparam/lr", lr, steps)
        writer.flush()
        window_stats = {key: 0.0 for key in window_stats}
        window_count = 0

    save_every = int(control["save_every"])
    while steps < target_steps:
        try:
            batch = next(data_loader)
        except StopIteration as exc:
            raise RuntimeError(
                f"Mortal offline dataset ended at steps={steps} before target_steps={target_steps}"
            ) from exc
        data_batches_consumed += 1
        obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = batch
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
            batch_metrics = {
                "dqn_loss": float(dqn_loss.detach().cpu()),
                "cql_loss": float(cql_loss.detach().cpu()),
                "next_rank_loss": float(next_rank_loss.detach().cpu()),
                "total_loss": float(loss.detach().cpu()),
                "next_rank_acc": float((next_rank_logits.argmax(-1) == player_ranks).to(torch.float64).mean().detach().cpu()),
                "q_mean": float(q.detach().to(torch.float32).mean().cpu()),
                "target_mean": float(q_target_mc.detach().to(torch.float32).mean().cpu()),
                "q_abs_err": float((q.detach().to(torch.float32) - q_target_mc.detach().to(torch.float32)).abs().mean().cpu()),
            }
            for key, value in batch_metrics.items():
                stats[key] += value
                window_stats[key] += value
            window_count += 1

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
        if trained_steps == 1 or trained_steps % log_every == 0 or steps >= target_steps:
            log_train_metrics()
        if (save_every > 0 and steps % save_every == 0) or steps in archive_steps_set:
            save_checkpoint()

    log_train_metrics(prefix="Mortal final train metrics")
    if trained_steps:
        for key, value in stats.items():
            namespace = "acc" if key.endswith("_acc") else ("q" if key.startswith("q_") or key.startswith("target_") else "loss")
            writer.add_scalar(f"{namespace}/{key}", value / trained_steps, steps)
        writer.add_scalar("hparam/lr", scheduler.get_last_lr()[0], steps)
        writer.flush()
    writer.close()

    save_checkpoint()
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
