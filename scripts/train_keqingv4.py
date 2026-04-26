#!/usr/bin/env python3
"""keqingv4 training entrypoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml
from torch.utils.data import DataLoader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train keqingv4 from cached npz files")
    parser.add_argument("--config", default="configs/keqingv4_default.yaml")
    parser.add_argument("--data-dirs", "--data_dirs", nargs="+", default=None)
    parser.add_argument("--output-dir", "--output_dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--weights-only", action="store_true")
    parser.add_argument("--allow-stale-cache", action="store_true")
    return parser.parse_args()


def _load_cfg(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _resolve_paths(root: Path, values: list[str] | None) -> list[Path]:
    resolved: list[Path] = []
    for raw in values or []:
        path = Path(raw)
        resolved.append(path if path.is_absolute() else root / path)
    return resolved


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))

    from keqingv4.cache_contract import (
        assert_keqingv4_contract,
        inspect_keqingv4_contract,
    )
    from keqingv4.checkpoint import keqingv4_checkpoint_contract_summary
    from keqingv4.cached_dataset import CachedMjaiDatasetV4, split_cached_files
    from keqingv4.model import KeqingV4Model
    from keqingv4.trainer import train

    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = _load_cfg(cfg_path if cfg_path.is_absolute() else root / cfg_path)
    cfg["model_name"] = "keqingv4"

    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.data_dirs is not None:
        cfg["data_dirs"] = list(args.data_dirs)

    device = args.device or cfg.get("device", "cuda")

    if args.smoke:
        cfg["output_dir"] = cfg.get("output_dir", "artifacts/models/keqingv4") + "_smoke"
        cfg["num_epochs"] = 1
        cfg["batch_size"] = 64
        cfg["num_workers"] = 0
        cfg["buffer_size"] = 64
        cfg["prefetch_factor"] = 2
        cfg["pin_memory"] = False
        cfg["persistent_workers"] = False
        cfg["warmup_steps"] = 1
        cfg["steps_per_epoch"] = 2
        cfg["files_per_epoch_ratio"] = 1.0
        cfg["log_interval"] = 1

    data_dirs = _resolve_paths(root, cfg.get("data_dirs", []))
    if not data_dirs:
        raise ValueError("config.data_dirs is required")

    train_files, val_files = split_cached_files(
        data_dirs,
        val_ratio=float(cfg.get("val_ratio", 0.05)),
        seed=args.seed,
    )
    if args.smoke:
        all_files = train_files + [p for p in val_files if p not in train_files]
        if not all_files:
            raise RuntimeError("no cached files available for keqingv4 smoke training")
        train_files = (train_files or all_files)[:2]
        val_files = (val_files or all_files[:1])[:1]
    elif not train_files or not val_files:
        raise RuntimeError(f"insufficient cached files: train={len(train_files)} val={len(val_files)}")

    inspected = inspect_keqingv4_contract(train_files + [p for p in val_files if p not in train_files])
    inspected_dims = sorted(inspected["summary_dims"])
    call_slots = sorted(inspected["call_summary_slots"])
    special_slots = sorted(inspected["special_summary_slots"])
    opportunity_shapes = sorted(inspected["opportunity_shapes"])
    print(
        "keqingv4 cache-inspect: "
        f"scanned={inspected['files_scanned']} "
        f"summary_dims={inspected_dims or ['unknown']} "
        f"call_slots={call_slots or ['unknown']} "
        f"special_slots={special_slots or ['unknown']} "
        f"pts_win_files={inspected['pts_given_win_files']} "
        f"pts_dealin_files={inspected['pts_given_dealin_files']} "
        f"opp_tenpai_files={inspected['opp_tenpai_files']} "
        f"event_history_files={inspected['event_history_files']} "
        f"opportunity_shapes={opportunity_shapes or ['unknown']}",
        flush=True,
    )
    contract = keqingv4_checkpoint_contract_summary()
    print(
        "keqingv4 checkpoint-contract: "
        f"{contract['schema_name']}@{contract['schema_version']} "
        f"summary_dim={contract['summary_dim']} "
        f"call_slots={contract['call_summary_slots']} "
        f"special_slots={contract['special_summary_slots']} "
        f"event_history=({contract['event_history_len']},{contract['event_history_dim']}) "
        f"opportunity_dim={contract['opportunity_dim']} strict_load=True",
        flush=True,
    )
    if inspected["npz_problems"]:
        print("keqingv4 cache warning: scanned npz issues detected:", flush=True)
        for problem in inspected["npz_problems"]:
            print(f"  - {problem}", flush=True)
    if inspected["manifest_problems"]:
        print("keqingv4 cache warning: manifest issues detected:", flush=True)
        for problem in inspected["manifest_problems"]:
            print(f"  - {problem}", flush=True)
    assert_keqingv4_contract(
        inspected,
        smoke=args.smoke,
        allow_stale_cache=args.allow_stale_cache,
    )

    batch_size = int(cfg.get("batch_size", 1024))
    num_workers = int(cfg.get("num_workers", 4))
    aug_perms = int(cfg.get("aug_perms", 1))
    pin_memory = bool(cfg.get("pin_memory", device == "cuda"))
    persistent_workers = bool(cfg.get("persistent_workers", num_workers > 0))
    prefetch_factor = int(cfg.get("prefetch_factor", 2))
    buffer_size = int(cfg.get("buffer_size", 512))

    val_ds = CachedMjaiDatasetV4(
        val_files,
        shuffle=False,
        seed=args.seed,
        aug_perms=aug_perms,
        buffer_size=buffer_size,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        collate_fn=CachedMjaiDatasetV4.collate,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(persistent_workers and num_workers > 0),
        prefetch_factor=prefetch_factor if num_workers > 0 else None,
    )

    model = KeqingV4Model(
        hidden_dim=int(cfg.get("hidden_dim", 320)),
        num_res_blocks=int(cfg.get("num_res_blocks", 6)),
        action_embed_dim=int(cfg.get("action_embed_dim", 64)),
        context_dim=int(cfg.get("context_dim", 32)),
        dropout=float(cfg.get("dropout", 0.1)),
    )

    output_dir_value = Path(cfg.get("output_dir", "artifacts/models/keqingv4"))
    output_dir = output_dir_value if output_dir_value.is_absolute() else root / output_dir_value
    resume_path = None
    if args.resume:
        resume_value = Path(args.resume)
        resume_path = resume_value if resume_value.is_absolute() else root / resume_value

    print(
        f"keqingv4 train: data_dirs={[str(p) for p in data_dirs]} "
        f"train_files={len(train_files)} val_files={len(val_files)} "
        f"batch={batch_size} epochs={cfg.get('num_epochs')} device={device} output={output_dir}"
    )

    train(
        model=model,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=output_dir,
        resume_path=resume_path,
        weights_only=args.weights_only,
        device_str=device,
        train_files=train_files,
        seed=args.seed,
        use_cuda=(device == "cuda"),
        aug_perms=aug_perms,
        batch_size=batch_size,
        num_workers=num_workers,
        files_per_epoch_ratio=float(cfg.get("files_per_epoch_ratio", 1.0)),
    )


if __name__ == "__main__":
    main()
