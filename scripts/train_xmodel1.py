#!/usr/bin/env python3
"""Train Xmodel1 from candidate-centric cached npz files."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Xmodel1 from cached npz files")
    parser.add_argument("--config", default="configs/xmodel1_default.yaml")
    parser.add_argument("--data-dirs", "--data_dirs", nargs="+", default=None)
    parser.add_argument("--output-dir", "--output_dir", default=None)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--strict-cache-scan", action="store_true")
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


def _resolve_resume_path(root: Path, raw: str | None, output_dir: Path) -> Path | None:
    if raw:
        if raw == "auto":
            candidate = output_dir / "last.pth"
            return candidate if candidate.exists() else None
        path = Path(raw)
        return path if path.is_absolute() else root / path
    candidate = output_dir / "last.pth"
    return candidate if candidate.exists() else None


def _required_shards_from_data_dirs(data_dirs: list[Path]) -> list[str]:
    return sorted({path.name for path in data_dirs if path.name.startswith("ds")})


def _validate_batch_gate(batch: dict, *, inferred_dims: dict[str, int] | None) -> None:
    required = (
        "state_tile_feat",
        "state_scalar",
        "candidate_feat",
        "candidate_mask",
        "candidate_flags",
        "special_candidate_feat",
        "special_candidate_mask",
        "event_history",
        "opp_tenpai_target",
        "pts_given_win_target",
        "pts_given_dealin_target",
    )
    missing = [key for key in required if key not in batch]
    if missing:
        raise RuntimeError(f"xmodel1 train gate: batch missing required fields {missing}")
    batch_size = int(batch["candidate_feat"].shape[0])
    if batch_size <= 0:
        raise RuntimeError("xmodel1 train gate: first batch is empty")
    if inferred_dims is not None:
        expected_candidate_dim = int(inferred_dims["candidate_feature_dim"])
        expected_flag_dim = int(inferred_dims["candidate_flag_dim"])
        expected_special_dim = int(inferred_dims["special_candidate_feature_dim"])
        if int(batch["candidate_feat"].shape[-1]) != expected_candidate_dim:
            raise RuntimeError(
                f"xmodel1 train gate: candidate_feat last dim {batch['candidate_feat'].shape[-1]} != {expected_candidate_dim}"
            )
        if int(batch["candidate_flags"].shape[-1]) != expected_flag_dim:
            raise RuntimeError(
                f"xmodel1 train gate: candidate_flags last dim {batch['candidate_flags'].shape[-1]} != {expected_flag_dim}"
            )
        if int(batch["special_candidate_feat"].shape[-1]) != expected_special_dim:
            raise RuntimeError(
                f"xmodel1 train gate: special_candidate_feat last dim {batch['special_candidate_feat'].shape[-1]} != {expected_special_dim}"
            )
    if tuple(batch["event_history"].shape[1:]) != (48, 5):
        raise RuntimeError(f"xmodel1 train gate: event_history shape {tuple(batch['event_history'].shape)} != (N, 48, 5)")
    if int(batch["opp_tenpai_target"].shape[-1]) != 3:
        raise RuntimeError(f"xmodel1 train gate: opp_tenpai_target shape {tuple(batch['opp_tenpai_target'].shape)} != (N, 3)")


def _sample_train_files_for_epoch(
    train_files: list[Path],
    *,
    epoch: int,
    seed: int,
    files_per_epoch_ratio: float,
    files_per_epoch_count: int | None,
) -> list[Path]:
    if not train_files:
        return []
    if files_per_epoch_count is not None and files_per_epoch_count > 0:
        n = min(len(train_files), files_per_epoch_count)
    elif 0.0 < files_per_epoch_ratio < 1.0:
        n = max(1, int(len(train_files) * files_per_epoch_ratio))
    else:
        n = len(train_files)
    if n >= len(train_files):
        return list(train_files)
    rng = random.Random(seed + epoch * 1_000_003)
    sampled = list(train_files)
    rng.shuffle(sampled)
    return sampled[:n]


def _estimate_file_subset_summary(
    file_paths: list[Path],
    *,
    total_train_files: int,
    total_train_samples: int,
) -> dict[str, float | int]:
    sampled_files = len(file_paths)
    if total_train_files <= 0 or total_train_samples <= 0:
        sampled_samples = 0
    else:
        avg_samples_per_file = total_train_samples / total_train_files
        sampled_samples = int(round(sampled_files * avg_samples_per_file))
    return {
        "sampled_files": sampled_files,
        "sampled_samples": sampled_samples,
    }


def _auto_steps_from_sampled_summary(
    sampled_summary: dict[str, float | int] | None,
    *,
    batch_size: int,
) -> int | None:
    if not sampled_summary:
        return None
    sampled_samples = int(sampled_summary.get("sampled_samples", 0) or 0)
    if sampled_samples <= 0:
        return None
    return max(1, math.ceil(sampled_samples / max(1, batch_size)))


def _expected_manifest_summary_for_cache_files(
    manifest: dict,
    cache_files: list[Path],
    data_dirs: list[Path],
) -> dict[str, object] | None:
    if not cache_files or any(path.is_file() for path in data_dirs):
        return None
    shard_file_counts = manifest.get("shard_file_counts")
    shard_sample_counts = manifest.get("shard_sample_counts")
    if not isinstance(shard_file_counts, dict) or not isinstance(shard_sample_counts, dict):
        return None
    requested_shards = sorted({path.parent.name if path.parent.name else "." for path in cache_files})
    if any(shard not in shard_file_counts or shard not in shard_sample_counts for shard in requested_shards):
        return None
    expected_file_counts = {shard: int(shard_file_counts[shard]) for shard in requested_shards}
    expected_sample_counts = {shard: int(shard_sample_counts[shard]) for shard in requested_shards}
    return {
        "num_files": sum(expected_file_counts.values()),
        "num_samples": sum(expected_sample_counts.values()),
        "shard_file_counts": expected_file_counts,
        "shard_sample_counts": expected_sample_counts,
    }


def _validate_manifest_against_cache_files(
    manifest_map: dict[Path, dict],
    *,
    cache_files: list[Path],
    data_dirs: list[Path],
    summarize_cached_files_fn,
) -> None:
    for manifest_path, manifest in manifest_map.items():
        manifest_cache_files = [
            path
            for path in cache_files
            if _find_manifest_for_cache_file(path, {manifest_path: manifest}) is not None
        ]
        expected = _expected_manifest_summary_for_cache_files(manifest, manifest_cache_files, data_dirs)
        if expected is None:
            continue
        actual = summarize_cached_files_fn(manifest_cache_files)
        if actual != expected:
            raise RuntimeError(
                f"xmodel1 train: manifest/cache mismatch for {manifest_path}; "
                f"expected {expected}, got {actual}"
            )


def _find_manifest_for_cache_file(path: Path, manifest_map: dict[Path, dict]) -> tuple[Path, dict] | None:
    resolved = path.resolve()
    matches: list[tuple[int, Path, dict]] = []
    for manifest_path, manifest in manifest_map.items():
        manifest_root = manifest_path.parent.resolve()
        try:
            resolved.relative_to(manifest_root)
        except ValueError:
            continue
        matches.append((len(manifest_root.parts), manifest_path, manifest))
    if not matches:
        return None
    _, manifest_path, manifest = max(matches, key=lambda item: item[0])
    return manifest_path, manifest


def _summarize_files_from_manifest(
    file_paths: list[Path],
    *,
    manifest_map: dict[Path, dict],
) -> dict[str, object] | None:
    shard_file_counts: dict[str, int] = {}
    shard_sample_counts: dict[str, int] = {}
    grouped: dict[Path, dict[str, int]] = {}
    manifests_used: set[str] = set()
    sample_count_estimated = False

    for path in file_paths:
        match = _find_manifest_for_cache_file(path, manifest_map)
        if match is None:
            return None
        manifest_path, _ = match
        shard = path.parent.name if path.parent.name else "."
        shard_file_counts[shard] = shard_file_counts.get(shard, 0) + 1
        grouped.setdefault(manifest_path, {})
        grouped[manifest_path][shard] = grouped[manifest_path].get(shard, 0) + 1
        manifests_used.add(str(manifest_path))

    total_samples = 0
    for manifest_path, per_manifest_counts in grouped.items():
        manifest = manifest_map[manifest_path]
        manifest_file_counts = manifest.get("shard_file_counts")
        manifest_sample_counts = manifest.get("shard_sample_counts")
        if not isinstance(manifest_file_counts, dict) or not isinstance(manifest_sample_counts, dict):
            return None
        for shard, selected_file_count in per_manifest_counts.items():
            if shard not in manifest_file_counts or shard not in manifest_sample_counts:
                return None
            shard_total_files = int(manifest_file_counts[shard])
            shard_total_samples = int(manifest_sample_counts[shard])
            if shard_total_files <= 0:
                return None
            if selected_file_count > shard_total_files:
                raise RuntimeError(
                    f"xmodel1 train: selected files exceed manifest shard count for {manifest_path} shard={shard}"
                )
            estimated_samples = int(round((selected_file_count / shard_total_files) * shard_total_samples))
            if selected_file_count != shard_total_files:
                sample_count_estimated = True
            shard_sample_counts[shard] = shard_sample_counts.get(shard, 0) + estimated_samples
            total_samples += estimated_samples

    return {
        "num_files": len(file_paths),
        "num_samples": total_samples,
        "shard_file_counts": shard_file_counts,
        "shard_sample_counts": shard_sample_counts,
        "sample_count_source": "manifest_estimate" if sample_count_estimated else "manifest_exact",
        "is_sample_count_estimated": sample_count_estimated,
        "manifest_paths": sorted(manifests_used),
    }


def _build_dataset_summary(
    *,
    manifest_map: dict[Path, dict],
    selected_files: list[Path],
    train_files: list[Path],
    val_files: list[Path],
    summarize_cached_files_fn,
) -> dict[str, object]:
    selected = _summarize_files_from_manifest(selected_files, manifest_map=manifest_map)
    train = _summarize_files_from_manifest(train_files, manifest_map=manifest_map)
    val = _summarize_files_from_manifest(val_files, manifest_map=manifest_map)
    if selected is None or train is None or val is None:
        return {
            "selected": {
                **summarize_cached_files_fn(selected_files),
                "sample_count_source": "cache_scan",
                "is_sample_count_estimated": False,
            },
            "train": {
                **summarize_cached_files_fn(train_files),
                "sample_count_source": "cache_scan",
                "is_sample_count_estimated": False,
            },
            "val": {
                **summarize_cached_files_fn(val_files),
                "sample_count_source": "cache_scan",
                "is_sample_count_estimated": False,
            },
        }
    return {
        "selected": selected,
        "train": train,
        "val": val,
    }


def _write_training_inputs(
    *,
    output_dir: Path,
    cfg: dict,
    data_dirs: list[Path],
    manifest_paths: list[str],
    train_files: list[Path],
    val_files: list[Path],
    inferred_dims: dict[str, int] | None,
    dataset_summary: dict[str, object] | None,
    synthetic_smoke: bool,
    device: str,
    seed: int,
    resume_path: Path | None,
    strict_cache_scan: bool,
) -> None:
    payload = {
        "model_version": "xmodel1",
        "synthetic_smoke": bool(synthetic_smoke),
        "device": device,
        "seed": int(seed),
        "resume_path": str(resume_path) if resume_path is not None else None,
        "strict_cache_scan": bool(strict_cache_scan),
        "data_dirs": [str(path) for path in data_dirs],
        "manifest_paths": manifest_paths,
        "train_files": [str(path) for path in train_files],
        "val_files": [str(path) for path in val_files],
        "inferred_dims": inferred_dims,
        "dataset_summary": dataset_summary,
        "cfg": cfg,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train_inputs.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))

    from xmodel1.cached_dataset import (
        Xmodel1DiscardDataset,
        discover_cached_files,
        find_export_manifests,
        infer_cached_dimensions,
        probe_cached_samples,
        split_cached_files,
        summarize_cached_files,
        validate_export_manifest,
    )
    from xmodel1.model import Xmodel1Model
    from xmodel1.schema import (
        XMODEL1_CANDIDATE_FEATURE_DIM,
        XMODEL1_CANDIDATE_FLAG_DIM,
    )
    from xmodel1.trainer import train

    args = _parse_args()
    cfg_path = Path(args.config)
    cfg = _load_cfg(cfg_path if cfg_path.is_absolute() else root / cfg_path)
    cfg["model_name"] = "xmodel1"
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.data_dirs is not None:
        cfg["data_dirs"] = list(args.data_dirs)

    device = args.device or cfg.get("device", "cuda")
    synthetic_smoke = False
    if args.smoke:
        cfg["output_dir"] = cfg.get("output_dir", "artifacts/models/xmodel1") + "_smoke"
        cfg["num_epochs"] = 1
        cfg["batch_size"] = 8
        cfg["num_workers"] = 0
        cfg["buffer_size"] = 16
        cfg["log_interval"] = 1
        cfg["steps_per_epoch"] = 117
        cfg["val_steps_per_epoch"] = 63

    data_dirs = _resolve_paths(root, cfg.get("data_dirs", []))
    manifest_map = find_export_manifests(data_dirs)
    required_shards = _required_shards_from_data_dirs(data_dirs)
    if data_dirs and not manifest_map and not args.smoke:
        raise RuntimeError("xmodel1 train gate: no export manifest found for configured data_dirs")
    for manifest_path in manifest_map:
        validate_export_manifest(manifest_path, required_shards=required_shards)

    batch_size = int(cfg.get("batch_size", 256))
    num_workers = int(cfg.get("num_workers", 2))
    buffer_size = int(cfg.get("buffer_size", 512))
    pin_memory = bool(cfg.get("pin_memory", device == "cuda"))
    persistent_workers = bool(cfg.get("persistent_workers", False))
    prefetch_factor = int(cfg.get("prefetch_factor", 2))
    files_per_epoch_ratio = float(cfg.get("files_per_epoch_ratio", 1.0))
    files_per_epoch_count_cfg = cfg.get("files_per_epoch_count", None)
    files_per_epoch_count = (
        int(files_per_epoch_count_cfg)
        if files_per_epoch_count_cfg is not None and int(files_per_epoch_count_cfg) > 0
        else None
    )

    train_loader = None
    train_loader_factory = None
    train_loader_summary_factory = None
    val_loader = None
    train_files: list[Path] = []
    val_files: list[Path] = []
    inferred_dims = None
    dataset_summary = None

    cache_files = discover_cached_files(data_dirs)
    if cache_files:
        probe_summary = probe_cached_samples(cache_files)
        if args.strict_cache_scan:
            _validate_manifest_against_cache_files(
                manifest_map,
                cache_files=cache_files,
                data_dirs=data_dirs,
                summarize_cached_files_fn=summarize_cached_files,
            )
        print(
            f"xmodel1 train gate: cache_probe files={probe_summary['num_files']} "
            f"rows={probe_summary['rows_probed']} dims={probe_summary['dims']} "
            f"required_shards={required_shards}",
            flush=True,
        )

        train_files, val_files = split_cached_files(
            data_dirs,
            val_ratio=float(cfg.get("val_ratio", 0.05)),
            seed=args.seed,
        )
        if args.smoke:
            all_files = train_files + [path for path in val_files if path not in train_files]
            if not all_files:
                raise RuntimeError("no cached files available for Xmodel1 smoke training")
            train_files = (train_files or all_files)[:2]
            val_files = (val_files or all_files[:1])[:1]
        elif not train_files or not val_files:
            raise RuntimeError(f"insufficient cached files: train={len(train_files)} val={len(val_files)}")

        selected_files = train_files + [path for path in val_files if path not in train_files]
        inferred_dims = infer_cached_dimensions(train_files or val_files)
        dataset_summary = _build_dataset_summary(
            manifest_map=manifest_map,
            selected_files=selected_files,
            train_files=train_files,
            val_files=val_files,
            summarize_cached_files_fn=summarize_cached_files,
        )

        def _make_train_loader(epoch: int):
            epoch_files = _sample_train_files_for_epoch(
                train_files,
                epoch=epoch,
                seed=args.seed,
                files_per_epoch_ratio=files_per_epoch_ratio,
                files_per_epoch_count=files_per_epoch_count,
            )
            return DataLoader(
                Xmodel1DiscardDataset(epoch_files, shuffle=True, buffer_size=buffer_size, seed=args.seed + epoch),
                batch_size=batch_size,
                collate_fn=Xmodel1DiscardDataset.collate,
                num_workers=num_workers,
                pin_memory=pin_memory,
                persistent_workers=(persistent_workers and num_workers > 0),
                prefetch_factor=prefetch_factor if num_workers > 0 else None,
            )

        def _make_train_loader_summary(epoch: int):
            epoch_files = _sample_train_files_for_epoch(
                train_files,
                epoch=epoch,
                seed=args.seed,
                files_per_epoch_ratio=files_per_epoch_ratio,
                files_per_epoch_count=files_per_epoch_count,
            )
            summary = _estimate_file_subset_summary(
                epoch_files,
                total_train_files=len(train_files),
                total_train_samples=int((dataset_summary or {}).get("train", {}).get("num_samples", 0) or 0),
            )
            base_samples = int((dataset_summary or {}).get("train", {}).get("num_samples", 0) or 0)
            summary["sampled_ratio"] = (float(summary["sampled_samples"]) / float(base_samples)) if base_samples > 0 else 0.0
            return summary

        train_loader_factory = _make_train_loader
        train_loader_summary_factory = _make_train_loader_summary
        steps_per_epoch_cfg = cfg.get("steps_per_epoch", None)
        if steps_per_epoch_cfg is None or int(steps_per_epoch_cfg) <= 0:
            auto_train_summary = train_loader_summary_factory(0)
            auto_steps_per_epoch = _auto_steps_from_sampled_summary(
                auto_train_summary,
                batch_size=batch_size,
            )
            if auto_steps_per_epoch is not None:
                cfg["steps_per_epoch"] = auto_steps_per_epoch
                print(
                    "xmodel1 train: auto steps_per_epoch="
                    f"{auto_steps_per_epoch} from sampled_files={auto_train_summary.get('sampled_files')} "
                    f"sampled_samples={auto_train_summary.get('sampled_samples')} batch_size={batch_size}",
                    flush=True,
                )
        train_loader = train_loader_factory(0)
        val_loader = DataLoader(
            Xmodel1DiscardDataset(val_files, shuffle=False, buffer_size=buffer_size, seed=args.seed),
            batch_size=batch_size,
            collate_fn=Xmodel1DiscardDataset.collate,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=(persistent_workers and num_workers > 0),
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
        )
        first_batch = next(iter(train_loader), None)
        if first_batch is None:
            raise RuntimeError("xmodel1 train gate: train loader produced no batches")
        _validate_batch_gate(first_batch, inferred_dims=inferred_dims)
    elif args.smoke:
        synthetic_smoke = True

        class _SyntheticDataset(torch.utils.data.Dataset):
            def __len__(self):
                return 32

            def __getitem__(self, idx):
                rng = np.random.default_rng(seed=args.seed + idx)
                candidate_mask = np.zeros((14,), dtype=np.uint8)
                candidate_mask[:5] = 1
                chosen = int(idx % 5)
                quality = np.zeros((14,), dtype=np.float32)
                quality[chosen] = 1.0
                rank_bucket = np.zeros((14,), dtype=np.int64)
                rank_bucket[chosen] = 3
                return {
                    "state_tile_feat": rng.random((57, 34), dtype=np.float32),
                    "state_scalar": rng.random((int(cfg.get("state_scalar_dim", 56)),), dtype=np.float32),
                    "candidate_feat": rng.random((14, XMODEL1_CANDIDATE_FEATURE_DIM), dtype=np.float32),
                    "candidate_tile_id": np.array([0, 1, 2, 3, 4] + [-1] * 9, dtype=np.int16),
                    "candidate_mask": candidate_mask,
                    "candidate_flags": np.zeros((14, XMODEL1_CANDIDATE_FLAG_DIM), dtype=np.uint8),
                    "chosen_candidate_idx": chosen,
                    "candidate_quality_score": quality,
                    "candidate_rank_bucket": rank_bucket,
                    "candidate_hard_bad_flag": np.zeros((14,), dtype=np.float32),
                    "global_value_target": np.float32(0.0),
                    "score_delta_target": np.float32(0.0),
                    "win_target": np.float32(0.0),
                    "dealin_target": np.float32(0.0),
                    "offense_quality_target": np.float32(1.0),
                }

        def _collate(batch):
            out = {}
            for key in batch[0].keys():
                values = [item[key] for item in batch]
                if key == "chosen_candidate_idx":
                    out[key] = torch.tensor(values, dtype=torch.long)
                else:
                    out[key] = torch.from_numpy(np.stack(values))
            return out

        train_loader = DataLoader(_SyntheticDataset(), batch_size=8, shuffle=False, collate_fn=_collate)
        val_loader = DataLoader(_SyntheticDataset(), batch_size=8, shuffle=False, collate_fn=_collate)
    else:
        raise ValueError("config.data_dirs is required and must contain exported Xmodel1 caches unless --smoke is used")

    if train_files or val_files:
        for key in (
            "state_tile_channels",
            "state_scalar_dim",
            "candidate_feature_dim",
            "candidate_flag_dim",
            "special_candidate_feature_dim",
        ):
            cfg_value = cfg.get(key)
            if key not in inferred_dims:
                continue
            inferred_value = inferred_dims[key]
            if cfg_value is not None and int(cfg_value) != inferred_value:
                print(
                    f"xmodel1 train: overriding {key} from config={cfg_value} to cache={inferred_value}",
                    flush=True,
                )
            cfg[key] = inferred_value

    model = Xmodel1Model(
        state_tile_channels=int(cfg.get("state_tile_channels", 57)),
        state_scalar_dim=int(cfg.get("state_scalar_dim", 64)),
        candidate_feature_dim=int(cfg.get("candidate_feature_dim", XMODEL1_CANDIDATE_FEATURE_DIM)),
        candidate_flag_dim=int(cfg.get("candidate_flag_dim", XMODEL1_CANDIDATE_FLAG_DIM)),
        special_candidate_feature_dim=int(cfg.get("special_candidate_feature_dim", 16)),
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_res_blocks=int(cfg.get("num_res_blocks", 4)),
        dropout=float(cfg.get("dropout", 0.1)),
    )

    output_dir_path = Path(cfg.get("output_dir", "artifacts/models/xmodel1"))
    output_dir = output_dir_path if output_dir_path.is_absolute() else root / output_dir_path
    resume_path = _resolve_resume_path(root, args.resume, output_dir)
    manifest_paths = [str(path) for path in sorted(manifest_map)]
    _write_training_inputs(
        output_dir=output_dir,
        cfg=cfg,
        data_dirs=data_dirs,
        manifest_paths=manifest_paths,
        train_files=train_files,
        val_files=val_files,
        inferred_dims=inferred_dims,
        dataset_summary=dataset_summary,
        synthetic_smoke=synthetic_smoke,
        device=device,
        seed=args.seed,
        resume_path=resume_path,
        strict_cache_scan=args.strict_cache_scan,
    )
    print(
        f"xmodel1 train: synthetic_smoke={synthetic_smoke} data_dirs={[str(p) for p in data_dirs]} "
        f"train_files={len(train_files)} val_files={len(val_files)} manifests={manifest_paths} "
        f"batch={batch_size} epochs={cfg.get('num_epochs')} "
        f"steps_per_epoch={cfg.get('steps_per_epoch')} val_steps_per_epoch={cfg.get('val_steps_per_epoch')} "
        f"files_per_epoch_ratio={files_per_epoch_ratio} files_per_epoch_count={files_per_epoch_count} "
        f"buffer_size={buffer_size} num_workers={num_workers} pin_memory={pin_memory} "
        f"strict_cache_scan={args.strict_cache_scan} "
        f"device={device} output={output_dir} resume={resume_path}",
        flush=True,
    )
    train(
        model=model,
        train_loader=train_loader,
        train_loader_factory=train_loader_factory,
        train_loader_summary_factory=train_loader_summary_factory,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=output_dir,
        resume_path=resume_path,
        device_str=device,
        dataset_summary=dataset_summary,
    )


if __name__ == "__main__":
    main()
