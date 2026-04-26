#!/usr/bin/env python3
"""Pack many small xmodel1 cache npz files into larger shard npz files."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pack xmodel1 cache files into larger shard files")
    parser.add_argument("--input-dir", "--input_dir", required=True)
    parser.add_argument("--output-dir", "--output_dir", required=True)
    parser.add_argument("--max-samples-per-shard", type=int, default=20_000)
    parser.add_argument("--max-files-per-shard", type=int, default=256)
    parser.add_argument("--compress", action="store_true")
    return parser.parse_args()


def _concat_group(group: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = list(group[0].keys())
    return {key: np.concatenate([item[key] for item in group], axis=0) for key in keys}


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))

    from training.cache_schema import (
        XMODEL1_AUX_TARGET_FIELDS,
        XMODEL1_BASE_FIELDS,
        XMODEL1_CANDIDATE_FEATURE_DIM,
        XMODEL1_CANDIDATE_FLAG_DIM,
        XMODEL1_MAX_CANDIDATES,
        XMODEL1_METADATA_FIELDS,
        XMODEL1_SCHEMA_NAME,
        XMODEL1_SCHEMA_VERSION,
        XMODEL1_TEACHER_FIELDS,
    )
    from xmodel1.cached_dataset import discover_cached_files

    args = _parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cache_files = discover_cached_files([input_dir])
    if not cache_files:
        raise RuntimeError(f"no xmodel1 cache files found under {input_dir}")

    required_fields = (
        *XMODEL1_BASE_FIELDS,
        *XMODEL1_TEACHER_FIELDS,
        *XMODEL1_AUX_TARGET_FIELDS,
        *XMODEL1_METADATA_FIELDS,
    )
    shard_groups: dict[str, list[Path]] = defaultdict(list)
    for path in cache_files:
        shard_groups[path.parent.name].append(path)

    manifest_files: list[str] = []
    shard_file_counts: dict[str, int] = {}
    shard_sample_counts: dict[str, int] = {}
    total_samples = 0

    for ds_name, files in sorted(shard_groups.items()):
        current_group: list[dict[str, np.ndarray]] = []
        current_samples = 0
        shard_index = 0

        def flush_group() -> None:
            nonlocal current_group, current_samples, shard_index, total_samples
            if not current_group:
                return
            merged = _concat_group(current_group)
            ds_out = output_dir / ds_name
            ds_out.mkdir(parents=True, exist_ok=True)
            out_path = ds_out / f"shard_{shard_index:05d}.npz"
            if args.compress:
                np.savez_compressed(out_path, **merged)
            else:
                np.savez(out_path, **merged)
            sample_count = int(merged["state_tile_feat"].shape[0])
            shard_index += 1
            total_samples += sample_count
            shard_file_counts[ds_name] = shard_file_counts.get(ds_name, 0) + 1
            shard_sample_counts[ds_name] = shard_sample_counts.get(ds_name, 0) + sample_count
            manifest_files.append(str(out_path.relative_to(output_dir)))
            current_group = []
            current_samples = 0

        for path in files:
            with np.load(path, allow_pickle=False) as data:
                missing = [field for field in required_fields if field not in data]
                if missing:
                    raise RuntimeError(f"{path} missing required fields: {missing}")
                sample_count = int(data["state_tile_feat"].shape[0])
                if current_group and (
                    current_samples + sample_count > args.max_samples_per_shard
                    or len(current_group) >= args.max_files_per_shard
                ):
                    flush_group()
                current_group.append({field: np.array(data[field]) for field in required_fields})
                current_samples += sample_count
        flush_group()

    manifest = {
        "schema_name": XMODEL1_SCHEMA_NAME,
        "schema_version": XMODEL1_SCHEMA_VERSION,
        "max_candidates": XMODEL1_MAX_CANDIDATES,
        "candidate_feature_dim": XMODEL1_CANDIDATE_FEATURE_DIM,
        "candidate_flag_dim": XMODEL1_CANDIDATE_FLAG_DIM,
        "file_count": len(manifest_files),
        "exported_file_count": len(manifest_files),
        "exported_sample_count": total_samples,
        "processed_file_count": len(manifest_files),
        "skipped_existing_file_count": 0,
        "shard_file_counts": shard_file_counts,
        "shard_sample_counts": shard_sample_counts,
        "used_fallback": False,
        "export_mode": "xmodel1_packed_shards",
        "files": manifest_files,
        "source_input_dir": str(input_dir),
    }
    (output_dir / "xmodel1_export_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        f"xmodel1 shard pack: input_files={len(cache_files)} output_files={len(manifest_files)} "
        f"samples={total_samples} output_dir={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
