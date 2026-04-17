"""Strict Xmodel1 v2 cached dataset loader."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from xmodel1.schema import (
    XMODEL1_DISCARD_REQUIRED_FIELDS,
    XMODEL1_MAX_SPECIAL_CANDIDATES,
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
    XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
)

DEFAULT_BUFFER_SIZE = 512
_MANIFEST_NAME = "xmodel1_export_manifest.json"

EVENT_HISTORY_LEN = 48
EVENT_HISTORY_FEATURE_DIM = 5
EVENT_TYPE_PAD = 0
EVENT_NO_ACTOR = 4
EVENT_NO_TILE = -1


def _empty_event_history() -> np.ndarray:
    out = np.zeros((EVENT_HISTORY_LEN, EVENT_HISTORY_FEATURE_DIM), dtype=np.int16)
    out[:, 0] = EVENT_NO_ACTOR
    out[:, 1] = EVENT_TYPE_PAD
    out[:, 2] = EVENT_NO_TILE
    return out


def _validate_xmodel1_cache_contract(data: np.lib.npyio.NpzFile, path: Path) -> None:
    schema_name = str(data["schema_name"].item()) if "schema_name" in data else None
    schema_version = int(data["schema_version"].item()) if "schema_version" in data else None
    if schema_name != XMODEL1_SCHEMA_NAME or schema_version != XMODEL1_SCHEMA_VERSION:
        raise ValueError(
            f"Xmodel1 cache schema mismatch in {path}: got {schema_name}@{schema_version}, "
            f"expected {XMODEL1_SCHEMA_NAME}@{XMODEL1_SCHEMA_VERSION}; rerun preprocess"
        )
    missing = [field for field in XMODEL1_DISCARD_REQUIRED_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Xmodel1 cache missing required fields {missing} in {path}; rerun preprocess")
    if tuple(data["special_candidate_feat"].shape[1:]) != (
        XMODEL1_MAX_SPECIAL_CANDIDATES,
        XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
    ):
        raise ValueError(
            f"special_candidate_feat shape {data['special_candidate_feat'].shape} incompatible with "
            f"v2 contract ({XMODEL1_MAX_SPECIAL_CANDIDATES}, {XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM}) in {path}; rerun preprocess"
        )
    for field in (
        "special_candidate_type_id",
        "special_candidate_mask",
        "special_candidate_quality_score",
        "special_candidate_rank_bucket",
        "special_candidate_hard_bad_flag",
    ):
        if int(data[field].shape[1]) != XMODEL1_MAX_SPECIAL_CANDIDATES:
            raise ValueError(f"{field} shape {data[field].shape} incompatible with v2 special slots in {path}")
    if tuple(data["event_history"].shape[1:]) != (EVENT_HISTORY_LEN, EVENT_HISTORY_FEATURE_DIM):
        raise ValueError(
            f"event_history shape {data['event_history'].shape} != ({EVENT_HISTORY_LEN}, {EVENT_HISTORY_FEATURE_DIM}) in {path}; rerun preprocess"
        )


def discover_cached_files(root_dirs: List[Path]) -> List[Path]:
    files: List[Path] = []
    seen: set[str] = set()
    for root in root_dirs:
        if root.is_file():
            candidates = [root] if root.suffix == ".npz" else []
        elif root.is_dir():
            candidates = sorted(root.rglob("*.npz"))
        else:
            candidates = []
        for path in candidates:
            key = str(path.resolve()) if path.exists() else str(path)
            if key in seen:
                continue
            seen.add(key)
            files.append(path)
    return files


def find_export_manifests(root_dirs: List[Path]) -> Dict[Path, Dict]:
    manifests: Dict[Path, Dict] = {}
    for root in root_dirs:
        start = root if root.is_dir() else root.parent
        for parent in (start, *start.parents):
            candidate = parent / _MANIFEST_NAME
            if not candidate.exists():
                continue
            if candidate not in manifests:
                with candidate.open("r", encoding="utf-8") as f:
                    manifests[candidate] = json.load(f)
            break
    return manifests


def summarize_cached_files(file_paths: List[Path]) -> Dict[str, object]:
    shard_file_counts: Dict[str, int] = {}
    shard_sample_counts: Dict[str, int] = {}
    total_samples = 0
    for path in file_paths:
        shard = path.parent.name if path.parent.name else "."
        shard_file_counts[shard] = shard_file_counts.get(shard, 0) + 1
        try:
            with np.load(path, allow_pickle=False) as data:
                _validate_xmodel1_cache_contract(data, path)
                sample_count = int(data["state_tile_feat"].shape[0])
        except Exception:
            sample_count = 0
        shard_sample_counts[shard] = shard_sample_counts.get(shard, 0) + sample_count
        total_samples += sample_count
    return {
        "num_files": len(file_paths),
        "num_samples": total_samples,
        "shard_file_counts": shard_file_counts,
        "shard_sample_counts": shard_sample_counts,
    }


def split_cached_files(
    root_dirs: List[Path],
    val_ratio: float = 0.05,
    seed: int = 42,
) -> Tuple[List[Path], List[Path]]:
    all_files = discover_cached_files(root_dirs)
    rng = random.Random(seed)
    rng.shuffle(all_files)
    if not all_files:
        return [], []
    n_val = max(1, int(len(all_files) * val_ratio))
    if len(all_files) == 1:
        return all_files, all_files
    if n_val >= len(all_files):
        n_val = len(all_files) - 1
    return all_files[n_val:], all_files[:n_val]


def infer_cached_dimensions(file_paths: List[Path]) -> Dict[str, int]:
    for path in file_paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                _validate_xmodel1_cache_contract(data, path)
                return {
                    "state_tile_channels": int(data["state_tile_feat"].shape[1]),
                    "state_scalar_dim": int(data["state_scalar"].shape[1]),
                    "candidate_feature_dim": int(data["candidate_feat"].shape[2]),
                    "candidate_flag_dim": int(data["candidate_flags"].shape[2]),
                    "max_candidates": int(data["candidate_mask"].shape[1]),
                    "special_candidate_feature_dim": int(data["special_candidate_feat"].shape[2]),
                    "max_special_candidates": int(data["special_candidate_feat"].shape[1]),
                }
        except Exception:
            continue
    raise FileNotFoundError("no readable Xmodel1 v2 cache files were found")


class Xmodel1DiscardDataset(IterableDataset):
    def __init__(
        self,
        file_paths: List[Path],
        *,
        shuffle: bool = True,
        buffer_size: int = DEFAULT_BUFFER_SIZE,
        seed: Optional[int] = None,
    ):
        self.file_paths = file_paths
        self.shuffle = shuffle
        self.buffer_size = buffer_size
        self.seed = seed if seed is not None else random.randint(0, 2**31)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _iter_file_rows(self, path: Path) -> Iterator[Tuple]:
        with np.load(path, allow_pickle=False) as data:
            _validate_xmodel1_cache_contract(data, path)
            sample_count = int(data["state_tile_feat"].shape[0])
            for i in range(sample_count):
                yield (
                    data["state_tile_feat"][i],
                    data["state_scalar"][i],
                    data["candidate_feat"][i],
                    data["candidate_tile_id"][i],
                    data["candidate_mask"][i],
                    data["candidate_flags"][i],
                    int(data["chosen_candidate_idx"][i]),
                    int(data["sample_type"][i]),
                    int(data["action_idx_target"][i]),
                    data["candidate_quality_score"][i],
                    data["candidate_rank_bucket"][i],
                    data["candidate_hard_bad_flag"][i],
                    data["special_candidate_feat"][i],
                    data["special_candidate_type_id"][i],
                    data["special_candidate_mask"][i],
                    data["special_candidate_quality_score"][i],
                    data["special_candidate_rank_bucket"][i],
                    data["special_candidate_hard_bad_flag"][i],
                    int(data["chosen_special_candidate_idx"][i]),
                    np.float32(data["score_delta_target"][i]),
                    np.float32(data["win_target"][i]),
                    np.float32(data["dealin_target"][i]),
                    np.float32(data["pts_given_win_target"][i]),
                    np.float32(data["pts_given_dealin_target"][i]),
                    np.asarray(data["opp_tenpai_target"][i], dtype=np.float32).reshape(3),
                    np.asarray(data["event_history"][i], dtype=np.int16).reshape(EVENT_HISTORY_LEN, EVENT_HISTORY_FEATURE_DIM),
                )

    def __iter__(self) -> Iterator[Tuple]:
        worker_info = torch.utils.data.get_worker_info()
        paths = self.file_paths
        if worker_info is not None:
            paths = paths[worker_info.id :: worker_info.num_workers]
        base_seed = self.seed + self.epoch * 1_000_003
        if worker_info is not None:
            base_seed += worker_info.id
        rng = random.Random(base_seed)
        paths = list(paths)
        if self.shuffle:
            rng.shuffle(paths)
        buffer: List[Tuple] = []
        for path in paths:
            for row in self._iter_file_rows(path):
                buffer.append(row)
                while len(buffer) >= self.buffer_size:
                    idx = rng.randrange(len(buffer))
                    yield buffer[idx]
                    buffer[idx] = buffer[-1]
                    buffer.pop()
        if self.shuffle:
            rng.shuffle(buffer)
        for row in buffer:
            yield row

    @staticmethod
    def collate(batch: List[Tuple]) -> Dict[str, torch.Tensor]:
        (
            state_tile_feat,
            state_scalar,
            candidate_feat,
            candidate_tile_id,
            candidate_mask,
            candidate_flags,
            chosen_candidate_idx,
            sample_type,
            action_idx_target,
            candidate_quality_score,
            candidate_rank_bucket,
            candidate_hard_bad_flag,
            special_candidate_feat,
            special_candidate_type_id,
            special_candidate_mask,
            special_candidate_quality_score,
            special_candidate_rank_bucket,
            special_candidate_hard_bad_flag,
            chosen_special_candidate_idx,
            score_delta_target,
            win_target,
            dealin_target,
            pts_given_win_target,
            pts_given_dealin_target,
            opp_tenpai_target,
            event_history,
        ) = zip(*batch)
        return {
            "state_tile_feat": torch.from_numpy(np.stack(state_tile_feat)),
            "state_scalar": torch.from_numpy(np.stack(state_scalar)),
            "candidate_feat": torch.from_numpy(np.stack(candidate_feat)),
            "candidate_tile_id": torch.from_numpy(np.stack(candidate_tile_id)).long(),
            "candidate_mask": torch.from_numpy(np.stack(candidate_mask)),
            "candidate_flags": torch.from_numpy(np.stack(candidate_flags)),
            "chosen_candidate_idx": torch.tensor(chosen_candidate_idx, dtype=torch.long),
            "sample_type": torch.tensor(sample_type, dtype=torch.long),
            "action_idx_target": torch.tensor(action_idx_target, dtype=torch.long),
            "candidate_quality_score": torch.from_numpy(np.stack(candidate_quality_score)).float(),
            "candidate_rank_bucket": torch.from_numpy(np.stack(candidate_rank_bucket)).long(),
            "candidate_hard_bad_flag": torch.from_numpy(np.stack(candidate_hard_bad_flag)).float(),
            "special_candidate_feat": torch.from_numpy(np.stack(special_candidate_feat)),
            "special_candidate_type_id": torch.from_numpy(np.stack(special_candidate_type_id)).long(),
            "special_candidate_mask": torch.from_numpy(np.stack(special_candidate_mask)),
            "special_candidate_quality_score": torch.from_numpy(np.stack(special_candidate_quality_score)).float(),
            "special_candidate_rank_bucket": torch.from_numpy(np.stack(special_candidate_rank_bucket)).long(),
            "special_candidate_hard_bad_flag": torch.from_numpy(np.stack(special_candidate_hard_bad_flag)).float(),
            "chosen_special_candidate_idx": torch.tensor(chosen_special_candidate_idx, dtype=torch.long),
            "score_delta_target": torch.tensor(score_delta_target, dtype=torch.float32),
            "win_target": torch.tensor(win_target, dtype=torch.float32),
            "dealin_target": torch.tensor(dealin_target, dtype=torch.float32),
            "pts_given_win_target": torch.tensor(pts_given_win_target, dtype=torch.float32),
            "pts_given_dealin_target": torch.tensor(pts_given_dealin_target, dtype=torch.float32),
            "opp_tenpai_target": torch.from_numpy(np.stack(opp_tenpai_target)).float(),
            "event_history": torch.from_numpy(np.stack(event_history)).long(),
        }


__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "EVENT_HISTORY_FEATURE_DIM",
    "EVENT_HISTORY_LEN",
    "EVENT_NO_ACTOR",
    "EVENT_NO_TILE",
    "EVENT_TYPE_PAD",
    "_empty_event_history",
    "Xmodel1DiscardDataset",
    "discover_cached_files",
    "find_export_manifests",
    "infer_cached_dimensions",
    "split_cached_files",
    "summarize_cached_files",
]
