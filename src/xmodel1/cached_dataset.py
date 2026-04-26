"""Strict Xmodel1 cached dataset loader."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset

from xmodel1.schema import (
    XMODEL1_DISCARD_REQUIRED_FIELDS,
    XMODEL1_MAX_RESPONSE_CANDIDATES,
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
)
from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_HISTORY_SUMMARY_DIM,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_RULE_CONTEXT_DIM,
)

DEFAULT_BUFFER_SIZE = 512
_MANIFEST_NAME = "xmodel1_export_manifest.json"
_MANIFEST_REQUIRED_KEYS = (
    "schema_name",
    "schema_version",
    "file_count",
    "exported_file_count",
    "exported_sample_count",
    "processed_file_count",
    "skipped_existing_file_count",
    "shard_file_counts",
    "shard_sample_counts",
)

def _empty_history_summary() -> np.ndarray:
    return np.zeros((XMODEL1_HISTORY_SUMMARY_DIM,), dtype=np.float16)


def _default_rule_context() -> np.ndarray:
    return np.asarray((1.0, 0.5, 0.0, -1.5, 0.0, 1.0), dtype=np.float32)


def _validate_xmodel1_cache_contract(data: np.lib.npyio.NpzFile, path: Path) -> None:
    schema_name = str(data["schema_name"].item()) if "schema_name" in data else None
    schema_version = int(data["schema_version"].item()) if "schema_version" in data else None
    if schema_name != XMODEL1_SCHEMA_NAME or schema_version != XMODEL1_SCHEMA_VERSION:
        raise ValueError(
            f"Xmodel1 cache schema mismatch in {path}: got {schema_name}@{schema_version}, "
            f"expected {XMODEL1_SCHEMA_NAME}@{XMODEL1_SCHEMA_VERSION}; "
            f"rerun preprocess for {XMODEL1_SCHEMA_NAME}"
        )
    missing = [field for field in XMODEL1_DISCARD_REQUIRED_FIELDS if field not in data]
    if missing:
        raise ValueError(f"Xmodel1 cache missing required fields {missing} in {path}; rerun preprocess")
    if tuple(data["history_summary"].shape[1:]) != (XMODEL1_HISTORY_SUMMARY_DIM,):
        raise ValueError(
            f"history_summary shape {data['history_summary'].shape} != ({XMODEL1_HISTORY_SUMMARY_DIM},) in {path}; rerun preprocess"
        )
    if len(data["final_rank_target"].shape) != 1:
        raise ValueError(
            f"final_rank_target shape {data['final_rank_target'].shape} incompatible in {path}"
        )
    if len(data["final_score_delta_points_target"].shape) != 1:
        raise ValueError(
            f"final_score_delta_points_target shape {data['final_score_delta_points_target'].shape} incompatible in {path}"
        )
    if tuple(data["response_action_idx"].shape[1:]) != (XMODEL1_MAX_RESPONSE_CANDIDATES,):
        raise ValueError(
            f"response_action_idx shape {data['response_action_idx'].shape} incompatible with "
            f"v4 response slots ({XMODEL1_MAX_RESPONSE_CANDIDATES},) in {path}"
        )
    if tuple(data["response_action_mask"].shape[1:]) != (XMODEL1_MAX_RESPONSE_CANDIDATES,):
        raise ValueError(
            f"response_action_mask shape {data['response_action_mask'].shape} incompatible with "
            f"v4 response slots ({XMODEL1_MAX_RESPONSE_CANDIDATES},) in {path}"
        )
    if tuple(data["response_post_candidate_feat"].shape[1:]) != (
        XMODEL1_MAX_RESPONSE_CANDIDATES,
        XMODEL1_MAX_CANDIDATES,
        XMODEL1_CANDIDATE_FEATURE_DIM,
    ):
        raise ValueError(
            f"response_post_candidate_feat shape {data['response_post_candidate_feat'].shape} incompatible in {path}"
        )
    if tuple(data["response_post_candidate_tile_id"].shape[1:]) != (
        XMODEL1_MAX_RESPONSE_CANDIDATES,
        XMODEL1_MAX_CANDIDATES,
    ):
        raise ValueError(
            f"response_post_candidate_tile_id shape {data['response_post_candidate_tile_id'].shape} incompatible in {path}"
        )
    if tuple(data["response_post_candidate_mask"].shape[1:]) != (
        XMODEL1_MAX_RESPONSE_CANDIDATES,
        XMODEL1_MAX_CANDIDATES,
    ):
        raise ValueError(
            f"response_post_candidate_mask shape {data['response_post_candidate_mask'].shape} incompatible in {path}"
        )
    if tuple(data["response_post_candidate_flags"].shape[1:]) != (
        XMODEL1_MAX_RESPONSE_CANDIDATES,
        XMODEL1_MAX_CANDIDATES,
        XMODEL1_CANDIDATE_FLAG_DIM,
    ):
        raise ValueError(
            f"response_post_candidate_flags shape {data['response_post_candidate_flags'].shape} incompatible in {path}"
        )
    if tuple(data["response_post_candidate_quality_score"].shape[1:]) != (
        XMODEL1_MAX_RESPONSE_CANDIDATES,
        XMODEL1_MAX_CANDIDATES,
    ):
        raise ValueError(
            f"response_post_candidate_quality_score shape {data['response_post_candidate_quality_score'].shape} incompatible in {path}"
        )
    if tuple(data["response_post_candidate_hard_bad_flag"].shape[1:]) != (
        XMODEL1_MAX_RESPONSE_CANDIDATES,
        XMODEL1_MAX_CANDIDATES,
    ):
        raise ValueError(
            f"response_post_candidate_hard_bad_flag shape {data['response_post_candidate_hard_bad_flag'].shape} incompatible in {path}"
        )
    if tuple(data["response_teacher_discard_idx"].shape[1:]) != (
        XMODEL1_MAX_RESPONSE_CANDIDATES,
    ):
        raise ValueError(
            f"response_teacher_discard_idx shape {data['response_teacher_discard_idx'].shape} incompatible in {path}"
        )
    if "response_human_discard_idx" in data and tuple(data["response_human_discard_idx"].shape[1:]) != (
        XMODEL1_MAX_RESPONSE_CANDIDATES,
    ):
        raise ValueError(
            f"response_human_discard_idx shape {data['response_human_discard_idx'].shape} incompatible in {path}"
        )
    if "rule_context" in data and tuple(data["rule_context"].shape[1:]) != (
        XMODEL1_RULE_CONTEXT_DIM,
    ):
        raise ValueError(
            f"rule_context shape {data['rule_context'].shape} incompatible in {path}"
        )


def load_export_manifest(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_export_manifest(path: Path, *, required_shards: List[str] | None = None) -> Dict:
    manifest = load_export_manifest(path)
    missing = [key for key in _MANIFEST_REQUIRED_KEYS if key not in manifest]
    if missing:
        raise ValueError(f"Xmodel1 export manifest missing keys {missing} in {path}")
    if manifest.get("schema_name") != XMODEL1_SCHEMA_NAME or int(manifest.get("schema_version", -1)) != XMODEL1_SCHEMA_VERSION:
        raise ValueError(
            f"Xmodel1 export manifest schema mismatch in {path}: "
            f"got {manifest.get('schema_name')}@{manifest.get('schema_version')}, "
            f"expected {XMODEL1_SCHEMA_NAME}@{XMODEL1_SCHEMA_VERSION}"
        )
    shard_file_counts = manifest.get("shard_file_counts")
    shard_sample_counts = manifest.get("shard_sample_counts")
    if not isinstance(shard_file_counts, dict) or not isinstance(shard_sample_counts, dict):
        raise ValueError(f"Xmodel1 export manifest shard counts must be dicts in {path}")
    missing_shards = [
        shard
        for shard in (required_shards or [])
        if shard not in shard_file_counts or shard not in shard_sample_counts
    ]
    if missing_shards:
        raise ValueError(f"Xmodel1 export manifest missing requested shards {missing_shards} in {path}")
    return manifest


def probe_cached_samples(
    file_paths: List[Path],
    *,
    max_files: int = 3,
    rows_per_file: int = 1,
) -> Dict[str, object]:
    selected = list(file_paths[: max(0, max_files)])
    if not selected:
        raise ValueError("no Xmodel1 cache files available for contract probe")
    rows_probed = 0
    for path in selected:
        with np.load(path, allow_pickle=False) as data:
            _validate_xmodel1_cache_contract(data, path)
            sample_count = int(data["state_tile_feat"].shape[0])
            if sample_count <= 0:
                raise ValueError(f"Xmodel1 cache has no samples in {path}")
            rows_probed += min(sample_count, max(1, rows_per_file))
            if int(data["opp_tenpai_target"].shape[1]) != 3:
                raise ValueError(f"opp_tenpai_target shape {data['opp_tenpai_target'].shape} incompatible in {path}")
            if int(data["history_summary"].shape[1]) != XMODEL1_HISTORY_SUMMARY_DIM:
                raise ValueError(f"history_summary shape {data['history_summary'].shape} incompatible in {path}")
            if len(data["pts_given_win_target"].shape) != 1:
                raise ValueError(f"pts_given_win_target shape {data['pts_given_win_target'].shape} incompatible in {path}")
            if len(data["pts_given_dealin_target"].shape) != 1:
                raise ValueError(f"pts_given_dealin_target shape {data['pts_given_dealin_target'].shape} incompatible in {path}")
            if len(data["final_rank_target"].shape) != 1:
                raise ValueError(f"final_rank_target shape {data['final_rank_target'].shape} incompatible in {path}")
            if len(data["final_score_delta_points_target"].shape) != 1:
                raise ValueError(
                    f"final_score_delta_points_target shape {data['final_score_delta_points_target'].shape} incompatible in {path}"
                )
    dims = infer_cached_dimensions(selected)
    return {
        "num_files": len(selected),
        "rows_probed": rows_probed,
        "dims": dims,
    }


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


def infer_cached_dimensions(file_paths: List[Path], *, strict: bool = False) -> Dict[str, int]:
    skipped_errors: list[str] = []
    for path in file_paths:
        try:
            with np.load(path, allow_pickle=False) as data:
                _validate_xmodel1_cache_contract(data, path)
                dims = {
                    "state_tile_channels": int(data["state_tile_feat"].shape[1]),
                    "state_scalar_dim": int(data["state_scalar"].shape[1]),
                    "candidate_feature_dim": int(data["candidate_feat"].shape[2]),
                    "candidate_flag_dim": int(data["candidate_flags"].shape[2]),
                    "max_candidates": int(data["candidate_mask"].shape[1]),
                }
                if skipped_errors:
                    print(
                        f"xmodel1 cache scan: skipped {len(skipped_errors)} unreadable cache file(s) "
                        f"before inferring dims from {path}",
                        file=sys.stderr,
                        flush=True,
                    )
                return dims
        except Exception as exc:
            skipped_errors.append(f"{path}: {exc}")
            print(
                f"xmodel1 cache scan: skipping unreadable cache {path}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            if strict:
                raise RuntimeError(
                    f"xmodel1 strict cache scan failed on {path}: {exc}"
                ) from exc
            continue
    if skipped_errors:
        raise FileNotFoundError(
            "no readable Xmodel1 cache files were found; skipped errors: "
            + " | ".join(skipped_errors)
        )
    raise FileNotFoundError("no readable Xmodel1 cache files were found")


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
            state_tile_feat = data["state_tile_feat"]
            state_scalar = data["state_scalar"]
            candidate_feat = data["candidate_feat"]
            candidate_tile_id = data["candidate_tile_id"]
            candidate_mask = data["candidate_mask"]
            candidate_flags = data["candidate_flags"]
            chosen_candidate_idx = data["chosen_candidate_idx"]
            sample_type = data["sample_type"]
            action_idx_target = data["action_idx_target"]
            sample_count = int(state_tile_feat.shape[0])
            candidate_quality_score = data["candidate_quality_score"]
            candidate_hard_bad_flag = data["candidate_hard_bad_flag"]
            response_action_idx = data["response_action_idx"]
            response_action_mask = data["response_action_mask"]
            chosen_response_action_idx = data["chosen_response_action_idx"]
            response_post_candidate_feat = data["response_post_candidate_feat"]
            response_post_candidate_tile_id = data["response_post_candidate_tile_id"]
            response_post_candidate_mask = data["response_post_candidate_mask"]
            response_post_candidate_flags = data["response_post_candidate_flags"]
            response_post_candidate_quality_score = data["response_post_candidate_quality_score"]
            response_post_candidate_hard_bad_flag = data["response_post_candidate_hard_bad_flag"]
            response_teacher_discard_idx = data["response_teacher_discard_idx"]
            response_human_discard_idx = (
                data["response_human_discard_idx"]
                if "response_human_discard_idx" in data
                else np.full((sample_count, XMODEL1_MAX_RESPONSE_CANDIDATES), -1, dtype=np.int16)
            )
            win_target = data["win_target"]
            dealin_target = data["dealin_target"]
            pts_given_win_target = data["pts_given_win_target"]
            pts_given_dealin_target = data["pts_given_dealin_target"]
            opp_tenpai_target = data["opp_tenpai_target"]
            final_rank_target = data["final_rank_target"]
            final_score_delta_points_target = data["final_score_delta_points_target"]
            history_summary = data["history_summary"]
            rule_context = (
                data["rule_context"]
                if "rule_context" in data
                else np.tile(_default_rule_context(), (sample_count, 1)).astype(np.float32)
            )
            event_index = data["event_index"]
            replay_id = data["replay_id"] if "replay_id" in data else None
            sample_id = data["sample_id"] if "sample_id" in data else None
            for i in range(sample_count):
                replay_id_value = str(replay_id[i]) if replay_id is not None else path.stem
                sample_id_value = (
                    str(sample_id[i])
                    if sample_id is not None
                    else f"{replay_id_value}:{int(event_index[i])}"
                )
                yield (
                    state_tile_feat[i],
                    state_scalar[i],
                    candidate_feat[i],
                    candidate_tile_id[i],
                    candidate_mask[i],
                    candidate_flags[i],
                    int(chosen_candidate_idx[i]),
                    int(sample_type[i]),
                    int(action_idx_target[i]),
                    candidate_quality_score[i],
                    candidate_hard_bad_flag[i],
                    response_action_idx[i],
                    response_action_mask[i],
                    int(chosen_response_action_idx[i]),
                    response_post_candidate_feat[i],
                    response_post_candidate_tile_id[i],
                    response_post_candidate_mask[i],
                    response_post_candidate_flags[i],
                    response_post_candidate_quality_score[i],
                    response_post_candidate_hard_bad_flag[i],
                    response_teacher_discard_idx[i],
                    response_human_discard_idx[i],
                    np.float32(win_target[i]),
                    np.float32(dealin_target[i]),
                    np.float32(pts_given_win_target[i]),
                    np.float32(pts_given_dealin_target[i]),
                    np.asarray(opp_tenpai_target[i], dtype=np.float32).reshape(3),
                    np.int8(final_rank_target[i]),
                    np.int32(final_score_delta_points_target[i]),
                    np.asarray(history_summary[i], dtype=np.float16).reshape(XMODEL1_HISTORY_SUMMARY_DIM),
                    np.asarray(rule_context[i], dtype=np.float32).reshape(XMODEL1_RULE_CONTEXT_DIM),
                    replay_id_value,
                    sample_id_value,
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
            candidate_hard_bad_flag,
            response_action_idx,
            response_action_mask,
            chosen_response_action_idx,
            response_post_candidate_feat,
            response_post_candidate_tile_id,
            response_post_candidate_mask,
            response_post_candidate_flags,
            response_post_candidate_quality_score,
            response_post_candidate_hard_bad_flag,
            response_teacher_discard_idx,
            response_human_discard_idx,
            win_target,
            dealin_target,
            pts_given_win_target,
            pts_given_dealin_target,
            opp_tenpai_target,
            final_rank_target,
            final_score_delta_points_target,
            history_summary,
            rule_context,
            replay_id,
            sample_id,
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
            "candidate_hard_bad_flag": torch.from_numpy(np.stack(candidate_hard_bad_flag)).float(),
            "response_action_idx": torch.from_numpy(np.stack(response_action_idx)).long(),
            "response_action_mask": torch.from_numpy(np.stack(response_action_mask)),
            "chosen_response_action_idx": torch.tensor(chosen_response_action_idx, dtype=torch.long),
            "response_post_candidate_feat": torch.from_numpy(np.stack(response_post_candidate_feat)),
            "response_post_candidate_tile_id": torch.from_numpy(np.stack(response_post_candidate_tile_id)).long(),
            "response_post_candidate_mask": torch.from_numpy(np.stack(response_post_candidate_mask)),
            "response_post_candidate_flags": torch.from_numpy(np.stack(response_post_candidate_flags)),
            "response_post_candidate_quality_score": torch.from_numpy(np.stack(response_post_candidate_quality_score)).float(),
            "response_post_candidate_hard_bad_flag": torch.from_numpy(np.stack(response_post_candidate_hard_bad_flag)).float(),
            "response_teacher_discard_idx": torch.from_numpy(np.stack(response_teacher_discard_idx)).long(),
            "response_human_discard_idx": torch.from_numpy(np.stack(response_human_discard_idx)).long(),
            "win_target": torch.tensor(win_target, dtype=torch.float32),
            "dealin_target": torch.tensor(dealin_target, dtype=torch.float32),
            "pts_given_win_target": torch.tensor(pts_given_win_target, dtype=torch.float32),
            "pts_given_dealin_target": torch.tensor(pts_given_dealin_target, dtype=torch.float32),
            "opp_tenpai_target": torch.from_numpy(np.stack(opp_tenpai_target)).float(),
            "final_rank_target": torch.tensor(final_rank_target, dtype=torch.long),
            "final_score_delta_points_target": torch.tensor(
                final_score_delta_points_target,
                dtype=torch.int32,
            ),
            "history_summary": torch.from_numpy(np.stack(history_summary)).float(),
            "rule_context": torch.from_numpy(np.stack(rule_context)).float(),
            "replay_id": list(replay_id),
            "sample_id": list(sample_id),
        }


__all__ = [
    "DEFAULT_BUFFER_SIZE",
    "_empty_history_summary",
    "Xmodel1DiscardDataset",
    "discover_cached_files",
    "find_export_manifests",
    "infer_cached_dimensions",
    "load_export_manifest",
    "probe_cached_samples",
    "split_cached_files",
    "summarize_cached_files",
    "validate_export_manifest",
]
