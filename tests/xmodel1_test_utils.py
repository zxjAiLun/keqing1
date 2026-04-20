from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_HISTORY_SUMMARY_DIM,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_MAX_SPECIAL_CANDIDATES,
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
    XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
)


def make_xmodel1_v3_payload(
    *,
    n: int,
    state_scalar_dim: int = 64,
    candidate_active: int = 3,
    sample_type: np.ndarray | None = None,
    chosen_candidate_idx: np.ndarray | None = None,
    chosen_special_candidate_idx: np.ndarray | None = None,
    replay_ids: list[str] | None = None,
    sample_ids: list[str] | None = None,
) -> dict[str, np.ndarray]:
    candidate_mask = np.zeros((n, XMODEL1_MAX_CANDIDATES), dtype=np.uint8)
    candidate_mask[:, : max(0, min(candidate_active, XMODEL1_MAX_CANDIDATES))] = 1
    candidate_tile_id = np.full((n, XMODEL1_MAX_CANDIDATES), -1, dtype=np.int16)
    for slot in range(min(candidate_active, XMODEL1_MAX_CANDIDATES)):
        candidate_tile_id[:, slot] = slot
    special_mask = np.zeros((n, XMODEL1_MAX_SPECIAL_CANDIDATES), dtype=np.uint8)
    special_type_id = np.full((n, XMODEL1_MAX_SPECIAL_CANDIDATES), -1, dtype=np.int16)
    payload: dict[str, np.ndarray] = {
        "schema_name": np.array(XMODEL1_SCHEMA_NAME, dtype=np.str_),
        "schema_version": np.array(XMODEL1_SCHEMA_VERSION, dtype=np.int32),
        "state_tile_feat": np.zeros((n, 57, 34), dtype=np.float16),
        "state_scalar": np.zeros((n, state_scalar_dim), dtype=np.float16),
        "candidate_feat": np.zeros(
            (n, XMODEL1_MAX_CANDIDATES, XMODEL1_CANDIDATE_FEATURE_DIM),
            dtype=np.float16,
        ),
        "candidate_tile_id": candidate_tile_id,
        "candidate_mask": candidate_mask,
        "candidate_flags": np.zeros(
            (n, XMODEL1_MAX_CANDIDATES, XMODEL1_CANDIDATE_FLAG_DIM),
            dtype=np.uint8,
        ),
        "chosen_candidate_idx": (
            np.zeros((n,), dtype=np.int16)
            if chosen_candidate_idx is None
            else np.asarray(chosen_candidate_idx, dtype=np.int16)
        ),
        "candidate_quality_score": np.zeros((n, XMODEL1_MAX_CANDIDATES), dtype=np.float32),
        "candidate_hard_bad_flag": np.zeros((n, XMODEL1_MAX_CANDIDATES), dtype=np.uint8),
        "special_candidate_feat": np.zeros(
            (n, XMODEL1_MAX_SPECIAL_CANDIDATES, XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM),
            dtype=np.float16,
        ),
        "special_candidate_type_id": special_type_id,
        "special_candidate_mask": special_mask,
        "special_candidate_quality_score": np.zeros((n, XMODEL1_MAX_SPECIAL_CANDIDATES), dtype=np.float32),
        "special_candidate_hard_bad_flag": np.zeros((n, XMODEL1_MAX_SPECIAL_CANDIDATES), dtype=np.uint8),
        "chosen_special_candidate_idx": (
            np.full((n,), -1, dtype=np.int16)
            if chosen_special_candidate_idx is None
            else np.asarray(chosen_special_candidate_idx, dtype=np.int16)
        ),
        "win_target": np.zeros((n,), dtype=np.float32),
        "dealin_target": np.zeros((n,), dtype=np.float32),
        "pts_given_win_target": np.zeros((n,), dtype=np.float32),
        "pts_given_dealin_target": np.zeros((n,), dtype=np.float32),
        "opp_tenpai_target": np.zeros((n, 3), dtype=np.float32),
        "history_summary": np.zeros((n, XMODEL1_HISTORY_SUMMARY_DIM), dtype=np.float16),
        "sample_type": (
            np.zeros((n,), dtype=np.int8)
            if sample_type is None
            else np.asarray(sample_type, dtype=np.int8)
        ),
        "action_idx_target": np.zeros((n,), dtype=np.int16),
        "actor": np.zeros((n,), dtype=np.int8),
        "event_index": np.arange(n, dtype=np.int32),
        "kyoku": np.ones((n,), dtype=np.int8),
        "honba": np.zeros((n,), dtype=np.int8),
        "is_open_hand": np.zeros((n,), dtype=np.uint8),
    }
    if replay_ids is not None:
        payload["replay_id"] = np.asarray(replay_ids, dtype=np.str_)
    if sample_ids is not None:
        payload["sample_id"] = np.asarray(sample_ids, dtype=np.str_)
    return payload


def write_xmodel1_v3_npz(path: Path, **kwargs) -> dict[str, np.ndarray]:
    payload = make_xmodel1_v3_payload(**kwargs)
    np.savez(path, **payload)
    return payload


def write_xmodel1_v3_manifest(
    root: Path,
    *,
    shard_file_counts: dict[str, int],
    shard_sample_counts: dict[str, int],
) -> None:
    (root / "xmodel1_export_manifest.json").write_text(
        json.dumps(
            {
                "schema_name": XMODEL1_SCHEMA_NAME,
                "schema_version": XMODEL1_SCHEMA_VERSION,
                "max_candidates": XMODEL1_MAX_CANDIDATES,
                "candidate_feature_dim": XMODEL1_CANDIDATE_FEATURE_DIM,
                "candidate_flag_dim": XMODEL1_CANDIDATE_FLAG_DIM,
                "file_count": sum(shard_file_counts.values()),
                "exported_file_count": sum(shard_file_counts.values()),
                "exported_sample_count": sum(shard_sample_counts.values()),
                "processed_file_count": sum(shard_file_counts.values()),
                "skipped_existing_file_count": 0,
                "shard_file_counts": shard_file_counts,
                "shard_sample_counts": shard_sample_counts,
                "used_fallback": False,
                "export_mode": "rust_full_npz_export",
                "files": [
                    f"{shard}/{idx}.mjson"
                    for shard, count in shard_file_counts.items()
                    for idx in range(count)
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
