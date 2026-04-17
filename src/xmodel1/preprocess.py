"""Python fallback preprocess for Xmodel1.

Long-term ownership belongs to Rust (`keqing_core`). This module exists so the
candidate-centric schema can be exercised end-to-end before the Rust export path
is fully implemented.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from typing import Dict, List, Optional

import numpy as np

from keqingv1.action_space import action_to_idx
from keqing_core import validate_xmodel1_discard_record, xmodel1_schema_info
from mahjong_env.replay import build_supervised_samples, read_mjai_jsonl
from mahjong_env.tiles import tile_to_34
from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_MAX_SPECIAL_CANDIDATES,
    XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
)
from xmodel1.candidate_quality import (
    build_candidate_features,
    build_special_candidate_arrays,
    iter_legal_discards,
)
from xmodel1.features import compute_event_history, encode
from xmodel1.schema import (
    XMODEL1_SAMPLE_TYPE_CALL,
    XMODEL1_SAMPLE_TYPE_DISCARD,
    XMODEL1_SAMPLE_TYPE_RIICHI,
)


def _write_python_export_manifest(
    output_dir: Path,
    *,
    files: List[str],
    exported_file_count: int,
    exported_sample_count: int,
    shard_file_counts: Dict[str, int],
    shard_sample_counts: Dict[str, int],
) -> None:
    schema_name, schema_version, max_candidates, candidate_dim, flag_dim = xmodel1_schema_info()
    manifest: Dict[str, Any] = {
        "schema_name": schema_name,
        "schema_version": int(schema_version),
        "max_candidates": int(max_candidates),
        "candidate_feature_dim": int(candidate_dim),
        "candidate_flag_dim": int(flag_dim),
        "file_count": len(files),
        "exported_file_count": int(exported_file_count),
        "exported_sample_count": int(exported_sample_count),
        "processed_file_count": len(files),
        "skipped_existing_file_count": 0,
        "shard_file_counts": shard_file_counts,
        "shard_sample_counts": shard_sample_counts,
        "used_fallback": True,
        "export_mode": "python_full_export",
        "files": files,
    }
    manifest_path = output_dir / "xmodel1_export_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def events_to_xmodel1_arrays(events, *, replay_id: str) -> Optional[Dict[str, np.ndarray]]:
    samples = build_supervised_samples(events, value_strategy="mc_return", strict_legal_labels=False)
    rows: dict[str, list] = {
        "state_tile_feat": [],
        "state_scalar": [],
        "candidate_feat": [],
        "candidate_tile_id": [],
        "candidate_mask": [],
        "candidate_flags": [],
        "chosen_candidate_idx": [],
        "candidate_quality_score": [],
        "candidate_rank_bucket": [],
        "candidate_hard_bad_flag": [],
        "special_candidate_feat": [],
        "special_candidate_type_id": [],
        "special_candidate_mask": [],
        "special_candidate_quality_score": [],
        "special_candidate_rank_bucket": [],
        "special_candidate_hard_bad_flag": [],
        "chosen_special_candidate_idx": [],
        "score_delta_target": [],
        "win_target": [],
        "dealin_target": [],
        "pts_given_win_target": [],
        "pts_given_dealin_target": [],
        # Stage 2 Python 原型:3 家对手的 tenpai 标签 (1.0 表示该对手当前向听 ≤ 0)。
        # 相对 actor 顺序为 (actor+1,actor+2,actor+3)%4。Rust preprocess 迁移后保持同样含义。
        "opp_tenpai_target": [],
        "event_history": [],
        "sample_type": [],
        "action_idx_target": [],
        "actor": [],
        "event_index": [],
        "kyoku": [],
        "honba": [],
        "is_open_hand": [],
        "replay_id": [],
        "sample_id": [],
    }

    for sample_index, sample in enumerate(samples):
        event_index = int(getattr(sample, "event_index", sample_index))
        tile_feat, scalar = encode(sample.state, sample.actor, state_scalar_dim=56)
        label_type = sample.label_action.get("type")

        candidate_feat = np.zeros((XMODEL1_MAX_CANDIDATES, XMODEL1_CANDIDATE_FEATURE_DIM), dtype=np.float16)
        candidate_tile_id = np.full((XMODEL1_MAX_CANDIDATES,), -1, dtype=np.int16)
        candidate_mask = np.zeros((XMODEL1_MAX_CANDIDATES,), dtype=np.uint8)
        candidate_flags = np.zeros((XMODEL1_MAX_CANDIDATES, 10), dtype=np.uint8)
        candidate_quality = np.zeros((XMODEL1_MAX_CANDIDATES,), dtype=np.float32)
        candidate_rank = np.zeros((XMODEL1_MAX_CANDIDATES,), dtype=np.int8)
        candidate_hard_bad = np.zeros((XMODEL1_MAX_CANDIDATES,), dtype=np.uint8)
        (
            special_candidate_feat,
            special_candidate_type_id,
            special_candidate_mask,
            special_candidate_quality,
            special_candidate_rank,
            special_candidate_hard_bad,
            chosen_special_idx,
        ) = build_special_candidate_arrays(
            sample.state,
            sample.actor,
            sample.legal_actions,
            sample.label_action,
            max_candidates=XMODEL1_MAX_SPECIAL_CANDIDATES,
            feature_dim=XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
            include_terminal_actions=True,
        )

        chosen_idx = None
        action_idx_target = int(action_to_idx(sample.label_action))
        sample_type = XMODEL1_SAMPLE_TYPE_DISCARD

        if label_type == "dahai":
            legal_discards = iter_legal_discards(sample.legal_actions)
            if not legal_discards:
                continue
            for cidx, action in enumerate(legal_discards[:XMODEL1_MAX_CANDIDATES]):
                feat, flags, quality, rank_bucket, hard_bad = build_candidate_features(sample.state, sample.actor, action)
                candidate_feat[cidx] = feat.astype(np.float16)
                candidate_tile_id[cidx] = int(tile_to_34(action["pai"]))
                candidate_mask[cidx] = 1
                candidate_flags[cidx] = flags
                candidate_quality[cidx] = quality
                candidate_rank[cidx] = rank_bucket
                candidate_hard_bad[cidx] = hard_bad
                if action == sample.label_action:
                    chosen_idx = cidx
            if chosen_idx is None:
                continue
            validate_xmodel1_discard_record(chosen_idx, candidate_mask.tolist(), candidate_tile_id.tolist())
        elif label_type == "reach":
            sample_type = XMODEL1_SAMPLE_TYPE_RIICHI
            chosen_idx = -1
            if chosen_special_idx < 0:
                continue
        elif label_type in {"chi", "pon", "daiminkan", "ankan", "kakan", "none"}:
            sample_type = XMODEL1_SAMPLE_TYPE_CALL
            chosen_idx = -1
            if chosen_special_idx < 0:
                continue
        else:
            continue

        rows["state_tile_feat"].append(tile_feat.astype(np.float16))
        rows["state_scalar"].append(scalar.astype(np.float16))
        rows["candidate_feat"].append(candidate_feat)
        rows["candidate_tile_id"].append(candidate_tile_id)
        rows["candidate_mask"].append(candidate_mask)
        rows["candidate_flags"].append(candidate_flags)
        rows["chosen_candidate_idx"].append(np.int16(chosen_idx if chosen_idx is not None else -1))
        rows["candidate_quality_score"].append(candidate_quality)
        rows["candidate_rank_bucket"].append(candidate_rank)
        rows["candidate_hard_bad_flag"].append(candidate_hard_bad)
        rows["special_candidate_feat"].append(special_candidate_feat.astype(np.float16))
        rows["special_candidate_type_id"].append(special_candidate_type_id)
        rows["special_candidate_mask"].append(special_candidate_mask)
        rows["special_candidate_quality_score"].append(special_candidate_quality)
        rows["special_candidate_rank_bucket"].append(special_candidate_rank)
        rows["special_candidate_hard_bad_flag"].append(special_candidate_hard_bad)
        rows["chosen_special_candidate_idx"].append(np.int16(chosen_special_idx))
        rows["score_delta_target"].append(np.float32(sample.score_delta_target))
        rows["win_target"].append(np.float32(sample.win_target))
        rows["dealin_target"].append(np.float32(sample.dealin_target))
        rows["pts_given_win_target"].append(np.float32(sample.pts_given_win_target))
        rows["pts_given_dealin_target"].append(np.float32(sample.pts_given_dealin_target))
        opp_tenpai = getattr(sample, "opp_tenpai_target", (0.0, 0.0, 0.0))
        rows["opp_tenpai_target"].append(
            np.asarray(opp_tenpai, dtype=np.float32).reshape(3)
        )
        rows["event_history"].append(compute_event_history(events, event_index))
        rows["sample_type"].append(np.int8(sample_type))
        rows["action_idx_target"].append(np.int16(action_idx_target))
        rows["actor"].append(np.int8(sample.actor))
        rows["event_index"].append(np.int32(event_index))
        rows["kyoku"].append(np.int8(sample.state.get("kyoku", 1)))
        rows["honba"].append(np.int8(sample.state.get("honba", 0)))
        rows["is_open_hand"].append(np.uint8(len((sample.state.get("melds") or [[], [], [], []])[sample.actor]) > 0))
        rows["replay_id"].append(replay_id)
        rows["sample_id"].append(f"{replay_id}:{event_index}")

    if not rows["state_tile_feat"]:
        return None

    schema_name, schema_version, max_candidates, candidate_dim, flag_dim = xmodel1_schema_info()
    stacked_keys = {
        "state_tile_feat",
        "state_scalar",
        "candidate_feat",
        "candidate_tile_id",
        "candidate_mask",
        "candidate_flags",
        "candidate_quality_score",
        "candidate_rank_bucket",
        "candidate_hard_bad_flag",
        "special_candidate_feat",
        "special_candidate_type_id",
        "special_candidate_mask",
        "special_candidate_quality_score",
        "special_candidate_rank_bucket",
        "special_candidate_hard_bad_flag",
        "opp_tenpai_target",
        "event_history",
    }
    out: Dict[str, np.ndarray] = {}
    for key, value in rows.items():
        out[key] = np.stack(value) if key in stacked_keys else np.array(value)
    out["schema_name"] = np.array(schema_name, dtype=np.str_)
    out["schema_version"] = np.array(schema_version, dtype=np.int32)
    out["max_candidates"] = np.array(max_candidates, dtype=np.int32)
    out["candidate_dim"] = np.array(candidate_dim, dtype=np.int32)
    out["candidate_flag_dim"] = np.array(flag_dim, dtype=np.int32)
    return out


def preprocess_files(
    *,
    data_dirs: List[Path],
    output_dir: Path,
    overwrite: bool = False,
    limit_files: int | None = None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_files: List[str] = []
    exported_file_count = 0
    exported_sample_count = 0
    shard_file_counts: Dict[str, int] = {}
    shard_sample_counts: Dict[str, int] = {}
    remaining = None if limit_files is None or int(limit_files) <= 0 else int(limit_files)
    for data_dir in data_dirs:
        out_ds = output_dir / data_dir.name
        out_ds.mkdir(parents=True, exist_ok=True)
        for replay_path in sorted(data_dir.glob("*.mjson")):
            if remaining is not None and remaining <= 0:
                break
            processed_files.append(str(replay_path))
            out_path = out_ds / f"{replay_path.stem}.npz"
            if out_path.exists() and not overwrite:
                if remaining is not None:
                    remaining -= 1
                continue
            events = read_mjai_jsonl(replay_path)
            arrays = events_to_xmodel1_arrays(events, replay_id=str(replay_path))
            if arrays is None:
                if remaining is not None:
                    remaining -= 1
                continue
            np.savez(out_path, **arrays)
            exported_file_count += 1
            sample_count = int(arrays["state_tile_feat"].shape[0])
            exported_sample_count += sample_count
            shard = data_dir.name
            shard_file_counts[shard] = shard_file_counts.get(shard, 0) + 1
            shard_sample_counts[shard] = shard_sample_counts.get(shard, 0) + sample_count
            if remaining is not None:
                remaining -= 1
        if remaining is not None and remaining <= 0:
            break
    _write_python_export_manifest(
        output_dir,
        files=processed_files,
        exported_file_count=exported_file_count,
        exported_sample_count=exported_sample_count,
        shard_file_counts=shard_file_counts,
        shard_sample_counts=shard_sample_counts,
    )


__all__ = ["events_to_xmodel1_arrays", "preprocess_files"]
