"""Python fallback preprocess for Xmodel1.

Long-term ownership belongs to Rust (`keqing_core`). This module exists so the
candidate-centric schema can be exercised end-to-end before the Rust export path
is fully implemented.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from keqing_core import validate_xmodel1_discard_record, xmodel1_schema_info
from mahjong_env.replay import build_supervised_samples, read_mjai_jsonl
from mahjong_env.tiles import tile_to_34
from training.cache_schema import XMODEL1_MAX_CANDIDATES
from xmodel1.candidate_quality import build_candidate_features, iter_legal_discards
from xmodel1.features import encode
from xmodel1.schema import XMODEL1_SAMPLE_TYPE_DISCARD


def events_to_xmodel1_arrays(events, *, replay_id: str) -> Optional[Dict[str, np.ndarray]]:
    samples = build_supervised_samples(events, value_strategy="mc_return", strict_legal_labels=True)
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
        "global_value_target": [],
        "score_delta_target": [],
        "win_target": [],
        "dealin_target": [],
        "offense_quality_target": [],
        "sample_type": [],
        "actor": [],
        "event_index": [],
        "kyoku": [],
        "honba": [],
        "is_open_hand": [],
        "replay_id": [],
        "sample_id": [],
    }

    for event_index, sample in enumerate(samples):
        if sample.label_action.get("type") != "dahai":
            continue
        legal_discards = iter_legal_discards(sample.legal_actions)
        if not legal_discards:
            continue
        tile_feat, scalar = encode(sample.state, sample.actor, state_scalar_dim=56)

        candidate_feat = np.zeros((XMODEL1_MAX_CANDIDATES, 21), dtype=np.float16)
        candidate_tile_id = np.full((XMODEL1_MAX_CANDIDATES,), -1, dtype=np.int16)
        candidate_mask = np.zeros((XMODEL1_MAX_CANDIDATES,), dtype=np.uint8)
        candidate_flags = np.zeros((XMODEL1_MAX_CANDIDATES, 10), dtype=np.uint8)
        candidate_quality = np.zeros((XMODEL1_MAX_CANDIDATES,), dtype=np.float32)
        candidate_rank = np.zeros((XMODEL1_MAX_CANDIDATES,), dtype=np.int8)
        candidate_hard_bad = np.zeros((XMODEL1_MAX_CANDIDATES,), dtype=np.uint8)

        chosen_idx = None
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

        rows["state_tile_feat"].append(tile_feat.astype(np.float16))
        rows["state_scalar"].append(scalar.astype(np.float16))
        rows["candidate_feat"].append(candidate_feat)
        rows["candidate_tile_id"].append(candidate_tile_id)
        rows["candidate_mask"].append(candidate_mask)
        rows["candidate_flags"].append(candidate_flags)
        rows["chosen_candidate_idx"].append(np.int16(chosen_idx))
        rows["candidate_quality_score"].append(candidate_quality)
        rows["candidate_rank_bucket"].append(candidate_rank)
        rows["candidate_hard_bad_flag"].append(candidate_hard_bad)
        rows["global_value_target"].append(np.float32(sample.value_target))
        rows["score_delta_target"].append(np.float32(sample.score_delta_target))
        rows["win_target"].append(np.float32(sample.win_target))
        rows["dealin_target"].append(np.float32(sample.dealin_target))
        rows["offense_quality_target"].append(np.float32(candidate_quality[chosen_idx]))
        rows["sample_type"].append(np.int8(XMODEL1_SAMPLE_TYPE_DISCARD))
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
    }
    out: Dict[str, np.ndarray] = {}
    for key, value in rows.items():
        out[key] = np.stack(value) if key in stacked_keys else np.array(value)
    out["schema_name"] = np.array(schema_name, dtype=object)
    out["schema_version"] = np.array(schema_version, dtype=np.int32)
    out["max_candidates"] = np.array(max_candidates, dtype=np.int32)
    out["candidate_dim"] = np.array(candidate_dim, dtype=np.int32)
    out["candidate_flag_dim"] = np.array(flag_dim, dtype=np.int32)
    return out


def preprocess_files(*, data_dirs: List[Path], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for data_dir in data_dirs:
        out_ds = output_dir / data_dir.name
        out_ds.mkdir(parents=True, exist_ok=True)
        for replay_path in sorted(data_dir.glob("*.mjson")):
            out_path = out_ds / f"{replay_path.stem}.npz"
            if out_path.exists():
                continue
            events = read_mjai_jsonl(replay_path)
            arrays = events_to_xmodel1_arrays(events, replay_id=str(replay_path))
            if arrays is None:
                continue
            np.savez(out_path, **arrays)


__all__ = ["events_to_xmodel1_arrays", "preprocess_files"]
