"""Legacy Python parity oracle for Xmodel1 candidate arrays.

Production preprocess ownership belongs to Rust (`keqing_core` /
`scripts/preprocess_xmodel1.py`). This module is kept only so tests can compare
Rust export output against a Python reference implementation of the candidate
array schema.
"""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from mahjong_env.action_space import action_to_idx
from keqing_core import validate_xmodel1_discard_record, xmodel1_schema_info
from mahjong_env.replay import build_replay_samples_mc_return
from mahjong_env.replay_normalizer import normalize_replay_events
from mahjong_env.tiles import tile_to_34
from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_HISTORY_SUMMARY_DIM,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_MAX_SPECIAL_CANDIDATES,
    XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
)
from xmodel1.candidate_quality import (
    build_candidate_features,
    build_special_candidate_arrays,
    iter_legal_discards,
)
from xmodel1.features import compute_history_summary, encode
from xmodel1.schema import (
    XMODEL1_SAMPLE_TYPE_CALL,
    XMODEL1_SAMPLE_TYPE_DISCARD,
    XMODEL1_SAMPLE_TYPE_HORA,
    XMODEL1_SAMPLE_TYPE_RIICHI,
)


def events_to_xmodel1_arrays(events, *, replay_id: str) -> Optional[Dict[str, np.ndarray]]:
    samples = build_replay_samples_mc_return(events, strict_legal_labels=False)
    rows: dict[str, list] = {
        "state_tile_feat": [],
        "state_scalar": [],
        "candidate_feat": [],
        "candidate_tile_id": [],
        "candidate_mask": [],
        "candidate_flags": [],
        "chosen_candidate_idx": [],
        "candidate_quality_score": [],
        "candidate_hard_bad_flag": [],
        "special_candidate_feat": [],
        "special_candidate_type_id": [],
        "special_candidate_mask": [],
        "special_candidate_quality_score": [],
        "special_candidate_hard_bad_flag": [],
        "chosen_special_candidate_idx": [],
        "win_target": [],
        "dealin_target": [],
        "pts_given_win_target": [],
        "pts_given_dealin_target": [],
        # Stage 2 Python 原型:3 家对手的 tenpai 标签 (1.0 表示该对手当前向听 ≤ 0)。
        # 相对 actor 顺序为 (actor+1,actor+2,actor+3)%4。Rust preprocess 迁移后保持同样含义。
        "opp_tenpai_target": [],
        "history_summary": [],
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

    normalized_events = normalize_replay_events(events)

    for sample_index, sample in enumerate(samples):
        event_index = int(getattr(sample, "event_index", sample_index))
        tile_feat, scalar = encode(sample.state, sample.actor, state_scalar_dim=56)
        label_type = sample.label_action.get("type")

        candidate_feat = np.zeros((XMODEL1_MAX_CANDIDATES, XMODEL1_CANDIDATE_FEATURE_DIM), dtype=np.float16)
        candidate_tile_id = np.full((XMODEL1_MAX_CANDIDATES,), -1, dtype=np.int16)
        candidate_mask = np.zeros((XMODEL1_MAX_CANDIDATES,), dtype=np.uint8)
        candidate_flags = np.zeros((XMODEL1_MAX_CANDIDATES, XMODEL1_CANDIDATE_FLAG_DIM), dtype=np.uint8)
        candidate_quality = np.zeros((XMODEL1_MAX_CANDIDATES,), dtype=np.float32)
        candidate_hard_bad = np.zeros((XMODEL1_MAX_CANDIDATES,), dtype=np.uint8)
        (
            special_candidate_feat,
            special_candidate_type_id,
            special_candidate_mask,
            special_candidate_quality,
            _special_candidate_rank,
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
                feat, flags, quality, _rank_bucket, hard_bad = build_candidate_features(
                    sample.state,
                    sample.actor,
                    action,
                )
                candidate_feat[cidx] = feat.astype(np.float16)
                candidate_tile_id[cidx] = int(tile_to_34(action["pai"]))
                candidate_mask[cidx] = 1
                candidate_flags[cidx] = flags
                candidate_quality[cidx] = quality
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
        elif label_type == "hora":
            sample_type = XMODEL1_SAMPLE_TYPE_HORA
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
        rows["candidate_hard_bad_flag"].append(candidate_hard_bad)
        rows["special_candidate_feat"].append(special_candidate_feat.astype(np.float16))
        rows["special_candidate_type_id"].append(special_candidate_type_id)
        rows["special_candidate_mask"].append(special_candidate_mask)
        rows["special_candidate_quality_score"].append(special_candidate_quality)
        rows["special_candidate_hard_bad_flag"].append(special_candidate_hard_bad)
        rows["chosen_special_candidate_idx"].append(np.int16(chosen_special_idx))
        rows["win_target"].append(np.float32(sample.win_target))
        rows["dealin_target"].append(np.float32(sample.dealin_target))
        rows["pts_given_win_target"].append(np.float32(sample.pts_given_win_target))
        rows["pts_given_dealin_target"].append(np.float32(sample.pts_given_dealin_target))
        opp_tenpai = getattr(sample, "opp_tenpai_target", (0.0, 0.0, 0.0))
        rows["opp_tenpai_target"].append(
            np.asarray(opp_tenpai, dtype=np.float32).reshape(3)
        )
        rows["history_summary"].append(
            compute_history_summary(normalized_events, event_index, sample.actor)
        )
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
        "candidate_hard_bad_flag",
        "special_candidate_feat",
        "special_candidate_type_id",
        "special_candidate_mask",
        "special_candidate_quality_score",
        "special_candidate_hard_bad_flag",
        "opp_tenpai_target",
        "history_summary",
    }
    out: Dict[str, np.ndarray] = {}
    for key, value in rows.items():
        out[key] = np.stack(value) if key in stacked_keys else np.array(value)
    out["schema_name"] = np.array(schema_name, dtype=np.str_)
    out["schema_version"] = np.array(schema_version, dtype=np.int32)
    out["max_candidates"] = np.array(max_candidates, dtype=np.int32)
    out["candidate_dim"] = np.array(candidate_dim, dtype=np.int32)
    out["candidate_flag_dim"] = np.array(flag_dim, dtype=np.int32)
    if out["history_summary"].shape[1] != XMODEL1_HISTORY_SUMMARY_DIM:
        raise RuntimeError(
            f"xmodel1 parity oracle history_summary drift: {out['history_summary'].shape}"
        )
    return out


__all__ = ["events_to_xmodel1_arrays"]
