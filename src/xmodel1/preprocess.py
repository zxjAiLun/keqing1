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
from mahjong_env.replay_normalizer import normalize_replay_events, replay_label_matches_legal
from mahjong_env.tiles import tile_to_34
from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_HISTORY_SUMMARY_DIM,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_MAX_RESPONSE_CANDIDATES,
    XMODEL1_RULE_CONTEXT_DIM,
)
from xmodel1.call_response import build_response_action_states
from xmodel1.candidate_quality import (
    build_candidate_features,
    iter_legal_discards,
)
from xmodel1.features import compute_history_summary, default_rule_context, encode
from xmodel1.schema import (
    XMODEL1_SAMPLE_TYPE_CALL,
    XMODEL1_SAMPLE_TYPE_DISCARD,
    XMODEL1_SAMPLE_TYPE_HORA,
    XMODEL1_SAMPLE_TYPE_RIICHI,
)


def _resolve_human_response_discard_idx(sample, response_states, normalized_events) -> np.ndarray:
    human_idx = np.full((XMODEL1_MAX_RESPONSE_CANDIDATES,), -1, dtype=np.int16)
    label_type = sample.label_action.get("type")
    if label_type not in {"reach", "chi", "pon", "daiminkan"}:
        return human_idx
    chosen_slot = -1
    for response_idx, response_state in enumerate(response_states[:XMODEL1_MAX_RESPONSE_CANDIDATES]):
        if replay_label_matches_legal(sample.label_action, [response_state.action]):
            chosen_slot = response_idx
            break
    if chosen_slot < 0:
        return human_idx
    for event in normalized_events[int(getattr(sample, "event_index", 0)) + 1 :]:
        event_type = event.get("type")
        if event_type in {"reach_accepted", "dora", "new_dora", "tsumo"}:
            continue
        if event_type != "dahai" or int(event.get("actor", -1)) != int(sample.actor):
            return human_idx
        actions = response_states[chosen_slot].post_discard_actions[:XMODEL1_MAX_CANDIDATES]
        for discard_idx, action in enumerate(actions):
            if replay_label_matches_legal(event, [action]):
                human_idx[chosen_slot] = np.int16(discard_idx)
                return human_idx
        return human_idx
    return human_idx


def _build_response_semantics(sample, normalized_events) -> dict[str, np.ndarray]:
    response_action_idx = np.full((XMODEL1_MAX_RESPONSE_CANDIDATES,), -1, dtype=np.int16)
    response_action_mask = np.zeros((XMODEL1_MAX_RESPONSE_CANDIDATES,), dtype=np.uint8)
    chosen_response_action_idx = np.int16(-1)
    response_post_candidate_feat = np.zeros(
        (
            XMODEL1_MAX_RESPONSE_CANDIDATES,
            XMODEL1_MAX_CANDIDATES,
            XMODEL1_CANDIDATE_FEATURE_DIM,
        ),
        dtype=np.float16,
    )
    response_post_candidate_tile_id = np.full(
        (XMODEL1_MAX_RESPONSE_CANDIDATES, XMODEL1_MAX_CANDIDATES),
        -1,
        dtype=np.int16,
    )
    response_post_candidate_mask = np.zeros(
        (XMODEL1_MAX_RESPONSE_CANDIDATES, XMODEL1_MAX_CANDIDATES),
        dtype=np.uint8,
    )
    response_post_candidate_flags = np.zeros(
        (
            XMODEL1_MAX_RESPONSE_CANDIDATES,
            XMODEL1_MAX_CANDIDATES,
            XMODEL1_CANDIDATE_FLAG_DIM,
        ),
        dtype=np.uint8,
    )
    response_post_candidate_quality_score = np.zeros(
        (XMODEL1_MAX_RESPONSE_CANDIDATES, XMODEL1_MAX_CANDIDATES),
        dtype=np.float32,
    )
    response_post_candidate_hard_bad_flag = np.zeros(
        (XMODEL1_MAX_RESPONSE_CANDIDATES, XMODEL1_MAX_CANDIDATES),
        dtype=np.uint8,
    )
    response_teacher_discard_idx = np.full(
        (XMODEL1_MAX_RESPONSE_CANDIDATES,),
        -1,
        dtype=np.int16,
    )

    response_states = build_response_action_states(
        sample.state,
        sample.actor,
        sample.legal_actions,
    )
    for response_idx, response_state in enumerate(
        response_states[:XMODEL1_MAX_RESPONSE_CANDIDATES]
    ):
        response_action_idx[response_idx] = np.int16(response_state.action_idx)
        response_action_mask[response_idx] = 1
        if replay_label_matches_legal(sample.label_action, [response_state.action]):
            chosen_response_action_idx = np.int16(response_idx)
        if response_state.after_snapshot is None:
            continue
        best_teacher_idx = -1
        best_teacher_quality = -1e9
        for discard_idx, action in enumerate(
            response_state.post_discard_actions[:XMODEL1_MAX_CANDIDATES]
        ):
            feat, flags, quality, _rank_bucket, hard_bad = build_candidate_features(
                response_state.after_snapshot,
                sample.actor,
                action,
            )
            response_post_candidate_feat[response_idx, discard_idx] = feat.astype(
                np.float16
            )
            response_post_candidate_tile_id[response_idx, discard_idx] = int(
                tile_to_34(action["pai"])
            )
            response_post_candidate_mask[response_idx, discard_idx] = 1
            response_post_candidate_flags[response_idx, discard_idx] = flags
            response_post_candidate_quality_score[response_idx, discard_idx] = quality
            response_post_candidate_hard_bad_flag[response_idx, discard_idx] = hard_bad
            if quality > best_teacher_quality:
                best_teacher_quality = float(quality)
                best_teacher_idx = discard_idx
        response_teacher_discard_idx[response_idx] = np.int16(best_teacher_idx)
    response_human_discard_idx = _resolve_human_response_discard_idx(
        sample,
        response_states,
        normalized_events,
    )

    return {
        "response_action_idx": response_action_idx,
        "response_action_mask": response_action_mask,
        "chosen_response_action_idx": np.array(chosen_response_action_idx, dtype=np.int16),
        "response_post_candidate_feat": response_post_candidate_feat,
        "response_post_candidate_tile_id": response_post_candidate_tile_id,
        "response_post_candidate_mask": response_post_candidate_mask,
        "response_post_candidate_flags": response_post_candidate_flags,
        "response_post_candidate_quality_score": response_post_candidate_quality_score,
        "response_post_candidate_hard_bad_flag": response_post_candidate_hard_bad_flag,
        "response_teacher_discard_idx": response_teacher_discard_idx,
        "response_human_discard_idx": response_human_discard_idx,
    }


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
        "response_action_idx": [],
        "response_action_mask": [],
        "chosen_response_action_idx": [],
        "response_post_candidate_feat": [],
        "response_post_candidate_tile_id": [],
        "response_post_candidate_mask": [],
        "response_post_candidate_flags": [],
        "response_post_candidate_quality_score": [],
        "response_post_candidate_hard_bad_flag": [],
        "response_teacher_discard_idx": [],
        "response_human_discard_idx": [],
        "win_target": [],
        "dealin_target": [],
        "pts_given_win_target": [],
        "pts_given_dealin_target": [],
        # Stage 2 Python 原型:3 家对手的 tenpai 标签 (1.0 表示该对手当前向听 ≤ 0)。
        # 相对 actor 顺序为 (actor+1,actor+2,actor+3)%4。Rust preprocess 迁移后保持同样含义。
        "opp_tenpai_target": [],
        "final_rank_target": [],
        "final_score_delta_points_target": [],
        "history_summary": [],
        "rule_context": [],
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
        legal_discards = iter_legal_discards(sample.legal_actions)

        candidate_feat = np.zeros((XMODEL1_MAX_CANDIDATES, XMODEL1_CANDIDATE_FEATURE_DIM), dtype=np.float16)
        candidate_tile_id = np.full((XMODEL1_MAX_CANDIDATES,), -1, dtype=np.int16)
        candidate_mask = np.zeros((XMODEL1_MAX_CANDIDATES,), dtype=np.uint8)
        candidate_flags = np.zeros((XMODEL1_MAX_CANDIDATES, XMODEL1_CANDIDATE_FLAG_DIM), dtype=np.uint8)
        candidate_quality = np.zeros((XMODEL1_MAX_CANDIDATES,), dtype=np.float32)
        candidate_hard_bad = np.zeros((XMODEL1_MAX_CANDIDATES,), dtype=np.uint8)
        response_semantics = _build_response_semantics(sample, normalized_events)

        chosen_idx = None
        action_idx_target = int(action_to_idx(sample.label_action))
        sample_type = XMODEL1_SAMPLE_TYPE_DISCARD

        if legal_discards:
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
                if label_type == "dahai" and action == sample.label_action:
                    chosen_idx = cidx
        if label_type == "dahai":
            if not legal_discards:
                continue
            if chosen_idx is None:
                continue
            validate_xmodel1_discard_record(chosen_idx, candidate_mask.tolist(), candidate_tile_id.tolist())
        elif label_type == "reach":
            sample_type = XMODEL1_SAMPLE_TYPE_RIICHI
            chosen_idx = -1
            if int(response_semantics["chosen_response_action_idx"]) < 0:
                continue
        elif label_type == "hora":
            sample_type = XMODEL1_SAMPLE_TYPE_HORA
            chosen_idx = -1
            if int(response_semantics["chosen_response_action_idx"]) < 0:
                continue
        elif label_type in {"chi", "pon", "daiminkan", "ankan", "kakan", "none"}:
            sample_type = XMODEL1_SAMPLE_TYPE_CALL
            chosen_idx = -1
            if int(response_semantics["chosen_response_action_idx"]) < 0:
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
        rows["response_action_idx"].append(response_semantics["response_action_idx"])
        rows["response_action_mask"].append(response_semantics["response_action_mask"])
        rows["chosen_response_action_idx"].append(
            response_semantics["chosen_response_action_idx"]
        )
        rows["response_post_candidate_feat"].append(
            response_semantics["response_post_candidate_feat"]
        )
        rows["response_post_candidate_tile_id"].append(
            response_semantics["response_post_candidate_tile_id"]
        )
        rows["response_post_candidate_mask"].append(
            response_semantics["response_post_candidate_mask"]
        )
        rows["response_post_candidate_flags"].append(
            response_semantics["response_post_candidate_flags"]
        )
        rows["response_post_candidate_quality_score"].append(
            response_semantics["response_post_candidate_quality_score"]
        )
        rows["response_post_candidate_hard_bad_flag"].append(
            response_semantics["response_post_candidate_hard_bad_flag"]
        )
        rows["response_teacher_discard_idx"].append(
            response_semantics["response_teacher_discard_idx"]
        )
        rows["response_human_discard_idx"].append(
            response_semantics["response_human_discard_idx"]
        )
        rows["win_target"].append(np.float32(sample.win_target))
        rows["dealin_target"].append(np.float32(sample.dealin_target))
        rows["pts_given_win_target"].append(np.float32(sample.pts_given_win_target))
        rows["pts_given_dealin_target"].append(np.float32(sample.pts_given_dealin_target))
        opp_tenpai = getattr(sample, "opp_tenpai_target", (0.0, 0.0, 0.0))
        rows["opp_tenpai_target"].append(
            np.asarray(opp_tenpai, dtype=np.float32).reshape(3)
        )
        rows["final_rank_target"].append(
            np.int8(getattr(sample, "final_rank_target", 0))
        )
        rows["final_score_delta_points_target"].append(
            np.int32(getattr(sample, "final_score_delta_points_target", 0))
        )
        rows["history_summary"].append(
            compute_history_summary(normalized_events, event_index, sample.actor)
        )
        rows["rule_context"].append(default_rule_context().reshape(XMODEL1_RULE_CONTEXT_DIM))
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
        "response_action_idx",
        "response_action_mask",
        "response_post_candidate_feat",
        "response_post_candidate_tile_id",
        "response_post_candidate_mask",
        "response_post_candidate_flags",
        "response_post_candidate_quality_score",
        "response_post_candidate_hard_bad_flag",
        "response_teacher_discard_idx",
        "response_human_discard_idx",
        "opp_tenpai_target",
        "history_summary",
        "rule_context",
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
