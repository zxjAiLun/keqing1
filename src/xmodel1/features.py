"""Xmodel1 runtime/state feature helpers."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from keqing_core import (
    build_xmodel1_runtime_tensors as _build_xmodel1_runtime_tensors,
    is_missing_rust_capability_error,
)
from mahjong_env.history_summary import (
    HISTORY_SUMMARY_DIM,
    compute_history_summary,
    empty_history_summary,
)
from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.tiles import tile_to_34
from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_MAX_RESPONSE_CANDIDATES,
    XMODEL1_MAX_SPECIAL_CANDIDATES,
    XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
)
from training.state_features import C_TILE, N_SCALAR, encode as _encode_state, encode_with_timings
from xmodel1.call_response import build_response_action_states
from xmodel1.candidate_quality import build_candidate_features, build_special_candidate_arrays, iter_legal_discards


def encode(state: dict, actor: int, *, state_scalar_dim: int = N_SCALAR):
    tile_feat, scalar = _encode_state(state, actor)
    if scalar.shape[0] < state_scalar_dim:
        padded = np.zeros((state_scalar_dim,), dtype=np.float32)
        padded[: scalar.shape[0]] = scalar
        scalar = padded
    elif scalar.shape[0] > state_scalar_dim:
        scalar = scalar[:state_scalar_dim]
    return tile_feat, scalar.astype(np.float32)


def resolve_runtime_history_summary(snap: dict) -> np.ndarray:
    value = snap.get("history_summary")
    if value is None:
        return empty_history_summary()
    arr = np.asarray(value, dtype=np.float16)
    if arr.shape != (HISTORY_SUMMARY_DIM,):
        raise ValueError(f"history_summary shape {arr.shape} != ({HISTORY_SUMMARY_DIM},)")
    return arr


def _normalize_legal_actions(
    state: dict,
    actor: int,
    legal_actions: Iterable[dict] | None,
) -> list[dict]:
    if legal_actions is None:
        return [a.to_mjai() for a in enumerate_legal_actions(state, actor)]
    return [dict(a) for a in legal_actions]


def _python_runtime_tensor_payload(
    state: dict,
    actor: int,
    legal_actions: list[dict],
    *,
    max_candidates: int,
    candidate_feature_dim: int,
    candidate_flag_dim: int,
) -> dict[str, np.ndarray]:
    discard_actions = iter_legal_discards(legal_actions)
    candidate_feat = np.zeros((max_candidates, candidate_feature_dim), dtype=np.float32)
    candidate_tile_id = np.full((max_candidates,), -1, dtype=np.int16)
    candidate_mask = np.zeros((max_candidates,), dtype=np.uint8)
    candidate_flags = np.zeros((max_candidates, candidate_flag_dim), dtype=np.uint8)
    for slot, action in enumerate(discard_actions[:max_candidates]):
        feat, flags, _quality, _rank, _hard_bad = build_candidate_features(state, actor, action)
        candidate_feat[slot] = feat.astype(np.float32, copy=False)
        candidate_tile_id[slot] = int(tile_to_34(action["pai"]))
        candidate_mask[slot] = 1
        candidate_flags[slot] = flags.astype(np.uint8, copy=False)

    response_action_idx = np.full((XMODEL1_MAX_RESPONSE_CANDIDATES,), -1, dtype=np.int16)
    response_action_mask = np.zeros((XMODEL1_MAX_RESPONSE_CANDIDATES,), dtype=np.uint8)
    response_post_candidate_feat = np.zeros(
        (
            XMODEL1_MAX_RESPONSE_CANDIDATES,
            max_candidates,
            candidate_feature_dim,
        ),
        dtype=np.float32,
    )
    response_post_candidate_tile_id = np.full(
        (XMODEL1_MAX_RESPONSE_CANDIDATES, max_candidates),
        -1,
        dtype=np.int16,
    )
    response_post_candidate_mask = np.zeros(
        (XMODEL1_MAX_RESPONSE_CANDIDATES, max_candidates),
        dtype=np.uint8,
    )
    response_post_candidate_flags = np.zeros(
        (
            XMODEL1_MAX_RESPONSE_CANDIDATES,
            max_candidates,
            candidate_flag_dim,
        ),
        dtype=np.uint8,
    )
    response_post_candidate_quality_score = np.zeros(
        (XMODEL1_MAX_RESPONSE_CANDIDATES, max_candidates),
        dtype=np.float32,
    )
    response_post_candidate_hard_bad_flag = np.zeros(
        (XMODEL1_MAX_RESPONSE_CANDIDATES, max_candidates),
        dtype=np.uint8,
    )
    response_teacher_discard_idx = np.full(
        (XMODEL1_MAX_RESPONSE_CANDIDATES,),
        -1,
        dtype=np.int16,
    )
    response_states = build_response_action_states(state, actor, legal_actions)
    for response_idx, response_state in enumerate(response_states[:XMODEL1_MAX_RESPONSE_CANDIDATES]):
        response_action_idx[response_idx] = np.int16(response_state.action_idx)
        response_action_mask[response_idx] = 1
        best_teacher_idx = -1
        best_teacher_quality = -1e9
        for discard_idx, action in enumerate(response_state.post_discard_actions[:max_candidates]):
            feat, flags, quality, _rank, hard_bad = build_candidate_features(
                response_state.after_snapshot,
                actor,
                action,
            )
            response_post_candidate_feat[response_idx, discard_idx] = feat.astype(np.float32, copy=False)
            response_post_candidate_tile_id[response_idx, discard_idx] = int(tile_to_34(action["pai"]))
            response_post_candidate_mask[response_idx, discard_idx] = 1
            response_post_candidate_flags[response_idx, discard_idx] = flags.astype(np.uint8, copy=False)
            response_post_candidate_quality_score[response_idx, discard_idx] = quality
            response_post_candidate_hard_bad_flag[response_idx, discard_idx] = hard_bad
            if quality > best_teacher_quality:
                best_teacher_quality = float(quality)
                best_teacher_idx = discard_idx
        response_teacher_discard_idx[response_idx] = np.int16(best_teacher_idx)
    return {
        "candidate_feat": candidate_feat,
        "candidate_tile_id": candidate_tile_id,
        "candidate_mask": candidate_mask,
        "candidate_flags": candidate_flags,
        "response_action_idx": response_action_idx,
        "response_action_mask": response_action_mask,
        "response_post_candidate_feat": response_post_candidate_feat,
        "response_post_candidate_tile_id": response_post_candidate_tile_id,
        "response_post_candidate_mask": response_post_candidate_mask,
        "response_post_candidate_flags": response_post_candidate_flags,
        "response_post_candidate_quality_score": response_post_candidate_quality_score,
        "response_post_candidate_hard_bad_flag": response_post_candidate_hard_bad_flag,
        "response_teacher_discard_idx": response_teacher_discard_idx,
        "history_summary": resolve_runtime_history_summary(state),
    }


def resolve_runtime_tensor_payload(
    state: dict,
    actor: int,
    legal_actions: Iterable[dict] | None = None,
    *,
    max_candidates: int = XMODEL1_MAX_CANDIDATES,
    candidate_feature_dim: int = XMODEL1_CANDIDATE_FEATURE_DIM,
    candidate_flag_dim: int = XMODEL1_CANDIDATE_FLAG_DIM,
) -> dict[str, np.ndarray]:
    resolved_legal_actions = _normalize_legal_actions(state, actor, legal_actions)
    try:
        payload = _build_xmodel1_runtime_tensors(
            state,
            actor,
            resolved_legal_actions,
        )
    except RuntimeError as exc:
        message = str(exc)
        if (
            not is_missing_rust_capability_error(exc)
            and "failed to parse xmodel1 runtime snapshot" not in message
        ):
            raise
    else:
        return {
            "candidate_feat": np.asarray(payload["candidate_feat"], dtype=np.float32).reshape(
                max_candidates,
                candidate_feature_dim,
            ),
            "candidate_tile_id": np.asarray(payload["candidate_tile_id"], dtype=np.int16).reshape(max_candidates),
            "candidate_mask": np.asarray(payload["candidate_mask"], dtype=np.uint8).reshape(max_candidates),
            "candidate_flags": np.asarray(payload["candidate_flags"], dtype=np.uint8).reshape(
                max_candidates,
                candidate_flag_dim,
            ),
            "response_action_idx": np.asarray(
                payload.get("response_action_idx", np.full((XMODEL1_MAX_RESPONSE_CANDIDATES,), -1, dtype=np.int16)),
                dtype=np.int16,
            ).reshape(XMODEL1_MAX_RESPONSE_CANDIDATES),
            "response_action_mask": np.asarray(
                payload.get("response_action_mask", np.zeros((XMODEL1_MAX_RESPONSE_CANDIDATES,), dtype=np.uint8)),
                dtype=np.uint8,
            ).reshape(XMODEL1_MAX_RESPONSE_CANDIDATES),
            "response_post_candidate_feat": np.asarray(
                payload.get(
                    "response_post_candidate_feat",
                    np.zeros(
                        (
                            XMODEL1_MAX_RESPONSE_CANDIDATES,
                            max_candidates,
                            candidate_feature_dim,
                        ),
                        dtype=np.float32,
                    ),
                ),
                dtype=np.float32,
            ).reshape(XMODEL1_MAX_RESPONSE_CANDIDATES, max_candidates, candidate_feature_dim),
            "response_post_candidate_tile_id": np.asarray(
                payload.get(
                    "response_post_candidate_tile_id",
                    np.full((XMODEL1_MAX_RESPONSE_CANDIDATES, max_candidates), -1, dtype=np.int16),
                ),
                dtype=np.int16,
            ).reshape(XMODEL1_MAX_RESPONSE_CANDIDATES, max_candidates),
            "response_post_candidate_mask": np.asarray(
                payload.get(
                    "response_post_candidate_mask",
                    np.zeros((XMODEL1_MAX_RESPONSE_CANDIDATES, max_candidates), dtype=np.uint8),
                ),
                dtype=np.uint8,
            ).reshape(XMODEL1_MAX_RESPONSE_CANDIDATES, max_candidates),
            "response_post_candidate_flags": np.asarray(
                payload.get(
                    "response_post_candidate_flags",
                    np.zeros(
                        (
                            XMODEL1_MAX_RESPONSE_CANDIDATES,
                            max_candidates,
                            candidate_flag_dim,
                        ),
                        dtype=np.uint8,
                    ),
                ),
                dtype=np.uint8,
            ).reshape(XMODEL1_MAX_RESPONSE_CANDIDATES, max_candidates, candidate_flag_dim),
            "response_post_candidate_quality_score": np.asarray(
                payload.get(
                    "response_post_candidate_quality_score",
                    np.zeros((XMODEL1_MAX_RESPONSE_CANDIDATES, max_candidates), dtype=np.float32),
                ),
                dtype=np.float32,
            ).reshape(XMODEL1_MAX_RESPONSE_CANDIDATES, max_candidates),
            "response_post_candidate_hard_bad_flag": np.asarray(
                payload.get(
                    "response_post_candidate_hard_bad_flag",
                    np.zeros((XMODEL1_MAX_RESPONSE_CANDIDATES, max_candidates), dtype=np.uint8),
                ),
                dtype=np.uint8,
            ).reshape(XMODEL1_MAX_RESPONSE_CANDIDATES, max_candidates),
            "response_teacher_discard_idx": np.asarray(
                payload.get(
                    "response_teacher_discard_idx",
                    np.full((XMODEL1_MAX_RESPONSE_CANDIDATES,), -1, dtype=np.int16),
                ),
                dtype=np.int16,
            ).reshape(XMODEL1_MAX_RESPONSE_CANDIDATES),
            "history_summary": np.asarray(payload["history_summary"], dtype=np.float16).reshape(HISTORY_SUMMARY_DIM),
        }
    return _python_runtime_tensor_payload(
        state,
        actor,
        resolved_legal_actions,
        max_candidates=max_candidates,
        candidate_feature_dim=candidate_feature_dim,
        candidate_flag_dim=candidate_flag_dim,
    )


def build_runtime_candidate_arrays(
    state: dict,
    actor: int,
    legal_actions: Iterable[dict] | None = None,
    *,
    max_candidates: int = XMODEL1_MAX_CANDIDATES,
    candidate_feature_dim: int = XMODEL1_CANDIDATE_FEATURE_DIM,
    candidate_flag_dim: int = XMODEL1_CANDIDATE_FLAG_DIM,
):
    payload = resolve_runtime_tensor_payload(
        state,
        actor,
        legal_actions,
        max_candidates=max_candidates,
        candidate_feature_dim=candidate_feature_dim,
        candidate_flag_dim=candidate_flag_dim,
    )
    return (
        payload["candidate_feat"],
        payload["candidate_tile_id"],
        payload["candidate_mask"],
        payload["candidate_flags"],
    )


def build_runtime_special_candidate_arrays(
    state: dict,
    actor: int,
    legal_actions: Iterable[dict] | None = None,
    *,
    max_candidates: int = XMODEL1_MAX_SPECIAL_CANDIDATES,
    feature_dim: int = XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
):
    resolved_legal_actions = _normalize_legal_actions(state, actor, legal_actions)
    feat, type_id, mask, _quality, _rank, _hard_bad, _chosen_idx = build_special_candidate_arrays(
        state,
        actor,
        resolved_legal_actions,
        chosen_action=None,
        max_candidates=max_candidates,
        feature_dim=feature_dim,
        include_terminal_actions=True,
    )
    return feat, type_id, mask


__all__ = [
    "C_TILE",
    "N_SCALAR",
    "HISTORY_SUMMARY_DIM",
    "compute_history_summary",
    "empty_history_summary",
    "resolve_runtime_history_summary",
    "resolve_runtime_tensor_payload",
    "encode",
    "encode_with_timings",
    "build_runtime_candidate_arrays",
    "build_runtime_special_candidate_arrays",
]
