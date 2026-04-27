"""Discard-only Mortal teacher mapping utilities.

This module intentionally maps only Mortal's discard tile ids. It does not
bridge Mortal calls, riichi, kan decisions, agari, ryukyoku, or pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import torch

from keqingrl.actions import ACTION_FLAG_TSUMOGIRI, ActionSpec, ActionType
from mahjong_env.action_space import TILE_NAME_TO_IDX
from mahjong_env.tiles import normalize_tile

MORTAL_ACTION_SPACE = 46
MORTAL_DISCARD_ACTION_COUNT = 37
MORTAL_DISCARD_TEACHER_CONTRACT_VERSION = "mortal_discard_teacher_v1"
MORTAL_Q_VALUES_EXTRA_KEY = "mortal_q_values"
MORTAL_ACTION_MASK_EXTRA_KEY = "mortal_action_mask"

MORTAL_DISCARD_ID_TO_TILE = (
    "1m",
    "2m",
    "3m",
    "4m",
    "5m",
    "6m",
    "7m",
    "8m",
    "9m",
    "1p",
    "2p",
    "3p",
    "4p",
    "5p",
    "6p",
    "7p",
    "8p",
    "9p",
    "1s",
    "2s",
    "3s",
    "4s",
    "5s",
    "6s",
    "7s",
    "8s",
    "9s",
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
    "5mr",
    "5pr",
    "5sr",
)


class MortalTeacherMappingError(ValueError):
    """Raised when Mortal discard ids cannot be aligned to KeqingRL actions."""


@dataclass(frozen=True)
class MortalDiscardTeacherScores:
    scores: torch.Tensor
    score_mask: torch.BoolTensor
    source_action_ids: tuple[tuple[int, ...], ...]
    missing_legal_keys: tuple[str, ...] = ()
    extra_mortal_discard_ids: tuple[int, ...] = ()


def mortal_discard_tile_name(action_id: int) -> str:
    if not 0 <= int(action_id) < MORTAL_DISCARD_ACTION_COUNT:
        raise MortalTeacherMappingError(f"Mortal discard action id out of range: {action_id}")
    return MORTAL_DISCARD_ID_TO_TILE[int(action_id)]


def mortal_discard_tile_id(action_id: int) -> int:
    return int(TILE_NAME_TO_IDX[normalize_tile(mortal_discard_tile_name(action_id))])


def mortal_discard_action_spec(action_id: int, *, tsumogiri: bool = False) -> ActionSpec:
    flags = ACTION_FLAG_TSUMOGIRI if bool(tsumogiri) else 0
    return ActionSpec(ActionType.DISCARD, tile=mortal_discard_tile_id(action_id), flags=flags)


def mortal_discard_ids_for_tile_id(tile_id: int) -> tuple[int, ...]:
    tile_id = int(tile_id)
    ids = tuple(
        action_id
        for action_id, tile_name in enumerate(MORTAL_DISCARD_ID_TO_TILE)
        if int(TILE_NAME_TO_IDX[normalize_tile(tile_name)]) == tile_id
    )
    if not ids:
        raise MortalTeacherMappingError(f"tile id has no Mortal discard id: {tile_id}")
    return ids


def mortal_discard_mask_tile_ids(mortal_mask: torch.Tensor | Sequence[bool]) -> set[int]:
    mask = _as_1d_bool_tensor(mortal_mask, field_name="mortal_mask")
    return {
        mortal_discard_tile_id(action_id)
        for action_id in range(MORTAL_DISCARD_ACTION_COUNT)
        if bool(mask[action_id].item())
    }


def keqing_legal_discard_tile_ids(legal_actions: Sequence[ActionSpec]) -> set[int]:
    return {
        int(action.tile)
        for action in legal_actions
        if action.action_type == ActionType.DISCARD and action.tile is not None
    }


def assert_mortal_discard_mask_parity(
    mortal_mask: torch.Tensor | Sequence[bool],
    legal_actions: Sequence[ActionSpec],
) -> None:
    mortal_tiles = mortal_discard_mask_tile_ids(mortal_mask)
    keqing_tiles = keqing_legal_discard_tile_ids(legal_actions)
    missing = sorted(keqing_tiles - mortal_tiles)
    extra = sorted(mortal_tiles - keqing_tiles)
    if missing or extra:
        raise MortalTeacherMappingError(
            "Mortal/KeqingRL discard mask mismatch: "
            f"missing_keqing_tiles={missing}, extra_mortal_tiles={extra}"
        )


def mortal_discard_scores_for_legal_actions(
    q_values: torch.Tensor | Sequence[float],
    mortal_mask: torch.Tensor | Sequence[bool],
    legal_actions: Sequence[ActionSpec],
    *,
    strict_mask: bool = True,
) -> MortalDiscardTeacherScores:
    q = _as_1d_float_tensor(q_values, field_name="q_values")
    mask = _as_1d_bool_tensor(mortal_mask, field_name="mortal_mask")
    if strict_mask:
        assert_mortal_discard_mask_parity(mask, legal_actions)

    scores = torch.full((len(legal_actions),), float("-inf"), dtype=q.dtype, device=q.device)
    score_mask = torch.zeros((len(legal_actions),), dtype=torch.bool, device=q.device)
    source_action_ids: list[tuple[int, ...]] = []
    missing_legal_keys: list[str] = []

    for index, action in enumerate(legal_actions):
        if action.action_type != ActionType.DISCARD or action.tile is None:
            source_action_ids.append(())
            continue
        candidate_ids = mortal_discard_ids_for_tile_id(int(action.tile))
        legal_candidate_ids = tuple(action_id for action_id in candidate_ids if bool(mask[action_id].item()))
        source_action_ids.append(legal_candidate_ids)
        if not legal_candidate_ids:
            missing_legal_keys.append(action.canonical_key)
            continue
        candidate_tensor = torch.tensor(legal_candidate_ids, dtype=torch.long, device=q.device)
        scores[index] = q.index_select(0, candidate_tensor).max()
        score_mask[index] = True

    extra_mortal_ids = _extra_mortal_discard_ids(mask, legal_actions)
    if strict_mask and (missing_legal_keys or extra_mortal_ids):
        raise MortalTeacherMappingError(
            "Mortal/KeqingRL discard score mapping mismatch: "
            f"missing_legal_keys={missing_legal_keys}, extra_mortal_discard_ids={extra_mortal_ids}"
        )
    return MortalDiscardTeacherScores(
        scores=scores,
        score_mask=score_mask,
        source_action_ids=tuple(source_action_ids),
        missing_legal_keys=tuple(missing_legal_keys),
        extra_mortal_discard_ids=tuple(extra_mortal_ids),
    )


def mortal_discard_topk_teacher_context(
    *,
    prior_logits: torch.Tensor,
    legal_action_mask: torch.Tensor,
    legal_actions: Sequence[Sequence[ActionSpec]],
    q_values: torch.Tensor,
    mortal_masks: torch.Tensor,
    topk: int,
    teacher_temperature: float,
    strict_mask: bool = True,
) -> dict[str, torch.Tensor]:
    if float(teacher_temperature) <= 0.0:
        raise ValueError(f"teacher_temperature must be positive, got {teacher_temperature}")
    if int(topk) <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if prior_logits.ndim != 2 or legal_action_mask.ndim != 2:
        raise ValueError("prior_logits and legal_action_mask must be rank-2")
    if q_values.ndim != 2 or mortal_masks.ndim != 2:
        raise ValueError("q_values and mortal_masks must be rank-2")
    batch_size = int(prior_logits.shape[0])
    if len(legal_actions) != batch_size:
        raise ValueError("legal_actions length must match prior_logits batch size")
    if q_values.shape[0] != batch_size or mortal_masks.shape[0] != batch_size:
        raise ValueError("Mortal q/mask batch size must match prior_logits")

    mask = legal_action_mask.bool()
    masked_prior = prior_logits.float().masked_fill(~mask, torch.finfo(torch.float32).min)
    min_legal_count = int(mask.sum(dim=-1).min().item())
    if min_legal_count <= 0:
        raise ValueError("every row must have at least one legal action")
    k = min(int(topk), min_legal_count)
    topk_prior_values, topk_indices = torch.topk(masked_prior, k=k, dim=-1)

    row_scores: list[torch.Tensor] = []
    for row_idx, row_actions in enumerate(legal_actions):
        mapped = mortal_discard_scores_for_legal_actions(
            q_values[row_idx],
            mortal_masks[row_idx],
            row_actions,
            strict_mask=strict_mask,
        )
        topk_scores = mapped.scores.gather(0, topk_indices[row_idx].to(mapped.scores.device))
        if not torch.isfinite(topk_scores).all():
            raise MortalTeacherMappingError(f"Mortal teacher is missing at least one topK action in row {row_idx}")
        row_scores.append(topk_scores.to(device=prior_logits.device, dtype=prior_logits.dtype))

    teacher_topk_scores = torch.stack(row_scores, dim=0)
    teacher_probs = torch.softmax(teacher_topk_scores / float(teacher_temperature), dim=-1)
    teacher_log_probs = torch.log(teacher_probs.clamp_min(1e-12))
    teacher_argmax = teacher_topk_scores.argmax(dim=-1)
    teacher_prior_agreement = (teacher_argmax == 0).float()
    teacher_rule_top1_rank = 1.0 + (teacher_topk_scores > teacher_topk_scores[:, :1]).sum(dim=-1).float()
    if k > 1:
        teacher_top2 = torch.topk(teacher_topk_scores, k=2, dim=-1).values
        teacher_margin = teacher_top2[:, 0] - teacher_top2[:, 1]
    else:
        teacher_margin = torch.zeros((batch_size,), device=prior_logits.device, dtype=prior_logits.dtype)
    teacher_entropy = -(teacher_probs * teacher_log_probs).sum(dim=-1)
    return {
        "topk_indices": topk_indices,
        "topk_prior_values": topk_prior_values,
        "teacher_topk_scores": teacher_topk_scores,
        "teacher_probs": teacher_probs,
        "teacher_log_probs": teacher_log_probs,
        "teacher_argmax": teacher_argmax,
        "teacher_prior_agreement": teacher_prior_agreement,
        "teacher_rule_top1_rank": teacher_rule_top1_rank,
        "teacher_margin": teacher_margin,
        "teacher_entropy": teacher_entropy,
    }


def mortal_discard_teacher_tensors_from_extras(
    extras: Mapping[str, torch.Tensor],
    *,
    q_values_key: str = MORTAL_Q_VALUES_EXTRA_KEY,
    action_mask_key: str = MORTAL_ACTION_MASK_EXTRA_KEY,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    missing = [key for key in (q_values_key, action_mask_key) if key not in extras]
    if missing:
        raise MortalTeacherMappingError(f"Mortal teacher extras missing required keys: {missing}")
    q_values = _as_2d_float_tensor(extras[q_values_key], field_name=q_values_key)
    action_mask = _as_2d_bool_tensor(extras[action_mask_key], field_name=action_mask_key)
    if int(q_values.shape[0]) != int(action_mask.shape[0]):
        raise MortalTeacherMappingError(
            "Mortal teacher extras batch size mismatch: "
            f"{q_values_key}={tuple(q_values.shape)}, {action_mask_key}={tuple(action_mask.shape)}"
        )
    return q_values, action_mask


def _extra_mortal_discard_ids(mask: torch.Tensor, legal_actions: Sequence[ActionSpec]) -> tuple[int, ...]:
    legal_tile_ids = keqing_legal_discard_tile_ids(legal_actions)
    return tuple(
        action_id
        for action_id in range(MORTAL_DISCARD_ACTION_COUNT)
        if bool(mask[action_id].item()) and mortal_discard_tile_id(action_id) not in legal_tile_ids
    )


def _as_1d_float_tensor(value: torch.Tensor | Sequence[float], *, field_name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 1:
        raise ValueError(f"{field_name} must be rank-1")
    if int(tensor.numel()) < MORTAL_ACTION_SPACE:
        raise ValueError(f"{field_name} must have at least {MORTAL_ACTION_SPACE} entries")
    return tensor


def _as_2d_float_tensor(value: torch.Tensor | Sequence[Sequence[float]], *, field_name: str) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32)
    if tensor.ndim != 2:
        raise MortalTeacherMappingError(f"{field_name} must be rank-2")
    if int(tensor.shape[-1]) < MORTAL_ACTION_SPACE:
        raise MortalTeacherMappingError(f"{field_name} must have at least {MORTAL_ACTION_SPACE} entries")
    return tensor


def _as_1d_bool_tensor(value: torch.Tensor | Sequence[bool], *, field_name: str) -> torch.BoolTensor:
    tensor = torch.as_tensor(value, dtype=torch.bool)
    if tensor.ndim != 1:
        raise ValueError(f"{field_name} must be rank-1")
    if int(tensor.numel()) < MORTAL_ACTION_SPACE:
        raise ValueError(f"{field_name} must have at least {MORTAL_ACTION_SPACE} entries")
    return tensor


def _as_2d_bool_tensor(value: torch.Tensor | Sequence[Sequence[bool]], *, field_name: str) -> torch.BoolTensor:
    tensor = torch.as_tensor(value, dtype=torch.bool)
    if tensor.ndim != 2:
        raise MortalTeacherMappingError(f"{field_name} must be rank-2")
    if int(tensor.shape[-1]) < MORTAL_ACTION_SPACE:
        raise MortalTeacherMappingError(f"{field_name} must have at least {MORTAL_ACTION_SPACE} entries")
    return tensor


__all__ = [
    "MORTAL_ACTION_MASK_EXTRA_KEY",
    "MORTAL_ACTION_SPACE",
    "MORTAL_DISCARD_ACTION_COUNT",
    "MORTAL_DISCARD_ID_TO_TILE",
    "MORTAL_DISCARD_TEACHER_CONTRACT_VERSION",
    "MORTAL_Q_VALUES_EXTRA_KEY",
    "MortalDiscardTeacherScores",
    "MortalTeacherMappingError",
    "assert_mortal_discard_mask_parity",
    "keqing_legal_discard_tile_ids",
    "mortal_discard_action_spec",
    "mortal_discard_ids_for_tile_id",
    "mortal_discard_mask_tile_ids",
    "mortal_discard_scores_for_legal_actions",
    "mortal_discard_teacher_tensors_from_extras",
    "mortal_discard_tile_id",
    "mortal_discard_tile_name",
    "mortal_discard_topk_teacher_context",
]
