"""Rule-context and terminal reward helpers for keqingrl."""

from __future__ import annotations

from typing import Sequence

import torch

from inference.pt_map import expected_pt_for_all_seats, validate_pt_map


DEFAULT_PT_MAP = (90.0, 45.0, 0.0, -135.0)
RULE_CONTEXT_DIM = 6


def pt_map_scale(pt_map: Sequence[int | float]) -> float:
    normalized = validate_pt_map(pt_map)
    first_place_scale = abs(normalized[0])
    if first_place_scale > 0.0:
        return float(first_place_scale)
    fallback = max(abs(value) for value in normalized)
    return float(fallback if fallback > 0.0 else 1.0)


def normalize_pt_map(pt_map: Sequence[int | float]) -> tuple[float, float, float, float]:
    normalized = validate_pt_map(pt_map)
    scale = pt_map_scale(normalized)
    return tuple(value / scale for value in normalized)  # type: ignore[return-value]


def terminal_rank_rewards(
    scores: Sequence[int | float],
    pt_map: Sequence[int | float],
    *,
    initial_oya: int = 0,
    normalize: bool = True,
) -> tuple[float, float, float, float]:
    rewards = expected_pt_for_all_seats(scores, pt_map, initial_oya=initial_oya)
    if not normalize:
        return rewards
    scale = pt_map_scale(pt_map)
    return tuple(value / scale for value in rewards)  # type: ignore[return-value]


def build_rule_context(
    pt_map: Sequence[int | float],
    *,
    rank_score_scale: float = 0.0,
    is_hanchan: bool = True,
) -> torch.Tensor:
    normalized_pt = normalize_pt_map(pt_map)
    return torch.tensor(
        [
            normalized_pt[0],
            normalized_pt[1],
            normalized_pt[2],
            normalized_pt[3],
            float(rank_score_scale),
            1.0 if is_hanchan else 0.0,
        ],
        dtype=torch.float32,
    )


__all__ = [
    "DEFAULT_PT_MAP",
    "RULE_CONTEXT_DIM",
    "build_rule_context",
    "normalize_pt_map",
    "pt_map_scale",
    "terminal_rank_rewards",
]
