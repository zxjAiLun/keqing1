"""Rule-context and terminal reward helpers for keqingrl."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch

from inference.pt_map import expected_pt_for_all_seats, validate_pt_map


DEFAULT_PT_MAP = (90.0, 45.0, 0.0, -135.0)
RULE_CONTEXT_DIM = 6
RULE_CONTEXT_ENCODING_VERSION = "keqingrl_rule_context_v1"


@dataclass(frozen=True)
class RuleContext:
    pt_map: tuple[float, float, float, float] = DEFAULT_PT_MAP
    pt_norm: float = 90.0
    rank_score_scale: float = 0.0
    game_type: str = "hanchan"

    def to_tensor(self) -> torch.Tensor:
        normalized_pt = tuple(float(value) / float(self.pt_norm) for value in self.pt_map)
        return torch.tensor(
            [
                normalized_pt[0],
                normalized_pt[1],
                normalized_pt[2],
                normalized_pt[3],
                float(self.rank_score_scale),
                _game_type_id(self.game_type),
            ],
            dtype=torch.float32,
        )


@dataclass(frozen=True)
class RewardSpec:
    pt_map: tuple[float, float, float, float] = DEFAULT_PT_MAP
    pt_norm: float = 90.0
    terminal_only: bool = True
    round_score_delta_weight: float = 0.0
    gamma: float = 1.0
    gae_lambda: float = 0.95
    terminal_handling: str = "final_rank_pt"
    abortive_draw_handling: str = "continue_or_terminal_by_env"
    tie_break: str = "game_start_oya"
    metadata: dict[str, float | str | bool] = field(default_factory=dict)


DEFAULT_RULE_CONTEXT = RuleContext()
DEFAULT_REWARD_SPEC = RewardSpec()


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
    pt_map: Sequence[int | float] | RuleContext = DEFAULT_PT_MAP,
    *,
    rank_score_scale: float = 0.0,
    is_hanchan: bool = True,
    pt_norm: float | None = None,
) -> torch.Tensor:
    if isinstance(pt_map, RuleContext):
        return pt_map.to_tensor()
    context = RuleContext(
        pt_map=tuple(float(value) for value in validate_pt_map(pt_map)),  # type: ignore[arg-type]
        pt_norm=float(pt_norm) if pt_norm is not None else pt_map_scale(pt_map),
        rank_score_scale=float(rank_score_scale),
        game_type="hanchan" if is_hanchan else "tonpuu",
    )
    return context.to_tensor()


def _game_type_id(game_type: str) -> float:
    normalized = str(game_type).lower()
    if normalized in {"hanchan", "half", "south"}:
        return 1.0
    if normalized in {"tonpuu", "east"}:
        return 0.0
    raise ValueError(f"unsupported rule_context game_type: {game_type!r}")


__all__ = [
    "DEFAULT_PT_MAP",
    "DEFAULT_REWARD_SPEC",
    "DEFAULT_RULE_CONTEXT",
    "RULE_CONTEXT_DIM",
    "RULE_CONTEXT_ENCODING_VERSION",
    "RewardSpec",
    "RuleContext",
    "build_rule_context",
    "normalize_pt_map",
    "pt_map_scale",
    "terminal_rank_rewards",
]
