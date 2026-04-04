from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass(frozen=True)
class ModelAuxOutputs:
    score_delta: float = 0.0
    win_prob: float = 0.0
    dealin_prob: float = 0.0


@dataclass(frozen=True)
class ModelForwardResult:
    policy_logits: np.ndarray
    value: float
    aux: ModelAuxOutputs = field(default_factory=ModelAuxOutputs)


@dataclass(frozen=True)
class DecisionContext:
    actor: int
    event: dict[str, Any]
    runtime_snap: dict[str, Any]
    model_snap: dict[str, Any]
    legal_actions: list[dict[str, Any]]


@dataclass(frozen=True)
class ScoredCandidate:
    action: dict[str, Any]
    logit: float
    final_score: float
    beam_score: Optional[float] = None
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionResult:
    chosen: dict[str, Any]
    candidates: list[ScoredCandidate]
    model_value: float
    model_aux: ModelAuxOutputs = field(default_factory=ModelAuxOutputs)
