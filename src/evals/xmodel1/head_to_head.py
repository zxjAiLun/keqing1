"""Head-to-head summary protocol for Xmodel1."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class HeadToHeadSummary:
    rounds: int
    avg_score_delta: float
    hora_rate: float
    dealin_rate: float
    riichi_rate: float
    call_rate: float


__all__ = ["HeadToHeadSummary"]
