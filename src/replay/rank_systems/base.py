"""Versioned rank system profile contracts.

A ``RankSystem`` encapsulates the full semantics of a competitive ranking /
rating system (rank ladder, PT tables, rating formula).  Report builders and
ladder loaders depend only on this interface, so new systems (Tenhou ranked,
Majsoul, custom league pts, ...) can be added as new implementations without
rewriting the report pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any, Literal, Protocol, Sequence


@dataclass(frozen=True)
class PlayerRankState:
    """Immutable per-player rank state snapshot (used for pre-match inputs)."""

    rank_id: str
    pt: int
    rating: Decimal
    games: int


@dataclass(frozen=True)
class RankMeta:
    """Display / progression metadata for a single rank."""

    rank_id: str
    rank_name: str
    ordinal: int
    initial_pt: int | None
    target_pt: int | None
    is_tenhou: bool = False


@dataclass(frozen=True)
class TableContext:
    """Resolved table for one game.

    ``avg_rating`` is the raw pre-match table average; profiles that floor the
    average (e.g. Tenhou's ``max(avg, 1500)``) apply the floor inside
    ``apply_result`` so the legacy profile can keep its exact historical
    behavior.
    """

    room: str
    game_length: str
    positive_pt: tuple[int, int, int]
    avg_rating: Decimal
    strict: bool = field(default=False)


Transition = Literal["none", "promotion", "demotion", "tenhou"]


@dataclass(frozen=True)
class RankUpdate:
    """One player's rank/rating transition for a single game."""

    rank_before: str
    pt_before: int
    pt_delta: int

    transition: Transition

    rank_after: str
    pt_after: int

    rating_before: Decimal
    rating_delta_raw: Decimal
    rating_after: Decimal


class RankResolutionError(ValueError):
    """No table could be resolved for the given players under the room policy."""


class RankSystem(Protocol):
    """Common interface for a ranked scoring profile."""

    system_id: str
    version: str

    def initial_state(self) -> PlayerRankState: ...
    def resolve_table(self, players: Sequence[PlayerRankState]) -> TableContext: ...
    def apply_result(
        self,
        state: PlayerRankState,
        *,
        placement: int,
        table: TableContext,
    ) -> RankUpdate: ...
    def rank_meta(self, rank_id: str) -> RankMeta: ...
    def scoring_block(self) -> dict[str, Any]: ...
