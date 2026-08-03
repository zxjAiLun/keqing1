"""Legacy fixed-profile rank system.

Preserves the exact historical semantics of the pre-Round-8 ladder:

- every account sits at a fixed rank (default ``七段``) with a fixed initial PT
  and target PT;
- placement PT comes from a fixed four-tuple (default 鳳凰 東南 ``90,45,0,-135``);
- rating uses the historical formula *without* the Tenhou table-average floor
  or two-decimal rounding, so re-building old seasons reproduces identical
  numbers.

This profile is intentionally frozen: it exists so existing 1000-hanchan
snapshots are never re-interpreted as "start from 新人".
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Sequence

from .base import (
    PlayerRankState,
    RankMeta,
    RankUpdate,
    TableContext,
)

_RATING_RESULTS = (30, 10, -10, -30)


@dataclass
class LegacyFixedProfile:
    """Fixed-rank PT/Rating profile (``tenhou_houou_7dan_fixed``)."""

    system_id = "tenhou_houou_7dan_fixed"

    version: str = "2026-07-01"
    game_length: str = "hanchan"
    room: str = "houou"
    initial_rank: str = "7dan"
    initial_pt: float = 1400.0
    initial_rating: float = 1500.0
    target_pt: float = 2800.0
    rank_points: tuple[int, int, int, int] = (90, 45, 0, -135)

    def __post_init__(self) -> None:
        if len(self.rank_points) != 4:
            raise ValueError(f"rank_points must contain four values, got {self.rank_points!r}")

    @property
    def room_policy(self) -> str:
        return "fixed"

    @property
    def membership(self) -> str | None:
        return None

    @property
    def initial_rating_decimal(self) -> Decimal:
        return Decimal(str(self.initial_rating))

    # --- RankSystem protocol ------------------------------------------------

    def initial_state(self) -> PlayerRankState:
        return PlayerRankState(
            rank_id=self.initial_rank,
            pt=int(self.initial_pt),
            rating=Decimal(str(self.initial_rating)),
            games=0,
        )

    def rank_meta(self, rank_id: str) -> RankMeta:
        if rank_id != self.initial_rank:
            # The legacy profile only ever tracks its single fixed rank.
            raise ValueError(f"legacy profile only knows rank {self.initial_rank!r}, got {rank_id!r}")
        return RankMeta(
            rank_id=self.initial_rank,
            rank_name="七段" if self.initial_rank == "7dan" else self.initial_rank,
            ordinal=17,
            initial_pt=int(self.initial_pt),
            target_pt=int(self.target_pt),
        )

    def resolve_table(self, players: Sequence[PlayerRankState]) -> TableContext:
        raw_avg = sum(player.rating for player in players) / 4
        return TableContext(
            room=self.room,
            game_length=self.game_length,
            positive_pt=tuple(self.rank_points[:3]),
            avg_rating=raw_avg,
            strict=False,
        )

    def apply_result(
        self,
        state: PlayerRankState,
        *,
        placement: int,
        table: TableContext,
    ) -> RankUpdate:
        if placement not in (1, 2, 3, 4):
            raise ValueError(f"placement must be 1..4, got {placement}")
        pt_delta = int(self.rank_points[placement - 1])
        pt_after = int(state.pt) + pt_delta

        # Historical rating formula: no 1500 floor, no rounding.
        correction = (
            Decimal("1") - Decimal(state.games) * Decimal("0.002")
            if state.games < 400
            else Decimal("0.2")
        )
        delta_raw = correction * (
            Decimal(_RATING_RESULTS[placement - 1])
            + (table.avg_rating - state.rating) / Decimal(40)
        )
        rating_after = state.rating + delta_raw

        return RankUpdate(
            rank_before=state.rank_id,
            pt_before=int(state.pt),
            pt_delta=pt_delta,
            transition="none",
            rank_after=state.rank_id,
            pt_after=pt_after,
            rating_before=state.rating,
            rating_delta_raw=delta_raw,
            rating_after=rating_after,
        )

    def scoring_block(self) -> dict[str, Any]:
        return {
            "system": self.system_id,
            "version": self.version,
            "game_length": self.game_length,
            "room_policy": self.room_policy,
            "membership": self.membership,
            "initial_rank": self.initial_rank,
            "initial_rating": float(self.initial_rating),
        }
