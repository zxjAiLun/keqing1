"""Tenhou four-player ranked ladder profile.

Implements the official Tenhou rank/PT progression (four-player mahjong):

- rank ladder: ``新人 -> 9級..1級 -> 初段..十段 -> 天鳳位``
- rooms: 一般 / 上級 / 特上 / 鳳凰, each for tonpuu (東風) and hanchan (東南)
- promotion / demotion discards overflow: promoted players restart at the new
  rank's initial PT; 初段~十段 demote when PT goes negative
- kyu ranks never demote: PT is floored at 0
- rating follows the official Tenhou formula with the table-average floor of
  ``max(avg, 1500)`` and round-up to two decimals
- room selection under ``highest_common_eligible`` is strict: if the four
  players share no eligible room the table fails loudly instead of silently
  reusing an arbitrary PT table

Sources: https://tenhou.net/man/index.html
"""

from __future__ import annotations

from decimal import ROUND_CEILING, Decimal
from typing import Any, Sequence

from .base import (
    PlayerRankState,
    RankMeta,
    RankResolutionError,
    RankUpdate,
    TableContext,
)

# --- Rank ladder -----------------------------------------------------------

# 1-based ordinals: newcomer=1 ... 1kyu=10, 初段=11 ... 十段=20, 天鳳位=21.
RANK_NAMES: dict[str, str] = {
    "newcomer": "新人",
    "9kyu": "9级",
    "8kyu": "8级",
    "7kyu": "7级",
    "6kyu": "6级",
    "5kyu": "5级",
    "4kyu": "4级",
    "3kyu": "3级",
    "2kyu": "2级",
    "1kyu": "1级",
    "1dan": "初段",
    "2dan": "二段",
    "3dan": "三段",
    "4dan": "四段",
    "5dan": "五段",
    "6dan": "六段",
    "7dan": "七段",
    "8dan": "八段",
    "9dan": "九段",
    "10dan": "十段",
    "tenhou": "天凤位",
}
RANK_ORDER: tuple[str, ...] = tuple(RANK_NAMES)
RANK_ORDINALS: dict[str, int] = {rank_id: index + 1 for index, rank_id in enumerate(RANK_ORDER)}

# --- PT tables -------------------------------------------------------------

# Positive placement PT by (game_length, room). 3rd place is always 0.
POSITIVE_PT: dict[str, dict[str, tuple[int, int, int]]] = {
    "hanchan": {
        "ippan": (30, 15, 0),
        "joukyuu": (60, 15, 0),
        "tokujou": (75, 30, 0),
        "houou": (90, 45, 0),
    },
    "tonpuu": {
        "ippan": (20, 10, 0),
        "joukyuu": (40, 10, 0),
        "tokujou": (50, 20, 0),
        "houou": (60, 30, 0),
    },
}

# Promotion threshold (PT to exceed the rank) for newcomer/kyu ranks.
KYU_PROMOTION_PT: dict[str, int] = {
    "newcomer": 20,
    "9kyu": 20,
    "8kyu": 20,
    "7kyu": 20,
    "6kyu": 40,
    "5kyu": 60,
    "4kyu": 80,
    "3kyu": 100,
    "2kyu": 100,
    "1kyu": 100,
}

# 4th-place penalty for kyu ranks. Kyu ranks never demote: negative PT is
# floored at 0. 新人~3級 take no penalty at all.
KYU_FOURTH_PT: dict[str, dict[str, int]] = {
    "tonpuu": {
        "newcomer": 0, "9kyu": 0, "8kyu": 0, "7kyu": 0, "6kyu": 0, "5kyu": 0,
        "4kyu": 0, "3kyu": 0, "2kyu": -10, "1kyu": -20,
    },
    "hanchan": {
        "newcomer": 0, "9kyu": 0, "8kyu": 0, "7kyu": 0, "6kyu": 0, "5kyu": 0,
        "4kyu": 0, "3kyu": 0, "2kyu": -15, "1kyu": -30,
    },
}

# 初段~十段:  initial PT = 200n, promotion threshold = 400n,
#            4th place = -10(n+2) (tonpuu) / -15(n+2) (hanchan)
DAN_INITIAL_PT: dict[str, int] = {f"{n}dan": 200 * n for n in range(1, 11)}
DAN_PROMOTION_PT: dict[str, int] = {f"{n}dan": 400 * n for n in range(1, 11)}
DAN_FOURTH_PT: dict[str, dict[str, int]] = {
    "tonpuu": {f"{n}dan": -10 * (n + 2) for n in range(1, 11)},
    "hanchan": {f"{n}dan": -15 * (n + 2) for n in range(1, 11)},
}

TENHOU_INITIAL_PT = 4000  # 十段 promotion threshold; 天鳳位 has no target

# --- Rating ----------------------------------------------------------------

RATING_PLACEMENT_POINTS = (30, 10, -10, -30)
RATING_FLOOR = Decimal("1500")


def _rating_correction(games_before: int) -> Decimal:
    """``1 - games*0.002`` before 400 games, fixed ``0.2`` afterwards."""
    if games_before < 400:
        return Decimal("1") - Decimal(games_before) * Decimal("0.002")
    return Decimal("0.2")


def _round_up_2dp(value: Decimal) -> Decimal:
    """Round the third decimal place and below up (切り上げ)."""
    return (value * 100).to_integral_value(rounding=ROUND_CEILING) / 100


def _rank_number(rank_id: str) -> int:
    """Rank as a comparable number: 0 for kyu/newcomer, n for dan, 20 for tenhou."""
    if rank_id == "tenhou":
        return 20
    if rank_id in DAN_INITIAL_PT:
        return int(rank_id.removesuffix("dan"))
    return 0


def _is_dan(rank_id: str) -> bool:
    return rank_id in DAN_INITIAL_PT


def _next_rank(rank_id: str) -> str:
    index = RANK_ORDER.index(rank_id)
    return RANK_ORDER[index + 1]


class Tenhou4pRanked:
    """Tenhou four-player ranked progression (``tenhou_4p_ranked``).

    ``room_policy``:
    - ``highest_common_eligible`` (default, strict): pick the highest room all
      four players are eligible for; fail loudly when none exists.
    - ``fixed``: always use the configured room; marked in the manifest as not
      a full Tenhou match simulation.
    """

    system_id = "tenhou_4p_ranked"

    def __init__(
        self,
        version: str,
        *,
        game_length: str = "hanchan",
        room_policy: str = "highest_common_eligible",
        room: str | None = None,
        membership: str = "premium",
        initial_rank: str = "newcomer",
        initial_rating: float = 1500.0,
    ):
        if game_length not in POSITIVE_PT:
            raise ValueError(f"unknown game_length: {game_length!r}")
        if room_policy not in {"highest_common_eligible", "fixed"}:
            raise ValueError(f"unknown room_policy: {room_policy!r}")
        if initial_rank not in RANK_NAMES:
            raise ValueError(f"unknown initial_rank: {initial_rank!r}")
        self.version = str(version)
        self.game_length = game_length
        self.room_policy = room_policy
        self.membership = membership
        self.initial_rank = initial_rank
        self.initial_rating = Decimal(str(initial_rating))
        # ``fixed`` room policy requires an explicit room.
        self.fixed_room = room or ("houou" if room_policy == "fixed" else None)
        if room_policy == "fixed" and self.fixed_room not in POSITIVE_PT[game_length]:
            raise ValueError(f"unknown fixed room: {self.fixed_room!r}")

    # --- RankSystem protocol ------------------------------------------------

    def initial_state(self) -> PlayerRankState:
        return PlayerRankState(
            rank_id=self.initial_rank,
            pt=self.rank_meta(self.initial_rank).initial_pt or 0,
            rating=self.initial_rating,
            games=0,
        )

    def rank_meta(self, rank_id: str) -> RankMeta:
        if rank_id not in RANK_NAMES:
            raise ValueError(f"unknown rank_id: {rank_id!r}")
        ordinal = RANK_ORDINALS[rank_id]
        if rank_id == "tenhou":
            return RankMeta(
                rank_id=rank_id,
                rank_name=RANK_NAMES[rank_id],
                ordinal=ordinal,
                initial_pt=TENHOU_INITIAL_PT,
                target_pt=None,
                is_tenhou=True,
            )
        if _is_dan(rank_id):
            return RankMeta(
                rank_id=rank_id,
                rank_name=RANK_NAMES[rank_id],
                ordinal=ordinal,
                initial_pt=DAN_INITIAL_PT[rank_id],
                target_pt=DAN_PROMOTION_PT[rank_id],
            )
        return RankMeta(
            rank_id=rank_id,
            rank_name=RANK_NAMES[rank_id],
            ordinal=ordinal,
            initial_pt=0,
            target_pt=KYU_PROMOTION_PT[rank_id],
        )

    def _eligible(self, state: PlayerRankState, room: str) -> bool:
        """Tenhou room admission based on pre-match rank and rating."""
        rank_num = _rank_number(state.rank_id)
        rating = float(state.rating)
        if room == "ippan":
            if rank_num == 0:
                return True
            if rank_num <= 3:
                return True
            if rank_num == 4:
                return rating < 1800
            return False
        if room == "joukyuu":
            return 1 <= rank_num <= 7 and rating < 2000
        if room == "tokujou":
            return rank_num >= 4 and rating >= 1800
        if room == "houou":
            return rank_num >= 7 and rating >= 2000 and self.membership == "premium"
        raise ValueError(f"unknown room: {room!r}")

    def _fixed_room(self) -> str:
        if self.fixed_room is None:
            raise RankResolutionError("fixed room policy requires a configured room")
        return self.fixed_room

    def resolve_table(self, players: Sequence[PlayerRankState]) -> TableContext:
        if len(players) != 4:
            raise ValueError(f"expected four players, got {len(players)}")
        raw_avg = sum(player.rating for player in players) / 4
        if self.room_policy == "fixed":
            room = self._fixed_room()
            if self.game_length not in POSITIVE_PT or room not in POSITIVE_PT[self.game_length]:
                raise RankResolutionError(f"no PT table for room {room!r} / {self.game_length}")
            return TableContext(
                room=room,
                game_length=self.game_length,
                positive_pt=POSITIVE_PT[self.game_length][room],
                avg_rating=raw_avg,
                strict=False,
            )
        # highest_common_eligible: 鳳凰 -> 特上 -> 上級 -> 一般
        for room in ("houou", "tokujou", "joukyuu", "ippan"):
            if all(self._eligible(player, room) for player in players):
                return TableContext(
                    room=room,
                    game_length=self.game_length,
                    positive_pt=POSITIVE_PT[self.game_length][room],
                    avg_rating=raw_avg,
                    strict=True,
                )
        ranks = ", ".join(player.rank_id for player in players)
        raise RankResolutionError(
            f"no common eligible room for players [{ranks}] "
            f"(ratings {[float(p.rating) for p in players]}) under highest_common_eligible"
        )

    def _fourth_pt(self, rank_id: str, game_length: str) -> int:
        if rank_id == "tenhou":
            return 0
        if _is_dan(rank_id):
            return DAN_FOURTH_PT[game_length][rank_id]
        return KYU_FOURTH_PT[game_length][rank_id]

    def apply_result(
        self,
        state: PlayerRankState,
        *,
        placement: int,
        table: TableContext,
    ) -> RankUpdate:
        if placement not in (1, 2, 3, 4):
            raise ValueError(f"placement must be 1..4, got {placement}")
        pt_delta = (
            int(table.positive_pt[placement - 1])
            if placement < 4
            else self._fourth_pt(state.rank_id, table.game_length)
        )
        raw_pt = int(state.pt) + pt_delta

        rank_before = state.rank_id
        rank_after = state.rank_id
        pt_after = raw_pt
        transition = "none"

        if rank_before == "tenhou":
            pt_after = int(state.pt)
        elif _is_dan(rank_before):
            n = int(rank_before.removesuffix("dan"))
            if raw_pt >= DAN_PROMOTION_PT[rank_before]:
                rank_after = "tenhou" if n == 10 else f"{n + 1}dan"
                pt_after = self.rank_meta(rank_after).initial_pt or 0
                transition = "tenhou" if n == 10 else "promotion"
            elif raw_pt < 0:
                rank_after = "1kyu" if n == 1 else f"{n - 1}dan"
                pt_after = self.rank_meta(rank_after).initial_pt or 0
                transition = "demotion"
            else:
                pt_after = raw_pt
        else:  # newcomer / kyu: never demote, PT floored at 0
            if raw_pt >= KYU_PROMOTION_PT[rank_before]:
                rank_after = _next_rank(rank_before)
                pt_after = self.rank_meta(rank_after).initial_pt or 0
                transition = "promotion"
            else:
                pt_after = max(0, raw_pt)

        # Rating: official Tenhou formula with table-average floor + round-up.
        avg_effective = max(table.avg_rating, RATING_FLOOR)
        correction = _rating_correction(state.games)
        delta_raw = correction * (
            Decimal(RATING_PLACEMENT_POINTS[placement - 1])
            + (avg_effective - state.rating) / Decimal(40)
        )
        delta = _round_up_2dp(delta_raw)
        rating_after = state.rating + delta

        return RankUpdate(
            rank_before=rank_before,
            pt_before=int(state.pt),
            pt_delta=pt_delta,
            transition=transition,
            rank_after=rank_after,
            pt_after=pt_after,
            rating_before=state.rating,
            rating_delta_raw=delta_raw,
            rating_after=rating_after,
        )

    def scoring_block(self) -> dict[str, Any]:
        """Serializable scoring description for the report manifest."""
        return {
            "system": self.system_id,
            "version": self.version,
            "game_length": self.game_length,
            "room_policy": self.room_policy,
            "membership": self.membership,
            "initial_rank": self.initial_rank,
            "initial_rating": float(self.initial_rating),
            "room": self.fixed_room if self.room_policy == "fixed" else None,
        }
