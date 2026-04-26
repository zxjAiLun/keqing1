"""Opponent-pool helpers for keqingrl self-play."""

from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Sequence

from keqingrl.policy import InteractivePolicy


@dataclass(frozen=True)
class SeatPolicyAssignment:
    policy: InteractivePolicy
    policy_version: int = 0
    greedy: bool = False
    name: str | None = None


@dataclass(frozen=True)
class OpponentPoolEntry:
    policy: InteractivePolicy
    policy_version: int = 0
    greedy: bool = True
    weight: float = 1.0
    name: str | None = None

    def __post_init__(self) -> None:
        if float(self.weight) <= 0.0:
            raise ValueError(f"opponent pool weight must be positive, got {self.weight}")

    def to_assignment(self) -> SeatPolicyAssignment:
        return SeatPolicyAssignment(
            policy=self.policy,
            policy_version=self.policy_version,
            greedy=self.greedy,
            name=self.name,
        )


class OpponentPool:
    def __init__(self, entries: Sequence[OpponentPoolEntry]) -> None:
        if not entries:
            raise ValueError("opponent pool must not be empty")
        self._entries = tuple(entries)
        self._total_weight = float(sum(float(entry.weight) for entry in self._entries))

    @property
    def entries(self) -> tuple[OpponentPoolEntry, ...]:
        return self._entries

    def sample(self, *, rng: random.Random | None = None) -> OpponentPoolEntry:
        source = random if rng is None else rng
        target = source.random() * self._total_weight
        cumulative = 0.0
        for entry in self._entries:
            cumulative += float(entry.weight)
            if target <= cumulative:
                return entry
        return self._entries[-1]


def build_selfplay_seat_assignments(
    *,
    learner_policy: InteractivePolicy,
    learner_policy_version: int = 0,
    learner_greedy: bool = False,
    learner_name: str | None = "learner",
    learner_seats: Sequence[int] = (0, 1, 2, 3),
    seat_count: int = 4,
    opponent_pool: OpponentPool | None = None,
    rng: random.Random | None = None,
) -> tuple[SeatPolicyAssignment, ...]:
    if seat_count <= 0:
        raise ValueError(f"seat_count must be positive, got {seat_count}")

    learner_seat_tuple = tuple(int(seat) for seat in learner_seats)
    if not learner_seat_tuple:
        raise ValueError("learner_seats must not be empty")
    if len(set(learner_seat_tuple)) != len(learner_seat_tuple):
        raise ValueError("learner_seats must not contain duplicates")
    if any(seat < 0 or seat >= seat_count for seat in learner_seat_tuple):
        raise ValueError(
            f"learner_seats must stay within [0, {seat_count - 1}], got {learner_seat_tuple}"
        )
    if opponent_pool is None and len(learner_seat_tuple) != seat_count:
        raise ValueError("opponent_pool is required when learner_seats does not cover every seat")

    learner_assignment = SeatPolicyAssignment(
        policy=learner_policy,
        policy_version=learner_policy_version,
        greedy=learner_greedy,
        name=learner_name,
    )
    learner_seat_set = set(learner_seat_tuple)

    seat_assignments: list[SeatPolicyAssignment] = []
    for seat in range(seat_count):
        if seat in learner_seat_set:
            seat_assignments.append(learner_assignment)
            continue
        if opponent_pool is None:
            raise ValueError(f"missing opponent policy for non-learner seat {seat}")
        opponent_assignment = opponent_pool.sample(rng=rng).to_assignment()
        if opponent_assignment.name is None:
            opponent_assignment = SeatPolicyAssignment(
                policy=opponent_assignment.policy,
                policy_version=opponent_assignment.policy_version,
                greedy=opponent_assignment.greedy,
                name="opponent",
            )
        seat_assignments.append(opponent_assignment)
    return tuple(seat_assignments)


__all__ = [
    "OpponentPool",
    "OpponentPoolEntry",
    "SeatPolicyAssignment",
    "build_selfplay_seat_assignments",
]
