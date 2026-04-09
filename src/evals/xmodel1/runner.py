"""Case benchmark runner for Xmodel1."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class CaseEvalRecord:
    case_id: str
    category: str
    chosen_action: str
    preferred_actions: tuple[str, ...]
    acceptable_actions: tuple[str, ...]
    unacceptable_actions: tuple[str, ...]


@dataclass(frozen=True)
class CaseEvalSummary:
    total: int
    preferred: int
    acceptable: int
    bad: int
    blunder: int

    @property
    def preferred_rate(self) -> float:
        return self.preferred / self.total if self.total else 0.0

    @property
    def acceptable_rate(self) -> float:
        return self.acceptable / self.total if self.total else 0.0

    @property
    def bad_rate(self) -> float:
        return self.bad / self.total if self.total else 0.0

    @property
    def blunder_rate(self) -> float:
        return self.blunder / self.total if self.total else 0.0


def load_case_records(path: str | Path) -> list[dict]:
    records: list[dict] = []
    p = Path(path)
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _normalize_action(value) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    raise TypeError(f"unsupported action value type: {type(value)!r}")


def record_from_case(case: dict, *, chosen_action) -> CaseEvalRecord:
    return CaseEvalRecord(
        case_id=str(case["case_id"]),
        category=str(case["category"]),
        chosen_action=_normalize_action(chosen_action),
        preferred_actions=tuple(_normalize_action(v) for v in case.get("preferred_actions", [])),
        acceptable_actions=tuple(_normalize_action(v) for v in case.get("acceptable_actions", [])),
        unacceptable_actions=tuple(_normalize_action(v) for v in case.get("unacceptable_actions", [])),
    )


def _score_record(record: CaseEvalRecord) -> str:
    if record.chosen_action in record.unacceptable_actions:
        return "blunder"
    if record.chosen_action in record.preferred_actions:
        return "preferred"
    if record.chosen_action in record.acceptable_actions:
        return "acceptable"
    return "bad"


def evaluate_case_records(records: Iterable[CaseEvalRecord]) -> CaseEvalSummary:
    counts = {"preferred": 0, "acceptable": 0, "bad": 0, "blunder": 0}
    total = 0
    for record in records:
        verdict = _score_record(record)
        counts[verdict] += 1
        total += 1
    return CaseEvalSummary(
        total=total,
        preferred=counts["preferred"],
        acceptable=counts["acceptable"],
        bad=counts["bad"],
        blunder=counts["blunder"],
    )


__all__ = [
    "CaseEvalRecord",
    "CaseEvalSummary",
    "load_case_records",
    "record_from_case",
    "evaluate_case_records",
]
