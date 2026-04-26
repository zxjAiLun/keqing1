"""Review export utilities for Xmodel1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

from mahjong_env.action_space import IDX_TO_TILE_NAME


@dataclass(frozen=True)
class ReviewCandidate:
    action: str
    score: float
    quality_score: float | None = None
    rank_bucket: int | None = None
    hard_bad: bool | None = None


@dataclass(frozen=True)
class ReviewRecord:
    sample_id: str
    replay_id: str
    category: str
    chosen_action: str
    top_k: tuple[ReviewCandidate, ...]
    note: str = ""


def export_review_records(records: Iterable[ReviewRecord], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for record in records:
            payload = asdict(record)
            payload["top_k"] = [asdict(item) for item in record.top_k]
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


_TILE34_LABELS = tuple(IDX_TO_TILE_NAME[i] for i in range(34))


def tile34_to_action_label(tile34: int) -> str:
    if 0 <= int(tile34) < len(_TILE34_LABELS):
        return f"dahai:{_TILE34_LABELS[int(tile34)]}"
    return "padding"


def topk_candidates_from_row(
    *,
    scores: Sequence[float],
    candidate_tile_ids: Sequence[int],
    candidate_mask: Sequence[int],
    quality_scores: Sequence[float] | None = None,
    rank_buckets: Sequence[int] | None = None,
    hard_bad_flags: Sequence[int] | None = None,
    k: int = 3,
) -> tuple[ReviewCandidate, ...]:
    indexed = []
    for i, (score, tile_id, mask) in enumerate(zip(scores, candidate_tile_ids, candidate_mask)):
        if int(mask) <= 0:
            continue
        indexed.append(
            (
                float(score),
                ReviewCandidate(
                    action=tile34_to_action_label(int(tile_id)),
                    score=float(score),
                    quality_score=None if quality_scores is None else float(quality_scores[i]),
                    rank_bucket=None if rank_buckets is None else int(rank_buckets[i]),
                    hard_bad=None if hard_bad_flags is None else bool(hard_bad_flags[i]),
                ),
            )
        )
    indexed.sort(key=lambda item: item[0], reverse=True)
    return tuple(candidate for _score, candidate in indexed[:k])


__all__ = [
    "ReviewCandidate",
    "ReviewRecord",
    "export_review_records",
    "tile34_to_action_label",
    "topk_candidates_from_row",
]
