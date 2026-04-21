"""Review export utilities for Xmodel1."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Iterable, Sequence

from mahjong_env.action_space import IDX_TO_TILE_NAME
from xmodel1.schema import XMODEL1_SPECIAL_TYPE_NAMES


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


def special_type_to_action_label(special_type: int) -> str:
    special_type = int(special_type)
    if 0 <= special_type < len(XMODEL1_SPECIAL_TYPE_NAMES):
        return XMODEL1_SPECIAL_TYPE_NAMES[special_type]
    return "padding"


def action_idx_to_action_label(action_idx: int) -> str:
    action_idx = int(action_idx)
    if 0 <= action_idx < 34:
        return tile34_to_action_label(action_idx)
    mapping = {
        34: "reach",
        35: "chi_low",
        36: "chi_mid",
        37: "chi_high",
        38: "pon",
        39: "daiminkan",
        40: "ankan",
        41: "kakan",
        42: "hora",
        43: "ryukyoku",
        44: "none",
    }
    return mapping.get(action_idx, "padding")


def _topk_from_candidates(
    *,
    scores: Sequence[float],
    labels: Sequence[str],
    mask: Sequence[int],
    quality_scores: Sequence[float] | None = None,
    rank_buckets: Sequence[int] | None = None,
    hard_bad_flags: Sequence[int] | None = None,
    k: int,
) -> tuple[ReviewCandidate, ...]:
    indexed = []
    for i, (score, label, active) in enumerate(zip(scores, labels, mask)):
        if int(active) <= 0:
            continue
        indexed.append(
            (
                float(score),
                ReviewCandidate(
                    action=label,
                    score=float(score),
                    quality_score=None if quality_scores is None else float(quality_scores[i]),
                    rank_bucket=None if rank_buckets is None else int(rank_buckets[i]),
                    hard_bad=None if hard_bad_flags is None else bool(hard_bad_flags[i]),
                ),
            )
        )
    indexed.sort(key=lambda item: item[0], reverse=True)
    return tuple(candidate for _score, candidate in indexed[:k])


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
    return _topk_from_candidates(
        scores=scores,
        labels=[tile34_to_action_label(int(tile_id)) for tile_id in candidate_tile_ids],
        mask=candidate_mask,
        quality_scores=quality_scores,
        rank_buckets=rank_buckets,
        hard_bad_flags=hard_bad_flags,
        k=k,
    )


def topk_special_candidates_from_row(
    *,
    scores: Sequence[float],
    special_type_ids: Sequence[int],
    special_mask: Sequence[int],
    quality_scores: Sequence[float] | None = None,
    rank_buckets: Sequence[int] | None = None,
    hard_bad_flags: Sequence[int] | None = None,
    k: int = 3,
) -> tuple[ReviewCandidate, ...]:
    return _topk_from_candidates(
        scores=scores,
        labels=[special_type_to_action_label(int(special_type)) for special_type in special_type_ids],
        mask=special_mask,
        quality_scores=quality_scores,
        rank_buckets=rank_buckets,
        hard_bad_flags=hard_bad_flags,
        k=k,
    )


def topk_response_candidates_from_row(
    *,
    scores: Sequence[float],
    response_action_idx: Sequence[int],
    response_mask: Sequence[int],
    quality_scores: Sequence[float] | None = None,
    hard_bad_flags: Sequence[int] | None = None,
    k: int = 3,
) -> tuple[ReviewCandidate, ...]:
    return _topk_from_candidates(
        scores=scores,
        labels=[action_idx_to_action_label(int(action_idx)) for action_idx in response_action_idx],
        mask=response_mask,
        quality_scores=quality_scores,
        rank_buckets=None,
        hard_bad_flags=hard_bad_flags,
        k=k,
    )


__all__ = [
    "ReviewCandidate",
    "ReviewRecord",
    "action_idx_to_action_label",
    "export_review_records",
    "special_type_to_action_label",
    "tile34_to_action_label",
    "topk_candidates_from_row",
    "topk_response_candidates_from_row",
    "topk_special_candidates_from_row",
]
