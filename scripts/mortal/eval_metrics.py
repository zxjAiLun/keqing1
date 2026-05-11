"""Shared metric export helpers for Mortal evaluation scripts."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

METRICS_SCHEMA = "keqing.mortal.eval.metrics.v1"
TENHOU_RANK_POINTS = (90.0, 45.0, 0.0, -135.0)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def summarize_rank_counts(rank_counts: Sequence[int | float]) -> dict[str, Any]:
    counts = [int(value) for value in rank_counts]
    if len(counts) != 4:
        raise ValueError(f"rank_counts must have length 4, got {len(counts)}")
    games = sum(counts)
    if games <= 0:
        return {
            "games": 0,
            "rank_counts": counts,
            "avg_rank": None,
            "avg_rank_pt": None,
        }
    return {
        "games": games,
        "rank_counts": counts,
        "avg_rank": sum((idx + 1) * count for idx, count in enumerate(counts)) / games,
        "avg_rank_pt": sum(TENHOU_RANK_POINTS[idx] * count for idx, count in enumerate(counts)) / games,
    }


def build_metrics_document(
    *,
    run: Mapping[str, Any],
    metrics: Mapping[str, Any],
    artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema": METRICS_SCHEMA,
        "created_at": utc_now_iso(),
        "run": dict(run),
        "metrics": dict(metrics),
        "artifacts": dict(artifacts or {}),
    }


def write_metrics(path: str | Path, document: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
