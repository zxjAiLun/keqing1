#!/usr/bin/env python3
"""Summarize keqingv4 train_log.jsonl with emphasis on B3 metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SLICE_KEYS = [
    "val_reach_chosen_rate",
    "val_meld_chosen_rate",
    "val_reach_slice_acc",
    "val_meld_slice_acc",
    "val_special_slice_acc",
    "val_typed_rank_loss",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize keqingv4 training log")
    parser.add_argument("path", help="train_log.jsonl path")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _best_row(rows: list[dict], key: str, mode: str) -> dict | None:
    valid = [row for row in rows if row.get(key) is not None]
    if not valid:
        return None
    reverse = mode == "max"
    return sorted(valid, key=lambda row: float(row[key]), reverse=reverse)[0]


def main() -> None:
    args = _parse_args()
    path = Path(args.path)
    rows = _load_rows(path)
    if not rows:
        raise RuntimeError(f"empty training log: {path}")

    last = rows[-1]
    print(f"# {path}")
    print(f"epochs_logged: {len(rows)}")
    print(
        f"last_epoch={last.get('epoch')} "
        f"train_objective={last.get('train_objective'):.4f} "
        f"val_objective={last.get('val_objective'):.4f}"
    )
    for key in SLICE_KEYS:
        if key in last:
            value = last.get(key)
            if value is not None:
                print(f"{key}={float(value):.4f}")

    objective_best = _best_row(rows, "val_objective", "min")
    if objective_best is not None:
        print(
            f"best_val_objective: epoch={objective_best['epoch']} "
            f"value={float(objective_best['val_objective']):.4f}"
        )

    for key in ("val_reach_slice_acc", "val_meld_slice_acc", "val_special_slice_acc"):
        best = _best_row(rows, key, "max")
        if best is not None:
            print(f"best_{key}: epoch={best['epoch']} value={float(best[key]):.4f}")

    best_rank = _best_row(rows, "val_typed_rank_loss", "min")
    if best_rank is not None:
        print(f"best_val_typed_rank_loss: epoch={best_rank['epoch']} value={float(best_rank['val_typed_rank_loss']):.4f}")


if __name__ == "__main__":
    main()

