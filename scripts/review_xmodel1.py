#!/usr/bin/env python3
"""Export structured Xmodel1 review results from cached npz samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review Xmodel1 checkpoint on cached Xmodel1 samples")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", nargs="+", required=True, help="npz files or directories")
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))

    from evals.xmodel1.review_export import ReviewRecord, export_review_records
    from xmodel1.adapter import Xmodel1Adapter
    from xmodel1.cached_dataset import Xmodel1DiscardDataset

    args = _parse_args()
    data_paths: list[Path] = []
    for item in args.data:
        p = Path(item)
        if p.is_dir():
            data_paths.extend(sorted(p.glob("*.npz")))
        else:
            data_paths.append(p)
    if not data_paths:
        raise RuntimeError("no Xmodel1 cache files found for review")

    adapter = Xmodel1Adapter.from_checkpoint(args.checkpoint, device="cpu")
    dataset = Xmodel1DiscardDataset(data_paths, shuffle=False, buffer_size=1024, seed=42)
    rows = list(dataset)
    if not rows:
        raise RuntimeError("review dataset produced no rows")
    batch = Xmodel1DiscardDataset.collate(rows)
    scored = adapter.score_batch(batch)
    records: list[ReviewRecord] = []
    for idx in range(len(rows)):
        review = adapter.scored_row_to_review(scored, idx, k=args.topk)
        records.append(
            ReviewRecord(
                sample_id=str(idx),
                replay_id="cached",
                category="xmodel1-review",
                chosen_action=review.chosen_action,
                top_k=review.top_k,
                note=(
                    f"score_delta={review.score_delta:.4f} "
                    f"win_prob={review.win_prob:.4f} "
                    f"dealin_prob={review.dealin_prob:.4f} "
                    f"offense_quality={review.offense_quality:.4f}"
                ),
            )
        )
    export_review_records(records, args.output)
    print(f"exported {len(records)} review rows -> {args.output}")


if __name__ == "__main__":
    main()
