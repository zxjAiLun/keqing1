#!/usr/bin/env python3
"""Export structured Xmodel1 review results from cached npz samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from torch.utils.data import DataLoader


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Review Xmodel1 checkpoint on cached Xmodel1 samples")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", nargs="+", required=True, help="npz files or directories")
    parser.add_argument("--output", required=True)
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))

    from xmodel1.adapter import Xmodel1Adapter
    from xmodel1.cached_dataset import Xmodel1DiscardDataset
    from xmodel1.review_export import ReviewRecord, export_review_records
    from xmodel1.schema import (
        XMODEL1_SAMPLE_TYPE_CALL,
        XMODEL1_SAMPLE_TYPE_DISCARD,
        XMODEL1_SAMPLE_TYPE_HORA,
        XMODEL1_SAMPLE_TYPE_RIICHI,
    )

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
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        collate_fn=Xmodel1DiscardDataset.collate,
        num_workers=0,
    )

    def _category(sample_type: int) -> str:
        if sample_type == XMODEL1_SAMPLE_TYPE_DISCARD:
            return "xmodel1-discard-review"
        if sample_type == XMODEL1_SAMPLE_TYPE_RIICHI:
            return "xmodel1-riichi-review"
        if sample_type == XMODEL1_SAMPLE_TYPE_CALL:
            return "xmodel1-call-review"
        if sample_type == XMODEL1_SAMPLE_TYPE_HORA:
            return "xmodel1-hora-review"
        return "xmodel1-review"

    def _iter_records():
        exported = 0
        for batch in loader:
            scored = adapter.score_batch(batch)
            batch_size = int(batch["state_tile_feat"].shape[0])
            for idx in range(batch_size):
                review = adapter.scored_row_to_review(scored, idx, k=args.topk)
                sample_type = int(batch["sample_type"][idx].item())
                exported += 1
                yield ReviewRecord(
                    sample_id=str(batch["sample_id"][idx]),
                    replay_id=str(batch["replay_id"][idx]),
                    category=_category(sample_type),
                    chosen_action=review.chosen_action,
                    top_k=review.top_k,
                    note=json.dumps(
                        {
                            "win_prob": round(review.win_prob, 6),
                            "dealin_prob": round(review.dealin_prob, 6),
                            "pts_given_win": round(review.pts_given_win, 6),
                            "pts_given_dealin": round(review.pts_given_dealin, 6),
                            "composed_ev": round(review.composed_ev, 6),
                        },
                        ensure_ascii=False,
                    ),
                )
        if exported <= 0:
            raise RuntimeError("review dataset produced no rows")

    records = _iter_records()
    export_review_records(records, args.output)
    line_count = sum(1 for line in Path(args.output).read_text(encoding="utf-8").splitlines() if line.strip())
    print(f"exported {line_count} review rows -> {args.output}")


if __name__ == "__main__":
    main()
