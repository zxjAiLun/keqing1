"""Unified slice report helpers for Xmodel1 review/train acceptance."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from torch.utils.data import DataLoader

from xmodel1.schema import (
    XMODEL1_CHI_SPECIAL_TYPES,
    XMODEL1_KAN_SPECIAL_TYPES,
    XMODEL1_SPECIAL_TYPE_CHI_HIGH,
    XMODEL1_SPECIAL_TYPE_CHI_LOW,
    XMODEL1_SPECIAL_TYPE_CHI_MID,
    XMODEL1_SPECIAL_TYPE_HORA,
    XMODEL1_SPECIAL_TYPE_NONE,
    XMODEL1_SPECIAL_TYPE_PON,
    XMODEL1_SPECIAL_TYPE_REACH,
)

REACH_ACTION_IDX = 34


@dataclass(frozen=True)
class SliceStat:
    name: str
    count: int
    correct: int
    accuracy: float


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def chosen_special_type_ids(
    special_candidate_type_id,
    chosen_special_candidate_idx,
) -> np.ndarray:
    type_ids = _to_numpy(special_candidate_type_id)
    chosen_idx = _to_numpy(chosen_special_candidate_idx).astype(np.int64)
    out = np.full((chosen_idx.shape[0],), -1, dtype=np.int16)
    valid = (chosen_idx >= 0) & (chosen_idx < type_ids.shape[1])
    if np.any(valid):
        rows = np.nonzero(valid)[0]
        out[rows] = type_ids[rows, chosen_idx[rows]]
    return out


def build_slice_masks(
    *,
    action_idx_target,
    special_candidate_type_id,
    chosen_special_candidate_idx,
) -> dict[str, np.ndarray]:
    action_targets = _to_numpy(action_idx_target).astype(np.int64)
    chosen_special_types = chosen_special_type_ids(
        special_candidate_type_id,
        chosen_special_candidate_idx,
    )
    call_types = set(XMODEL1_CHI_SPECIAL_TYPES) | {XMODEL1_SPECIAL_TYPE_PON} | set(XMODEL1_KAN_SPECIAL_TYPES)
    return {
        "reach": (action_targets == REACH_ACTION_IDX) | (chosen_special_types == XMODEL1_SPECIAL_TYPE_REACH),
        "call": np.isin(chosen_special_types, np.array(sorted(call_types), dtype=np.int16)),
        "none": chosen_special_types == XMODEL1_SPECIAL_TYPE_NONE,
        "hora": chosen_special_types == XMODEL1_SPECIAL_TYPE_HORA,
        "chi_low": chosen_special_types == XMODEL1_SPECIAL_TYPE_CHI_LOW,
        "chi_mid": chosen_special_types == XMODEL1_SPECIAL_TYPE_CHI_MID,
        "chi_high": chosen_special_types == XMODEL1_SPECIAL_TYPE_CHI_HIGH,
    }


def build_slice_report(
    *,
    action_idx_target,
    action_idx_pred,
    special_candidate_type_id,
    chosen_special_candidate_idx,
) -> dict[str, SliceStat]:
    targets = _to_numpy(action_idx_target).astype(np.int64)
    preds = _to_numpy(action_idx_pred).astype(np.int64)
    masks = build_slice_masks(
        action_idx_target=targets,
        special_candidate_type_id=special_candidate_type_id,
        chosen_special_candidate_idx=chosen_special_candidate_idx,
    )
    report: dict[str, SliceStat] = {}
    for name, mask in masks.items():
        mask = np.asarray(mask, dtype=bool)
        count = int(mask.sum())
        correct = int(((preds == targets) & mask).sum())
        report[name] = SliceStat(
            name=name,
            count=count,
            correct=correct,
            accuracy=(correct / count) if count else 0.0,
        )
    return report


def merge_slice_reports(*reports: Mapping[str, SliceStat]) -> dict[str, SliceStat]:
    merged_counts: dict[str, tuple[int, int]] = {}
    for report in reports:
        for name, stat in report.items():
            prev_count, prev_correct = merged_counts.get(name, (0, 0))
            merged_counts[name] = (prev_count + int(stat.count), prev_correct + int(stat.correct))
    return {
        name: SliceStat(
            name=name,
            count=count,
            correct=correct,
            accuracy=(correct / count) if count else 0.0,
        )
        for name, (count, correct) in merged_counts.items()
    }


def format_slice_report(report: Mapping[str, SliceStat]) -> dict[str, dict[str, float | int | str]]:
    return {name: asdict(stat) for name, stat in report.items()}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export unified Xmodel1 slice accuracy report")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", nargs="+", required=True, help="npz files or directories")
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    return parser.parse_args()


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    import sys

    sys.path.insert(0, str(root / "src"))
    from xmodel1.adapter import Xmodel1Adapter
    from xmodel1.cached_dataset import Xmodel1DiscardDataset

    args = _parse_args()
    data_paths: list[Path] = []
    for item in args.data:
        path = Path(item)
        if path.is_dir():
            data_paths.extend(sorted(path.glob("*.npz")))
        else:
            data_paths.append(path)
    if not data_paths:
        raise RuntimeError("no Xmodel1 cache files found for slice report")

    adapter = Xmodel1Adapter.from_checkpoint(args.checkpoint, device="cpu")
    dataset = Xmodel1DiscardDataset(data_paths, shuffle=False, buffer_size=1024, seed=42)
    loader = DataLoader(
        dataset,
        batch_size=max(1, int(args.batch_size)),
        collate_fn=Xmodel1DiscardDataset.collate,
        num_workers=0,
    )
    partial_reports: list[dict[str, SliceStat]] = []
    for batch in loader:
        scored = adapter.score_batch(batch)
        partial_reports.append(
            build_slice_report(
                action_idx_target=batch["action_idx_target"],
                action_idx_pred=scored["action_logits"].argmax(dim=-1),
                special_candidate_type_id=batch["special_candidate_type_id"],
                chosen_special_candidate_idx=batch["chosen_special_candidate_idx"],
            )
        )
    if not partial_reports:
        raise RuntimeError("slice report dataset produced no rows")
    report = merge_slice_reports(*partial_reports)
    payload = format_slice_report(report)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"exported xmodel1 slice report -> {output_path}")


__all__ = [
    "SliceStat",
    "build_slice_masks",
    "build_slice_report",
    "chosen_special_type_ids",
    "format_slice_report",
    "merge_slice_reports",
]


if __name__ == "__main__":
    main()
