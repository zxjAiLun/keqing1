#!/usr/bin/env python3
"""Compare two keqingv4 train logs and propose the next B3 iteration move."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


KEYS = [
    "val_objective",
    "val_typed_rank_loss",
    "val_reach_chosen_rate",
    "val_meld_chosen_rate",
    "val_reach_slice_acc",
    "val_meld_slice_acc",
    "val_special_slice_acc",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare baseline and probe keqingv4 train logs")
    parser.add_argument("baseline", help="baseline train_log.jsonl")
    parser.add_argument("probe", help="probe train_log.jsonl")
    parser.add_argument("--json", action="store_true", help="print JSON output")
    return parser.parse_args()


def _load_rows(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise RuntimeError(f"empty training log: {path}")
    return rows


def _best(rows: list[dict], key: str, mode: str) -> float | None:
    vals = [row.get(key) for row in rows if row.get(key) is not None]
    if not vals:
        return None
    vals = [float(v) for v in vals]
    return min(vals) if mode == "min" else max(vals)


def _extract_summary(rows: list[dict]) -> dict[str, float | None]:
    summary: dict[str, float | None] = {}
    for key in KEYS:
        mode = "min" if "loss" in key or key == "val_objective" else "max"
        summary[key] = _best(rows, key, mode)
    return summary


def _delta(probe: float | None, base: float | None) -> float | None:
    if probe is None or base is None:
        return None
    return float(probe) - float(base)


def _recommend(base: dict[str, float | None], probe: dict[str, float | None], delta: dict[str, float | None]) -> list[str]:
    recs: list[str] = []

    special_acc_delta = delta.get("val_special_slice_acc")
    rank_loss_delta = delta.get("val_typed_rank_loss")
    reach_rate_delta = delta.get("val_reach_chosen_rate")
    meld_rate_delta = delta.get("val_meld_chosen_rate")
    objective_delta = delta.get("val_objective")

    if special_acc_delta is not None and special_acc_delta > 0.01 and rank_loss_delta is not None and rank_loss_delta < -0.01:
        recs.append("B3 signal looks healthy: keep typed ranking loss on and scale the probe up before touching the model shell again.")
    if special_acc_delta is not None and special_acc_delta <= 0.01 and rank_loss_delta is not None and rank_loss_delta < -0.01:
        recs.append("Ranking loss is optimizing but slice accuracy is not moving: next iteration should raise opportunity weighting or add sharper pairwise targets, not more generic heads.")
    if special_acc_delta is not None and special_acc_delta < -0.01:
        recs.append("Special slice accuracy regressed: reduce aggressive opportunity weighting or inspect whether CE is fighting the typed objective.")
    if reach_rate_delta is not None and reach_rate_delta < -0.02:
        recs.append("Reach chosen rate dropped: next iteration should increase reach-side weighting or inspect dama-vs-reach comparison quality.")
    if meld_rate_delta is not None and meld_rate_delta < -0.02:
        recs.append("Meld chosen rate dropped: next iteration should strengthen call-vs-none pressure or audit call summary semantics.")
    if objective_delta is not None and objective_delta > 0.05 and not recs:
        recs.append("Global objective got worse without clear slice benefit: hold this probe and avoid scaling it up until the slice-level cause is understood.")
    if not recs:
        recs.append("No strong signal either way: run a slightly longer probe or compare on real boundary-case review slices before changing architecture.")

    return recs


def main() -> None:
    args = _parse_args()
    baseline_rows = _load_rows(Path(args.baseline))
    probe_rows = _load_rows(Path(args.probe))
    baseline_summary = _extract_summary(baseline_rows)
    probe_summary = _extract_summary(probe_rows)
    delta_summary = {key: _delta(probe_summary[key], baseline_summary[key]) for key in KEYS}
    recommendations = _recommend(baseline_summary, probe_summary, delta_summary)

    payload = {
        "baseline": baseline_summary,
        "probe": probe_summary,
        "delta": delta_summary,
        "recommendations": recommendations,
    }

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    print("# keqingv4 B3 comparison")
    for key in KEYS:
        print(
            f"{key}: baseline={baseline_summary[key]} "
            f"probe={probe_summary[key]} delta={delta_summary[key]}"
        )
    print("\nnext-iteration recommendations:")
    for rec in recommendations:
        print(f"- {rec}")


if __name__ == "__main__":
    main()

