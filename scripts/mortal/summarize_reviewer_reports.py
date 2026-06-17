#!/usr/bin/env python3
"""Summarize Mortal reviewer report JSON files into model comparison tables."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        action="append",
        required=True,
        help="LABEL:PLAYER_ID:PATH, e.g. 70k:0:report.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def _action_label(action: dict[str, Any] | None) -> str:
    if not action:
        return ""
    typ = str(action.get("type", ""))
    if typ == "dahai":
        suffix = "*" if action.get("tsumogiri") else ""
        return f"dahai {action.get('pai')}{suffix}"
    if typ in {"chi", "pon", "ankan", "kakan", "daiminkan"}:
        pai = action.get("pai")
        return f"{typ} {pai}" if pai is not None else typ
    return typ


def _action_kind(action: dict[str, Any] | None) -> str:
    if not action:
        return ""
    return str(action.get("type", ""))


def _entry_loss(entry: dict[str, Any]) -> tuple[float | None, float | None, float | None, dict[str, Any] | None]:
    details = list(entry.get("details") or [])
    if not details:
        return None, None, None, None
    best = details[0]
    actual_index = entry.get("actual_index")
    actual = None
    if isinstance(actual_index, int) and 0 <= actual_index < len(details):
        actual = details[actual_index]
    if actual is None:
        actual_action = json.dumps(entry.get("actual"), sort_keys=True, ensure_ascii=False)
        for detail in details:
            if json.dumps(detail.get("action"), sort_keys=True, ensure_ascii=False) == actual_action:
                actual = detail
                break
    if actual is None:
        return None, best.get("q_value"), None, best
    best_q = best.get("q_value")
    actual_q = actual.get("q_value")
    if best_q is None or actual_q is None:
        return None, best_q, actual_q, best
    return float(best_q) - float(actual_q), float(best_q), float(actual_q), best


def _parse_report_spec(spec: str) -> tuple[str, int, Path]:
    parts = spec.split(":", 2)
    if len(parts) != 3:
        raise ValueError(f"--report must be LABEL:PLAYER_ID:PATH, got {spec!r}")
    label, player_id, path = parts
    return label, int(player_id), Path(path)


def _collect_entries(label: str, player_id: int, report_path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for kyoku in report.get("review", {}).get("kyokus", []):
        end_status = kyoku.get("end_status") or []
        for entry_index, entry in enumerate(kyoku.get("entries") or []):
            q_loss, best_q, actual_q, best = _entry_loss(entry)
            expected = entry.get("expected")
            actual = entry.get("actual")
            row = {
                "model": label,
                "player_id": player_id,
                "report_path": str(report_path),
                "kyoku": int(kyoku.get("kyoku", 0)),
                "honba": int(kyoku.get("honba", 0)),
                "entry_index": entry_index,
                "junme": entry.get("junme"),
                "tiles_left": entry.get("tiles_left"),
                "shanten": entry.get("shanten"),
                "is_equal": bool(entry.get("is_equal")),
                "expected": _action_label(expected),
                "actual": _action_label(actual),
                "expected_kind": _action_kind(expected),
                "actual_kind": _action_kind(actual),
                "best_action": _action_label(best.get("action") if best else expected),
                "q_loss": q_loss,
                "best_q": best_q,
                "actual_q": actual_q,
                "best_prob": best.get("prob") if best else None,
                "actual_index": entry.get("actual_index"),
                "at_self_chi_pon": bool(entry.get("at_self_chi_pon")),
                "at_self_riichi": bool(entry.get("at_self_riichi")),
                "at_opponent_kakan": bool(entry.get("at_opponent_kakan")),
                "end_status_type": str(end_status[0].get("type", "")) if end_status else "",
            }
            rows.append(row)
    return report, rows


def _summarize(label: str, player_id: int, report: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    reviewed = len(rows)
    mismatches = [row for row in rows if not row["is_equal"]]
    losses = [float(row["q_loss"]) for row in rows if row["q_loss"] is not None]
    mismatch_losses = [float(row["q_loss"]) for row in mismatches if row["q_loss"] is not None]
    transition_counts = Counter(f"{row['actual_kind']}->{row['expected_kind']}" for row in mismatches)
    flag_counts = {
        "after_call_entries": sum(1 for row in rows if row["at_self_chi_pon"]),
        "after_riichi_entries": sum(1 for row in rows if row["at_self_riichi"]),
        "call_mismatches": sum(1 for row in mismatches if row["expected_kind"] in {"chi", "pon"} or row["actual_kind"] in {"chi", "pon", "none"}),
    }
    buckets = {
        "loss_ge_1.0": sum(1 for value in mismatch_losses if value >= 1.0),
        "loss_ge_0.5": sum(1 for value in mismatch_losses if value >= 0.5),
        "loss_ge_0.2": sum(1 for value in mismatch_losses if value >= 0.2),
        "loss_gt_0": sum(1 for value in mismatch_losses if value > 0.0),
    }
    return {
        "model": label,
        "player_id": player_id,
        "engine": report.get("engine"),
        "version": report.get("version"),
        "review_time": report.get("review_time"),
        "reviewed": reviewed,
        "matches": sum(1 for row in rows if row["is_equal"]),
        "mismatches": len(mismatches),
        "match_rate": (reviewed - len(mismatches)) / reviewed if reviewed else None,
        "rating": report.get("review", {}).get("rating"),
        "avg_loss_all": mean(losses) if losses else None,
        "avg_loss_mismatch": mean(mismatch_losses) if mismatch_losses else None,
        "sum_loss": sum(losses),
        "sum_loss_mismatch": sum(mismatch_losses),
        **buckets,
        **flag_counts,
        "transition_counts": dict(sorted(transition_counts.items())),
    }


def _format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _write_markdown(path: Path, summaries: list[dict[str, Any]], top_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# 4.1b Reviewer Comparison",
        "",
        "| model | player | reviewed | match rate | rating | avg q-loss | mismatch q-loss | loss >=0.5 | loss >=0.2 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summaries:
        lines.append(
            " | ".join(
                [
                    f"| {row['model']}",
                    str(row["player_id"]),
                    str(row["reviewed"]),
                    _format_float(row["match_rate"]),
                    _format_float(row["rating"]),
                    _format_float(row["avg_loss_all"]),
                    _format_float(row["avg_loss_mismatch"]),
                    str(row["loss_ge_0.5"]),
                    f"{row['loss_ge_0.2']} |",
                ]
            )
        )
    lines.extend(
        [
            "",
            "## Top q-loss mismatches",
            "",
            "| model | kyoku | junme | shanten | expected | actual | q-loss | best q | actual q |",
            "| --- | ---: | ---: | ---: | --- | --- | ---: | ---: | ---: |",
        ]
    )
    for row in top_rows:
        lines.append(
            " | ".join(
                [
                    f"| {row['model']}",
                    f"{row['kyoku']}-{row['honba']}",
                    str(row.get("junme")),
                    str(row.get("shanten")),
                    str(row["expected"]),
                    str(row["actual"]),
                    _format_float(row["q_loss"]),
                    _format_float(row["best_q"]),
                    f"{_format_float(row['actual_q'])} |",
                ]
            )
        )
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    all_entries: list[dict[str, Any]] = []
    grouped_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for spec in args.report:
        label, player_id, report_path = _parse_report_spec(spec)
        report, entries = _collect_entries(label, player_id, report_path)
        summaries.append(_summarize(label, player_id, report, entries))
        all_entries.extend(entries)
        grouped_entries[label].extend(entries)

    top_rows = sorted(
        [row for row in all_entries if not row["is_equal"] and row["q_loss"] is not None],
        key=lambda row: float(row["q_loss"]),
        reverse=True,
    )[: max(0, int(args.top))]

    (args.output_dir / "reviewer_comparison_summary.json").write_text(
        json.dumps({"summaries": summaries, "top_mismatches": top_rows}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with (args.output_dir / "reviewer_entries.jsonl").open("w", encoding="utf-8") as handle:
        for row in all_entries:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    with (args.output_dir / "reviewer_comparison_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "model",
            "player_id",
            "reviewed",
            "matches",
            "mismatches",
            "match_rate",
            "rating",
            "avg_loss_all",
            "avg_loss_mismatch",
            "loss_ge_0.5",
            "loss_ge_0.2",
            "sum_loss",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summaries:
            writer.writerow({field: row.get(field) for field in fields})
    _write_markdown(args.output_dir / "reviewer_comparison.md", summaries, top_rows)
    print(json.dumps({"output_dir": str(args.output_dir), "summaries": summaries}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
