#!/usr/bin/env python3
"""Parse archived mjai reviewer reports into teacher-probe decision tables."""

from __future__ import annotations

import argparse
from collections import Counter
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
import statistics
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_REPORT_MANIFEST = Path(
    "artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_smoke/report_manifest.jsonl"
)
DEFAULT_OUTPUT_DIR = Path("artifacts/experiments/reviewer_teacher_probe_2026_05/R1_parser_smoke")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-manifest", type=Path, default=DEFAULT_REPORT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--high-confidence-prob", type=float, default=0.60)
    parser.add_argument("--high-confidence-margin", type=float, default=0.20)
    parser.add_argument("--top-k", type=int, default=50)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key)) for key in fieldnames})


def csv_value(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    return value


def source_input_key(path: str) -> str:
    name = Path(path).name
    if name.endswith(".tenhou6.json"):
        return name[: -len(".tenhou6.json")]
    return Path(name).stem


def network_key(network: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", network).strip("_")


def action_key(action: Any) -> str:
    return json.dumps(action, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def action_family(action: Any) -> str:
    if not isinstance(action, dict):
        return "unknown"
    action_type = str(action.get("type", "unknown"))
    if action_type == "dahai":
        return "discard"
    if action_type in {"chi", "pon", "daiminkan"}:
        return "call"
    if action_type == "reach":
        return "riichi"
    if action_type in {"ankan", "kakan"}:
        return "kan"
    if action_type == "none":
        return "pass"
    if action_type in {"hora", "ryukyoku"}:
        return action_type
    return action_type


def detail_lookup(details: list[dict[str, Any]], action: Any) -> dict[str, Any] | None:
    target = action_key(action)
    for detail in details:
        if action_key(detail.get("action")) == target:
            return detail
    return None


def sorted_details(details: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(details, key=lambda item: float(item.get("prob", float("-inf"))), reverse=True)


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def build_decision_rows(report_manifest: Path) -> list[dict[str, Any]]:
    manifest_rows = load_jsonl(report_manifest)
    rows: list[dict[str, Any]] = []
    for manifest_row in manifest_rows:
        report_path = Path(str(manifest_row["report_json_path"]))
        report = json.loads(report_path.read_text(encoding="utf-8"))
        network = str(manifest_row["network"])
        report_id = str(manifest_row["report_id"])
        player_id = int(report.get("player_id", manifest_row["target_player"]))
        input_key = source_input_key(str(manifest_row["source_tenhou6_path"]))
        for kyoku_index, kyoku in enumerate(report["review"].get("kyokus", [])):
            for entry_index, entry in enumerate(kyoku.get("entries", [])):
                details = list(entry.get("details", []))
                details_by_prob = sorted_details(details)
                top1 = details_by_prob[0] if details_by_prob else None
                top2 = details_by_prob[1] if len(details_by_prob) > 1 else None
                actual = entry.get("actual")
                expected = entry.get("expected")
                actual_detail = detail_lookup(details, actual)
                expected_detail = detail_lookup(details, expected)
                top1_prob = float_or_none(top1.get("prob") if top1 else None)
                top2_prob = float_or_none(top2.get("prob") if top2 else None)
                top1_q = float_or_none(top1.get("q_value") if top1 else None)
                top2_q = float_or_none(top2.get("q_value") if top2 else None)
                actual_prob = float_or_none(actual_detail.get("prob") if actual_detail else None)
                expected_prob = float_or_none(expected_detail.get("prob") if expected_detail else None)
                prob_margin = None if top1_prob is None or top2_prob is None else top1_prob - top2_prob
                q_margin = None if top1_q is None or top2_q is None else top1_q - top2_q
                is_match = action_key(actual) == action_key(expected)
                rows.append(
                    {
                        "schema": "keqing.mortal.reviewer_teacher_decision.v1",
                        "source_report_path": str(report_path),
                        "report_id": report_id,
                        "network": network,
                        "player_id": player_id,
                        "source_tenhou6_path": str(manifest_row["source_tenhou6_path"]),
                        "source_input_key": input_key,
                        "source_log": manifest_row.get("source_log"),
                        "kyoku_index": kyoku_index,
                        "kyoku": kyoku.get("kyoku"),
                        "honba": kyoku.get("honba"),
                        "entry_index": entry_index,
                        "junme": entry.get("junme"),
                        "tiles_left": entry.get("tiles_left"),
                        "actual_action": actual,
                        "expected_action": expected,
                        "actual_action_key": action_key(actual),
                        "expected_action_key": action_key(expected),
                        "is_match": is_match,
                        "detail_count": len(details),
                        "expected_prob": expected_prob,
                        "actual_prob": actual_prob,
                        "expected_q": float_or_none(expected_detail.get("q_value") if expected_detail else None),
                        "actual_q": float_or_none(actual_detail.get("q_value") if actual_detail else None),
                        "top1_prob": top1_prob,
                        "top2_prob": top2_prob,
                        "prob_margin": prob_margin,
                        "top1_q": top1_q,
                        "top2_q": top2_q,
                        "q_margin": q_margin,
                        "top1_action": top1.get("action") if top1 else None,
                        "top2_action": top2.get("action") if top2 else None,
                        "action_family_actual": action_family(actual),
                        "action_family_expected": action_family(expected),
                        "shanten": entry.get("shanten"),
                        "at_self_chi_pon": entry.get("at_self_chi_pon"),
                        "at_self_riichi": entry.get("at_self_riichi"),
                        "at_furiten": entry.get("at_furiten"),
                    }
                )
    return rows


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def summarize_rows(
    rows: list[dict[str, Any]],
    *,
    high_confidence_prob: float,
    high_confidence_margin: float,
) -> dict[str, Any]:
    by_network: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_network[str(row["network"])].append(row)

    network_summaries: dict[str, Any] = {}
    for network, network_rows in sorted(by_network.items()):
        total = len(network_rows)
        matches = sum(1 for row in network_rows if row["is_match"])
        high_conf = [
            row
            for row in network_rows
            if (not row["is_match"])
            and (row.get("expected_prob") is not None)
            and float(row["expected_prob"]) >= high_confidence_prob
            and (row.get("prob_margin") is not None)
            and float(row["prob_margin"]) >= high_confidence_margin
        ]
        actual_probs = [float(row["actual_prob"]) for row in network_rows if row.get("actual_prob") is not None]
        mismatch_families = Counter(row["action_family_expected"] for row in network_rows if not row["is_match"])
        high_conf_families = Counter(row["action_family_expected"] for row in high_conf)
        network_summaries[network] = {
            "decision_count": total,
            "match_count": matches,
            "match_rate": matches / total if total else None,
            "mismatch_count": total - matches,
            "high_confidence_disagreement_count": len(high_conf),
            "high_confidence_disagreement_rate": len(high_conf) / total if total else None,
            "actual_prob_mean": mean(actual_probs),
            "actual_prob_min": min(actual_probs) if actual_probs else None,
            "actual_prob_max": max(actual_probs) if actual_probs else None,
            "mismatch_by_expected_family": dict(sorted(mismatch_families.items())),
            "high_confidence_disagreement_by_expected_family": dict(sorted(high_conf_families.items())),
        }
    return network_summaries


def build_aligned_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        key = (row["source_input_key"], row["player_id"], row["kyoku_index"], row["entry_index"])
        grouped[key].append(row)

    aligned: list[dict[str, Any]] = []
    for key, group in sorted(grouped.items()):
        if len(group) < 2:
            continue
        by_network = {str(row["network"]): row for row in group}
        if len(by_network) < 2:
            continue
        base = group[0]
        row: dict[str, Any] = {
            "schema": "keqing.mortal.reviewer_teacher_aligned_decision.v1",
            "source_input_key": key[0],
            "player_id": key[1],
            "kyoku_index": key[2],
            "entry_index": key[3],
            "actual_action": base["actual_action"],
            "actual_action_key": base["actual_action_key"],
            "action_family_actual": base["action_family_actual"],
            "networks": sorted(by_network),
        }
        expected_keys = []
        for network, decision in sorted(by_network.items()):
            suffix = network_key(network)
            row[f"expected_{suffix}"] = decision["expected_action"]
            row[f"expected_key_{suffix}"] = decision["expected_action_key"]
            row[f"match_actual_{suffix}"] = decision["is_match"]
            row[f"expected_prob_{suffix}"] = decision["expected_prob"]
            row[f"actual_prob_{suffix}"] = decision["actual_prob"]
            row[f"prob_margin_{suffix}"] = decision["prob_margin"]
            row[f"action_family_expected_{suffix}"] = decision["action_family_expected"]
            expected_keys.append(decision["expected_action_key"])
        row["teacher_agree"] = len(set(expected_keys)) == 1
        row["teacher_agreement_key_count"] = len(set(expected_keys))
        aligned.append(row)
    return aligned


def summarize_aligned_rows(aligned_rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(aligned_rows)
    network_suffixes = sorted(
        {
            key[len("match_actual_") :]
            for row in aligned_rows
            for key in row
            if key.startswith("match_actual_")
        }
    )
    summary: dict[str, Any] = {
        "aligned_entry_count": total,
        "teacher_agreement_count": sum(1 for row in aligned_rows if row["teacher_agree"]),
        "teacher_agreement_rate": None,
        "match_pattern_counts": {},
    }
    if total:
        summary["teacher_agreement_rate"] = summary["teacher_agreement_count"] / total

    pattern_counts: Counter[str] = Counter()
    for row in aligned_rows:
        matched = [suffix for suffix in network_suffixes if row.get(f"match_actual_{suffix}")]
        if not matched:
            label = "matches_neither"
        elif len(matched) == len(network_suffixes):
            label = "matches_all"
        else:
            label = "matches_" + "_only_".join(matched) + "_only"
        pattern_counts[label] += 1
    summary["match_pattern_counts"] = dict(sorted(pattern_counts.items()))
    return summary


def top_disagreements(rows: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    candidates = [row for row in rows if not row["is_match"]]
    candidates.sort(
        key=lambda row: (
            row.get("expected_prob") if row.get("expected_prob") is not None else -1.0,
            row.get("prob_margin") if row.get("prob_margin") is not None else -1.0,
        ),
        reverse=True,
    )
    return candidates[:top_k]


def parse_reports(
    *,
    report_manifest: Path,
    output_dir: Path,
    high_confidence_prob: float,
    high_confidence_margin: float,
    top_k: int,
) -> dict[str, Any]:
    if top_k < 0:
        raise ValueError(f"top-k must be non-negative, got {top_k}")
    rows = build_decision_rows(report_manifest)
    aligned_rows = build_aligned_rows(rows)
    summary = {
        "schema": "keqing.mortal.reviewer_teacher_probe_summary.v1",
        "report_manifest": str(report_manifest),
        "output_dir": str(output_dir),
        "decision_count": len(rows),
        "aligned_entry_count": len(aligned_rows),
        "high_confidence_prob": high_confidence_prob,
        "high_confidence_margin": high_confidence_margin,
        "network_summaries": summarize_rows(
            rows,
            high_confidence_prob=high_confidence_prob,
            high_confidence_margin=high_confidence_margin,
        ),
        "aligned_summary": summarize_aligned_rows(aligned_rows),
        "outputs": {
            "decision_table_jsonl": str(output_dir / "decision_table.jsonl"),
            "decision_table_csv": str(output_dir / "decision_table.csv"),
            "aligned_decisions_jsonl": str(output_dir / "aligned_decisions.jsonl"),
            "aligned_decisions_csv": str(output_dir / "aligned_decisions.csv"),
            "top_disagreements_jsonl": str(output_dir / "top_disagreements.jsonl"),
            "summary_json": str(output_dir / "summary.json"),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "decision_table.jsonl", rows)
    write_csv(output_dir / "decision_table.csv", rows)
    write_jsonl(output_dir / "aligned_decisions.jsonl", aligned_rows)
    write_csv(output_dir / "aligned_decisions.csv", aligned_rows)
    write_jsonl(output_dir / "top_disagreements.jsonl", top_disagreements(rows, top_k))
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    args = parse_args()
    summary = parse_reports(
        report_manifest=args.report_manifest,
        output_dir=args.output_dir,
        high_confidence_prob=float(args.high_confidence_prob),
        high_confidence_margin=float(args.high_confidence_margin),
        top_k=int(args.top_k),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
