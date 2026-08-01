#!/usr/bin/env python3
"""Audit the V2/V3 descendant-view mix before D2 training.

This orchestrator runs the existing raw-event and parent-Q auditors on the two
fixed 3,000-hanchan indexes, then writes a weighted 50/50 D2 summary.  It does
not alter replay logs, checkpoints, or training state.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROOT = REPO_ROOT / "artifacts/experiments/model_pool_2026_07/D2_project_owned_descendant_view_mix_2026_08"
DEFAULT_PARENT = REPO_ROOT / "artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth"
DEFAULT_CONFIG = DEFAULT_ROOT / "training_prep_2026_08/D2_variant/seed_20260806/config.toml"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def counter_add(target: Counter[str], values: dict[str, Any]) -> None:
    for key, value in values.items():
        target[str(key)] += int(value)


def weighted_support(reports: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "behavior_action_legal_rate",
        "greedy_agreement_rate",
        "greedy_disagreement_rate",
        "mean_behavior_q_rank",
        "mean_q_regret_greedy_minus_behavior",
        "mean_greedy_margin",
        "mean_behavior_q",
        "mean_greedy_q",
        "mean_legal_q_abs",
    )
    sums = Counter()
    rank_counts = Counter()
    states = 0
    for report in reports:
        support = report["support_audit"]["overall"]
        count = int(support["states"])
        states += count
        for field in fields:
            sums[field] += count * float(support[field])
        counter_add(rank_counts, support["behavior_q_rank_counts"])
    return {"states": states, **{field: sums[field] / states for field in fields}, "behavior_q_rank_counts": dict(sorted(rank_counts.items()))}


def weighted_corpus(reports: list[dict[str, Any]]) -> dict[str, Any]:
    corpus = {key: 0 for key in ("files_selected", "hanchans", "trainable_perspectives", "total_decisions", "malformed_count")}
    final_ranks = Counter()
    targets = Counter()
    for report in reports:
        source = report["corpus"]
        for key in corpus:
            corpus[key] += int(source[key])
        counter_add(final_ranks, source["final_rank_counts"])
        counter_add(targets, source["target_counts"])
    corpus["final_rank_counts"] = dict(sorted(final_ranks.items()))
    corpus["target_counts"] = dict(sorted(targets.items()))
    return corpus


def weighted_decisions(reports: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    names = (
        "phase_counts",
        "current_rank_counts",
        "score_gap_counts",
        "own_riichi_counts",
        "action_counts",
        "legal_action_count_buckets",
        "shanten_buckets",
    )
    result = {}
    for name in names:
        values = Counter()
        for report in reports:
            counter_add(values, report["decision_distribution"][name])
        result[name] = dict(sorted(values.items()))
    return result


def combine_outcomes(reports: list[dict[str, Any]]) -> dict[str, Any]:
    event_keys = ("kyoku", "agari", "houjuu", "fuuro", "riichi", "ryukyoku", "dahai", "tsumo", "reach_accepted")
    events = Counter()
    hanchans_with = Counter()
    ranks = Counter()
    malformed = 0
    hanchans = 0
    decisions = 0
    for report in reports:
        hanchans += int(report["hanchans"])
        malformed += int(report["malformed_count"])
        decisions += int(report["decision_like_event_count"])
        counter_add(events, report["raw_event_counts"])
        counter_add(hanchans_with, report["hanchans_with_event"])
        counter_add(ranks, report["final_rank_counts"])
    rates = {f"{key}_per_hanchan": events[key] / hanchans for key in ("agari", "houjuu", "fuuro", "riichi")}
    rates.update({f"{key}_hanchan_rate": hanchans_with[key] / hanchans for key in ("agari", "houjuu", "fuuro", "riichi")})
    return {
        "hanchans": hanchans,
        "raw_event_counts": {key: int(events[key]) for key in event_keys},
        "hanchans_with_event": {key: int(hanchans_with[key]) for key in ("agari", "houjuu", "fuuro", "riichi")},
        "rates": rates,
        "final_rank_counts": {str(key): int(value) for key, value in sorted(ranks.items())},
        "decision_like_event_count": decisions,
        "malformed_count": malformed,
    }


def run_child(command: list[str]) -> None:
    print("[d2-audit] running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)


def markdown(report: dict[str, Any]) -> str:
    rows = report["views"]
    lines = [
        "# D2 Descendant-View Audit",
        "",
        "This is a pre-training audit of the fixed D2 50/50 V2/V3 view assignment. It does not select checkpoints or use outcomes for assignment.",
        "",
        "| view | files | decisions | 70k agreement | mean behavior-Q rank | mean Q regret | agari/houjuu/fuuro/riichi per hanchan | decision ESS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ("K0_70k_reference", "V2_74000", "V3_74000", "D2_50_50"):
        row = rows[name]
        events = row["outcomes"]
        lines.append(
            f"| {name} | {row['files']} | {row['decisions']} | {row['greedy_agreement_rate']:.4%} | "
            f"{row['mean_behavior_q_rank']:.4f} | {row['mean_q_regret_greedy_minus_behavior']:.6g} | "
            f"{events['rates']['agari_per_hanchan']:.3f}/{events['rates']['houjuu_per_hanchan']:.3f}/"
            f"{events['rates']['fuuro_per_hanchan']:.3f}/{events['rates']['riichi_per_hanchan']:.3f} | "
            f"{row['decision_weight_ess']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Gate",
            "",
            f"- Source hanchans: `{report['source']['source_file_count']}`; D2 assignment: `{report['assignment']['counts']}`.",
            f"- Single trainable perspective per file: `{report['assignment']['single_perspective_per_file']}`.",
            f"- Malformed V2/V3 reports: `{report['malformed_count']}`.",
            "- Agreement is a parent-support diagnostic, not a ground-truth quality score.",
            "- A high agreement result is not automatically a reason to continue training; inspect action support and Q regret together.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--parent", type=Path, default=DEFAULT_PARENT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--skip-q-audit", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    parent = args.parent.resolve()
    config = args.config.resolve()
    prep = root / "training_prep_2026_08"
    dataset = root / "dataset"
    output = prep / "distribution_d2"
    output.mkdir(parents=True, exist_ok=True)
    indexes = {label: dataset / f"file_index_{suffix}.pth" for label, suffix in (("V2_74000", "v2"), ("V3_74000", "v3"))}
    if any(not path.is_file() for path in indexes.values()):
        raise FileNotFoundError("D2 V2/V3 file indexes are missing; run prepare_d2_descendant_view_mix_2026_08.py first")

    for label, index in indexes.items():
        outcome_path = output / f"outcomes_{label}.json"
        run_child(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/mortal/audit_trainable_view_outcomes_2026_07.py"),
                "--file-index",
                str(index),
                "--model-label",
                label,
                "--output",
                str(outcome_path),
                "--progress-every",
                "250",
            ]
        )
        if not args.skip_q_audit:
            run_child(
                [
                    sys.executable,
                    str(REPO_ROOT / "scripts/mortal/audit_replay_distribution.py"),
                    "--file-index",
                    str(index),
                    "--parent",
                    str(parent),
                    "--config",
                    str(config),
                    "--output-dir",
                    str(output / label),
                    "--model-label",
                    label,
                    "--device",
                    "cuda",
                    "--require-cuda",
                    "--q-batch-size",
                    "4096",
                    "--file-batch-size",
                    "50",
                    "--progress-every",
                    "250",
                ]
            )

    reports = {}
    for label in indexes:
        outcome = load_json(output / f"outcomes_{label}.json")
        q_report_path = output / label / "data_distribution_audit.json"
        if args.skip_q_audit:
            q_report = None
        else:
            q_report = load_json(q_report_path)
        if outcome["hanchans"] != 3000 or outcome["malformed_count"] != 0:
            raise SystemExit(f"{label} outcome audit failed: {outcome}")
        if q_report is not None and (
            q_report["corpus"]["files_selected"] != 3000
            or q_report["corpus"]["trainable_perspectives"] != 3000
            or q_report["corpus"]["malformed_count"] != 0
        ):
            raise SystemExit(f"{label} Q audit failed corpus gate")
        reports[label] = {"outcomes": outcome, "q": q_report}

    if args.skip_q_audit:
        print(json.dumps({"status": "outcome_audit_only", "output": str(output)}, indent=2), flush=True)
        return

    view_rows = {}
    for label, value in reports.items():
        q = value["q"]
        support = q["support_audit"]["overall"]
        decisions = int(q["corpus"]["total_decisions"])
        contribution = q["hanchan_contribution"]
        view_rows[label] = {
            "files": int(q["corpus"]["files_selected"]),
            "decisions": decisions,
            "greedy_agreement_rate": float(support["greedy_agreement_rate"]),
            "mean_behavior_q_rank": float(support["mean_behavior_q_rank"]),
            "mean_q_regret_greedy_minus_behavior": float(support["mean_q_regret_greedy_minus_behavior"]),
            "mean_greedy_margin": float(support["mean_greedy_margin"]),
            "mean_behavior_q": float(support["mean_behavior_q"]),
            "mean_greedy_q": float(support["mean_greedy_q"]),
            "mean_legal_q_abs": float(support["mean_legal_q_abs"]),
            "decision_weight_ess": float(contribution["decision_weight_ess"]),
            "outcomes": value["outcomes"],
            "q_report": str((output / label / "data_distribution_audit.json").resolve()),
        }
    v2_q = reports["V2_74000"]["q"]
    v3_q = reports["V3_74000"]["q"]
    combined_support = weighted_support([v2_q, v3_q])
    combined_corpus = weighted_corpus([v2_q, v3_q])
    combined_outcomes = combine_outcomes([reports["V2_74000"]["outcomes"], reports["V3_74000"]["outcomes"]])
    combined_decisions = weighted_decisions([v2_q, v3_q])
    combined_contribution = {
        "hanchans": combined_corpus["hanchans"],
        "total_decisions": combined_corpus["total_decisions"],
        "decision_weight_ess": combined_corpus["total_decisions"] ** 2
        / sum(int(row["decisions"]) ** 2 for report in (v2_q, v3_q) for row in report["hanchans"]),
    }
    view_rows["D2_50_50"] = {
        "files": combined_corpus["files_selected"],
        "decisions": combined_corpus["total_decisions"],
        **{key: float(value) for key, value in combined_support.items() if key != "behavior_q_rank_counts"},
        "decision_weight_ess": float(combined_contribution["decision_weight_ess"]),
        "outcomes": combined_outcomes,
        "decision_distribution": combined_decisions,
    }

    d1_reference = prep / "distribution/summary/d1_distribution_d1.json"
    k0 = load_json(d1_reference) if d1_reference.is_file() else None
    if k0:
        k0_support = k0["support_audit_overall"]
        view_rows["K0_70k_reference"] = {
            "files": int(k0["corpus"]["files_selected"]),
            "decisions": int(k0["corpus"]["total_decisions"]),
            "greedy_agreement_rate": float(k0_support["greedy_agreement_rate"]),
            "mean_behavior_q_rank": float(k0_support["mean_behavior_q_rank"]),
            "mean_q_regret_greedy_minus_behavior": float(k0_support["mean_q_regret_greedy_minus_behavior"]),
            "mean_greedy_margin": float(k0_support["mean_greedy_margin"]),
            "mean_behavior_q": float(k0_support["mean_behavior_q"]),
            "mean_greedy_q": float(k0_support["mean_greedy_q"]),
            "mean_legal_q_abs": float(k0_support["mean_legal_q_abs"]),
            "decision_weight_ess": float(k0["hanchan_contribution"]["decision_weight_ess"]),
            "outcomes": k0["outcomes"],
            "reference": str(d1_reference.resolve()),
        }

    report = {
        "schema": "keqing.mortal.d2_descendant_view_audit.v1",
        "passed": all(
            row["files"] == (6000 if name == "D2_50_50" else 3000)
            for name, row in view_rows.items()
            if name in {"V2_74000", "V3_74000", "D2_50_50"}
        ),
        "source": {
            "d1_root": str(root),
            "source_file_count": 6000,
            "reuse_same_hanchans": True,
        },
        "assignment": {
            "method": "canonical_hash_parity_50_50",
            "counts": {"V2_74000": 3000, "V3_74000": 3000},
            "single_perspective_per_file": True,
        },
        "malformed_count": sum(int(value["outcomes"]["malformed_count"]) for value in reports.values()),
        "views": view_rows,
        "decision_distributions": {
            "V2_74000": reports["V2_74000"]["q"]["decision_distribution"],
            "V3_74000": reports["V3_74000"]["q"]["decision_distribution"],
            "D2_50_50": combined_decisions,
        },
        "notes": [
            "V2/V3 assignment is independent of outcome, rank, and target.",
            "D2_50_50 metrics are decision-weighted sums of the two fixed 3000-file views.",
            "Parent-Q regret is an internal support diagnostic, not a ground-truth regret estimate.",
        ],
    }
    json_path = output / "d2_descendant_view_audit.json"
    md_path = output / "d2_descendant_view_audit.md"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(markdown(report), encoding="utf-8")
    print(json.dumps({"passed": report["passed"], "json": str(json_path), "markdown": str(md_path)}, ensure_ascii=False, indent=2), flush=True)
    if not report["passed"]:
        raise SystemExit("D2 descendant-view audit failed")


if __name__ == "__main__":
    main()
