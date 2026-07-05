from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path("artifacts/experiments/teacher_transfer_2026_05")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def fnum(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    return f"{float(value):.{digits}f}"


def fpct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100.0:.{digits}f}%"


def fpp(value: Any, digits: int = 2) -> str:
    if value is None:
        return "NA"
    return f"{float(value) * 100.0:+.{digits}f}pp"


def score(value: Any, digits: int = 1) -> str:
    if value is None:
        return "NA"
    return f"{float(value):+.{digits}f}"


def table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" if i == 0 else "---:" for i in range(len(headers))) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return lines


def player_metric(detailed: dict[str, Any], model: str) -> dict[str, Any]:
    player = detailed["players"][model]
    out = dict(player["derived"])
    out["games"] = player["raw"]["game"]
    out["rounds"] = player["raw"]["round"]
    return out


def load_slice_rows(base: Path, ref: str) -> dict[str, Any]:
    path = base / f"outcome_risk_slices_T1_vs_{ref}_1000h" / "target_vs_reference_slices.csv"
    rows = read_csv_rows(path)
    by_slice = {row["slice"]: row for row in rows}
    return {"path": str(path), "rows": rows, "by_slice": by_slice}


def load_metric(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return read_json(path)


def challenger_rank_pt(metric: dict[str, Any] | None) -> float | None:
    if metric is None:
        return None
    challenger = (metric.get("metrics") or {}).get("challenger") or {}
    value = challenger.get("avg_rank_pt_tenhou_reference", challenger.get("avg_rank_pt"))
    if value is None:
        return None
    return float(value)


def row_float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key)
    if value is None or value == "":
        return None
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a compact T1 risk audit from existing evaluation outputs.")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "T1_risk_audit_2026_06")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--naga-summary",
        type=Path,
        default=Path("artifacts/tmp/naga_site/report_json/naga_117_summary.json"),
    )
    args = parser.parse_args()

    four_player_path = args.root / "four_player_native_random_1000h" / "detailed_stats.json"
    behavior_path = args.root / "behavior_readout_four_model_100h" / "metrics.json"
    naga_path = args.naga_summary
    four_player = read_json(four_player_path)
    behavior = read_json(behavior_path)
    naga = read_json(naga_path)
    gate_t1_vs_70k_path = args.root / "T1_teacher_ce_01" / "gate_5000h_final" / "Gate_T1_vs_70k" / "metrics.json"
    gate_70k_vs_t1_path = args.root / "T1_teacher_ce_01" / "gate_5000h_final" / "Gate_70k_vs_T1" / "metrics.json"
    gate_t1_vs_70k = load_metric(gate_t1_vs_70k_path)
    gate_70k_vs_t1 = load_metric(gate_70k_vs_t1_path)
    t1_vs_70k_pt = challenger_rank_pt(gate_t1_vs_70k)
    reverse_70k_vs_t1_pt = challenger_rank_pt(gate_70k_vs_t1)

    models = ["70k", "80k_game", "T1_71000", "model_v4"]
    native_rows: list[dict[str, Any]] = []
    for model in models:
        item = player_metric(four_player, model)
        native_rows.append(
            {
                "model": model,
                "avg_rank_pt": item["avg_rank_pt"],
                "avg_rank": item["avg_rank"],
                "agari_rate": item["agari_rate"],
                "houjuu_rate": item["houjuu_rate"],
                "fuuro_rate": item["fuuro_rate"],
                "riichi_rate": item["riichi_rate"],
                "avg_point_per_agari": item["avg_point_per_agari"],
                "agari_after_fuuro": item["agari_rate_after_fuuro"],
                "houjuu_after_fuuro": item["houjuu_rate_after_fuuro"],
                "agari_after_riichi": item["agari_rate_after_riichi"],
                "houjuu_after_riichi": item["houjuu_rate_after_riichi"],
            }
        )

    naga_by_model = {row["model"]: row for row in naga["models"]}
    naga_rows = [naga_by_model[model] for model in models]

    slices = {
        "70k": load_slice_rows(args.root, "70k"),
        "80k_game": load_slice_rows(args.root, "80k_game"),
        "model_v4": load_slice_rows(args.root, "v4"),
    }
    key_slices = [
        "round_all",
        "round_after_fuuro",
        "round_after_riichi",
        "action_discard_after_fuuro",
        "action_discard_vs_riichi",
        "action_discard_after_fuuro_vs_riichi",
        "round_score_ahead_big",
        "round_start_rank_1",
    ]
    slice_summary: dict[str, list[dict[str, Any]]] = {}
    for ref, payload in slices.items():
        rows = []
        by_slice = payload["by_slice"]
        for name in key_slices:
            row = by_slice.get(name)
            if not row:
                continue
            rows.append(
                {
                    "slice": name,
                    "target_count": int(row["target_count"]),
                    "reference_count": int(row["reference_count"]),
                    "delta_agari_rate": row_float(row, "delta_agari_rate"),
                    "delta_houjuu_rate": row_float(row, "delta_houjuu_rate"),
                    "delta_avg_delta_score": row_float(row, "delta_avg_delta_score"),
                }
            )
        slice_summary[ref] = rows

    audit = {
        "schema": "keqing.mortal.t1_risk_audit.v1",
        "sources": {
            "four_player_native_1000h": str(four_player_path),
            "behavior_readout_100h": str(behavior_path),
            "naga_117_summary": str(naga_path),
            "gate_t1_vs_70k_5000h": str(gate_t1_vs_70k_path),
            "gate_70k_vs_t1_5000h": str(gate_70k_vs_t1_path),
            "outcome_slices": {ref: payload["path"] for ref, payload in slices.items()},
        },
        "directional_gate_anchor": {
            "T1_vs_70k_5000h_challenger_pt": t1_vs_70k_pt,
            "70k_vs_T1_5000h_challenger_pt": reverse_70k_vs_t1_pt,
            "T1_vs_80k_5000h": "not complete in available artifacts",
        },
        "native_1000h": native_rows,
        "naga_117": naga_rows,
        "outcome_slice_summary": slice_summary,
        "conclusions": [
            "T1_71000 is not a pure failed checkpoint: in the 1000h four-model native arena it is above 70k and 80k_game by Tenhou rank points, but still below model_v4.",
            "T1_71000's main gap to model_v4 is risk calibration, especially overall houjuu, post-fuuro discard, and leading/dealer states; the post-fuuro-vs-riichi slice trades more agari for materially more houjuu.",
            "The 117-kyoku NAGA sample is a high-risk diagnostic subset, not a replacement for long-run arena strength. It confirms that T1 has lower NAGA similarity and more low-probability discards than the other baselines in these cases.",
            "T2 should not directly clone model_v4 or NAGA. Use them as risk filters and confidence signals while preserving T1's higher-value agari behavior.",
        ],
        "next_plan": {
            "step_1": "Use this audit as the fixed T1 risk baseline.",
            "step_2": "Design T2 as risk-constrained teacher transfer: keep useful teacher CE, but downweight or block it in high-risk contexts identified by outcome slices and NAGA low-probability flags.",
            "step_3": "Evaluate in layers: fast behavior/readout first, then native four-model arena, then targeted NAGA/reviewer sampling only for changed risk slices.",
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "t1_risk_audit.json"
    md_path = args.output_dir / "t1_risk_audit.md"
    json_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines: list[str] = [
        "# T1 Risk Audit",
        "",
        "## Sources",
        "",
        f"- Native four-model 1000h: `{four_player_path}`",
        f"- Four-model behavior 100h: `{behavior_path}`",
        f"- NAGA 117 kyoku: `{naga_path}`",
        f"- T1 vs 70k 5000h gate: `{gate_t1_vs_70k_path}` and `{gate_70k_vs_t1_path}`",
        "- Outcome risk slices: `outcome_risk_slices_T1_vs_{70k,80k_game,v4}_1000h`",
        "",
        "## Directional Gate Anchor",
        "",
        "| matchup | challenger avg pt | note |",
        "| --- | ---: | --- |",
        f"| T1 vs 3x70k | {fnum(t1_vs_70k_pt, 3)} | 5000 half-games |",
        f"| 70k vs 3xT1 | {fnum(reverse_70k_vs_t1_pt, 3)} | 5000 half-games |",
        "| T1 vs 80k_game | NA | 5000h artifact has progress only, no completed metrics |",
        "",
        "## Native Four-Model Arena, 1000h",
        "",
    ]
    lines.extend(
        table(
            [
                "model",
                "avg pt",
                "avg rank",
                "agari",
                "houjuu",
                "fuuro",
                "riichi",
                "win value",
                "after-fuuro A/H",
                "after-riichi A/H",
            ],
            [
                [
                    row["model"],
                    fnum(row["avg_rank_pt"], 3),
                    fnum(row["avg_rank"], 3),
                    fpct(row["agari_rate"]),
                    fpct(row["houjuu_rate"]),
                    fpct(row["fuuro_rate"]),
                    fpct(row["riichi_rate"]),
                    fnum(row["avg_point_per_agari"], 1),
                    f"{fpct(row['agari_after_fuuro'])} / {fpct(row['houjuu_after_fuuro'])}",
                    f"{fpct(row['agari_after_riichi'])} / {fpct(row['houjuu_after_riichi'])}",
                ]
                for row in native_rows
            ],
        )
    )
    lines.extend(
        [
            "",
            "## NAGA 117-Kyoku Diagnostic Sample",
            "",
        ]
    )
    lines.extend(
        table(
            [
                "model",
                "agari",
                "houjuu",
                "riichi",
                "fuuro kyoku",
                "Nishiki acc/sim/bad",
                "Kagashi acc/sim/bad",
                "Kagashi low p<.05",
            ],
            [
                [
                    row["model"],
                    f"{row['hora_rate']:.1f}%",
                    f"{row['hoju_rate']:.1f}%",
                    f"{row['reach_rate']:.1f}%",
                    f"{row['huro_kyoku_rate']:.1f}%",
                    f"{row['naga']['ニシキ']['accuracy']:.1f} / {row['naga']['ニシキ']['similarity']:.1f} / {row['naga']['ニシキ']['bad_move_rate']:.1f}",
                    f"{row['naga']['カガシ']['accuracy']:.1f} / {row['naga']['カガシ']['similarity']:.1f} / {row['naga']['カガシ']['bad_move_rate']:.1f}",
                    str(row["naga"]["カガシ"]["low_prob_005"]),
                ]
                for row in naga_rows
            ],
        )
    )
    lines.extend(["", "## T1 Outcome Slice Deltas", ""])
    for ref, rows in slice_summary.items():
        lines.extend([f"### Versus {ref}", ""])
        lines.extend(
            table(
                ["slice", "T1 n", "ref n", "agari d", "houjuu d", "delta d"],
                [
                    [
                        row["slice"],
                        str(row["target_count"]),
                        str(row["reference_count"]),
                        fpp(row["delta_agari_rate"]),
                        fpp(row["delta_houjuu_rate"]),
                        score(row["delta_avg_delta_score"]),
                    ]
                    for row in rows
                ],
            )
        )
        lines.append("")

    lines.extend(
        [
            "## Read",
            "",
            "- T1 is a valid own-trained candidate, not just a 70k-overfit checkpoint. Native four-model 1000h puts it above 70k and 80k_game, but below model_v4.",
            "- The useful part of T1 is value: higher average winning value and strong after-fuuro/after-riichi conversion. The weak part is calibration against risk, especially relative to model_v4.",
            "- The NAGA 117 sample should be treated as targeted risk evidence. It is too selected to replace arena strength, but it is good enough to define T2 guardrails.",
            "",
            "## T2 Direction",
            "",
            "1. Keep T1 as the starting baseline and do not switch to model_v4 as parent.",
            "2. Keep teacher CE as a weak auxiliary signal, but apply a risk gate instead of copying all teacher choices.",
            "3. Downweight or disable teacher CE in contexts with high T1-vs-v4 houjuu gap: post-fuuro discard, discard versus riichi, post-fuuro versus riichi, dealer/leading states.",
            "4. Add a reviewer/NAGA confidence filter only as a penalty or sample selector. Do not train toward NAGA/v4 top choice directly unless multiple signals agree and the local model probability is very low.",
            "",
            "## Evaluation Ladder",
            "",
            "1. 100h/200h native behavior readout after a short T2 run: agari, houjuu, fuuro, riichi, after-fuuro A/H, after-riichi A/H, win value.",
            "2. 1000h native four-model random-seat arena against 70k, 80k_game, T1, model_v4.",
            "3. Targeted outcome-risk slices from the same arena logs.",
            "4. Small NAGA/reviewer batch only for changed high-risk slices, not every checkpoint.",
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {md_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
