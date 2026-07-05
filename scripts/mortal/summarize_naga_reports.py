from __future__ import annotations

import argparse
import gzip
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


TILE34 = {
    **{f"{i}m": i - 1 for i in range(1, 10)},
    **{f"{i}p": 8 + i for i in range(1, 10)},
    **{f"{i}s": 17 + i for i in range(1, 10)},
    "E": 27,
    "S": 28,
    "W": 29,
    "N": 30,
    "P": 31,
    "F": 32,
    "C": 33,
}
TILE34.update({"5mr": 4, "5pr": 13, "5sr": 22})


def read_report(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":
        return json.loads(gzip.decompress(raw).decode("utf-8"))
    return json.loads(raw.decode("utf-8"))


def expand_report_paths(values: list[Path]) -> list[Path]:
    paths: list[Path] = []
    for value in values:
        text = str(value)
        if any(char in text for char in "*?[]"):
            matches = sorted(Path().glob(text))
            paths.extend(path for path in matches if path.is_file())
        elif value.is_file():
            paths.append(value)
    return paths


def pct(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value * 100.0, 3)


def round_or_none(value: float | None, digits: int = 3) -> float | None:
    if value is None:
        return None
    return round(value, digits)


def ensure_model(stats: dict[str, dict[str, Any]], model: str, naga_names: list[str]) -> dict[str, Any]:
    if model not in stats:
        stats[model] = {
            "model": model,
            "kyoku": 0,
            "dahai_total": 0,
            "dahai_decisions": 0,
            "hora": 0,
            "hora_point": 0,
            "hoju": 0,
            "hoju_point": 0,
            "reach": 0,
            "huro_total": 0,
            "huro_kyoku": 0,
            "naga": {
                name: {
                    "not_match": 0,
                    "bad_move": 0,
                    "prob_gap_sum": 0.0,
                    "actual_prob_sum": 0.0,
                    "top_prob_sum": 0.0,
                    "low_prob_005": 0,
                    "low_prob_020": 0,
                }
                for name in naga_names
            },
        }
    return stats[model]


def summarize_reports(paths: list[Path]) -> dict[str, Any]:
    stats: dict[str, dict[str, Any]] = {}
    total_kyoku = 0
    report_rows = []

    for path in paths:
        report = read_report(path)
        if "pred" not in report:
            continue
        pred = report.get("pred") or []
        names = (report.get("player_info") or report.get("playerInfo") or {}).get("name") or []
        naga_types = report.get("naga_types") or {}
        naga_names = [str(naga_types[str(i)]) for i in range(len(naga_types))]
        total_kyoku += len(pred)
        report_rows.append(
            {
                "path": str(path),
                "report_id": path.stem,
                "kyoku": len(pred),
                "names": names,
                "naga_types": naga_names,
            }
        )

        for name in names:
            ensure_model(stats, str(name), naga_names)

        for kyoku in pred:
            huro_seen = [False, False, False, False]
            first_msg = ((kyoku[0] if kyoku else {}).get("info") or {}).get("msg") or {}
            end_msgs = first_msg.get("end_msgs") or []
            for seat, name in enumerate(names):
                ensure_model(stats, str(name), naga_names)["kyoku"] += 1

            for event in kyoku:
                msg = (event.get("info") or {}).get("msg") or {}
                actor = msg.get("actor")
                msg_type = msg.get("type")
                if isinstance(actor, int) and 0 <= actor < len(names):
                    model = str(names[actor])
                    row = ensure_model(stats, model, naga_names)
                    if msg_type == "dahai":
                        row["dahai_total"] += 1
                    if msg_type == "reach_accepted":
                        row["reach"] += 1
                    if msg_type in {"chi", "pon", "daiminkan"}:
                        row["huro_total"] += 1
                        huro_seen[actor] = True

                    if (
                        msg_type in {"tsumo", "chi", "pon"}
                        and msg.get("real_dahai")
                        and msg.get("real_dahai") != "?"
                        and not msg.get("reached")
                    ):
                        real_tile = TILE34.get(str(msg.get("real_dahai")))
                        pred_dahai = msg.get("pred_dahai") or []
                        dahai_pred = event.get("dahai_pred") or []
                        if real_tile is None:
                            continue
                        row["dahai_decisions"] += 1
                        for idx, naga_name in enumerate(naga_names):
                            if idx >= len(pred_dahai) or idx >= len(dahai_pred):
                                continue
                            pred_tile = TILE34.get(str(pred_dahai[idx]))
                            probs = dahai_pred[idx]
                            if pred_tile is None or real_tile >= len(probs) or pred_tile >= len(probs):
                                continue
                            actual_prob = float(probs[real_tile]) / 10000.0
                            top_prob = float(probs[pred_tile]) / 10000.0
                            item = row["naga"][naga_name]
                            item["actual_prob_sum"] += actual_prob
                            item["top_prob_sum"] += top_prob
                            if real_tile != pred_tile:
                                gap = abs(actual_prob - top_prob)
                                item["not_match"] += 1
                                item["prob_gap_sum"] += gap
                                if actual_prob < 0.05:
                                    item["bad_move"] += 1
                                if actual_prob < 0.05:
                                    item["low_prob_005"] += 1
                                if actual_prob < 0.20:
                                    item["low_prob_020"] += 1

            for seat, seen in enumerate(huro_seen):
                if seen and seat < len(names):
                    ensure_model(stats, str(names[seat]), naga_names)["huro_kyoku"] += 1

            if end_msgs and end_msgs[0].get("type") == "hora":
                for end in end_msgs:
                    actor = end.get("actor")
                    target = end.get("target")
                    deltas = end.get("deltas") or []
                    if isinstance(actor, int) and 0 <= actor < len(names):
                        row = ensure_model(stats, str(names[actor]), naga_names)
                        row["hora"] += 1
                        if actor < len(deltas):
                            row["hora_point"] += int(deltas[actor])
                    if isinstance(target, int) and target != actor and 0 <= target < len(names):
                        row = ensure_model(stats, str(names[target]), naga_names)
                        row["hoju"] += 1
                        if target < len(deltas):
                            row["hoju_point"] += -int(deltas[target])

    model_rows = []
    for model, row in sorted(stats.items()):
        decisions = row["dahai_decisions"]
        naga_out = {}
        for naga_name, item in row["naga"].items():
            naga_out[naga_name] = {
                "accuracy": None if decisions == 0 else round((decisions - item["not_match"]) / decisions * 100, 3),
                "similarity": None if decisions == 0 else round((decisions - item["prob_gap_sum"]) / decisions * 100, 3),
                "bad_move_rate": None if decisions == 0 else round(item["bad_move"] / decisions * 100, 3),
                "avg_actual_prob": None if decisions == 0 else round(item["actual_prob_sum"] / decisions, 5),
                "avg_top_prob": None if decisions == 0 else round(item["top_prob_sum"] / decisions, 5),
                "low_prob_005": item["low_prob_005"],
                "low_prob_020": item["low_prob_020"],
            }
        model_rows.append(
            {
                "model": model,
                "kyoku": row["kyoku"],
                "dahai_total": row["dahai_total"],
                "dahai_decisions": decisions,
                "hora": row["hora"],
                "hora_rate": None if row["kyoku"] == 0 else round(row["hora"] / row["kyoku"] * 100, 3),
                "hora_point_avg": None if row["hora"] == 0 else round(row["hora_point"] / row["hora"], 1),
                "hoju": row["hoju"],
                "hoju_rate": None if row["kyoku"] == 0 else round(row["hoju"] / row["kyoku"] * 100, 3),
                "hoju_point_avg": None if row["hoju"] == 0 else round(row["hoju_point"] / row["hoju"], 1),
                "reach": row["reach"],
                "reach_rate": None if row["kyoku"] == 0 else round(row["reach"] / row["kyoku"] * 100, 3),
                "huro_total": row["huro_total"],
                "huro_kyoku": row["huro_kyoku"],
                "huro_kyoku_rate": None if row["kyoku"] == 0 else round(row["huro_kyoku"] / row["kyoku"] * 100, 3),
                "naga": naga_out,
            }
        )

    return {"total_kyoku": total_kyoku, "reports": report_rows, "models": model_rows}


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize NAGA report JSON files with NAGA-style similarity.")
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    reports = expand_report_paths(args.reports)
    if not reports:
        raise SystemExit("no report files matched")
    summary = summarize_reports(reports)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.output_md:
        lines = [
            f"# NAGA Report Summary",
            "",
            f"Total kyoku: {summary['total_kyoku']}",
            "",
            "| model | kyoku | agari% | houjuu% | riichi% | fuuro kyoku% | dahai decisions | ニシキ acc/sim/bad | カガシ acc/sim/bad |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in summary["models"]:
            def naga_cell(name: str) -> str:
                item = row["naga"].get(name) or {}
                values = [item.get("accuracy"), item.get("similarity"), item.get("bad_move_rate")]
                return " / ".join("—" if value is None else f"{value:.1f}" for value in values)

            lines.append(
                "| {model} | {kyoku} | {hora_rate:.1f} | {hoju_rate:.1f} | {reach_rate:.1f} | {huro_kyoku_rate:.1f} | {dahai_decisions} | {nishiki} | {kagashi} |".format(
                    model=row["model"],
                    kyoku=row["kyoku"],
                    hora_rate=row["hora_rate"] or 0.0,
                    hoju_rate=row["hoju_rate"] or 0.0,
                    reach_rate=row["reach_rate"] or 0.0,
                    huro_kyoku_rate=row["huro_kyoku_rate"] or 0.0,
                    dahai_decisions=row["dahai_decisions"],
                    nishiki=naga_cell("ニシキ"),
                    kagashi=naga_cell("カガシ"),
                )
            )
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
