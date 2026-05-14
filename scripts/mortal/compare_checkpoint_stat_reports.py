"""Compare two Mortal detailed stat reports and write a compact delta report."""

from __future__ import annotations

import argparse
import json
from math import isfinite
from pathlib import Path
from typing import Any


METRICS: tuple[tuple[str, str, str, str], ...] = (
    ("Win rate", "derived", "agari_rate", "rate"),
    ("Deal-in rate", "derived", "houjuu_rate", "rate"),
    ("Call rate", "derived", "fuuro_rate", "rate"),
    ("Riichi rate", "derived", "riichi_rate", "rate"),
    ("Ryukyoku rate", "derived", "ryukyoku_rate", "rate"),
    ("Avg winning delta score", "derived", "avg_point_per_agari", "score"),
    ("Avg deal-in delta score", "derived", "avg_point_per_houjuu", "score"),
    ("Winning rate after riichi", "derived", "agari_rate_after_riichi", "rate"),
    ("Deal-in rate after riichi", "derived", "houjuu_rate_after_riichi", "rate"),
    ("Winning rate after call", "derived", "agari_rate_after_fuuro", "rate"),
    ("Deal-in rate after call", "derived", "houjuu_rate_after_fuuro", "rate"),
    ("Avg riichi turn", "derived", "avg_riichi_jun", "turn"),
    ("Avg deal-in turn", "derived", "avg_houjuu_jun", "turn"),
    ("Avg number of calls", "derived", "avg_fuuro_num", "count"),
    ("Chasing riichi rate", "derived", "chasing_riichi_rate", "rate"),
    ("Riichi chased rate", "derived", "riichi_chased_rate", "rate"),
    ("Avg rank", "derived", "avg_rank", "rank"),
    ("Tenhou avg rank pt", "derived", "avg_rank_pt", "pt"),
)


def load_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path)
    if report_path.is_dir():
        report_path = report_path / "detailed_stats.json"
    return json.loads(report_path.read_text(encoding="utf-8"))


def select_player(report: dict[str, Any], label: str | None) -> tuple[str, dict[str, Any]]:
    players = report.get("players", {})
    if not players:
        raise ValueError("report contains no players")
    if label:
        if label not in players:
            raise ValueError(f"player label {label!r} not found; available: {', '.join(players)}")
        return label, players[label]
    if len(players) != 1:
        raise ValueError(f"report has multiple players; pass a label. available: {', '.join(players)}")
    only_label = next(iter(players))
    return only_label, players[only_label]


def metric_value(player: dict[str, Any], group: str, key: str) -> float | None:
    value = player.get(group, {}).get(key)
    if isinstance(value, int | float) and isfinite(float(value)):
        return float(value)
    return None


def build_delta(
    *,
    left_report: dict[str, Any],
    right_report: dict[str, Any],
    left_label: str,
    right_label: str,
    left_player_label: str | None = None,
    right_player_label: str | None = None,
) -> dict[str, Any]:
    selected_left_label, left_player = select_player(left_report, left_player_label)
    selected_right_label, right_player = select_player(right_report, right_player_label)

    rows: list[dict[str, Any]] = []
    for title, group, key, kind in METRICS:
        left = metric_value(left_player, group, key)
        right = metric_value(right_player, group, key)
        delta = None if left is None or right is None else right - left
        rows.append(
            {
                "metric": title,
                "group": group,
                "key": key,
                "kind": kind,
                "left": left,
                "right": right,
                "delta": delta,
            }
        )

    return {
        "schema": "keqing.mortal.behavior_stat_delta.v1",
        "left_label": left_label,
        "right_label": right_label,
        "left_player_label": selected_left_label,
        "right_player_label": selected_right_label,
        "left_log_dir": left_report.get("log_dir"),
        "right_log_dir": right_report.get("log_dir"),
        "rank_points_profile": right_report.get("rank_points_profile", left_report.get("rank_points_profile")),
        "rank_points_values": right_report.get("rank_points_values", left_report.get("rank_points_values")),
        "rows": rows,
        "diagnosis": diagnose(rows, left_label=left_label, right_label=right_label),
    }


def diagnose(rows: list[dict[str, Any]], *, left_label: str, right_label: str) -> list[str]:
    by_key = {row["key"]: row for row in rows}

    def delta(key: str) -> float | None:
        return by_key.get(key, {}).get("delta")

    notes: list[str] = []
    agari = delta("agari_rate")
    houjuu = delta("houjuu_rate")
    fuuro = delta("fuuro_rate")
    riichi = delta("riichi_rate")
    after_fuuro_houjuu = delta("houjuu_rate_after_fuuro")
    after_riichi_houjuu = delta("houjuu_rate_after_riichi")
    after_fuuro_agari = delta("agari_rate_after_fuuro")
    after_riichi_agari = delta("agari_rate_after_riichi")

    if fuuro is not None and fuuro > 0:
        notes.append(f"{right_label} calls more often than {left_label} (+{format_percent(fuuro)}).")
    if riichi is not None and riichi > 0:
        notes.append(f"{right_label} declares riichi more often than {left_label} (+{format_percent(riichi)}).")
    if agari is not None and houjuu is not None:
        notes.append(
            f"{right_label} win rate changes by {format_signed_percent(agari)}, "
            f"while deal-in rate changes by {format_signed_percent(houjuu)}."
        )
    if after_fuuro_houjuu is not None and after_fuuro_agari is not None:
        notes.append(
            f"After calls, win rate changes by {format_signed_percent(after_fuuro_agari)} and "
            f"deal-in rate changes by {format_signed_percent(after_fuuro_houjuu)}."
        )
    if after_riichi_houjuu is not None and after_riichi_agari is not None:
        notes.append(
            f"After riichi, win rate changes by {format_signed_percent(after_riichi_agari)} and "
            f"deal-in rate changes by {format_signed_percent(after_riichi_houjuu)}."
        )
    if houjuu is not None and agari is not None and houjuu > agari:
        notes.append("The added aggression looks risk-heavy at L1: deal-in rose more than win rate.")
    return notes


def format_markdown(delta: dict[str, Any]) -> str:
    left_label = delta["left_label"]
    right_label = delta["right_label"]
    lines = [
        "# Default Checkpoint Behavior Stat Delta",
        "",
        f"- Left baseline: `{left_label}` (`{delta['left_log_dir']}`)",
        f"- Right candidate: `{right_label}` (`{delta['right_log_dir']}`)",
        f"- Rank point profile: `{delta.get('rank_points_profile', 'unknown')}`",
        "",
        "| Metric | "
        + left_label
        + " | "
        + right_label
        + " | Delta (right-left) |",
        "| --- | ---: | ---: | ---: |",
    ]
    for row in delta["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["metric"],
                    format_metric(row["left"], row["kind"]),
                    format_metric(row["right"], row["kind"]),
                    format_delta(row["delta"], row["kind"]),
                ]
            )
            + " |"
        )
    lines.extend(["", "## L1 Diagnosis", ""])
    for note in delta["diagnosis"]:
        lines.append(f"- {note}")
    lines.append("")
    return "\n".join(lines)


def format_metric(value: float | None, kind: str) -> str:
    if value is None:
        return "NA"
    if kind == "rate":
        return format_percent(value)
    if kind in {"score", "pt"}:
        return f"{value:.2f}"
    if kind == "turn":
        return f"{value:.2f}"
    if kind == "rank":
        return f"{value:.4f}"
    return f"{value:.4f}"


def format_delta(value: float | None, kind: str) -> str:
    if value is None:
        return "NA"
    if kind == "rate":
        return format_signed_percent(value)
    if kind in {"score", "pt"}:
        return f"{value:+.2f}"
    if kind == "turn":
        return f"{value:+.2f}"
    if kind == "rank":
        return f"{value:+.4f}"
    return f"{value:+.4f}"


def format_percent(value: float) -> str:
    return f"{value * 100.0:.2f}%"


def format_signed_percent(value: float) -> str:
    return f"{value * 100.0:+.2f}pp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--left-player-label")
    parser.add_argument("--right-player-label")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    delta = build_delta(
        left_report=load_report(args.left_report),
        right_report=load_report(args.right_report),
        left_label=args.left_label,
        right_label=args.right_label,
        left_player_label=args.left_player_label,
        right_player_label=args.right_player_label,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stat_delta.json").write_text(
        json.dumps(delta, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "stat_delta.md").write_text(format_markdown(delta), encoding="utf-8")
    print(format_markdown(delta), end="")


if __name__ == "__main__":
    main()
