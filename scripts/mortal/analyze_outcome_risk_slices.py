#!/usr/bin/env python3
"""Outcome-first risk slices from Mortal/libriichi mjai logs.

This is diagnostic only: it compares long-run outcomes after broad action
contexts. It does not treat any reference model's move as a correction label.
"""

from __future__ import annotations

import argparse
import csv
import glob
import gzip
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CALL_TYPES = {"chi", "pon", "daiminkan", "ankan", "kakan"}


@dataclass(frozen=True)
class Observation:
    model: str
    slice_name: str
    turn: int | None
    delta_score: float
    agari: bool
    houjuu: bool


@dataclass
class SeatState:
    model: str
    dealer: bool
    start_rank: int
    score_bucket: str
    turns: int = 0
    riichi: bool = False
    fuuro: bool = False
    agari: bool = False
    houjuu: bool = False
    delta_score: float = 0.0
    observations: list[tuple[str, int | None]] | None = None

    def __post_init__(self) -> None:
        if self.observations is None:
            self.observations = []


@dataclass
class Accumulator:
    count: int = 0
    agari: int = 0
    houjuu: int = 0
    delta_sum: float = 0.0
    turn_sum: float = 0.0
    turn_count: int = 0

    def add(self, obs: Observation) -> None:
        self.count += 1
        self.agari += int(obs.agari)
        self.houjuu += int(obs.houjuu)
        self.delta_sum += float(obs.delta_score)
        if obs.turn is not None:
            self.turn_sum += int(obs.turn)
            self.turn_count += 1

    def row(self, *, model: str, slice_name: str) -> dict[str, Any]:
        return {
            "model": model,
            "slice": slice_name,
            "count": self.count,
            "agari_rate": safe_div(self.agari, self.count),
            "houjuu_rate": safe_div(self.houjuu, self.count),
            "avg_delta_score": safe_div(self.delta_sum, self.count),
            "avg_turn": safe_div(self.turn_sum, self.turn_count),
        }


def safe_div(num: float, den: int) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def expand_logs(patterns: Sequence[str | Path]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            files.extend(sorted(path.glob("**/*.mjson")))
            files.extend(sorted(path.glob("**/*.jsonl")))
            files.extend(sorted(path.glob("**/*.json.gz")))
        else:
            files.extend(Path(match) for match in sorted(glob.glob(str(pattern), recursive=True)))
    return sorted(dict.fromkeys(files))


def iter_events(path: Path) -> Iterable[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def score_ranks(scores: Sequence[int]) -> list[int]:
    ordered = sorted(range(4), key=lambda seat: (-int(scores[seat]), seat))
    ranks = [0, 0, 0, 0]
    for rank, seat in enumerate(ordered, 1):
        ranks[seat] = rank
    return ranks


def score_bucket(diff_to_best_other: int) -> str:
    if diff_to_best_other >= 12000:
        return "ahead_big"
    if diff_to_best_other > 0:
        return "ahead_small"
    if diff_to_best_other >= -8000:
        return "near_even"
    if diff_to_best_other >= -18000:
        return "behind_small"
    return "behind_big"


def turn_bucket(turn: int) -> str:
    if turn <= 6:
        return "early_1_6"
    if turn <= 12:
        return "middle_7_12"
    return "late_13_plus"


def parse_deltas(event: Mapping[str, Any]) -> list[float]:
    raw = event.get("deltas", [0, 0, 0, 0])
    if not isinstance(raw, list) or len(raw) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(raw[seat] or 0.0) for seat in range(4)]


def add_obs(seat: SeatState, slice_name: str, turn: int | None = None) -> None:
    assert seat.observations is not None
    seat.observations.append((slice_name, turn))


def build_seats(event: Mapping[str, Any], names: Sequence[str]) -> list[SeatState]:
    scores = [int(score) for score in event.get("scores", [25000, 25000, 25000, 25000])]
    ranks = score_ranks(scores)
    oya = int(event.get("oya", 0))
    seats = []
    for seat in range(4):
        best_other = max(score for idx, score in enumerate(scores) if idx != seat)
        seats.append(
            SeatState(
                model=str(names[seat]),
                dealer=seat == oya,
                start_rank=ranks[seat],
                score_bucket=score_bucket(scores[seat] - best_other),
            )
        )
    return seats


def finalize_seats(seats: Sequence[SeatState]) -> list[Observation]:
    observations: list[Observation] = []
    for seat in seats:
        add_obs(seat, "round_all")
        add_obs(seat, "round_dealer" if seat.dealer else "round_nondealer")
        add_obs(seat, f"round_start_rank_{seat.start_rank}")
        add_obs(seat, f"round_score_{seat.score_bucket}")
        if seat.fuuro:
            add_obs(seat, "round_after_fuuro")
        if seat.riichi:
            add_obs(seat, "round_after_riichi")
        assert seat.observations is not None
        for slice_name, turn in seat.observations:
            observations.append(
                Observation(
                    model=seat.model,
                    slice_name=slice_name,
                    turn=turn,
                    delta_score=seat.delta_score,
                    agari=seat.agari,
                    houjuu=seat.houjuu,
                )
            )
    return observations


def parse_log(path: Path) -> tuple[list[Observation], bool]:
    names = [str(seat) for seat in range(4)]
    seats: list[SeatState] | None = None
    observations: list[Observation] = []
    saw_start_game = False
    for event in iter_events(path):
        event_type = str(event.get("type", ""))
        if event_type == "start_game":
            saw_start_game = True
            raw_names = event.get("names")
            if isinstance(raw_names, list) and len(raw_names) >= 4:
                names = [str(name) for name in raw_names[:4]]
        elif event_type == "start_kyoku":
            if seats is not None:
                observations.extend(finalize_seats(seats))
            seats = build_seats(event, names)
        elif seats is None:
            continue
        elif event_type == "tsumo":
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                seats[actor].turns += 1
        elif event_type == "reach":
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                seat = seats[actor]
                turn = max(1, seat.turns)
                if not seat.riichi:
                    add_obs(seat, "action_first_riichi", turn)
                    add_obs(seat, f"action_first_riichi_{turn_bucket(turn)}", turn)
                seat.riichi = True
        elif event_type in CALL_TYPES:
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                seat = seats[actor]
                turn = max(1, seat.turns)
                if not seat.fuuro:
                    add_obs(seat, "action_first_fuuro", turn)
                    add_obs(seat, f"action_first_fuuro_{turn_bucket(turn)}", turn)
                seat.fuuro = True
        elif event_type == "dahai":
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                seat = seats[actor]
                turn = max(1, seat.turns)
                opponents_riichi = any(other.riichi for idx, other in enumerate(seats) if idx != actor)
                add_obs(seat, "action_discard_all", turn)
                if bool(event.get("tsumogiri", False)):
                    add_obs(seat, "action_discard_tsumogiri", turn)
                else:
                    add_obs(seat, "action_discard_from_hand", turn)
                if seat.fuuro:
                    add_obs(seat, "action_discard_after_fuuro", turn)
                if seat.riichi:
                    add_obs(seat, "action_discard_after_riichi", turn)
                if opponents_riichi:
                    add_obs(seat, "action_discard_vs_riichi", turn)
                    add_obs(seat, f"action_discard_vs_riichi_{turn_bucket(turn)}", turn)
                    if seat.fuuro:
                        add_obs(seat, "action_discard_after_fuuro_vs_riichi", turn)
                if turn >= 13:
                    add_obs(seat, "action_discard_late_13_plus", turn)
        elif event_type == "hora":
            deltas = parse_deltas(event)
            for seat_idx, delta in enumerate(deltas):
                seats[seat_idx].delta_score += float(delta)
            actor = event.get("actor")
            target = event.get("target")
            if isinstance(actor, int) and 0 <= actor < 4:
                seats[actor].agari = True
            if isinstance(target, int) and 0 <= target < 4 and target != actor:
                seats[target].houjuu = True
        elif event_type == "ryukyoku":
            deltas = parse_deltas(event)
            for seat_idx, delta in enumerate(deltas):
                seats[seat_idx].delta_score += float(delta)
        elif event_type == "end_kyoku":
            observations.extend(finalize_seats(seats))
            seats = None
    if seats is not None:
        observations.extend(finalize_seats(seats))
    return observations, saw_start_game


def summarize(logs: Sequence[str | Path]) -> dict[str, Any]:
    files = expand_logs(logs)
    accs: dict[tuple[str, str], Accumulator] = defaultdict(Accumulator)
    games = 0
    obs_count = 0
    for path in files:
        observations, has_game = parse_log(path)
        games += int(has_game)
        obs_count += len(observations)
        for obs in observations:
            accs[(obs.model, obs.slice_name)].add(obs)
    rows = [acc.row(model=model, slice_name=slice_name) for (model, slice_name), acc in sorted(accs.items())]
    return {
        "schema": "keqing.mortal.outcome_risk_slices.v1",
        "log_files": len(files),
        "games": games,
        "observations": obs_count,
        "rows": rows,
    }


def metric_delta(left: Mapping[str, Any], right: Mapping[str, Any], key: str) -> float | None:
    left_value = left.get(key)
    right_value = right.get(key)
    if isinstance(left_value, int | float) and isinstance(right_value, int | float):
        return float(left_value) - float(right_value)
    return None


def compare_rows(
    rows: Sequence[Mapping[str, Any]], *, target_model: str, reference_model: str, min_count: int
) -> list[dict[str, Any]]:
    by_key = {(str(row["model"]), str(row["slice"])): row for row in rows}
    comparisons: list[dict[str, Any]] = []
    slices = sorted({str(row["slice"]) for row in rows})
    for slice_name in slices:
        target = by_key.get((target_model, slice_name))
        reference = by_key.get((reference_model, slice_name))
        if not target or not reference:
            continue
        if int(target.get("count", 0)) < min_count or int(reference.get("count", 0)) < min_count:
            continue
        comparisons.append(
            {
                "slice": slice_name,
                "target_count": target["count"],
                "reference_count": reference["count"],
                "target_agari_rate": target["agari_rate"],
                "reference_agari_rate": reference["agari_rate"],
                "delta_agari_rate": metric_delta(target, reference, "agari_rate"),
                "target_houjuu_rate": target["houjuu_rate"],
                "reference_houjuu_rate": reference["houjuu_rate"],
                "delta_houjuu_rate": metric_delta(target, reference, "houjuu_rate"),
                "target_avg_delta_score": target["avg_delta_score"],
                "reference_avg_delta_score": reference["avg_delta_score"],
                "delta_avg_delta_score": metric_delta(target, reference, "avg_delta_score"),
                "target_avg_turn": target["avg_turn"],
                "reference_avg_turn": reference["avg_turn"],
            }
        )
    return comparisons


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(value: Any) -> str:
    if not isinstance(value, int | float):
        return "NA"
    return f"{value * 100.0:.2f}%"


def fmt_delta_pct(value: Any) -> str:
    if not isinstance(value, int | float):
        return "NA"
    return f"{value * 100.0:+.2f}pp"


def fmt_score(value: Any) -> str:
    if not isinstance(value, int | float):
        return "NA"
    return f"{value:+.1f}"


def format_markdown(
    report: Mapping[str, Any],
    comparisons: Sequence[Mapping[str, Any]],
    *,
    target_model: str,
    reference_model: str,
) -> str:
    primary_slices = [
        "round_all",
        "round_after_fuuro",
        "round_after_riichi",
        "action_first_fuuro",
        "action_first_riichi",
        "action_discard_after_fuuro",
        "action_discard_vs_riichi",
        "action_discard_after_fuuro_vs_riichi",
        "action_discard_late_13_plus",
        "action_discard_vs_riichi_late_13_plus",
    ]
    by_slice = {str(row["slice"]): row for row in comparisons}
    lines = [
        "# Outcome Risk Slice Report",
        "",
        f"- Games: `{report['games']}`",
        f"- Log files: `{report['log_files']}`",
        f"- Target model: `{target_model}`",
        f"- Reference model: `{reference_model}`",
        "",
        "## Primary Slices",
        "",
        "| Slice | target n | ref n | target agari | ref agari | agari d | target houjuu | ref houjuu | houjuu d | target delta | ref delta | delta d |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for slice_name in primary_slices:
        row = by_slice.get(slice_name)
        if not row:
            continue
        lines.append(format_compare_row(row))
    lines.extend(["", "## Largest Target Disadvantages", ""])
    risk_rows = sorted(
        comparisons,
        key=lambda row: (
            float(row.get("delta_avg_delta_score") or 0.0),
            -float(row.get("delta_houjuu_rate") or 0.0),
        ),
    )[:12]
    lines.extend(
        [
            "| Slice | target n | ref n | agari d | houjuu d | delta d |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in risk_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["slice"]),
                    str(row["target_count"]),
                    str(row["reference_count"]),
                    fmt_delta_pct(row.get("delta_agari_rate")),
                    fmt_delta_pct(row.get("delta_houjuu_rate")),
                    fmt_score(row.get("delta_avg_delta_score")),
                ]
            )
            + " |"
        )
    lines.extend(["", "## Notes", ""])
    lines.append("- These are outcome-conditioned diagnostics, not move labels.")
    lines.append("- A bad slice means this action context deserves deeper sampling or controlled correction; it does not prove the reference model's action is correct.")
    lines.append("")
    return "\n".join(lines)


def format_compare_row(row: Mapping[str, Any]) -> str:
    return (
        "| "
        + " | ".join(
            [
                str(row["slice"]),
                str(row["target_count"]),
                str(row["reference_count"]),
                fmt_pct(row.get("target_agari_rate")),
                fmt_pct(row.get("reference_agari_rate")),
                fmt_delta_pct(row.get("delta_agari_rate")),
                fmt_pct(row.get("target_houjuu_rate")),
                fmt_pct(row.get("reference_houjuu_rate")),
                fmt_delta_pct(row.get("delta_houjuu_rate")),
                fmt_score(row.get("target_avg_delta_score")),
                fmt_score(row.get("reference_avg_delta_score")),
                fmt_score(row.get("delta_avg_delta_score")),
            ]
        )
        + " |"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", action="append", required=True, help="Log dir or glob. Repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--target-model", default="T1_71000")
    parser.add_argument("--reference-model", default="model_v4")
    parser.add_argument("--min-count", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = summarize(args.logs)
    comparisons = compare_rows(
        report["rows"],
        target_model=str(args.target_model),
        reference_model=str(args.reference_model),
        min_count=int(args.min_count),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "outcome_risk_slices.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "outcome_risk_slices.csv", report["rows"])
    write_csv(args.output_dir / "target_vs_reference_slices.csv", comparisons)
    markdown = format_markdown(
        report,
        comparisons,
        target_model=str(args.target_model),
        reference_model=str(args.reference_model),
    )
    (args.output_dir / "outcome_risk_slice_report.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
