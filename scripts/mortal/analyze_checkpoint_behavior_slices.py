"""Analyze L2 behavior slices from Mortal/libriichi mjai logs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import glob


METRIC_FIELDS = (
    "player_rounds",
    "agari_rate",
    "houjuu_rate",
    "fuuro_rate",
    "riichi_rate",
    "ryukyoku_rate",
    "agari_after_riichi_rate",
    "houjuu_after_riichi_rate",
    "agari_after_fuuro_rate",
    "houjuu_after_fuuro_rate",
    "avg_first_riichi_turn",
    "avg_first_fuuro_turn",
    "avg_agari_turn",
    "avg_houjuu_turn",
)


@dataclass
class PlayerRound:
    model_label: str
    seat: int
    dealer: bool
    round_stage: str
    start_rank: int
    score_bucket: str
    agari: bool = False
    houjuu: bool = False
    ryukyoku: bool = False
    riichi: bool = False
    fuuro: bool = False
    first_riichi_turn: int | None = None
    first_fuuro_turn: int | None = None
    agari_turn: int | None = None
    houjuu_turn: int | None = None


@dataclass
class KyokuState:
    names: list[str]
    oya: int
    bakaze: str
    kyoku: int
    scores: list[int]
    turns: list[int] = field(default_factory=lambda: [0, 0, 0, 0])
    records: list[PlayerRound] = field(default_factory=list)


@dataclass
class Accumulator:
    player_rounds: int = 0
    agari: int = 0
    houjuu: int = 0
    fuuro: int = 0
    riichi: int = 0
    ryukyoku: int = 0
    riichi_den: int = 0
    riichi_agari: int = 0
    riichi_houjuu: int = 0
    fuuro_den: int = 0
    fuuro_agari: int = 0
    fuuro_houjuu: int = 0
    riichi_turn_sum: float = 0.0
    riichi_turn_count: int = 0
    fuuro_turn_sum: float = 0.0
    fuuro_turn_count: int = 0
    agari_turn_sum: float = 0.0
    agari_turn_count: int = 0
    houjuu_turn_sum: float = 0.0
    houjuu_turn_count: int = 0

    def add(self, record: PlayerRound) -> None:
        self.player_rounds += 1
        self.agari += int(record.agari)
        self.houjuu += int(record.houjuu)
        self.fuuro += int(record.fuuro)
        self.riichi += int(record.riichi)
        self.ryukyoku += int(record.ryukyoku)
        if record.riichi:
            self.riichi_den += 1
            self.riichi_agari += int(record.agari)
            self.riichi_houjuu += int(record.houjuu)
        if record.fuuro:
            self.fuuro_den += 1
            self.fuuro_agari += int(record.agari)
            self.fuuro_houjuu += int(record.houjuu)
        if record.first_riichi_turn is not None:
            self.riichi_turn_sum += record.first_riichi_turn
            self.riichi_turn_count += 1
        if record.first_fuuro_turn is not None:
            self.fuuro_turn_sum += record.first_fuuro_turn
            self.fuuro_turn_count += 1
        if record.agari_turn is not None:
            self.agari_turn_sum += record.agari_turn
            self.agari_turn_count += 1
        if record.houjuu_turn is not None:
            self.houjuu_turn_sum += record.houjuu_turn
            self.houjuu_turn_count += 1

    def metrics(self) -> dict[str, Any]:
        return {
            "player_rounds": self.player_rounds,
            "agari_rate": safe_div(self.agari, self.player_rounds),
            "houjuu_rate": safe_div(self.houjuu, self.player_rounds),
            "fuuro_rate": safe_div(self.fuuro, self.player_rounds),
            "riichi_rate": safe_div(self.riichi, self.player_rounds),
            "ryukyoku_rate": safe_div(self.ryukyoku, self.player_rounds),
            "agari_after_riichi_rate": safe_div(self.riichi_agari, self.riichi_den),
            "houjuu_after_riichi_rate": safe_div(self.riichi_houjuu, self.riichi_den),
            "agari_after_fuuro_rate": safe_div(self.fuuro_agari, self.fuuro_den),
            "houjuu_after_fuuro_rate": safe_div(self.fuuro_houjuu, self.fuuro_den),
            "avg_first_riichi_turn": safe_div(self.riichi_turn_sum, self.riichi_turn_count),
            "avg_first_fuuro_turn": safe_div(self.fuuro_turn_sum, self.fuuro_turn_count),
            "avg_agari_turn": safe_div(self.agari_turn_sum, self.agari_turn_count),
            "avg_houjuu_turn": safe_div(self.houjuu_turn_sum, self.houjuu_turn_count),
        }


def safe_div(num: float, den: int) -> float | None:
    if den == 0:
        return None
    return num / den


def expand_logs(patterns: Sequence[str | Path]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            files.extend(sorted(path.glob("**/*.json.gz")))
        else:
            files.extend(Path(match) for match in sorted(glob.glob(str(pattern), recursive=True)))
    return sorted(dict.fromkeys(files))


def iter_events(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def analyze_logs(patterns: Sequence[str | Path], *, model_label: str) -> dict[str, Any]:
    files = expand_logs(patterns)
    records: list[PlayerRound] = []
    game_count = 0
    for path in files:
        game_records, has_game = parse_game(path, model_label=model_label)
        records.extend(game_records)
        game_count += int(has_game)
    return {
        "schema": "keqing.mortal.behavior_slices.v1",
        "model_label": model_label,
        "log_files": len(files),
        "game_count": game_count,
        "player_rounds": len(records),
        "rows": slice_rows(records, model_label=model_label),
    }


def parse_game(path: Path, *, model_label: str) -> tuple[list[PlayerRound], bool]:
    names = [str(i) for i in range(4)]
    state: KyokuState | None = None
    records: list[PlayerRound] = []
    saw_start_game = False
    for event in iter_events(path):
        event_type = event.get("type")
        if event_type == "start_game":
            saw_start_game = True
            names = [str(name) for name in event.get("names", names)]
        elif event_type == "start_kyoku":
            if state is not None:
                records.extend(finalize_kyoku(state, ryukyoku=False))
            state = start_kyoku(event, names=names, model_label=model_label)
        elif state is None:
            continue
        elif event_type == "tsumo":
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                state.turns[actor] += 1
        elif event_type == "reach":
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                record = state.records[actor]
                record.riichi = True
                record.first_riichi_turn = record.first_riichi_turn or max(1, state.turns[actor])
        elif event_type in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                record = state.records[actor]
                record.fuuro = True
                record.first_fuuro_turn = record.first_fuuro_turn or max(1, state.turns[actor])
        elif event_type == "hora":
            actor = event.get("actor")
            target = event.get("target")
            if isinstance(actor, int) and 0 <= actor < 4:
                state.records[actor].agari = True
                state.records[actor].agari_turn = max(1, state.turns[actor])
            if isinstance(target, int) and 0 <= target < 4 and target != actor:
                state.records[target].houjuu = True
                state.records[target].houjuu_turn = max(1, state.turns[target])
        elif event_type == "ryukyoku":
            for record in state.records:
                record.ryukyoku = True
        elif event_type == "end_kyoku":
            records.extend(finalize_kyoku(state, ryukyoku=any(record.ryukyoku for record in state.records)))
            state = None
    if state is not None:
        records.extend(finalize_kyoku(state, ryukyoku=False))
    return records, saw_start_game


def start_kyoku(event: dict[str, Any], *, names: list[str], model_label: str) -> KyokuState:
    scores = [int(score) for score in event.get("scores", [25000, 25000, 25000, 25000])]
    oya = int(event.get("oya", 0))
    bakaze = str(event.get("bakaze", "E"))
    kyoku = int(event.get("kyoku", 1))
    start_ranks = score_ranks(scores)
    records = [
        PlayerRound(
            model_label=model_label,
            seat=seat,
            dealer=seat == oya,
            round_stage=round_stage(bakaze, kyoku),
            start_rank=start_ranks[seat],
            score_bucket=score_bucket(scores[seat] - max(score for idx, score in enumerate(scores) if idx != seat)),
        )
        for seat in range(4)
    ]
    return KyokuState(names=names, oya=oya, bakaze=bakaze, kyoku=kyoku, scores=scores, records=records)


def finalize_kyoku(state: KyokuState, *, ryukyoku: bool) -> list[PlayerRound]:
    if ryukyoku:
        for record in state.records:
            record.ryukyoku = True
    return state.records


def score_ranks(scores: Sequence[int]) -> list[int]:
    ordered = sorted(range(4), key=lambda seat: (-scores[seat], seat))
    ranks = [0, 0, 0, 0]
    for rank, seat in enumerate(ordered, 1):
        ranks[seat] = rank
    return ranks


def round_stage(bakaze: str, kyoku: int) -> str:
    if bakaze == "S" and kyoku >= 4:
        return "all_last"
    if bakaze == "S":
        return "south"
    if bakaze == "E":
        return "east"
    return str(bakaze).lower() or "unknown"


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


def turn_bucket(turn: int | None) -> str | None:
    if turn is None:
        return None
    if turn <= 6:
        return "early_1_6"
    if turn <= 12:
        return "middle_7_12"
    return "late_13_plus"


def slice_rows(records: Sequence[PlayerRound], *, model_label: str) -> list[dict[str, Any]]:
    accs: dict[tuple[str, str], Accumulator] = defaultdict(Accumulator)
    for record in records:
        add_slices(accs, record)
    rows = []
    for (slice_dim, slice_value), acc in sorted(accs.items()):
        rows.append(
            {
                "model_label": model_label,
                "slice_dim": slice_dim,
                "slice_value": slice_value,
                **acc.metrics(),
            }
        )
    return rows


def add_slices(accs: dict[tuple[str, str], Accumulator], record: PlayerRound) -> None:
    values = [
        ("all", "all"),
        ("dealer", "dealer" if record.dealer else "nondealer"),
        ("start_rank", str(record.start_rank)),
        ("round_stage", record.round_stage),
        ("score_bucket", record.score_bucket),
    ]
    for bucket_name, turn in (
        ("first_riichi_turn_bucket", record.first_riichi_turn),
        ("first_fuuro_turn_bucket", record.first_fuuro_turn),
        ("agari_turn_bucket", record.agari_turn),
        ("houjuu_turn_bucket", record.houjuu_turn),
    ):
        bucket = turn_bucket(turn)
        if bucket is not None:
            values.append((bucket_name, bucket))
    for key in values:
        accs[key].add(record)


def diff_rows(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    left_rows = {(row["slice_dim"], row["slice_value"]): row for row in left["rows"]}
    right_rows = {(row["slice_dim"], row["slice_value"]): row for row in right["rows"]}
    rows = []
    for key in sorted(set(left_rows) | set(right_rows)):
        left_row = left_rows.get(key, {})
        right_row = right_rows.get(key, {})
        row = {"slice_dim": key[0], "slice_value": key[1]}
        for field_name in METRIC_FIELDS:
            left_value = left_row.get(field_name)
            right_value = right_row.get(field_name)
            row[f"left_{field_name}"] = left_value
            row[f"right_{field_name}"] = right_value
            row[f"delta_{field_name}"] = numeric_delta(left_value, right_value)
        rows.append(row)
    return rows


def numeric_delta(left: Any, right: Any) -> float | None:
    if isinstance(left, int | float) and isinstance(right, int | float):
        return float(right) - float(left)
    return None


def write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_markdown(left: dict[str, Any], right: dict[str, Any], diffs: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Default Checkpoint Behavior Slice Report",
        "",
        f"- Left baseline: `{left['model_label']}`",
        f"- Right candidate: `{right['model_label']}`",
        f"- Left games/logs/player-rounds: `{left['game_count']}` / `{left['log_files']}` / `{left['player_rounds']}`",
        f"- Right games/logs/player-rounds: `{right['game_count']}` / `{right['log_files']}` / `{right['player_rounds']}`",
        "",
        "## Primary Slice Deltas",
        "",
    ]
    for slice_dim in ("all", "dealer", "start_rank", "round_stage", "score_bucket"):
        rows = [row for row in diffs if row["slice_dim"] == slice_dim]
        if not rows:
            continue
        lines.extend(format_slice_table(slice_dim, rows))
    lines.extend(["", "## Turn Bucket Signals", ""])
    for slice_dim in ("first_fuuro_turn_bucket", "first_riichi_turn_bucket", "houjuu_turn_bucket"):
        rows = [row for row in diffs if row["slice_dim"] == slice_dim]
        if rows:
            lines.extend(format_slice_table(slice_dim, rows))
    lines.extend(["", "## L2 Diagnosis", ""])
    lines.extend(f"- {note}" for note in diagnose_diffs(diffs))
    lines.append("")
    return "\n".join(lines)


def format_slice_table(slice_dim: str, rows: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        f"### `{slice_dim}`",
        "",
        "| Slice | rounds | agari d | houjuu d | fuuro d | riichi d | after fuuro agari d | after fuuro houjuu d | after riichi houjuu d |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["slice_value"]),
                    format_rounds(row.get("right_player_rounds")),
                    fmt_pp(row.get("delta_agari_rate")),
                    fmt_pp(row.get("delta_houjuu_rate")),
                    fmt_pp(row.get("delta_fuuro_rate")),
                    fmt_pp(row.get("delta_riichi_rate")),
                    fmt_pp(row.get("delta_agari_after_fuuro_rate")),
                    fmt_pp(row.get("delta_houjuu_after_fuuro_rate")),
                    fmt_pp(row.get("delta_houjuu_after_riichi_rate")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def diagnose_diffs(diffs: Sequence[dict[str, Any]]) -> list[str]:
    by_key = {(row["slice_dim"], row["slice_value"]): row for row in diffs}
    notes: list[str] = []
    all_row = by_key.get(("all", "all"))
    if all_row:
        notes.append(
            "Overall 80k shifts toward more calls "
            f"({fmt_pp(all_row.get('delta_fuuro_rate'))}) and more deal-ins "
            f"({fmt_pp(all_row.get('delta_houjuu_rate'))})."
        )
        notes.append(
            "Post-call outcomes deteriorate more than post-riichi outcomes: "
            f"after-call agari {fmt_pp(all_row.get('delta_agari_after_fuuro_rate'))}, "
            f"after-call houjuu {fmt_pp(all_row.get('delta_houjuu_after_fuuro_rate'))}, "
            f"after-riichi houjuu {fmt_pp(all_row.get('delta_houjuu_after_riichi_rate'))}."
        )
    for slice_value in ("1", "2", "3", "4"):
        row = by_key.get(("start_rank", slice_value))
        if row and positive(row.get("delta_fuuro_rate")) and positive(row.get("delta_houjuu_rate")):
            notes.append(
                f"Start-rank {slice_value} shows both higher call rate "
                f"({fmt_pp(row.get('delta_fuuro_rate'))}) and higher deal-in rate "
                f"({fmt_pp(row.get('delta_houjuu_rate'))})."
            )
    dealer = by_key.get(("dealer", "dealer"))
    nondealer = by_key.get(("dealer", "nondealer"))
    if dealer and nondealer:
        notes.append(
            "Dealer/nondealer split: dealer houjuu "
            f"{fmt_pp(dealer.get('delta_houjuu_rate'))}, nondealer houjuu "
            f"{fmt_pp(nondealer.get('delta_houjuu_rate'))}."
        )
    return notes


def positive(value: Any) -> bool:
    return isinstance(value, int | float) and value > 0


def format_rounds(value: Any) -> str:
    if isinstance(value, int | float):
        return str(int(value))
    return "NA"


def fmt_pp(value: Any) -> str:
    if not isinstance(value, int | float):
        return "NA"
    return f"{value * 100.0:+.2f}pp"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-label", required=True)
    parser.add_argument("--left-logs", action="append", required=True)
    parser.add_argument("--right-label", required=True)
    parser.add_argument("--right-logs", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    left = analyze_logs(args.left_logs, model_label=args.left_label)
    right = analyze_logs(args.right_logs, model_label=args.right_label)
    diffs = diff_rows(left, right)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "behavior_slices_left.json").write_text(
        json.dumps(left, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "behavior_slices_right.json").write_text(
        json.dumps(right, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "slice_metrics.csv", [*left["rows"], *right["rows"]])
    write_csv(args.output_dir / "slice_diff.csv", diffs)
    report = format_markdown(left, right, diffs)
    (args.output_dir / "behavior_slice_report.md").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
