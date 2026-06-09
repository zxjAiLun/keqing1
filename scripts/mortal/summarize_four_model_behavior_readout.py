#!/usr/bin/env python3
"""Summarize behavior metrics from four-model arena mjai logs."""
from __future__ import annotations

import argparse
import glob
import gzip
import json
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


CALL_TYPES = {"chi", "pon", "daiminkan", "ankan", "kakan"}


@dataclass
class PlayerRound:
    label: str
    agari: bool = False
    dealin: bool = False
    fuuro: bool = False
    riichi: bool = False
    call_events: int = 0
    delta_score: float = 0.0
    agari_delta_score: float | None = None


@dataclass
class Accumulator:
    player_rounds: int = 0
    agari: int = 0
    dealin: int = 0
    fuuro_rounds: int = 0
    riichi_rounds: int = 0
    call_events: int = 0
    agari_after_fuuro: int = 0
    dealin_after_fuuro: int = 0
    agari_after_riichi: int = 0
    dealin_after_riichi: int = 0
    winning_delta_sum: float = 0.0
    winning_delta_count: int = 0
    open_winning_delta_sum: float = 0.0
    open_winning_delta_count: int = 0
    fuuro_delta_sum: float = 0.0
    fuuro_delta_count: int = 0
    round_delta_sum: float = 0.0

    def add(self, record: PlayerRound) -> None:
        self.player_rounds += 1
        self.agari += int(record.agari)
        self.dealin += int(record.dealin)
        self.fuuro_rounds += int(record.fuuro)
        self.riichi_rounds += int(record.riichi)
        self.call_events += int(record.call_events)
        self.round_delta_sum += float(record.delta_score)
        if record.fuuro:
            self.agari_after_fuuro += int(record.agari)
            self.dealin_after_fuuro += int(record.dealin)
            self.fuuro_delta_sum += float(record.delta_score)
            self.fuuro_delta_count += 1
        if record.riichi:
            self.agari_after_riichi += int(record.agari)
            self.dealin_after_riichi += int(record.dealin)
        if record.agari and record.agari_delta_score is not None:
            self.winning_delta_sum += float(record.agari_delta_score)
            self.winning_delta_count += 1
            if record.fuuro:
                self.open_winning_delta_sum += float(record.agari_delta_score)
                self.open_winning_delta_count += 1

    def metrics(self) -> dict[str, Any]:
        return {
            "player_rounds": self.player_rounds,
            "agari_count": self.agari,
            "dealin_count": self.dealin,
            "fuuro_round_count": self.fuuro_rounds,
            "riichi_round_count": self.riichi_rounds,
            "call_event_count": self.call_events,
            "agari_rate": div(self.agari, self.player_rounds),
            "dealin_rate": div(self.dealin, self.player_rounds),
            "fuuro_rate": div(self.fuuro_rounds, self.player_rounds),
            "riichi_rate": div(self.riichi_rounds, self.player_rounds),
            "call_events_per_round": div(self.call_events, self.player_rounds),
            "agari_rate_after_fuuro": div(self.agari_after_fuuro, self.fuuro_rounds),
            "dealin_rate_after_fuuro": div(self.dealin_after_fuuro, self.fuuro_rounds),
            "agari_rate_after_riichi": div(self.agari_after_riichi, self.riichi_rounds),
            "dealin_rate_after_riichi": div(self.dealin_after_riichi, self.riichi_rounds),
            "avg_delta_score_per_round": div(self.round_delta_sum, self.player_rounds),
            "avg_winning_delta_score": div(self.winning_delta_sum, self.winning_delta_count),
            "avg_open_winning_delta_score": div(self.open_winning_delta_sum, self.open_winning_delta_count),
            "avg_call_delta_score": div(self.fuuro_delta_sum, self.fuuro_delta_count),
        }


def div(num: float, den: int) -> float | None:
    if not den:
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


def parse_log(path: Path) -> tuple[list[PlayerRound], bool]:
    names = [str(seat) for seat in range(4)]
    current: list[PlayerRound] | None = None
    records: list[PlayerRound] = []
    saw_start_game = False
    for event in iter_events(path):
        event_type = str(event.get("type", ""))
        if event_type == "start_game":
            saw_start_game = True
            event_names = event.get("names")
            if isinstance(event_names, list) and len(event_names) >= 4:
                names = [str(name) for name in event_names[:4]]
        elif event_type == "start_kyoku":
            if current is not None:
                records.extend(current)
            current = [PlayerRound(label=names[seat]) for seat in range(4)]
        elif current is None:
            continue
        elif event_type in CALL_TYPES:
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                current[actor].fuuro = True
                current[actor].call_events += 1
        elif event_type == "reach":
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                current[actor].riichi = True
        elif event_type == "hora":
            actor = event.get("actor")
            target = event.get("target")
            deltas = parse_deltas(event)
            for seat, delta in enumerate(deltas):
                current[seat].delta_score += float(delta)
            if isinstance(actor, int) and 0 <= actor < 4:
                current[actor].agari = True
                current[actor].agari_delta_score = float(deltas[actor])
            if isinstance(target, int) and 0 <= target < 4 and target != actor:
                current[target].dealin = True
        elif event_type == "ryukyoku":
            deltas = parse_deltas(event)
            for seat, delta in enumerate(deltas):
                current[seat].delta_score += float(delta)
        elif event_type == "end_kyoku":
            records.extend(current)
            current = None
    if current is not None:
        records.extend(current)
    return records, saw_start_game


def parse_deltas(event: Mapping[str, Any]) -> list[float]:
    raw = event.get("deltas", [0, 0, 0, 0])
    if not isinstance(raw, list) or len(raw) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(raw[seat] or 0.0) for seat in range(4)]


def summarize(logs: Sequence[str | Path]) -> dict[str, Any]:
    files = expand_logs(logs)
    accs: dict[str, Accumulator] = defaultdict(Accumulator)
    games = 0
    for path in files:
        records, has_game = parse_log(path)
        games += int(has_game)
        for record in records:
            accs[record.label].add(record)
    return {
        "schema": "keqing.mortal.four_model_behavior_readout.v1",
        "log_files": len(files),
        "games": games,
        "players": {label: acc.metrics() for label, acc in sorted(accs.items())},
    }


def format_markdown(report: Mapping[str, Any]) -> str:
    players = dict(report["players"])
    labels = list(players)
    rows = [
        ("Player rounds", "player_rounds", "count"),
        ("Agari", "agari_rate", "rate"),
        ("Dealin", "dealin_rate", "rate"),
        ("Fuuro rate", "fuuro_rate", "rate"),
        ("Riichi rate", "riichi_rate", "rate"),
        ("Call events/round", "call_events_per_round", "count"),
        ("After-fuuro agari", "agari_rate_after_fuuro", "rate"),
        ("After-fuuro dealin", "dealin_rate_after_fuuro", "rate"),
        ("After-riichi agari", "agari_rate_after_riichi", "rate"),
        ("After-riichi dealin", "dealin_rate_after_riichi", "rate"),
        ("Avg winning delta score", "avg_winning_delta_score", "score"),
        ("Avg open winning delta score", "avg_open_winning_delta_score", "score"),
        ("Avg call delta score", "avg_call_delta_score", "score"),
        ("Avg round delta score", "avg_delta_score_per_round", "score"),
    ]
    lines = [
        "# Four-Model Behavior Readout",
        "",
        f"- Games: `{report['games']}`",
        f"- Log files: `{report['log_files']}`",
        "",
        "| Metric | " + " | ".join(f"`{label}`" for label in labels) + " |",
        "| --- | " + " | ".join("---:" for _ in labels) + " |",
    ]
    for title, key, kind in rows:
        lines.append("| " + " | ".join([title, *[format_value(players[label].get(key), kind) for label in labels]]) + " |")
    if "T1_71000" in players:
        lines.extend(format_t1_distance(players))
    lines.append("")
    return "\n".join(lines)


def format_t1_distance(players: Mapping[str, Mapping[str, Any]]) -> list[str]:
    metric_keys = (
        "agari_rate",
        "dealin_rate",
        "fuuro_rate",
        "riichi_rate",
        "agari_rate_after_fuuro",
        "dealin_rate_after_fuuro",
        "agari_rate_after_riichi",
        "dealin_rate_after_riichi",
    )
    t1 = players["T1_71000"]
    lines = ["", "## T1 Shape Distance", ""]
    for other in ("80k_game", "model_v4"):
        if other not in players:
            continue
        diffs = []
        for key in metric_keys:
            left = t1.get(key)
            right = players[other].get(key)
            if isinstance(left, int | float) and isinstance(right, int | float):
                diffs.append(abs(float(left) - float(right)))
        if diffs:
            lines.append(f"- Mean absolute rate gap to `{other}`: `{sum(diffs) / len(diffs) * 100.0:.2f}pp`")
    return lines


def format_value(value: Any, kind: str) -> str:
    if value is None:
        return "NA"
    if kind == "rate":
        return f"{float(value) * 100.0:.2f}%"
    if kind == "score":
        return f"{float(value):.1f}"
    if kind == "count" and isinstance(value, int):
        return str(value)
    return f"{float(value):.4f}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", action="append", required=True, help="Log directory or glob. Can be repeated.")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = summarize(args.logs)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "behavior_readout.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown = format_markdown(report)
    (args.output_dir / "behavior_readout.md").write_text(markdown, encoding="utf-8")
    print(markdown, end="")


if __name__ == "__main__":
    main()
