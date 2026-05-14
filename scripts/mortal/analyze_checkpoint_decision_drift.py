"""Analyze decision-level Q margins from Mortal arena mjai logs."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from inference.mortal_bot import MORTAL_ACTION_SPACE
from inference.mortal_bot import _mortal_action_ids_for_mjai
from scripts.mortal.analyze_checkpoint_behavior_slices import expand_logs
from scripts.mortal.analyze_checkpoint_behavior_slices import iter_events
from scripts.mortal.analyze_checkpoint_behavior_slices import round_stage
from scripts.mortal.analyze_checkpoint_behavior_slices import score_bucket
from scripts.mortal.analyze_checkpoint_behavior_slices import score_ranks
from scripts.mortal.analyze_checkpoint_behavior_slices import turn_bucket


CALL_TYPES = {"chi", "pon", "daiminkan"}
KAN_TYPES = {"ankan", "kakan"}


@dataclass
class DecisionRecord:
    model_label: str
    decision_kind: str
    action_type: str
    actor: int
    dealer: bool
    start_rank: int
    round_stage: str
    score_bucket: str
    turn: int
    turn_bucket: str
    margin: float | None
    chosen_q: float | None
    alternative_q: float | None
    shanten: int | None
    agari: bool = False
    houjuu: bool = False
    ryukyoku: bool = False


@dataclass
class KyokuContext:
    oya: int
    bakaze: str
    kyoku: int
    scores: list[int]
    turns: list[int]
    start_ranks: list[int]
    decisions: list[DecisionRecord]


@dataclass
class Aggregate:
    count: int = 0
    margin_sum: float = 0.0
    margin_count: int = 0
    margin_pos: int = 0
    agari: int = 0
    houjuu: int = 0
    ryukyoku: int = 0
    shanten_sum: float = 0.0
    shanten_count: int = 0

    def add(self, record: DecisionRecord) -> None:
        self.count += 1
        if record.margin is not None:
            self.margin_sum += record.margin
            self.margin_count += 1
            self.margin_pos += int(record.margin > 0)
        self.agari += int(record.agari)
        self.houjuu += int(record.houjuu)
        self.ryukyoku += int(record.ryukyoku)
        if record.shanten is not None:
            self.shanten_sum += record.shanten
            self.shanten_count += 1

    def metrics(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "avg_margin": safe_div(self.margin_sum, self.margin_count),
            "positive_margin_rate": safe_div(self.margin_pos, self.margin_count),
            "agari_rate": safe_div(self.agari, self.count),
            "houjuu_rate": safe_div(self.houjuu, self.count),
            "ryukyoku_rate": safe_div(self.ryukyoku, self.count),
            "avg_shanten": safe_div(self.shanten_sum, self.shanten_count),
        }


def safe_div(num: float, den: int) -> float | None:
    if den == 0:
        return None
    return num / den


def expand_compact_meta(meta: dict[str, Any]) -> tuple[list[float], list[bool]]:
    mask_bits = int(meta.get("mask_bits", 0) or 0)
    compact_q = [float(value) for value in meta.get("q_values", [])]
    q_values = [float("-inf")] * MORTAL_ACTION_SPACE
    action_mask = [False] * MORTAL_ACTION_SPACE
    compact_idx = 0
    for action_id in range(MORTAL_ACTION_SPACE):
        if not (mask_bits & (1 << action_id)):
            continue
        action_mask[action_id] = True
        if compact_idx >= len(compact_q):
            raise RuntimeError("compact q_values shorter than mask_bits")
        q_values[action_id] = compact_q[compact_idx]
        compact_idx += 1
    if compact_idx != len(compact_q):
        raise RuntimeError("compact q_values longer than mask_bits")
    return q_values, action_mask


def action_score(action: dict[str, Any], *, q_values: Sequence[float], action_mask: Sequence[bool]) -> float | None:
    ids = [action_id for action_id in _mortal_action_ids_for_mjai(action) if action_mask[action_id]]
    if not ids:
        return None
    finite = [float(q_values[action_id]) for action_id in ids if q_values[action_id] != float("-inf")]
    if not finite:
        return None
    return max(finite)


def best_discard_score(q_values: Sequence[float], action_mask: Sequence[bool]) -> float | None:
    scores = [float(q_values[action_id]) for action_id in range(37) if action_mask[action_id]]
    scores = [score for score in scores if score != float("-inf")]
    if not scores:
        return None
    return max(scores)


def analyze_logs(patterns: Sequence[str | Path], *, model_label: str) -> dict[str, Any]:
    files = expand_logs(patterns)
    records: list[DecisionRecord] = []
    for path in files:
        records.extend(parse_game(path, model_label=model_label))
    return {
        "schema": "keqing.mortal.decision_drift.v1",
        "model_label": model_label,
        "log_files": len(files),
        "decision_count": len(records),
        "rows": aggregate_rows(records),
    }


def parse_game(path: Path, *, model_label: str) -> list[DecisionRecord]:
    ctx: KyokuContext | None = None
    records: list[DecisionRecord] = []
    for event in iter_events(path):
        event_type = str(event.get("type", ""))
        if event_type == "start_kyoku":
            if ctx is not None:
                records.extend(ctx.decisions)
            ctx = start_context(event)
            continue
        if ctx is None:
            continue
        if event_type == "tsumo":
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                ctx.turns[actor] += 1
            continue
        if event_type in {"hora", "ryukyoku"}:
            apply_terminal(ctx, event)
            continue
        if event_type == "end_kyoku":
            records.extend(ctx.decisions)
            ctx = None
            continue
        decision = decision_from_event(event, ctx=ctx, model_label=model_label)
        if decision is not None:
            ctx.decisions.append(decision)
    if ctx is not None:
        records.extend(ctx.decisions)
    return records


def start_context(event: dict[str, Any]) -> KyokuContext:
    scores = [int(score) for score in event.get("scores", [25000, 25000, 25000, 25000])]
    return KyokuContext(
        oya=int(event.get("oya", 0)),
        bakaze=str(event.get("bakaze", "E")),
        kyoku=int(event.get("kyoku", 1)),
        scores=scores,
        turns=[0, 0, 0, 0],
        start_ranks=score_ranks(scores),
        decisions=[],
    )


def apply_terminal(ctx: KyokuContext, event: dict[str, Any]) -> None:
    event_type = str(event.get("type", ""))
    if event_type == "ryukyoku":
        for decision in ctx.decisions:
            decision.ryukyoku = True
        return
    actor = event.get("actor")
    target = event.get("target")
    for decision in ctx.decisions:
        if decision.actor == actor:
            decision.agari = True
        if isinstance(target, int) and target != actor and decision.actor == target:
            decision.houjuu = True


def decision_from_event(event: dict[str, Any], *, ctx: KyokuContext, model_label: str) -> DecisionRecord | None:
    event_type = str(event.get("type", ""))
    actor = event.get("actor")
    meta = event.get("meta")
    if not isinstance(actor, int) or not (0 <= actor < 4) or not isinstance(meta, dict):
        return None
    try:
        q_values, action_mask = expand_compact_meta(meta)
    except RuntimeError:
        return None

    decision_kind: str | None = None
    margin: float | None = None
    chosen_q: float | None = None
    alternative_q: float | None = None
    if event_type in CALL_TYPES:
        decision_kind = "chosen_call"
        chosen_q = action_score(event, q_values=q_values, action_mask=action_mask)
        alternative_q = q_values[45] if action_mask[45] else None
        margin = None if chosen_q is None or alternative_q is None else chosen_q - alternative_q
    elif event_type == "reach":
        decision_kind = "chosen_reach"
        chosen_q = q_values[37] if action_mask[37] else None
        alternative_q = best_discard_score(q_values, action_mask)
        margin = None if chosen_q is None or alternative_q is None else chosen_q - alternative_q
    elif event_type == "dahai" and action_mask[37]:
        decision_kind = "declined_reach"
        chosen_q = action_score(event, q_values=q_values, action_mask=action_mask)
        alternative_q = q_values[37]
        margin = None if chosen_q is None or alternative_q is None else alternative_q - chosen_q
    elif event_type in KAN_TYPES:
        decision_kind = "chosen_kan"
        chosen_q = action_score(event, q_values=q_values, action_mask=action_mask)
        alternative_q = best_discard_score(q_values, action_mask)
        margin = None if chosen_q is None or alternative_q is None else chosen_q - alternative_q
    if decision_kind is None:
        return None

    turn = max(1, ctx.turns[actor])
    return DecisionRecord(
        model_label=model_label,
        decision_kind=decision_kind,
        action_type=event_type,
        actor=actor,
        dealer=actor == ctx.oya,
        start_rank=ctx.start_ranks[actor],
        round_stage=round_stage(ctx.bakaze, ctx.kyoku),
        score_bucket=score_bucket(ctx.scores[actor] - max(score for idx, score in enumerate(ctx.scores) if idx != actor)),
        turn=turn,
        turn_bucket=turn_bucket(turn) or "unknown",
        margin=margin,
        chosen_q=chosen_q,
        alternative_q=alternative_q,
        shanten=int(meta["shanten"]) if isinstance(meta.get("shanten"), int) else None,
    )


def aggregate_rows(records: Sequence[DecisionRecord]) -> list[dict[str, Any]]:
    accs: dict[tuple[str, str, str], Aggregate] = defaultdict(Aggregate)
    for record in records:
        for slice_dim, slice_value in slices(record):
            accs[(record.decision_kind, slice_dim, slice_value)].add(record)
    rows = []
    for (decision_kind, slice_dim, slice_value), acc in sorted(accs.items()):
        rows.append(
            {
                "decision_kind": decision_kind,
                "slice_dim": slice_dim,
                "slice_value": slice_value,
                **acc.metrics(),
            }
        )
    return rows


def slices(record: DecisionRecord) -> list[tuple[str, str]]:
    return [
        ("all", "all"),
        ("dealer", "dealer" if record.dealer else "nondealer"),
        ("start_rank", str(record.start_rank)),
        ("round_stage", record.round_stage),
        ("score_bucket", record.score_bucket),
        ("turn_bucket", record.turn_bucket),
    ]


def diff_rows(left: dict[str, Any], right: dict[str, Any]) -> list[dict[str, Any]]:
    left_rows = {(row["decision_kind"], row["slice_dim"], row["slice_value"]): row for row in left["rows"]}
    right_rows = {(row["decision_kind"], row["slice_dim"], row["slice_value"]): row for row in right["rows"]}
    rows = []
    for key in sorted(set(left_rows) | set(right_rows)):
        left_row = left_rows.get(key, {})
        right_row = right_rows.get(key, {})
        row = {"decision_kind": key[0], "slice_dim": key[1], "slice_value": key[2]}
        for field_name in (
            "count",
            "avg_margin",
            "positive_margin_rate",
            "agari_rate",
            "houjuu_rate",
            "ryukyoku_rate",
            "avg_shanten",
        ):
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
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def format_markdown(left: dict[str, Any], right: dict[str, Any], diffs: Sequence[dict[str, Any]]) -> str:
    lines = [
        "# Default Checkpoint Decision Drift Report",
        "",
        f"- Left baseline: `{left['model_label']}`",
        f"- Right candidate: `{right['model_label']}`",
        f"- Left log files / decisions: `{left['log_files']}` / `{left['decision_count']}`",
        f"- Right log files / decisions: `{right['log_files']}` / `{right['decision_count']}`",
        "",
        "This report uses arena-log Q metadata for chosen actions. It can measure chosen call/reach margins, "
        "but it cannot fully count pass-over-call opportunities because arena logs do not emit pass events.",
        "",
    ]
    for decision_kind in ("chosen_call", "chosen_reach", "declined_reach"):
        rows = [row for row in diffs if row["decision_kind"] == decision_kind and row["slice_dim"] in {"all", "dealer", "start_rank"}]
        if rows:
            lines.extend(format_decision_table(decision_kind, rows))
    lines.extend(["", "## L3 Diagnosis", ""])
    lines.extend(f"- {note}" for note in diagnose(diffs))
    lines.append("")
    return "\n".join(lines)


def format_decision_table(decision_kind: str, rows: Sequence[dict[str, Any]]) -> list[str]:
    lines = [
        f"## `{decision_kind}`",
        "",
        "| Slice | 70k n | 80k n | count d | margin d | positive margin d | agari d | houjuu d | shanten d |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        label = f"{row['slice_dim']}={row['slice_value']}"
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    fmt_count(row.get("left_count")),
                    fmt_count(row.get("right_count")),
                    fmt_count(row.get("delta_count"), signed=True),
                    fmt_num(row.get("delta_avg_margin")),
                    fmt_pp(row.get("delta_positive_margin_rate")),
                    fmt_pp(row.get("delta_agari_rate")),
                    fmt_pp(row.get("delta_houjuu_rate")),
                    fmt_num(row.get("delta_avg_shanten")),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def diagnose(diffs: Sequence[dict[str, Any]]) -> list[str]:
    by_key = {(row["decision_kind"], row["slice_dim"], row["slice_value"]): row for row in diffs}
    notes: list[str] = []
    call_all = by_key.get(("chosen_call", "all", "all"))
    if call_all:
        notes.append(
            "Chosen calls increase in count and have worse outcomes: "
            f"count {fmt_count(call_all.get('delta_count'), signed=True)}, "
            f"margin {fmt_num(call_all.get('delta_avg_margin'))}, "
            f"houjuu {fmt_pp(call_all.get('delta_houjuu_rate'))}."
        )
    for slice_key in (("dealer", "dealer"), ("start_rank", "2")):
        row = by_key.get(("chosen_call", *slice_key))
        if row:
            notes.append(
                f"Chosen-call {slice_key[0]}={slice_key[1]}: count "
                f"{fmt_count(row.get('delta_count'), signed=True)}, margin "
                f"{fmt_num(row.get('delta_avg_margin'))}, houjuu "
                f"{fmt_pp(row.get('delta_houjuu_rate'))}."
            )
    reach_all = by_key.get(("chosen_reach", "all", "all"))
    if reach_all:
        notes.append(
            "Chosen riichi does not show the same degradation: "
            f"margin {fmt_num(reach_all.get('delta_avg_margin'))}, "
            f"houjuu {fmt_pp(reach_all.get('delta_houjuu_rate'))}."
        )
    return notes


def fmt_count(value: Any, *, signed: bool = False) -> str:
    if not isinstance(value, int | float):
        return "NA"
    if signed:
        return f"{int(value):+d}"
    return str(int(value))


def fmt_num(value: Any) -> str:
    if not isinstance(value, int | float):
        return "NA"
    return f"{value:+.4f}"


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
    (args.output_dir / "decision_drift_left.json").write_text(
        json.dumps(left, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "decision_drift_right.json").write_text(
        json.dumps(right, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(args.output_dir / "decision_metrics.csv", [*left["rows"], *right["rows"]])
    write_csv(args.output_dir / "decision_diff.csv", diffs)
    report = format_markdown(left, right, diffs)
    (args.output_dir / "decision_drift_report.md").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
