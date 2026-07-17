#!/usr/bin/env python3
"""Build platform-style account pt/rating reports from Mortal native logs."""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.stat_report import build_stat_report
from scripts.mortal.stat_report import format_markdown_report

TENHOU_RANK_RESULTS = (30.0, 10.0, -10.0, -30.0)
HOUOU_7DAN_HANCHAN_PT = (90.0, 45.0, 0.0, -135.0)
INITIAL_RATING = 1500.0
INITIAL_PT = 1400.0
PT_TARGET = 2800.0
RANK_NAME = "七段"
KNOWN_OUTPUT_FILES = (
    "account_summary.json",
    "account_summary.csv",
    "account_summary.md",
    "account_ledger.jsonl",
    "rating_curve.csv",
    "per_game_results.csv",
    "detailed_stats.json",
    "detailed_stats.md",
)


@dataclass
class AccountState:
    account_id: str
    model_label: str
    rating: float = INITIAL_RATING
    pt: float = INITIAL_PT
    games: int = 0
    rank_counts: list[int] | None = None

    def __post_init__(self) -> None:
        if self.rank_counts is None:
            self.rank_counts = [0, 0, 0, 0]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-dir", action="append", type=Path, required=True, help="Input logs directory. Repeatable.")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--platform-model-label", default=None, help="Force all seats to MODEL@01-04.")
    parser.add_argument("--rank-points", default="90,45,0,-135")
    parser.add_argument("--preserve-log-dir-order", action="store_true", help="process repeated --log-dir inputs in supplied order")
    parser.add_argument(
        "--interleave-log-dirs",
        action="store_true",
        help="interleave same-index games across log directories, rotating the directory order each round",
    )
    return parser.parse_args()


def parse_rank_points(value: str) -> tuple[float, float, float, float]:
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError(f"rank points must contain four numbers, got {value!r}")
    return tuple(float(part) for part in parts)  # type: ignore[return-value]


def iter_log_files(
    log_dirs: Sequence[Path],
    *,
    preserve_log_dir_order: bool = False,
    interleave_log_dirs: bool = False,
) -> list[Path]:
    files_by_dir = [sorted(log_dir.glob("*.json.gz")) for log_dir in log_dirs]
    if interleave_log_dirs:
        files: list[Path] = []
        max_games = max((len(items) for items in files_by_dir), default=0)
        dir_count = len(files_by_dir)
        for game_index in range(max_games):
            # Rotate the per-round directory order so early Tenhou-R games are
            # not systematically assigned to the first league lineups.
            for offset in range(dir_count):
                dir_index = (game_index + offset) % dir_count
                if game_index < len(files_by_dir[dir_index]):
                    files.append(files_by_dir[dir_index][game_index])
        return files
    files = [path for items in files_by_dir for path in items]
    if preserve_log_dir_order:
        return files
    return sorted(files, key=lambda path: (str(path.parent), path.name))


def read_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                events.append(json.loads(line))
    if not events or events[0].get("type") != "start_game":
        raise ValueError(f"missing start_game in {path}")
    return events


def stable_source_id(path: Path, seen: dict[str, int]) -> str:
    stem = path.name.removesuffix(".json.gz")
    safe_parent = re.sub(r"[^A-Za-z0-9_.-]+", "_", path.parent.parent.name or path.parent.name)
    base = f"{safe_parent}_{stem}"
    count = seen.get(base, 0)
    seen[base] = count + 1
    return base if count == 0 else f"{base}_{count + 1}"


def normalize_model_label(raw_name: str) -> str:
    if raw_name in {"ext_mortal", "weak_mortal"}:
        return "ext_mortal"
    suffix_match = re.fullmatch(r"(.+)_([ab])", raw_name)
    if suffix_match and suffix_match.group(1) in {"ext_mortal", "70k", "V2_74000", "V3_74000"}:
        return suffix_match.group(1)
    return raw_name


def account_ids_for_names(names: Sequence[str], *, platform_model_label: str | None) -> list[dict[str, str]]:
    if len(names) != 4:
        raise ValueError(f"expected four names, got {names!r}")
    raw_names = [str(raw_name) for raw_name in names]
    if platform_model_label:
        return [
            {
                "account_id": f"{platform_model_label}@{seat + 1:02d}",
                "model_label": platform_model_label,
                "raw_player_name": str(raw_name),
            }
            for seat, raw_name in enumerate(raw_names)
        ]

    raw_counts: dict[str, int] = {}
    unique_raws_by_model: dict[str, list[str]] = {}
    for raw_name in raw_names:
        raw_counts[raw_name] = raw_counts.get(raw_name, 0) + 1
        model_label = normalize_model_label(raw_name)
        unique_raws_by_model.setdefault(model_label, [])
        if raw_name not in unique_raws_by_model[model_label]:
            unique_raws_by_model[model_label].append(raw_name)
    for raws in unique_raws_by_model.values():
        raws.sort()

    model_counts: dict[str, int] = {}
    result: list[dict[str, str]] = []
    for raw_name in raw_names:
        model_label = normalize_model_label(raw_name)
        if raw_counts[raw_name] == 1:
            account_number = unique_raws_by_model[model_label].index(raw_name) + 1
        else:
            model_counts[model_label] = model_counts.get(model_label, 0) + 1
            account_number = model_counts[model_label]
        result.append(
            {
                "account_id": f"{model_label}@{account_number:02d}",
                "model_label": model_label,
                "raw_player_name": raw_name,
            }
        )
    return result


def initial_and_final_scores_from_events(events: Sequence[Mapping[str, Any]]) -> tuple[list[int], list[int]]:
    initial_scores: list[int] | None = None
    scores: list[int] | None = None
    for event in events:
        event_type = event.get("type")
        if event_type == "start_kyoku":
            raw_scores = event.get("scores")
            if isinstance(raw_scores, list) and len(raw_scores) == 4:
                scores = [int(value) for value in raw_scores]
                if initial_scores is None:
                    initial_scores = list(scores)
        elif event_type == "reach_accepted" and scores is not None:
            actor = event.get("actor")
            if actor is not None:
                scores[int(actor)] -= 1000
        elif event_type in {"hora", "ryukyoku"} and scores is not None:
            deltas = event.get("deltas")
            if isinstance(deltas, list) and len(deltas) == 4:
                scores = [int(score + int(delta)) for score, delta in zip(scores, deltas, strict=True)]
    if initial_scores is None or scores is None:
        raise ValueError("could not reconstruct final scores")
    total = sum(scores)
    if total < 100_000:
        ranks = ranks_from_scores(scores)
        scores[ranks.index(1)] += 100_000 - total
    return initial_scores, scores


def ranks_from_scores(scores: Sequence[int]) -> list[int]:
    ordered = sorted(range(4), key=lambda seat: (-int(scores[seat]), seat))
    ranks = [0, 0, 0, 0]
    for rank, seat in enumerate(ordered, 1):
        ranks[seat] = rank
    return ranks


def rating_correction(games_before: int) -> float:
    return 1.0 - float(games_before) * 0.002 if games_before < 400 else 0.2


def write_account_log(events: Sequence[Mapping[str, Any]], account_ids: Sequence[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(output_path, "wt", encoding="utf-8") as handle:
        for idx, event in enumerate(events):
            row = dict(event)
            if idx == 0 and row.get("type") == "start_game":
                row["names"] = list(account_ids)
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def prepare_output_dir(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename in KNOWN_OUTPUT_FILES:
        path = output_dir / filename
        if path.exists():
            path.unlink()
    account_log_dir = output_dir / "account_logs"
    if account_log_dir.exists():
        shutil.rmtree(account_log_dir)
    account_log_dir.mkdir(parents=True, exist_ok=True)
    return account_log_dir


def build_report(
    *,
    log_dirs: Sequence[Path],
    output_dir: Path,
    mortal_root: Path,
    platform_model_label: str | None,
    rank_points: tuple[float, float, float, float],
    preserve_log_dir_order: bool = False,
    interleave_log_dirs: bool = False,
) -> dict[str, Any]:
    account_log_dir = prepare_output_dir(output_dir)

    files = iter_log_files(
        log_dirs,
        preserve_log_dir_order=preserve_log_dir_order,
        interleave_log_dirs=interleave_log_dirs,
    )
    accounts: dict[str, AccountState] = {}
    per_game_rows: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    source_seen: dict[str, int] = {}

    for game_index, path in enumerate(files):
        events = read_events(path)
        raw_names = [str(name) for name in events[0]["names"]]
        seat_accounts = account_ids_for_names(raw_names, platform_model_label=platform_model_label)
        account_ids = [entry["account_id"] for entry in seat_accounts]
        source_id = stable_source_id(path, source_seen)
        write_account_log(events, account_ids, account_log_dir / f"{source_id}.json.gz")

        initial_scores, scores = initial_and_final_scores_from_events(events)
        ranks = ranks_from_scores(scores)
        table_account_states: list[AccountState] = []
        for seat, entry in enumerate(seat_accounts):
            account_id = entry["account_id"]
            if account_id not in accounts:
                accounts[account_id] = AccountState(account_id=account_id, model_label=entry["model_label"])
            table_account_states.append(accounts[account_id])

        pre_ratings = [state.rating for state in table_account_states]
        table_avg_rating = sum(pre_ratings) / 4.0
        updates: list[dict[str, Any]] = []
        for seat, state in enumerate(table_account_states):
            rank = ranks[seat]
            games_before = int(state.games)
            rating_before = float(state.rating)
            pt_before = float(state.pt)
            correction = rating_correction(games_before)
            rating_delta = correction * (TENHOU_RANK_RESULTS[rank - 1] + (table_avg_rating - rating_before) / 40.0)
            pt_delta = float(rank_points[rank - 1])
            updates.append(
                {
                    "seat": seat,
                    "state": state,
                    "rank": rank,
                    "final_score": int(scores[seat]),
                    "score_delta": int(scores[seat] - initial_scores[seat]),
                    "rating_before": rating_before,
                    "rating_delta": rating_delta,
                    "rating_after": rating_before + rating_delta,
                    "pt_before": pt_before,
                    "pt_delta": pt_delta,
                    "pt_after": pt_before + pt_delta,
                    "games_before": games_before,
                    "rating_correction": correction,
                }
            )

        for update in updates:
            state = update["state"]
            state.rating = float(update["rating_after"])
            state.pt = float(update["pt_after"])
            state.games += 1
            assert state.rank_counts is not None
            state.rank_counts[int(update["rank"]) - 1] += 1

            seat = int(update["seat"])
            row = {
                "game_index": game_index,
                "source_log": str(path),
                "seat": seat,
                "raw_player_name": seat_accounts[seat]["raw_player_name"],
                "account_id": state.account_id,
                "model_label": state.model_label,
                "rank": int(update["rank"]),
                "final_score": int(update["final_score"]),
                "score_delta": int(update["score_delta"]),
                "table_avg_rating_before": table_avg_rating,
                "rating_before": float(update["rating_before"]),
                "rating_delta": float(update["rating_delta"]),
                "rating_after": float(update["rating_after"]),
                "rating_correction": float(update["rating_correction"]),
                "pt_before": float(update["pt_before"]),
                "pt_delta": float(update["pt_delta"]),
                "pt_after": float(update["pt_after"]),
                "pt_target": PT_TARGET,
                "rank_name": RANK_NAME,
                "games_before": int(update["games_before"]),
                "games_after": int(update["games_before"]) + 1,
            }
            per_game_rows.append(row)
            ledger_rows.append(row)
            curve_rows.append(
                {
                    "game_index": game_index,
                    "account_id": state.account_id,
                    "model_label": state.model_label,
                    "rating": float(update["rating_after"]),
                    "pt": float(update["pt_after"]),
                    "rank_name": RANK_NAME,
                    "games": int(update["games_before"]) + 1,
                }
            )

    account_ids = sorted(accounts)
    stat_report = build_stat_report(
        log_dir=account_log_dir,
        players={account_id: account_id for account_id in account_ids},
        mortal_root=mortal_root,
        rank_pts=rank_points,
        rank_points_profile="houou_7dan_hanchan",
    )

    summary_rows: list[dict[str, Any]] = []
    for account_id in account_ids:
        state = accounts[account_id]
        stat_player = stat_report["players"].get(account_id, {})
        raw = stat_player.get("raw", {})
        derived = stat_player.get("derived", {})
        ranks = state.rank_counts or [0, 0, 0, 0]
        summary_rows.append(
            {
                "account_id": account_id,
                "model_label": state.model_label,
                "games": int(state.games),
                "rank_name": RANK_NAME,
                "pt_current": float(state.pt),
                "pt_target": PT_TARGET,
                "rating": float(state.rating),
                "rank_1": ranks[0],
                "rank_2": ranks[1],
                "rank_3": ranks[2],
                "rank_4": ranks[3],
                "avg_rank": derived.get("avg_rank"),
                "avg_rank_pt": derived.get("avg_rank_pt"),
                "agari_rate": derived.get("agari_rate"),
                "houjuu_rate": derived.get("houjuu_rate"),
                "fuuro_rate": derived.get("fuuro_rate"),
                "riichi_rate": derived.get("riichi_rate"),
                "agari_rate_after_fuuro": derived.get("agari_rate_after_fuuro"),
                "houjuu_rate_after_fuuro": derived.get("houjuu_rate_after_fuuro"),
                "agari_rate_after_riichi": derived.get("agari_rate_after_riichi"),
                "houjuu_rate_after_riichi": derived.get("houjuu_rate_after_riichi"),
                "avg_point_per_agari": derived.get("avg_point_per_agari"),
                "total_delta_score": raw.get("point"),
            }
        )

    report = {
        "schema": "keqing.mortal.platform_account_report.v1",
        "log_dirs": [str(path) for path in log_dirs],
        "output_dir": str(output_dir),
        "platform_model_label": platform_model_label,
        "preserve_log_dir_order": bool(preserve_log_dir_order),
        "interleave_log_dirs": bool(interleave_log_dirs),
        "scoring": {
            "rating_initial": INITIAL_RATING,
            "rating_rank_results": list(TENHOU_RANK_RESULTS),
            "rating_formula": "delta = game_count_correction * (rank_result + (table_avg_rating - player_rating) / 40)",
            "rating_game_count_correction": "1 - games * 0.002 if games < 400 else 0.2",
            "rating_scaling": 1.0,
            "pt_profile": "houou_7dan_hanchan",
            "pt_rank_deltas": list(rank_points),
            "pt_initial": INITIAL_PT,
            "pt_target": PT_TARGET,
            "rank_name": RANK_NAME,
            "sources": [
                "https://doramahjong.org/osusume/02/0012.html",
                "https://detail.chiebukuro.yahoo.co.jp/qa/question_detail/q14229956494",
                "https://tenhou.net/man/",
            ],
        },
        "games": len(files),
        "accounts": summary_rows,
        "stat_report": stat_report,
    }

    write_outputs(
        output_dir=output_dir,
        report=report,
        summary_rows=summary_rows,
        ledger_rows=ledger_rows,
        curve_rows=curve_rows,
        per_game_rows=per_game_rows,
    )
    return report


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def fmt_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def build_summary_markdown(report: Mapping[str, Any], summary_rows: Sequence[Mapping[str, Any]]) -> str:
    lines = [
        "# Platform Account Report",
        "",
        f"- Games: `{report['games']}`",
        f"- Platform model label override: `{report.get('platform_model_label')}`",
        f"- Pt profile: `{report['scoring']['pt_profile']}`",
        f"- Rating formula: `{report['scoring']['rating_formula']}`",
        "",
        "| Account | Model | Games | Pt | R | Rank counts | Avg rank | Agari | Houjuu | Fuuro | Riichi |",
        "| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        ranks = f"[{row['rank_1']},{row['rank_2']},{row['rank_3']},{row['rank_4']}]"
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["account_id"]),
                    str(row["model_label"]),
                    str(row["games"]),
                    fmt_float(row["pt_current"], 1),
                    fmt_float(row["rating"], 2),
                    ranks,
                    fmt_float(row["avg_rank"], 4),
                    fmt_float(row["agari_rate"], 4),
                    fmt_float(row["houjuu_rate"], 4),
                    fmt_float(row["fuuro_rate"], 4),
                    fmt_float(row["riichi_rate"], 4),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def write_outputs(
    *,
    output_dir: Path,
    report: Mapping[str, Any],
    summary_rows: Sequence[Mapping[str, Any]],
    ledger_rows: Sequence[Mapping[str, Any]],
    curve_rows: Sequence[Mapping[str, Any]],
    per_game_rows: Sequence[Mapping[str, Any]],
) -> None:
    (output_dir / "account_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_csv(output_dir / "account_summary.csv", summary_rows)
    (output_dir / "account_summary.md").write_text(build_summary_markdown(report, summary_rows), encoding="utf-8")
    write_jsonl(output_dir / "account_ledger.jsonl", ledger_rows)
    write_csv(output_dir / "rating_curve.csv", curve_rows)
    write_csv(output_dir / "per_game_results.csv", per_game_rows)
    (output_dir / "detailed_stats.json").write_text(
        json.dumps(report["stat_report"], ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "detailed_stats.md").write_text(format_markdown_report(report["stat_report"]), encoding="utf-8")


def main() -> None:
    args = parse_args()
    report = build_report(
        log_dirs=args.log_dir,
        output_dir=args.output_dir,
        mortal_root=args.mortal_root,
        platform_model_label=args.platform_model_label,
        rank_points=parse_rank_points(str(args.rank_points)),
        preserve_log_dir_order=bool(args.preserve_log_dir_order),
        interleave_log_dirs=bool(args.interleave_log_dirs),
    )
    print(json.dumps({"games": report["games"], "accounts": len(report["accounts"])}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
