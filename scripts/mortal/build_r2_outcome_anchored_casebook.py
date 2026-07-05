#!/usr/bin/env python3
"""Build the R2 outcome-anchored reviewer casebook.

The casebook is diagnostic-only. It selects outcome-anchored kyoku from existing
native arena logs, exports NAGA-friendly Tenhou6 single-kyoku JSON, runs local
checkpoint reviews at the focus decision, and imports each case into the
existing project replay UI storage.
"""

from __future__ import annotations

import argparse
import copy
import csv
import gzip
import json
import math
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
for _path in (str(_SRC_DIR), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from inference.review import same_action  # noqa: E402
from replay.api import run_replay_single_raw  # noqa: E402
from replay.bot import render_replay_json  # noqa: E402
from replay.normalize import normalize_replay_decisions  # noqa: E402
from replay.storage import get_storage  # noqa: E402
from tools.mjai_jsonl_to_tenhou6 import convert_mjai_jsonl_to_tenhou6  # noqa: E402


DEFAULT_OUTPUT = Path("artifacts/experiments/reviewer_teacher_probe_2026_05/R2_late_vs_riichi_casebook_2026_06")
DEFAULT_T2A_LOGS = Path(
    "artifacts/experiments/teacher_transfer_2026_05/T2a_risk_gated_teacher_ce_005/four_player_native_1000h/logs"
)
DEFAULT_T2B_LOGS = Path(
    "artifacts/experiments/teacher_transfer_2026_05/T2b_soft_risk_teacher_ce_005_002/four_player_native_250h_71400/logs"
)
MODEL_CHECKPOINTS = {
    "70k": Path("artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth"),
    "80k_game": Path("artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth"),
    "T1_71000": Path("artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth"),
    "T2a_71800": Path("artifacts/experiments/teacher_transfer_2026_05/T2a_risk_gated_teacher_ce_005/checkpoints/mortal_t2a_71800.pth"),
    "T2b_71400": Path("artifacts/experiments/teacher_transfer_2026_05/T2b_soft_risk_teacher_ce_005_002/checkpoints/mortal_t2b_71400.pth"),
}
EAST_SOUTH_WEST_NORTH = ["东家", "南家", "西家", "北家"]
TITLE_DISP = "玉の間四人南"
TITLE_DATE = "2026/6/24 14:30:00"
RATING_INFO = "[125,60,-5,-240],[125,60,-5,-195],[125,60,-5,-195],[125,60,-5,-180],1"
BUCKET_LIMITS = {
    "late_vs_riichi_negative": 20,
    "after_riichi_negative": 20,
    "after_fuuro_positive": 10,
    "neutral_control": 10,
}
CALL_TYPES = {"chi", "pon", "daiminkan", "ankan", "kakan"}
MORTAL_DISCARD_ID_TO_TILE = (
    "1m",
    "2m",
    "3m",
    "4m",
    "5m",
    "6m",
    "7m",
    "8m",
    "9m",
    "1p",
    "2p",
    "3p",
    "4p",
    "5p",
    "6p",
    "7p",
    "8p",
    "9p",
    "1s",
    "2s",
    "3s",
    "4s",
    "5s",
    "6s",
    "7s",
    "8s",
    "9s",
    "E",
    "S",
    "W",
    "N",
    "P",
    "F",
    "C",
    "5mr",
    "5pr",
    "5sr",
)


@dataclass
class FocusEvent:
    event_index: int
    turn: int
    event: dict[str, Any]
    slice_name: str


@dataclass
class SeatRound:
    model: str
    seat: int
    oya: int
    kyoku_index: int
    start_scores: list[int]
    turns: int = 0
    riichi: bool = False
    fuuro: bool = False
    agari: bool = False
    houjuu: bool = False
    delta_score: float = 0.0
    first_riichi: FocusEvent | None = None
    first_fuuro_discard: FocusEvent | None = None
    first_late_vs_riichi_discard: FocusEvent | None = None
    first_late_discard: FocusEvent | None = None
    neutral_mid_discard: FocusEvent | None = None
    candidate_events: list[FocusEvent] = field(default_factory=list)

    @property
    def seat_wind(self) -> str:
        return EAST_SOUTH_WEST_NORTH[(self.seat - self.oya) % 4]


@dataclass
class CandidateCase:
    source_group: str
    bucket: str
    source_log: Path
    kyoku_index: int
    target_model: str
    target_seat: int
    target_seat_wind: str
    player_names: list[str]
    start_scores: list[int]
    focus_event_index: int
    focus_turn: int
    focus_slice: str
    focus_action: dict[str, Any]
    agari: bool
    houjuu: bool
    delta_score: float
    peer_avg_delta: float
    peer_best_delta: float
    score: float
    tags: list[str]
    selected_index: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t2a-log-dir", type=Path, default=DEFAULT_T2A_LOGS)
    parser.add_argument("--t2b-log-dir", type=Path, default=DEFAULT_T2B_LOGS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-local-reviews", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path | str) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def iter_events(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def source_key(path: Path) -> tuple[int, str]:
    try:
        seed = int(path.name.split("_", 1)[0])
    except ValueError:
        seed = 0
    return seed, path.name


def parse_deltas(event: Mapping[str, Any]) -> list[float]:
    raw = event.get("deltas", [0, 0, 0, 0])
    if not isinstance(raw, list) or len(raw) < 4:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(raw[seat] or 0.0) for seat in range(4)]


def parse_log_rounds(path: Path, *, source_group: str) -> tuple[list[str], list[list[SeatRound]]]:
    events = iter_events(path)
    names = [str(i) for i in range(4)]
    rounds: list[list[SeatRound]] = []
    current: list[SeatRound] | None = None
    riichi = [False, False, False, False]
    kyoku_index = -1
    for event_index, event in enumerate(events):
        event_type = str(event.get("type", ""))
        if event_type == "start_game":
            raw_names = event.get("names")
            if isinstance(raw_names, list) and len(raw_names) >= 4:
                names = [str(name) for name in raw_names[:4]]
        elif event_type == "start_kyoku":
            if current is not None:
                rounds.append(current)
            kyoku_index += 1
            riichi = [False, False, False, False]
            scores = [int(score) for score in event.get("scores", [25000, 25000, 25000, 25000])[:4]]
            oya = int(event.get("oya", 0))
            current = [
                SeatRound(
                    model=names[seat],
                    seat=seat,
                    oya=oya,
                    kyoku_index=kyoku_index,
                    start_scores=scores,
                )
                for seat in range(4)
            ]
        elif current is None:
            continue
        elif event_type == "tsumo":
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                current[actor].turns += 1
        elif event_type == "reach":
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                seat = current[actor]
                riichi[actor] = True
                if not seat.riichi:
                    seat.riichi = True
                    seat.first_riichi = FocusEvent(event_index, max(1, seat.turns), dict(event), "action_first_riichi")
        elif event_type in CALL_TYPES:
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                current[actor].fuuro = True
        elif event_type == "dahai":
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                seat = current[actor]
                turn = max(1, seat.turns)
                opponents_riichi = any(value for idx, value in enumerate(riichi) if idx != actor)
                focus = FocusEvent(event_index, turn, dict(event), "action_discard_all")
                if turn >= 13:
                    seat.first_late_discard = seat.first_late_discard or FocusEvent(
                        event_index, turn, dict(event), "action_discard_late_13_plus"
                    )
                if opponents_riichi and turn >= 13:
                    seat.first_late_vs_riichi_discard = seat.first_late_vs_riichi_discard or FocusEvent(
                        event_index, turn, dict(event), "action_discard_vs_riichi_late_13_plus"
                    )
                if seat.fuuro:
                    seat.first_fuuro_discard = seat.first_fuuro_discard or FocusEvent(
                        event_index, turn, dict(event), "action_discard_after_fuuro"
                    )
                if 7 <= turn <= 12 and not opponents_riichi and not seat.riichi and not seat.fuuro:
                    seat.neutral_mid_discard = seat.neutral_mid_discard or FocusEvent(
                        event_index, turn, dict(event), "action_discard_middle_control"
                    )
                seat.candidate_events.append(focus)
        elif event_type == "hora":
            deltas = parse_deltas(event)
            for seat_idx, delta in enumerate(deltas):
                current[seat_idx].delta_score += float(delta)
            actor = event.get("actor")
            target = event.get("target")
            if isinstance(actor, int) and 0 <= actor < 4:
                current[actor].agari = True
            if isinstance(target, int) and 0 <= target < 4 and target != actor:
                current[target].houjuu = True
        elif event_type == "ryukyoku":
            deltas = parse_deltas(event)
            for seat_idx, delta in enumerate(deltas):
                current[seat_idx].delta_score += float(delta)
        elif event_type == "end_kyoku":
            rounds.append(current)
            current = None
    if current is not None:
        rounds.append(current)
    return names, rounds


def build_candidate(
    *,
    source_group: str,
    bucket: str,
    source_log: Path,
    target: SeatRound,
    focus: FocusEvent,
    round_seats: Sequence[SeatRound],
    player_names: list[str],
    score: float,
    tags: Sequence[str],
) -> CandidateCase:
    peer_deltas = [seat.delta_score for seat in round_seats if seat.seat != target.seat]
    return CandidateCase(
        source_group=source_group,
        bucket=bucket,
        source_log=source_log,
        kyoku_index=target.kyoku_index,
        target_model=target.model,
        target_seat=target.seat,
        target_seat_wind=target.seat_wind,
        player_names=list(player_names),
        start_scores=list(target.start_scores),
        focus_event_index=focus.event_index,
        focus_turn=focus.turn,
        focus_slice=focus.slice_name,
        focus_action=focus.event,
        agari=target.agari,
        houjuu=target.houjuu,
        delta_score=target.delta_score,
        peer_avg_delta=sum(peer_deltas) / len(peer_deltas) if peer_deltas else 0.0,
        peer_best_delta=max(peer_deltas) if peer_deltas else 0.0,
        score=score,
        tags=list(tags),
    )


def collect_candidates(log_dirs: Mapping[str, tuple[Path, str]]) -> dict[str, list[CandidateCase]]:
    buckets: dict[str, list[CandidateCase]] = defaultdict(list)
    seen: set[tuple[str, int, str, str]] = set()
    for source_group, (log_dir, target_model) in log_dirs.items():
        for source_log in sorted(resolve_path(log_dir).glob("*.json.gz"), key=source_key):
            player_names, rounds = parse_log_rounds(source_log, source_group=source_group)
            for round_seats in rounds:
                target = next((seat for seat in round_seats if seat.model == target_model), None)
                if target is None:
                    continue
                peer_avg = sum(seat.delta_score for seat in round_seats if seat.seat != target.seat) / 3.0
                peer_best = max(seat.delta_score for seat in round_seats if seat.seat != target.seat)
                if target.first_late_vs_riichi_discard is not None and (target.delta_score < peer_avg or target.houjuu):
                    focus_turn_bonus = min(max(target.first_late_vs_riichi_discard.turn - 12, 0), 6) * 150
                    score = (peer_avg - target.delta_score) + (3000 if target.houjuu else 0) + focus_turn_bonus
                    add_candidate(
                        buckets,
                        seen,
                        build_candidate(
                            source_group=source_group,
                            bucket="late_vs_riichi_negative",
                            source_log=source_log,
                            target=target,
                            focus=target.first_late_vs_riichi_discard,
                            round_seats=round_seats,
                            player_names=player_names,
                            score=score,
                            tags=["late_vs_riichi", "negative", source_group],
                        ),
                    )
                if target.riichi and target.first_riichi is not None and (target.delta_score < peer_avg or target.houjuu):
                    score = (peer_avg - target.delta_score) + (2500 if target.houjuu else 0) + (1500 if not target.agari else 0)
                    add_candidate(
                        buckets,
                        seen,
                        build_candidate(
                            source_group=source_group,
                            bucket="after_riichi_negative",
                            source_log=source_log,
                            target=target,
                            focus=target.first_riichi,
                            round_seats=round_seats,
                            player_names=player_names,
                            score=score,
                            tags=["after_riichi", "negative", source_group],
                        ),
                    )
                if target.fuuro and target.first_fuuro_discard is not None and target.delta_score > peer_avg and not target.houjuu:
                    score = (target.delta_score - peer_avg) + (2000 if target.agari else 0)
                    add_candidate(
                        buckets,
                        seen,
                        build_candidate(
                            source_group=source_group,
                            bucket="after_fuuro_positive",
                            source_log=source_log,
                            target=target,
                            focus=target.first_fuuro_discard,
                            round_seats=round_seats,
                            player_names=player_names,
                            score=score,
                            tags=["after_fuuro", "positive_control", source_group],
                        ),
                    )
                if target.neutral_mid_discard is not None and not target.agari and not target.houjuu and abs(target.delta_score) <= 1500:
                    score = -abs(target.delta_score) - abs(target.delta_score - peer_avg) * 0.25
                    add_candidate(
                        buckets,
                        seen,
                        build_candidate(
                            source_group=source_group,
                            bucket="neutral_control",
                            source_log=source_log,
                            target=target,
                            focus=target.neutral_mid_discard,
                            round_seats=round_seats,
                            player_names=player_names,
                            score=score,
                            tags=["neutral_control", source_group],
                        ),
                    )
    return buckets


def add_candidate(
    buckets: dict[str, list[CandidateCase]],
    seen: set[tuple[str, int, str, str]],
    case: CandidateCase,
) -> None:
    key = (str(case.source_log), case.kyoku_index, case.target_model, case.bucket)
    if key in seen:
        return
    seen.add(key)
    buckets[case.bucket].append(case)


def select_cases(buckets: dict[str, list[CandidateCase]]) -> list[CandidateCase]:
    selected: list[CandidateCase] = []
    for bucket, limit in BUCKET_LIMITS.items():
        candidates = sorted(
            buckets.get(bucket, []),
            key=lambda case: (-case.score, source_key(case.source_log), case.kyoku_index, case.target_model),
        )
        chosen = candidates[:limit]
        if len(chosen) != limit:
            raise RuntimeError(f"bucket {bucket} selected {len(chosen)} cases, expected {limit}")
        selected.extend(chosen)
    selected = sorted(selected, key=lambda case: (case.bucket, case.source_group, source_key(case.source_log), case.kyoku_index))
    for index, case in enumerate(selected):
        case.selected_index = index
    return selected


def normalize_tile(tile: str) -> str:
    if tile in {"5mr", "5pr", "5sr"}:
        return tile[0] + tile[2]
    return tile


def discard_action_ids(tile: str) -> tuple[int, ...]:
    exact = tuple(i for i, mortal_tile in enumerate(MORTAL_DISCARD_ID_TO_TILE) if mortal_tile == tile)
    if exact:
        return exact
    normalized = normalize_tile(tile)
    return tuple(
        i for i, mortal_tile in enumerate(MORTAL_DISCARD_ID_TO_TILE) if normalize_tile(mortal_tile) == normalized
    )


def action_ids_from_event(event: Mapping[str, Any]) -> tuple[int, ...]:
    event_type = str(event.get("type", ""))
    if event_type == "dahai":
        return discard_action_ids(str(event.get("pai", "")))
    if event_type == "reach":
        return (37,)
    if event_type == "hora":
        return (43,)
    if event_type == "ryukyoku":
        return (44,)
    if event_type == "pon":
        return (41,)
    if event_type in {"daiminkan", "ankan", "kakan"}:
        return (42,)
    if event_type == "chi":
        return (38, 39, 40)
    if event_type == "none":
        return (45,)
    return ()


def expand_compact_meta(meta: Mapping[str, Any]) -> list[dict[str, float | int]]:
    mask_bits = int(meta.get("mask_bits", 0) or 0)
    q_values = [float(value) for value in (meta.get("q_values") or [])]
    action_ids = [action_id for action_id in range(46) if mask_bits & (1 << action_id)]
    return [
        {"action_id": int(action_id), "q": float(q_values[idx])}
        for idx, action_id in enumerate(action_ids[: len(q_values)])
    ]


def softmax(values: Sequence[float]) -> list[float]:
    if not values:
        return []
    max_value = max(values)
    exps = [math.exp(value - max_value) for value in values]
    total = sum(exps)
    if total <= 0:
        return [0.0 for _ in values]
    return [value / total for value in exps]


def source_action_review(event: Mapping[str, Any]) -> dict[str, Any]:
    meta = event.get("meta") if isinstance(event.get("meta"), Mapping) else {}
    expanded = expand_compact_meta(meta)
    probs = softmax([float(item["q"]) for item in expanded])
    actual_ids = set(action_ids_from_event(event))
    rows = []
    actual: dict[str, Any] | None = None
    for item, prob in zip(expanded, probs):
        row = {"action_id": int(item["action_id"]), "q": float(item["q"]), "prob": float(prob)}
        rows.append(row)
        if int(item["action_id"]) in actual_ids and actual is None:
            actual = row
    best = max(rows, key=lambda row: row["q"]) if rows else None
    return {
        "actual_action": strip_meta_action(event),
        "actual_action_ids": sorted(actual_ids),
        "actual_q": actual.get("q") if actual else None,
        "actual_prob": actual.get("prob") if actual else None,
        "best_action_id": best.get("action_id") if best else None,
        "best_q": best.get("q") if best else None,
        "best_prob": best.get("prob") if best else None,
        "is_greedy": bool(meta.get("is_greedy", False)) if isinstance(meta, Mapping) else None,
        "shanten": meta.get("shanten") if isinstance(meta, Mapping) else None,
        "at_furiten": meta.get("at_furiten") if isinstance(meta, Mapping) else None,
    }


def strip_meta_action(event: Mapping[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(event, Mapping):
        return None
    return {key: value for key, value in event.items() if key != "meta"}


def load_case_events(case: CandidateCase) -> list[dict[str, Any]]:
    return iter_events(resolve_path(case.source_log))


def find_focus_step(decisions: Mapping[str, Any], focus_event_index: int) -> tuple[int | None, str]:
    log = decisions.get("log", [])
    for idx, entry in enumerate(log):
        if isinstance(entry, Mapping) and entry.get("source_event_index") == focus_event_index and not entry.get("is_obs"):
            return idx, "exact"
    candidates = [
        (abs(int(entry["source_event_index"]) - int(focus_event_index)), idx)
        for idx, entry in enumerate(log)
        if isinstance(entry, Mapping)
        and not entry.get("is_obs")
        and isinstance(entry.get("source_event_index"), int)
    ]
    if not candidates:
        return None, "missing"
    distance, idx = min(candidates, key=lambda item: (item[0], item[1]))
    return idx, f"nearest:{distance}"


def review_case_with_model(
    case: CandidateCase,
    *,
    model_label: str,
    checkpoint: Path,
) -> tuple[dict[str, Any], dict[str, Any] | None, tuple[int | None, str]]:
    events = load_case_events(case)
    bot = run_replay_single_raw(
        events,
        player_id=int(case.target_seat),
        checkpoint=resolve_path(checkpoint),
        input_type="mjai",
        bot_type="mortal",
    )
    decisions = normalize_replay_decisions(render_replay_json(bot))
    review, focus = extract_review_at_focus(
        case,
        model_label=model_label,
        checkpoint=checkpoint,
        decisions=decisions,
    )
    return review, decisions, focus


def extract_review_at_focus(
    case: CandidateCase,
    *,
    model_label: str,
    checkpoint: Path,
    decisions: Mapping[str, Any],
) -> tuple[dict[str, Any], tuple[int | None, str]]:
    step, resolution = find_focus_step(decisions, int(case.focus_event_index))
    entry = decisions.get("log", [])[step] if step is not None else None
    review = {
        "model": model_label,
        "checkpoint": str(checkpoint),
        "focus_step": step,
        "focus_resolution": resolution,
        "chosen": strip_meta_action(entry.get("chosen")) if isinstance(entry, Mapping) else None,
        "gt_action": strip_meta_action(entry.get("gt_action")) if isinstance(entry, Mapping) else None,
        "matches_actual": bool(same_action(entry.get("chosen"), entry.get("gt_action"))) if isinstance(entry, Mapping) else None,
        "top": None,
        "actual_candidate": None,
        "candidate_count": 0,
    }
    if isinstance(entry, Mapping):
        candidates = list(entry.get("candidates") or [])
        review["candidate_count"] = len(candidates)
        if candidates:
            review["top"] = compact_candidate(candidates[0])
            for candidate in candidates:
                if same_action(candidate.get("action"), entry.get("gt_action")):
                    review["actual_candidate"] = compact_candidate(candidate)
                    break
    return review, (step, resolution)


def compact_candidate(candidate: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "action": strip_meta_action(candidate.get("action")),
        "q": candidate.get("q", candidate.get("logit")),
        "prob": candidate.get("prob"),
        "rank": candidate.get("rank"),
        "final_score": candidate.get("final_score"),
    }


def save_replay_for_case(
    *,
    case: CandidateCase,
    events: list[dict[str, Any]],
    decisions: dict[str, Any],
    checkpoint: Path,
    focus_step: int | None,
    focus_resolution: str,
) -> str:
    decisions["r2_casebook_case"] = {
        "case_id": case_id(case),
        "bucket": case.bucket,
        "target_model": case.target_model,
        "target_seat": case.target_seat,
        "target_seat_wind": case.target_seat_wind,
        "kyoku_index": case.kyoku_index,
        "focus_event_index": case.focus_event_index,
        "focus_step": focus_step,
        "focus_resolution": focus_resolution,
    }
    return get_storage().save(
        events=events,
        decisions=decisions,
        bot_type=f"r2_{case.target_model}",
        player_names=case.player_names,
        checkpoint=str(resolve_path(checkpoint)),
    )


def case_id(case: CandidateCase) -> str:
    assert case.selected_index is not None
    return f"r2_case_{case.selected_index:03d}"


def tenhou6_payload(source: dict[str, Any], kyoku_log: list[Any]) -> dict[str, Any]:
    return {
        "title": [[TITLE_DISP, TITLE_DATE], RATING_INFO],
        "name": list(source.get("name", ["A", "B", "C", "D"]))[:4],
        "rule": {"disp": TITLE_DISP, "aka53": 1, "aka52": 1, "aka51": 1},
        "log": [kyoku_log],
    }


def compact_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def write_zip(zip_path: Path, files: list[Path], *, root: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(root))


def export_casebook(
    cases: list[CandidateCase],
    *,
    output_dir: Path,
    base_url: str,
    skip_local_reviews: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    hanchan_dir = output_dir / "hanchan_tenhou6"
    naga_dir = output_dir / "naga_kyoku_tenhou6"
    pending_dir = output_dir / "external_reviews_pending"
    for directory in (hanchan_dir, naga_dir, pending_dir):
        directory.mkdir(parents=True, exist_ok=True)

    tenhou_cache: dict[Path, tuple[Path, dict[str, Any]]] = {}
    events_cache: dict[Path, list[dict[str, Any]]] = {}
    review_cache: dict[tuple[Path, int, str], dict[str, Any]] = {}
    case_rows: list[dict[str, Any]] = []
    naga_blocks: list[str] = []
    naga_urls: list[str] = []
    bucket_counts: dict[str, int] = defaultdict(int)

    def events_for_case(case_: CandidateCase) -> list[dict[str, Any]]:
        if case_.source_log not in events_cache:
            events_cache[case_.source_log] = load_case_events(case_)
        return events_cache[case_.source_log]

    def decisions_for_case(case_: CandidateCase, model_label_: str) -> dict[str, Any]:
        key = (case_.source_log, int(case_.target_seat), model_label_)
        if key not in review_cache:
            print(
                f"[R2] review log={case_.source_log.name} seat={case_.target_seat} {case_.target_seat_wind} model={model_label_}",
                flush=True,
            )
            bot = run_replay_single_raw(
                events_for_case(case_),
                player_id=int(case_.target_seat),
                checkpoint=resolve_path(MODEL_CHECKPOINTS[model_label_]),
                input_type="mjai",
                bot_type="mortal",
            )
            review_cache[key] = normalize_replay_decisions(render_replay_json(bot))
        return review_cache[key]

    for idx, case in enumerate(cases, start=1):
        print(f"[R2] case {idx}/{len(cases)} {case_id(case)} {case.bucket} {case.target_model}", flush=True)
        if case.source_log not in tenhou_cache:
            events = events_for_case(case)
            tenhou6 = convert_mjai_jsonl_to_tenhou6(events)
            hanchan_path = hanchan_dir / f"{case.source_log.stem.removesuffix('.json')}.tenhou6.json"
            hanchan_path.write_text(compact_json(tenhou6) + "\n", encoding="utf-8")
            tenhou_cache[case.source_log] = (hanchan_path, tenhou6)
        hanchan_path, tenhou6 = tenhou_cache[case.source_log]
        kyoku_logs = list(tenhou6.get("log") or [])
        if not (0 <= case.kyoku_index < len(kyoku_logs)):
            raise RuntimeError(f"kyoku_index out of range for {case_id(case)}")
        single_kyoku = tenhou6_payload(tenhou6, kyoku_logs[case.kyoku_index])
        single_json = compact_json(single_kyoku)
        case_stem = f"{case_id(case)}_{case.source_group}_{case.source_log.stem.removesuffix('.json')}_kyoku_{case.kyoku_index:02d}"
        naga_path = naga_dir / f"{case_stem}.naga.tenhou6.json"
        naga_path.write_text(single_json + "\n", encoding="utf-8")
        naga_url = "https://tenhou.net/6/#json=" + single_json
        naga_urls.append(naga_url)
        naga_blocks.append(
            "\n".join(
                [
                    f"===== {case_id(case)} | {case.bucket} | model={case.target_model} | seat={case.target_seat} {case.target_seat_wind} | kyoku={case.kyoku_index} =====",
                    f"source_log: {case.source_log}",
                    f"outcome: agari={case.agari} houjuu={case.houjuu} delta={case.delta_score}",
                    naga_url,
                    "",
                ]
            )
        )

        local_reviews: dict[str, Any] = {}
        replay_id = None
        focus_step = None
        focus_resolution = "pending"
        review_models = ["70k", "80k_game", "T1_71000", case.target_model]
        for model_label in dict.fromkeys(review_models):
            if skip_local_reviews:
                local_reviews[model_label] = {"model": model_label, "status": "pending"}
                continue
            decisions = decisions_for_case(case, model_label)
            review, focus = extract_review_at_focus(
                case,
                model_label=model_label,
                checkpoint=MODEL_CHECKPOINTS[model_label],
                decisions=decisions,
            )
            local_reviews[model_label] = review
            if model_label == case.target_model:
                focus_step, focus_resolution = focus
                replay_id = save_replay_for_case(
                    case=case,
                    events=events_for_case(case),
                    decisions=copy.deepcopy(decisions),
                    checkpoint=MODEL_CHECKPOINTS[model_label],
                    focus_step=focus_step,
                    focus_resolution=focus_resolution,
                )

        row = {
            "schema": "keqing.mortal.r2_casebook_case.v1",
            "case_id": case_id(case),
            "case_index": case.selected_index,
            "bucket": case.bucket,
            "source_group": case.source_group,
            "source_log": str(case.source_log),
            "hanchan_tenhou6_path": str(hanchan_path),
            "naga_kyoku_tenhou6_path": str(naga_path),
            "naga_url": naga_url,
            "kyoku_index": case.kyoku_index,
            "target_model": case.target_model,
            "target_seat": case.target_seat,
            "target_seat_wind": case.target_seat_wind,
            "player_names": case.player_names,
            "start_scores": case.start_scores,
            "focus_event_index": case.focus_event_index,
            "focus_turn": case.focus_turn,
            "focus_slice": case.focus_slice,
            "focus_action": strip_meta_action(case.focus_action),
            "source_action_review": source_action_review(case.focus_action),
            "agari": case.agari,
            "houjuu": case.houjuu,
            "delta_score": case.delta_score,
            "peer_avg_delta": case.peer_avg_delta,
            "peer_best_delta": case.peer_best_delta,
            "tags": case.tags,
            "local_reviews": local_reviews,
            "external_reviews": {"naga": "pending", "mortal_4.1b": "pending"},
            "replay_id": replay_id,
            "replay_url": replay_url(base_url, replay_id, case.target_seat, focus_step) if replay_id else None,
            "focus_step": focus_step,
            "focus_resolution": focus_resolution,
        }
        case_rows.append(row)
        bucket_counts[case.bucket] += 1

    write_jsonl(output_dir / "case_manifest.jsonl", case_rows)
    (output_dir / "case_manifest.json").write_text(
        json.dumps(case_rows, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    (output_dir / "naga_kyoku_blocks.txt").write_text("\n".join(naga_blocks), encoding="utf-8")
    (output_dir / "naga_kyoku_urls.txt").write_text("\n".join(naga_urls) + "\n", encoding="utf-8")
    write_zip(output_dir / "hanchan_tenhou6_cases.zip", sorted(hanchan_dir.glob("*.json")), root=output_dir)
    write_zip(output_dir / "naga_kyoku_tenhou6_cases.zip", sorted(naga_dir.glob("*.json")), root=output_dir)
    write_csv(output_dir / "case_summary.csv", case_rows)
    (output_dir / "review_cases.html").write_text(build_html(case_rows), encoding="utf-8")
    (output_dir / "R2_summary.md").write_text(build_summary(case_rows, bucket_counts), encoding="utf-8")
    write_external_review_placeholders(pending_dir)


def replay_url(base_url: str, replay_id: str | None, player_id: int, focus_step: int | None) -> str | None:
    if not replay_id:
        return None
    url = f"{base_url.rstrip('/')}/game-replay?id={replay_id}&player_id={player_id}"
    if focus_step is not None:
        url += f"&focus_step={focus_step}&step={focus_step}&phase=pre"
    return url


def write_external_review_placeholders(pending_dir: Path) -> None:
    readme = """# R2 External Review Import

Put NAGA and Mortal 4.1b outputs for this casebook here, keyed by `case_id`.

Expected JSONL row shape:

```json
{"case_id":"r2_case_000","reviewer":"naga","status":"complete","top_action":null,"prob":null,"q":null,"margin":null,"raw_report_path":"path/to/report.json"}
```

Reviewer outputs are diagnostic only. Do not treat them as hard training labels without the R2 summary decision.
"""
    template = {
        "case_id": "r2_case_000",
        "reviewer": "naga",
        "status": "pending",
        "top_action": None,
        "prob": None,
        "q": None,
        "margin": None,
        "raw_report_path": None,
        "notes": None,
    }
    (pending_dir / "README.md").write_text(readme, encoding="utf-8")
    write_jsonl(pending_dir / "import_template.jsonl", [template])


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "case_id",
        "bucket",
        "source_group",
        "target_model",
        "target_seat",
        "target_seat_wind",
        "kyoku_index",
        "focus_slice",
        "focus_turn",
        "agari",
        "houjuu",
        "delta_score",
        "peer_avg_delta",
        "focus_action",
        "replay_url",
        "naga_url",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: json.dumps(row.get(field), ensure_ascii=False) if isinstance(row.get(field), (dict, list)) else row.get(field) for field in fields})


def build_summary(rows: list[dict[str, Any]], bucket_counts: Mapping[str, int]) -> str:
    lines = [
        "# R2 Outcome-Anchored Reviewer Casebook",
        "",
        f"- Cases: `{len(rows)}`",
        "- Purpose: diagnose late/vs-riichi and after-riichi failure modes before any T3 training.",
        "- Reviewer/NAGA status: pending; external reports are diagnostic only.",
        "",
        "## Buckets",
        "",
        "| Bucket | Count |",
        "| --- | ---: |",
    ]
    for bucket in BUCKET_LIMITS:
        lines.append(f"| `{bucket}` | {int(bucket_counts.get(bucket, 0))} |")
    lines.extend(
        [
            "",
            "## Artifacts",
            "",
            "- `case_manifest.jsonl` / `case_manifest.json`",
            "- `review_cases.html`",
            "- `naga_kyoku_blocks.txt`",
            "- `naga_kyoku_urls.txt`",
            "- `external_reviews_pending/`",
            "",
            "## Decision",
            "",
            "No training decision yet. Import NAGA/4.1b reports and human annotations first.",
            "Possible next steps are T3 targeted correction, riichi-aftermath diagnosis, or stopping reviewer-training if disagreements are style-only.",
            "",
        ]
    )
    return "\n".join(lines)


def build_html(rows: list[dict[str, Any]]) -> str:
    data = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>R2 Outcome-Anchored Casebook</title>
<style>
body {{ margin:0; font-family:Arial,"Microsoft YaHei",sans-serif; background:#f7f8fa; color:#15171a; }}
header {{ position:sticky; top:0; z-index:2; background:#fff; border-bottom:1px solid #d8dee7; padding:12px 16px; }}
h1 {{ margin:0 0 8px; font-size:18px; }}
.layout {{ display:grid; grid-template-columns: 390px minmax(0,1fr); min-height:calc(100vh - 70px); }}
.list {{ background:#fff; border-right:1px solid #d8dee7; overflow:auto; }}
.case {{ display:block; width:100%; border:0; border-bottom:1px solid #edf0f4; background:#fff; text-align:left; padding:10px; cursor:pointer; }}
.case.active {{ background:#eaf2ff; }}
.title {{ font-weight:700; }}
.meta {{ margin-top:4px; color:#5b6572; font-size:12px; line-height:1.45; }}
main {{ padding:16px; }}
.panel {{ background:#fff; border:1px solid #d8dee7; border-radius:6px; padding:12px; margin-bottom:12px; }}
.row {{ display:grid; grid-template-columns: 150px 1fr; gap:8px; padding:4px 0; }}
.actions {{ display:flex; gap:8px; flex-wrap:wrap; align-items:center; }}
button, a.button {{ height:34px; border:1px solid #b8c0cc; background:#fff; border-radius:6px; padding:0 10px; font-size:13px; display:inline-flex; align-items:center; text-decoration:none; color:#15171a; cursor:pointer; }}
.primary {{ background:#1f6feb !important; color:#fff !important; border-color:#1f6feb !important; }}
.danger {{ color:#b42318; }}
.good {{ color:#087443; }}
.pill {{ display:inline-flex; padding:2px 7px; border-radius:999px; background:#eef2f7; font-size:12px; margin-right:4px; }}
textarea {{ width:100%; min-height:84px; border:1px solid #d8dee7; border-radius:6px; padding:8px; }}
pre {{ white-space:pre-wrap; word-break:break-word; background:#f5f7fa; border:1px solid #e2e7ee; padding:8px; border-radius:6px; max-height:220px; overflow:auto; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
td, th {{ border-bottom:1px solid #edf0f4; padding:6px; text-align:left; vertical-align:top; }}
</style>
</head>
<body>
<header>
  <h1>R2 Outcome-Anchored Reviewer Casebook</h1>
  <div class="actions">
    <button id="exportJson">导出标注 JSON</button>
    <button id="exportCsv">导出标注 CSV</button>
    <button id="copyNaga">复制当前 NAGA URL</button>
  </div>
</header>
<div class="layout">
  <aside class="list" id="caseList"></aside>
  <main>
    <div class="panel" id="detail"></div>
    <div class="panel">
      <strong>本地四模型 review</strong>
      <div id="localReviews"></div>
    </div>
    <div class="panel">
      <strong>NAGA / Mortal 4.1b 输入</strong>
      <pre id="nagaUrl"></pre>
    </div>
    <div class="panel">
      <strong>人工标注</strong>
      <div class="actions" style="margin:10px 0">
        <button data-label="agree_local">agree_local</button>
        <button data-label="agree_reviewer">agree_reviewer</button>
        <button data-label="unclear">unclear</button>
        <button data-label="style_only">style_only</button>
      </div>
      <textarea id="note" placeholder="备注"></textarea>
    </div>
  </main>
</div>
<script id="caseData" type="application/json">{data}</script>
<script>
const cases = JSON.parse(document.getElementById('caseData').textContent);
const key = 'r2_outcome_casebook_annotations_v1';
let annotations = JSON.parse(localStorage.getItem(key) || '{{}}');
let current = 0;
const list = document.getElementById('caseList');
const detail = document.getElementById('detail');
const localReviews = document.getElementById('localReviews');
const nagaUrl = document.getElementById('nagaUrl');
const note = document.getElementById('note');
function ann(c) {{ annotations[c.case_id] ||= {{label:'', note:''}}; return annotations[c.case_id]; }}
function save() {{ localStorage.setItem(key, JSON.stringify(annotations)); }}
function saveNote() {{ ann(cases[current]).note = note.value; save(); }}
function actionText(a) {{ if(!a) return '-'; return [a.type, a.pai, Number.isInteger(a.actor)?'a'+a.actor:''].filter(Boolean).join(' '); }}
function fmt(v, n=3) {{ return Number.isFinite(Number(v)) ? Number(v).toFixed(n) : '-'; }}
function renderList() {{
  list.innerHTML = cases.map((c,i) => `<button class="case ${{i===current?'active':''}}" data-i="${{i}}">
    <div class="title">${{c.case_id}} · ${{c.bucket}}</div>
    <div class="meta">${{c.target_model}} · seat ${{c.target_seat}} ${{c.target_seat_wind}} · kyoku ${{c.kyoku_index}}<br>
    ${{c.focus_slice}} · delta=${{c.delta_score}} · houjuu=${{c.houjuu}}</div>
  </button>`).join('');
}}
function renderReviews(c) {{
  const rows = Object.values(c.local_reviews || {{}}).map(r => `<tr>
    <td>${{r.model || '-'}}</td>
    <td>${{actionText(r.chosen)}}</td>
    <td>${{r.top ? actionText(r.top.action) : '-'}}</td>
    <td>${{r.top ? fmt(r.top.q) : '-'}}</td>
    <td>${{r.top ? fmt(r.top.prob) : '-'}}</td>
    <td>${{r.actual_candidate ? fmt(r.actual_candidate.q) : '-'}}</td>
    <td>${{r.matches_actual}}</td>
  </tr>`).join('');
  localReviews.innerHTML = `<table><thead><tr><th>model</th><th>chosen</th><th>top</th><th>top q</th><th>top p</th><th>actual q</th><th>match actual</th></tr></thead><tbody>${{rows}}</tbody></table>`;
}}
function render() {{
  const c = cases[current];
  const a = ann(c);
  renderList();
  detail.innerHTML = `
    <div class="row"><strong>Case</strong><span>${{c.case_id}}</span></div>
    <div class="row"><strong>Bucket</strong><span>${{c.bucket}}</span></div>
    <div class="row"><strong>主视角</strong><span>${{c.target_model}} / seat ${{c.target_seat}} / ${{c.target_seat_wind}}</span></div>
    <div class="row"><strong>玩家</strong><span>${{c.player_names.join(' / ')}}</span></div>
    <div class="row"><strong>Focus</strong><span>${{c.focus_slice}} · event ${{c.focus_event_index}} · turn ${{c.focus_turn}}</span></div>
    <div class="row"><strong>Outcome</strong><span>agari=${{c.agari}} · houjuu=${{c.houjuu}} · delta=${{c.delta_score}} · peerAvg=${{fmt(c.peer_avg_delta,1)}}</span></div>
    <div class="row"><strong>Tags</strong><span>${{(c.tags||[]).map(t=>`<span class="pill">${{t}}</span>`).join('')}}</span></div>
    <div class="actions" style="margin-top:12px">
      ${{c.replay_url ? `<a class="button primary" target="_blank" rel="noopener" href="${{c.replay_url}}">打开项目原生牌桌</a>` : '<span class="danger">replay pending</span>'}}
      <button onclick="navigator.clipboard.writeText(cases[current].replay_url || '')">复制 Review URL</button>
    </div>`;
  renderReviews(c);
  nagaUrl.textContent = c.naga_url || '';
  [...document.querySelectorAll('button[data-label]')].forEach(b => b.classList.toggle('primary', b.dataset.label === a.label));
  note.value = a.note || '';
}}
list.onclick = e => {{ const b=e.target.closest('button[data-i]'); if(!b) return; saveNote(); current=Number(b.dataset.i); render(); }};
document.querySelectorAll('button[data-label]').forEach(b => b.onclick = () => {{ ann(cases[current]).label=b.dataset.label; save(); render(); }});
note.oninput = saveNote;
document.getElementById('copyNaga').onclick = () => navigator.clipboard.writeText(cases[current].naga_url || '');
function rows() {{ saveNote(); return cases.map(c => ({{...c, annotation: annotations[c.case_id] || {{label:'',note:''}}}})); }}
function download(name, text, type) {{ const blob=new Blob([text],{{type}}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=name; a.click(); URL.revokeObjectURL(url); }}
document.getElementById('exportJson').onclick = () => download('r2_casebook_annotations.json', JSON.stringify(rows(), null, 2), 'application/json');
document.getElementById('exportCsv').onclick = () => {{
  const h=['case_id','bucket','target_model','target_seat_wind','label','note','replay_url','naga_url'];
  const esc=v => '"' + String(v ?? '').replaceAll('"','""') + '"';
  const text=[h.join(','), ...rows().map(r => h.map(k => esc(k==='label'||k==='note' ? r.annotation[k] : r[k])).join(','))].join('\\n')+'\\n';
  download('r2_casebook_annotations.csv', text, 'text/csv');
}};
render();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output_dir = resolve_path(args.output_dir)
    if output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output dir already exists, use --overwrite: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    buckets = collect_candidates(
        {
            "T2a_1000h": (args.t2a_log_dir, "T2a_71800"),
            "T2b_250h": (args.t2b_log_dir, "T2b_71400"),
        }
    )
    cases = select_cases(buckets)
    export_casebook(
        cases,
        output_dir=output_dir,
        base_url=str(args.base_url),
        skip_local_reviews=bool(args.skip_local_reviews),
    )
    print(f"saved R2 casebook: {output_dir}", flush=True)
    print(f"cases: {len(cases)}", flush=True)


if __name__ == "__main__":
    main()
