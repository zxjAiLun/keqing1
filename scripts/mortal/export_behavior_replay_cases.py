"""Export representative behavior-drift replay cases for GUI review."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from scripts.mortal.analyze_checkpoint_behavior_slices import expand_logs
from scripts.mortal.analyze_checkpoint_behavior_slices import iter_events
from scripts.mortal.analyze_checkpoint_behavior_slices import round_stage
from scripts.mortal.analyze_checkpoint_behavior_slices import score_bucket
from scripts.mortal.analyze_checkpoint_behavior_slices import score_ranks
from scripts.mortal.analyze_checkpoint_decision_drift import CALL_TYPES
from scripts.mortal.analyze_checkpoint_decision_drift import action_score
from scripts.mortal.analyze_checkpoint_decision_drift import expand_compact_meta


@dataclass
class CaseCandidate:
    case_kind: str
    model_label: str
    source_log: str
    events: list[dict[str, Any]]
    focus_event_index: int
    focus_step: int
    actor: int
    action_type: str
    dealer: bool
    start_rank: int
    round_stage: str
    score_bucket: str
    turn: int
    margin: float
    chosen_q: float
    alternative_q: float
    shanten: int | None
    outcome: str
    agari: bool
    houjuu: bool
    ryukyoku: bool
    selection_score: float


def load_events(path: Path) -> list[dict[str, Any]]:
    return list(iter_events(path))


def collect_candidates(log_patterns: list[str], *, model_label: str) -> list[CaseCandidate]:
    candidates: list[CaseCandidate] = []
    for path in expand_logs(log_patterns):
        events = load_events(path)
        candidates.extend(collect_game_candidates(events, source_log=str(path), model_label=model_label))
    return candidates


def collect_game_candidates(
    events: list[dict[str, Any]], *, source_log: str, model_label: str
) -> list[CaseCandidate]:
    candidates: list[CaseCandidate] = []
    active: list[dict[str, Any]] = []
    context: dict[str, Any] | None = None
    for event_index, event in enumerate(events):
        event_type = str(event.get("type", ""))
        if event_type == "start_kyoku":
            active = []
            context = start_context(event)
            continue
        if context is None:
            continue
        if event_type == "tsumo":
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                context["turns"][actor] += 1
            continue
        if event_type in CALL_TYPES:
            candidate = candidate_from_event(
                event,
                event_index=event_index,
                events=events,
                context=context,
                source_log=source_log,
                model_label=model_label,
            )
            if candidate is not None:
                active.append(candidate)
            continue
        if event_type in {"hora", "ryukyoku"}:
            apply_terminal(active, event)
            continue
        if event_type == "end_kyoku":
            for raw in active:
                candidates.extend(materialize_case_kinds(raw))
            active = []
            context = None
    return candidates


def start_context(event: dict[str, Any]) -> dict[str, Any]:
    scores = [int(score) for score in event.get("scores", [25000, 25000, 25000, 25000])]
    return {
        "oya": int(event.get("oya", 0)),
        "bakaze": str(event.get("bakaze", "E")),
        "kyoku": int(event.get("kyoku", 1)),
        "scores": scores,
        "start_ranks": score_ranks(scores),
        "turns": [0, 0, 0, 0],
    }


def candidate_from_event(
    event: dict[str, Any],
    *,
    event_index: int,
    events: list[dict[str, Any]],
    context: dict[str, Any],
    source_log: str,
    model_label: str,
) -> dict[str, Any] | None:
    actor = event.get("actor")
    meta = event.get("meta")
    if not isinstance(actor, int) or not (0 <= actor < 4) or not isinstance(meta, dict):
        return None
    try:
        q_values, action_mask = expand_compact_meta(meta)
    except RuntimeError:
        return None
    chosen_q = action_score(event, q_values=q_values, action_mask=action_mask)
    pass_q = q_values[45] if action_mask[45] else None
    if chosen_q is None or pass_q is None:
        return None
    scores = context["scores"]
    turn = max(1, int(context["turns"][actor]))
    return {
        "model_label": model_label,
        "source_log": source_log,
        "events": events,
        "focus_event_index": event_index,
        "focus_step": event_index,
        "actor": actor,
        "action_type": str(event.get("type", "")),
        "dealer": actor == int(context["oya"]),
        "start_rank": int(context["start_ranks"][actor]),
        "round_stage": round_stage(str(context["bakaze"]), int(context["kyoku"])),
        "score_bucket": score_bucket(scores[actor] - max(score for idx, score in enumerate(scores) if idx != actor)),
        "turn": turn,
        "margin": float(chosen_q) - float(pass_q),
        "chosen_q": float(chosen_q),
        "alternative_q": float(pass_q),
        "shanten": int(meta["shanten"]) if isinstance(meta.get("shanten"), int) else None,
        "focus_event": dict(event),
        "agari": False,
        "houjuu": False,
        "ryukyoku": False,
    }


def apply_terminal(active: list[dict[str, Any]], terminal: dict[str, Any]) -> None:
    event_type = str(terminal.get("type", ""))
    if event_type == "ryukyoku":
        for candidate in active:
            candidate["ryukyoku"] = True
        return
    actor = terminal.get("actor")
    target = terminal.get("target")
    for candidate in active:
        if candidate["actor"] == actor:
            candidate["agari"] = True
        if isinstance(target, int) and target != actor and candidate["actor"] == target:
            candidate["houjuu"] = True


def materialize_case_kinds(raw: dict[str, Any]) -> list[CaseCandidate]:
    cases: list[CaseCandidate] = []
    bad = bool(raw["houjuu"] or not raw["agari"])
    good = bool(raw["agari"] and not raw["houjuu"])
    specs = []
    if raw["model_label"] == "80k" and raw["dealer"] and bad:
        specs.append(("80k_dealer_call_bad", bad_score(raw)))
    if raw["model_label"] == "80k" and raw["start_rank"] == 2 and bad:
        specs.append(("80k_rank2_call_bad", bad_score(raw)))
    if raw["model_label"] == "70k" and raw["dealer"] and good:
        specs.append(("70k_dealer_call_good", good_score(raw)))
    if raw["model_label"] == "70k" and raw["start_rank"] == 2 and good:
        specs.append(("70k_rank2_call_good", good_score(raw)))
    for case_kind, selection_score in specs:
        cases.append(
            CaseCandidate(
                case_kind=case_kind,
                model_label=raw["model_label"],
                source_log=raw["source_log"],
                events=raw["events"],
                focus_event_index=int(raw["focus_event_index"]),
                focus_step=int(raw["focus_step"]),
                actor=int(raw["actor"]),
                action_type=str(raw["action_type"]),
                dealer=bool(raw["dealer"]),
                start_rank=int(raw["start_rank"]),
                round_stage=str(raw["round_stage"]),
                score_bucket=str(raw["score_bucket"]),
                turn=int(raw["turn"]),
                margin=float(raw["margin"]),
                chosen_q=float(raw["chosen_q"]),
                alternative_q=float(raw["alternative_q"]),
                shanten=raw["shanten"],
                outcome=outcome_label(raw),
                agari=bool(raw["agari"]),
                houjuu=bool(raw["houjuu"]),
                ryukyoku=bool(raw["ryukyoku"]),
                selection_score=float(selection_score),
            )
        )
    return cases


def bad_score(raw: dict[str, Any]) -> float:
    return float(raw["margin"]) + 2.0 * int(bool(raw["houjuu"])) + 0.25 * int(not bool(raw["agari"]))


def good_score(raw: dict[str, Any]) -> float:
    return float(raw["margin"]) + 1.0 * int(bool(raw["agari"])) + 0.25 * int(not bool(raw["houjuu"]))


def outcome_label(raw: dict[str, Any]) -> str:
    if raw["houjuu"]:
        return "houjuu"
    if raw["agari"]:
        return "agari"
    if raw["ryukyoku"]:
        return "ryukyoku"
    return "not_agari"


def select_cases(candidates: list[CaseCandidate], *, top_k: int) -> list[CaseCandidate]:
    by_kind: dict[str, list[CaseCandidate]] = {}
    for candidate in candidates:
        by_kind.setdefault(candidate.case_kind, []).append(candidate)
    selected: list[CaseCandidate] = []
    for case_kind in sorted(by_kind):
        rows = sorted(by_kind[case_kind], key=lambda item: item.selection_score, reverse=True)
        selected.extend(rows[:top_k])
    return selected


def write_cases(cases: list[CaseCandidate], output_dir: Path) -> list[dict[str, Any]]:
    cases_dir = output_dir / "cases"
    review_dir = output_dir / "review_payloads"
    cases_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        case_id = f"case_{idx:04d}_{case.case_kind}"
        mjson_path = cases_dir / f"{case_id}.mjson"
        review_path = review_dir / f"{case_id}.review.json"
        write_mjson(mjson_path, case.events)
        review_payload = {
            "case_id": case_id,
            "focus_event": case.events[case.focus_event_index],
            "q": {
                "chosen_q": case.chosen_q,
                "alternative_action": "pass",
                "alternative_q": case.alternative_q,
                "margin": case.margin,
            },
        }
        review_path.write_text(json.dumps(review_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        manifest_row = manifest_row_for_case(
            case,
            case_id=case_id,
            mjson_path=mjson_path,
            review_path=review_path,
            output_dir=output_dir,
        )
        manifest.append(manifest_row)
    return manifest


def write_mjson(path: Path, events: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def manifest_row_for_case(
    case: CaseCandidate, *, case_id: str, mjson_path: Path, review_path: Path, output_dir: Path
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "case_kind": case.case_kind,
        "source_log": case.source_log,
        "mjson_path": str(mjson_path.relative_to(output_dir)),
        "review_payload_path": str(review_path.relative_to(output_dir)),
        "focus_event_index": case.focus_event_index,
        "focus_step": case.focus_step,
        "model_label": case.model_label,
        "checkpoint_path": checkpoint_path_for_model(case.model_label),
        "slice_tags": slice_tags(case),
        "decision_kind": "chosen_call",
        "action_type": case.action_type,
        "actor": case.actor,
        "turn": case.turn,
        "margin": case.margin,
        "chosen_q": case.chosen_q,
        "alternative_action": "pass",
        "alternative_q": case.alternative_q,
        "outcome": case.outcome,
        "agari": case.agari,
        "houjuu": case.houjuu,
        "ryukyoku": case.ryukyoku,
        "why_selected": why_selected(case),
        "selection_score": case.selection_score,
        "shanten": case.shanten,
    }


def checkpoint_path_for_model(model_label: str) -> str | None:
    if model_label == "70k":
        return "artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth"
    if model_label == "80k":
        return "artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth"
    return None


def slice_tags(case: CaseCandidate) -> list[str]:
    tags = [
        "chosen_call",
        "dealer" if case.dealer else "nondealer",
        f"start_rank_{case.start_rank}",
        case.round_stage,
        case.score_bucket,
    ]
    if case.houjuu:
        tags.append("houjuu")
    if case.agari:
        tags.append("agari")
    return tags


def why_selected(case: CaseCandidate) -> str:
    if case.case_kind.endswith("_bad"):
        return f"high_margin_bad_{case.case_kind.removeprefix('80k_').removesuffix('_bad')}"
    return f"high_margin_good_{case.case_kind.removeprefix('70k_').removesuffix('_good')}"


def write_manifest(output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    (output_dir / "manifest.json").write_text(
        json.dumps({"schema": "keqing.mortal.behavior_replay_cases.v1", "cases": manifest}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-70k", action="append", required=True)
    parser.add_argument("--logs-80k", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    candidates = [
        *collect_candidates(args.logs_70k, model_label="70k"),
        *collect_candidates(args.logs_80k, model_label="80k"),
    ]
    selected = select_cases(candidates, top_k=args.top_k)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_cases(selected, args.output_dir)
    write_manifest(args.output_dir, manifest)
    print(
        json.dumps(
            {
                "schema": "keqing.mortal.behavior_replay_cases.v1",
                "candidate_count": len(candidates),
                "exported_count": len(manifest),
                "output_dir": str(args.output_dir),
                "case_kinds": {kind: sum(row["case_kind"] == kind for row in manifest) for kind in sorted({row["case_kind"] for row in manifest})},
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
