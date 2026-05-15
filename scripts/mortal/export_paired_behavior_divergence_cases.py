"""Export paired 70k/80k first-divergence cases for GUI review."""

from __future__ import annotations

import argparse
import glob
import gzip
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.analyze_checkpoint_behavior_slices import round_stage
from scripts.mortal.analyze_checkpoint_behavior_slices import score_bucket
from scripts.mortal.analyze_checkpoint_behavior_slices import score_ranks


CALL_TYPES = {"chi", "pon", "daiminkan"}


@dataclass
class PairedCase:
    case_kind: str
    case_id_suffix: str
    left_source_log: Path
    right_source_log: Path
    left_events: list[dict[str, Any]]
    right_events: list[dict[str, Any]]
    left_focus_event_index: int
    right_focus_event_index: int
    prefix_match_event_count: int
    divergence_kind: str
    actor: int
    oya: int | None
    bakaze: str | None
    kyoku: int | None
    turn: int | None
    start_rank: int | None
    score_bucket: str | None
    left_action: dict[str, Any]
    right_action: dict[str, Any]
    downstream_summary: dict[str, Any]
    slice_tags: list[str]
    why_selected: str
    selection_score: float


def expand_logs(patterns: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in patterns:
        path = Path(pattern)
        if path.is_dir():
            files.extend(sorted(path.glob("**/*.json.gz")))
        else:
            files.extend(Path(match) for match in sorted(glob.glob(str(pattern), recursive=True)))
    return sorted(dict.fromkeys(files))


def load_events(path: Path) -> list[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def event_core(event: dict[str, Any]) -> dict[str, Any]:
    return {key: core_value(value) for key, value in event.items() if key != "meta"}


def core_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: core_value(val) for key, val in value.items() if key != "meta"}
    if isinstance(value, list):
        return [core_value(item) for item in value]
    return value


def pair_logs(left_logs: list[Path], right_logs: list[Path]) -> list[tuple[Path, Path]]:
    right_by_name = {path.name: path for path in right_logs}
    return [(left, right_by_name[left.name]) for left in left_logs if left.name in right_by_name]


def find_first_divergence(left_events: list[dict[str, Any]], right_events: list[dict[str, Any]]) -> int | None:
    for idx, (left, right) in enumerate(zip(left_events, right_events, strict=False)):
        if event_core(left) != event_core(right):
            return idx
    if len(left_events) != len(right_events):
        return min(len(left_events), len(right_events))
    return None


def previous_discard_index(events: list[dict[str, Any]], before_index: int) -> int:
    for idx in range(before_index - 1, -1, -1):
        if events[idx].get("type") == "dahai":
            return idx
    return max(0, before_index - 1)


def context_before(events: list[dict[str, Any]], before_index: int) -> dict[str, Any]:
    context: dict[str, Any] = {
        "oya": None,
        "bakaze": None,
        "kyoku": None,
        "scores": [25000, 25000, 25000, 25000],
        "start_ranks": [1, 2, 3, 4],
        "turns": [0, 0, 0, 0],
    }
    for event in events[:before_index]:
        event_type = event.get("type")
        if event_type == "start_kyoku":
            scores = [int(score) for score in event.get("scores", [25000, 25000, 25000, 25000])]
            context = {
                "oya": int(event.get("oya", 0)),
                "bakaze": str(event.get("bakaze", "E")),
                "kyoku": int(event.get("kyoku", 1)),
                "scores": scores,
                "start_ranks": score_ranks(scores),
                "turns": [0, 0, 0, 0],
            }
        elif event_type == "tsumo":
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                context["turns"][actor] += 1
    return context


def final_scores(events: list[dict[str, Any]]) -> list[int]:
    scores = [25000, 25000, 25000, 25000]
    for event in events:
        if event.get("type") == "start_kyoku":
            raw_scores = event.get("scores")
            if isinstance(raw_scores, list) and len(raw_scores) == 4:
                scores = [int(score) for score in raw_scores]
        elif event.get("type") == "hora":
            deltas = event.get("deltas")
            if isinstance(deltas, list) and len(deltas) == 4:
                scores = [int(score + delta) for score, delta in zip(scores, deltas, strict=True)]
    return scores


def kyoku_outcome_after(events: list[dict[str, Any]], start_index: int, actor: int) -> str:
    for offset, event in enumerate(events[start_index:]):
        event_type = event.get("type")
        if offset > 0 and event_type == "start_kyoku":
            return "not_terminal"
        if event_type == "hora":
            winner = event.get("actor")
            target = event.get("target")
            if winner == actor:
                return "agari"
            if isinstance(target, int) and target == actor and winner != actor:
                return "houjuu"
            return "other_agari"
        if event_type == "ryukyoku":
            return "ryukyoku"
        if event_type == "end_kyoku":
            return "not_terminal"
    return "not_terminal"


def classify_divergence(left: dict[str, Any] | None, right: dict[str, Any] | None) -> str:
    left_type = str((left or {}).get("type", "missing"))
    right_type = str((right or {}).get("type", "missing"))
    if right_type in CALL_TYPES and left_type not in CALL_TYPES:
        return f"left_pass_vs_right_{right_type}"
    if left_type in CALL_TYPES and right_type not in CALL_TYPES:
        return f"left_{left_type}_vs_right_pass"
    if left_type == "dahai" and right_type == "dahai":
        return "discard_difference"
    if left_type == "reach" or right_type == "reach":
        return "reach_difference"
    return f"{left_type}_vs_{right_type}"


def primary_case_kind(divergence_kind: str, *, dealer: bool, start_rank: int | None) -> str:
    if divergence_kind.startswith("left_pass_vs_right_"):
        if dealer:
            return "paired_dealer_call_divergence"
        if start_rank == 2:
            return "paired_rank2_call_divergence"
        return "paired_call_divergence"
    if divergence_kind == "discard_difference":
        return "paired_discard_divergence"
    if divergence_kind == "reach_difference":
        return "paired_reach_divergence"
    return "paired_first_divergence"


def selection_score(
    divergence_kind: str,
    *,
    dealer: bool,
    start_rank: int | None,
    left_outcome: str,
    right_outcome: str,
    left_rank: int | None,
    right_rank: int | None,
    prefix_count: int,
) -> float:
    score = 0.0
    if divergence_kind.startswith("left_pass_vs_right_"):
        score += 100.0
    if dealer:
        score += 20.0
    if start_rank == 2:
        score += 20.0
    if right_outcome == "houjuu":
        score += 10.0
    if left_outcome == "agari":
        score += 5.0
    if left_rank is not None and right_rank is not None:
        score += max(0, right_rank - left_rank) * 4.0
    score += max(0, 200 - prefix_count) / 200.0
    return score


def build_case(
    left_path: Path,
    right_path: Path,
    *,
    left_events: list[dict[str, Any]],
    right_events: list[dict[str, Any]],
) -> PairedCase | None:
    divergence_index = find_first_divergence(left_events, right_events)
    if divergence_index is None:
        return None
    left_event = left_events[divergence_index] if divergence_index < len(left_events) else None
    right_event = right_events[divergence_index] if divergence_index < len(right_events) else None
    divergence_kind = classify_divergence(left_event, right_event)
    focus_actor = actor_for_divergence(left_event, right_event)
    if focus_actor is None:
        return None
    ctx = context_before(left_events, divergence_index)
    oya = ctx["oya"] if isinstance(ctx["oya"], int) else None
    start_rank = int(ctx["start_ranks"][focus_actor]) if isinstance(ctx.get("start_ranks"), list) else None
    dealer = oya == focus_actor
    left_focus_event_index = divergence_index
    if divergence_kind.startswith("left_pass_vs_right_"):
        left_focus_event_index = previous_discard_index(left_events, divergence_index)
    right_focus_event_index = divergence_index
    left_scores = final_scores(left_events)
    right_scores = final_scores(right_events)
    left_ranks = score_ranks(left_scores)
    right_ranks = score_ranks(right_scores)
    left_rank = int(left_ranks[focus_actor])
    right_rank = int(right_ranks[focus_actor])
    left_outcome = kyoku_outcome_after(left_events, left_focus_event_index, focus_actor)
    right_outcome = kyoku_outcome_after(right_events, right_focus_event_index, focus_actor)
    case_kind = primary_case_kind(divergence_kind, dealer=dealer, start_rank=start_rank)
    bakaze = ctx["bakaze"] if isinstance(ctx["bakaze"], str) else None
    kyoku = int(ctx["kyoku"]) if isinstance(ctx["kyoku"], int) else None
    turn = int(ctx["turns"][focus_actor]) if isinstance(ctx.get("turns"), list) else None
    actor_score = int(ctx["scores"][focus_actor]) if isinstance(ctx.get("scores"), list) else 25000
    others = [int(score) for idx, score in enumerate(ctx["scores"]) if idx != focus_actor]
    bucket = score_bucket(actor_score - max(others)) if others else None
    tags = ["paired", case_kind]
    if divergence_kind.startswith("left_pass_vs_right_"):
        tags.append("right_call_left_pass")
    tags.append("dealer" if dealer else "nondealer")
    if start_rank is not None:
        tags.append(f"start_rank_{start_rank}")
    if bakaze is not None and kyoku is not None:
        tags.append(round_stage(bakaze, kyoku))
    if bucket is not None:
        tags.append(bucket)
    if right_outcome == "houjuu":
        tags.append("right_houjuu")
    if left_outcome == "agari":
        tags.append("left_agari")
    score = selection_score(
        divergence_kind,
        dealer=dealer,
        start_rank=start_rank,
        left_outcome=left_outcome,
        right_outcome=right_outcome,
        left_rank=left_rank,
        right_rank=right_rank,
        prefix_count=divergence_index,
    )
    left_action = event_core(left_event) if isinstance(left_event, dict) else {"type": "missing"}
    right_action = event_core(right_event) if isinstance(right_event, dict) else {"type": "missing"}
    if divergence_kind.startswith("left_pass_vs_right_"):
        left_action = {"type": "pass", "actor": focus_actor}
    elif divergence_kind.startswith("left_") and divergence_kind.endswith("_vs_right_pass"):
        right_action = {"type": "pass", "actor": focus_actor}

    return PairedCase(
        case_kind=case_kind,
        case_id_suffix=f"{left_path.stem.replace('.json', '')}_{divergence_index}",
        left_source_log=left_path,
        right_source_log=right_path,
        left_events=left_events,
        right_events=right_events,
        left_focus_event_index=left_focus_event_index,
        right_focus_event_index=right_focus_event_index,
        prefix_match_event_count=divergence_index,
        divergence_kind=divergence_kind,
        actor=focus_actor,
        oya=oya,
        bakaze=bakaze,
        kyoku=kyoku,
        turn=turn,
        start_rank=start_rank,
        score_bucket=bucket,
        left_action=left_action,
        right_action=right_action,
        downstream_summary={
            "left_kyoku_outcome": left_outcome,
            "right_kyoku_outcome": right_outcome,
            "left_final_scores": left_scores,
            "right_final_scores": right_scores,
            "left_actor_final_rank": left_rank,
            "right_actor_final_rank": right_rank,
        },
        slice_tags=tags,
        why_selected=why_selected(case_kind, divergence_kind),
        selection_score=score,
    )


def actor_for_divergence(left: dict[str, Any] | None, right: dict[str, Any] | None) -> int | None:
    right_actor = (right or {}).get("actor")
    if isinstance(right_actor, int) and 0 <= right_actor < 4:
        return right_actor
    left_actor = (left or {}).get("actor")
    if isinstance(left_actor, int) and 0 <= left_actor < 4:
        return left_actor
    return None


def why_selected(case_kind: str, divergence_kind: str) -> str:
    if case_kind == "paired_dealer_call_divergence":
        return "first divergence is right-model call while left passes, in dealer context"
    if case_kind == "paired_rank2_call_divergence":
        return "first divergence is right-model call while left passes, in start-rank-2 context"
    if case_kind == "paired_call_divergence":
        return "first divergence is right-model call while left passes"
    return f"first divergence type: {divergence_kind}"


def select_cases(cases: list[PairedCase], *, top_k: int) -> list[PairedCase]:
    by_kind: dict[str, list[PairedCase]] = {}
    for case in cases:
        by_kind.setdefault(case.case_kind, []).append(case)
    selected: list[PairedCase] = []
    for kind in sorted(by_kind):
        selected.extend(sorted(by_kind[kind], key=lambda item: item.selection_score, reverse=True)[:top_k])
    return selected


def write_mjson(path: Path, events: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_cases(cases: list[PairedCase], output_dir: Path, *, left_label: str, right_label: str) -> list[dict[str, Any]]:
    left_dir = output_dir / "pairs" / left_label
    right_dir = output_dir / "pairs" / right_label
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, 1):
        case_id = f"pair_{idx:04d}_{case.case_kind}_{case.case_id_suffix}"
        left_path = left_dir / f"{case_id}.{left_label}.mjson"
        right_path = right_dir / f"{case_id}.{right_label}.mjson"
        write_mjson(left_path, case.left_events)
        write_mjson(right_path, case.right_events)
        manifest.append(
            {
                "case_id": case_id,
                "case_kind": case.case_kind,
                "left_model": left_label,
                "right_model": right_label,
                "left_checkpoint_path": checkpoint_path_for_model(left_label),
                "right_checkpoint_path": checkpoint_path_for_model(right_label),
                "left_source_log": str(case.left_source_log),
                "right_source_log": str(case.right_source_log),
                "left_mjson_path": str(left_path.relative_to(output_dir)),
                "right_mjson_path": str(right_path.relative_to(output_dir)),
                "left_focus_event_index": case.left_focus_event_index,
                "right_focus_event_index": case.right_focus_event_index,
                "prefix_match_event_count": case.prefix_match_event_count,
                "divergence_kind": case.divergence_kind,
                "actor": case.actor,
                "oya": case.oya,
                "bakaze": case.bakaze,
                "kyoku": case.kyoku,
                "turn": case.turn,
                "start_rank": case.start_rank,
                "score_bucket": case.score_bucket,
                "left_action": case.left_action,
                "right_action": case.right_action,
                "downstream_summary": case.downstream_summary,
                "slice_tags": case.slice_tags,
                "why_selected": case.why_selected,
                "selection_score": case.selection_score,
            }
        )
    return manifest


def checkpoint_path_for_model(model_label: str) -> str | None:
    if model_label == "70k":
        return "artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth"
    if model_label == "80k":
        return "artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth"
    return None


def write_manifest(output_dir: Path, manifest: list[dict[str, Any]]) -> None:
    (output_dir / "paired_manifest.json").write_text(
        json.dumps({"schema": "keqing.mortal.paired_behavior_divergence_cases.v1", "cases": manifest}, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "paired_manifest.jsonl").open("w", encoding="utf-8") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-label", default="70k")
    parser.add_argument("--right-label", default="80k")
    parser.add_argument("--left-logs", action="append", required=True)
    parser.add_argument("--right-logs", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = pair_logs(expand_logs(args.left_logs), expand_logs(args.right_logs))
    candidates: list[PairedCase] = []
    for left_path, right_path in pairs:
        case = build_case(left_path, right_path, left_events=load_events(left_path), right_events=load_events(right_path))
        if case is not None:
            candidates.append(case)
    selected = select_cases(candidates, top_k=args.top_k)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = write_cases(selected, args.output_dir, left_label=args.left_label, right_label=args.right_label)
    write_manifest(args.output_dir, manifest)
    print(
        json.dumps(
            {
                "schema": "keqing.mortal.paired_behavior_divergence_cases.v1",
                "paired_log_count": len(pairs),
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
