#!/usr/bin/env python3
"""Export GUI review cases for a KeqingRL checkpoint on a small replay slice."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.run_keqingrl_mortal_imitation import (  # noqa: E402
    DeltaSupportProjectionPolicy,
    _build_replay_imitation_batch,
    _load_policy,
    _write_csv,
    _write_json,
    _write_jsonl,
    imitation_metrics,
    prepare_mortal_imitation_teacher_data,
)


_BOOLEAN_OPTIONAL_CONFIG_KEYS = {
    "mortal_teacher_strict_extra_mask",
    "replay_decision_sidecar_cache",
}


def _parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, remaining_argv = config_parser.parse_known_args()
    configured_argv: list[str] = []
    if config_args.config is not None:
        configured_argv = _config_mapping_to_argv(_load_json_config(config_args.config))

    parser = argparse.ArgumentParser(description="Export KeqingRL decision review cases")
    parser.add_argument("--config", type=Path, default=config_args.config)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mortal-teacher-checkpoint", type=Path, default=Path("artifacts/mortal_training/mortal.pth"))
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--actors", type=int, nargs="+", default=(0, 1, 2, 3))
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--teacher-support", choices=("topk", "adaptive-topk", "full-legal"), default="full-legal")
    parser.add_argument("--teacher-topk", type=int, default=3)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument("--mortal-teacher-strict-extra-mask", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--replay-decision-sidecar-cache", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--decision-review-case-limit", type=int, default=500)
    parser.add_argument("--rule-score-scale", type=float, default=0.0)
    parser.add_argument("--support-policy-mode", choices=("support-only-topk", "unrestricted"), default="unrestricted")
    parser.add_argument("--delta-support-mode", choices=("topk", "all"), default="all")
    parser.add_argument("--delta-support-topk", type=int, default=3)
    parser.add_argument("--delta-support-margin-threshold", type=float, default=0.75)
    parser.add_argument("--outside-support-delta-mode", choices=("zero", "negative-clip"), default="zero")
    parser.add_argument("--max-kyokus", type=int, default=0)
    parser.add_argument("--self-turn-action-types", nargs="+", default=("DISCARD", "REACH_DISCARD", "TSUMO", "ANKAN", "KAKAN", "RYUKYOKU"))
    parser.add_argument("--response-action-types", nargs="*", default=("PASS", "RON", "PON", "CHI", "DAIMINKAN"))
    parser.add_argument("--forced-autopilot-action-types", nargs="*", default=("TSUMO", "RON", "RYUKYOKU"))
    args = parser.parse_args(configured_argv + remaining_argv)
    args.learner_seats = tuple(int(actor) for actor in args.actors)
    args.export_decision_review_cases = True
    if int(args.limit) <= 0:
        raise ValueError("--limit must be positive for review export")
    return args


def _load_json_config(path: Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"config must be a JSON object: {path}")
    return dict(payload)


def _config_mapping_to_argv(config: Mapping[str, Any]) -> list[str]:
    argv: list[str] = []
    for key, value in config.items():
        if value is None:
            continue
        flag = f"--{str(key).replace('_', '-')}"
        if isinstance(value, bool):
            if key in _BOOLEAN_OPTIONAL_CONFIG_KEYS or key == "recursive":
                argv.append(flag if value else f"--no-{str(key).replace('_', '-')}")
                continue
            raise ValueError(f"boolean config key is not a known boolean CLI option: {key}")
        argv.append(flag)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            argv.extend(str(item) for item in value)
        else:
            argv.append(str(value))
    return argv


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    replay_paths = _replay_paths(args)
    if not replay_paths:
        raise RuntimeError(f"no .mjson files found under {args.replay_dir}")
    device = torch.device(args.device)
    current_candidate = _candidate_from_checkpoint(args.checkpoint, args.output_dir / "checkpoint_config.json")
    current_policy = _wrap_policy(args, _load_policy(current_candidate, device)).eval()
    parent_candidate = _parent_candidate_from_checkpoint(args.checkpoint, args.output_dir / "parent_checkpoint_config.json")
    parent_policy = _wrap_policy(args, _load_policy(parent_candidate, device)).eval() if parent_candidate is not None else current_policy

    steps, batch, replay_summary = _build_replay_imitation_batch(args, replay_paths=replay_paths, rollout_seed=0)
    batch = batch.to(device)
    teacher_data = prepare_mortal_imitation_teacher_data(
        batch.policy_input,
        prepared_steps=steps,
        strict_extra=bool(args.mortal_teacher_strict_extra_mask),
        teacher_support=str(args.teacher_support),
        teacher_topk=int(args.teacher_topk),
    )
    with torch.no_grad():
        output = current_policy(batch.policy_input)
        parent_output = parent_policy(batch.policy_input)
        metrics, changed_rows, disagreement_rows, review_rows, review_case_rows, action_breakdown = imitation_metrics(
            output,
            batch.policy_input,
            parent_output=parent_output,
            source_output=parent_output,
            prepared_steps=steps,
            teacher_support=str(args.teacher_support),
            teacher_topk=int(args.teacher_topk),
            teacher_temperature=float(args.teacher_temperature),
            teacher_batch=teacher_data.teacher_batch,
            mapping_summary=teacher_data.summary,
            export_decision_review_cases=True,
        )
    review_case_rows = _risk_sorted_cases(review_case_rows)[: int(args.decision_review_case_limit)]
    _write_jsonl(args.output_dir / "decision_review_cases.jsonl", review_case_rows)
    _write_csv(args.output_dir / "changed_decisions.csv", changed_rows)
    _write_csv(args.output_dir / "teacher_disagreements.csv", disagreement_rows)
    _write_csv(args.output_dir / "decision_review_candidates.csv", review_rows)
    _write_csv(args.output_dir / "imitation_action_type_breakdown.csv", action_breakdown)
    summary = {
        "checkpoint": str(args.checkpoint),
        "replay_dir": str(args.replay_dir),
        "replay_count": len(replay_paths),
        "row_count": int(batch.policy_input.legal_action_mask.shape[0]),
        "review_case_count": len(review_case_rows),
        **replay_summary,
        **teacher_data.summary,
        **metrics,
    }
    _write_csv(args.output_dir / "review_summary.csv", [summary])
    _write_json(args.output_dir / "review_summary.json", summary)
    print(
        f"review-export done replays={len(replay_paths)} rows={summary['row_count']} "
        f"cases={len(review_case_rows)} out={args.output_dir}",
        flush=True,
    )


def _wrap_policy(args: argparse.Namespace, base_policy):
    base_policy.rule_score_scale = float(args.rule_score_scale)
    return DeltaSupportProjectionPolicy(
        base_policy,
        support_mode=str(args.delta_support_mode),
        topk=int(args.delta_support_topk),
        margin_threshold=float(args.delta_support_margin_threshold),
        outside_support_delta_mode=str(args.outside_support_delta_mode),
        support_policy_mode=str(args.support_policy_mode),
    )


def _replay_paths(args: argparse.Namespace) -> list[Path]:
    paths = sorted(Path(args.replay_dir).rglob("*.mjson") if bool(args.recursive) else Path(args.replay_dir).glob("*.mjson"))
    if int(args.skip) > 0:
        paths = paths[int(args.skip) :]
    return paths[: int(args.limit)]


def _candidate_from_checkpoint(checkpoint: Path, config_out: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        raise RuntimeError(f"checkpoint missing config payload: {checkpoint}")
    config_out.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "source_config_id": 93,
        "rerun_config_id": 0,
        "checkpoint_path": str(checkpoint),
        "checkpoint_sha256": "",
        "config_path": str(config_out),
    }


def _parent_candidate_from_checkpoint(checkpoint: Path, config_out: Path) -> dict[str, Any] | None:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config_payload = payload.get("config")
    if not isinstance(config_payload, Mapping):
        return None
    parent_path = config_payload.get("parent_checkpoint_path")
    if not parent_path:
        return None
    parent_config = config_payload.get("parent_config_path") or config_payload.get("config_path")
    if parent_config and Path(str(parent_config)).exists():
        config_path = Path(str(parent_config))
    else:
        config_out.write_text(json.dumps(config_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        config_path = config_out
    return {
        "source_config_id": 93,
        "rerun_config_id": 0,
        "checkpoint_path": str(parent_path),
        "checkpoint_sha256": str(config_payload.get("parent_checkpoint_sha256", "")),
        "config_path": str(config_path),
    }


def _risk_sorted_cases(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    def score(row: Mapping[str, Any]) -> tuple[int, int, int, int]:
        reason = str(row.get("review_reason", ""))
        return (
            int(bool(row.get("selected_changed")) and bool(row.get("teacher_disagreed"))),
            int("regressed" in reason or "teacher_disagreement" in reason),
            int(any(token in reason for token in ("kan", "reach", "call", "ron"))),
            int(bool(row.get("teacher_disagreed"))),
        )

    return sorted((dict(row) for row in rows), key=score, reverse=True)


if __name__ == "__main__":
    main()
