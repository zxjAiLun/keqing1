#!/usr/bin/env python3
"""Generate Mortal selfplay replays with RiichiEnv."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, DefaultDict, Mapping, Sequence

import torch
from riichienv import RiichiEnv

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from inference.mortal_bot import MortalReviewBot
from keqingrl.mortal_teacher import MORTAL_ACTION_SPACE


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 4-Mortal RiichiEnv selfplay replays")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--game-mode", default="4p-red-half")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--trace-mode", choices=("compact", "full"), default="compact")
    return parser.parse_args()


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _event_equal(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    return _json_value_matches(left, right)


def _json_value_matches(left: Any, right: Any) -> bool:
    if left == "?" or right == "?":
        return True
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        if set(left.keys()) != set(right.keys()):
            return False
        return all(_json_value_matches(left[key], right[key]) for key in left)
    if isinstance(left, Sequence) and not isinstance(left, str) and isinstance(right, Sequence) and not isinstance(right, str):
        if len(left) != len(right):
            return False
        return all(_json_value_matches(lval, rval) for lval, rval in zip(left, right))
    return left == right


def _assign_event_indices(
    *,
    events: Sequence[Mapping[str, Any]],
    mjai_log: Sequence[Mapping[str, Any]],
    cursor: int,
) -> tuple[list[int], int, int]:
    indices: list[int] = []
    mismatch_count = 0
    next_cursor = int(cursor)
    for event in events:
        if next_cursor < len(mjai_log) and _event_equal(event, mjai_log[next_cursor]):
            indices.append(next_cursor)
            next_cursor += 1
            continue
        found = None
        for candidate in range(next_cursor + 1, len(mjai_log)):
            if _event_equal(event, mjai_log[candidate]):
                found = candidate
                break
        if found is None:
            indices.append(-1)
            mismatch_count += 1
            continue
        indices.append(found)
        next_cursor = found + 1
        mismatch_count += 1
    return indices, next_cursor, mismatch_count


def _sidecar_entry(
    *,
    actor: int,
    event_index: int,
    event: Mapping[str, Any],
    reaction: Mapping[str, Any] | None,
    meta: Mapping[str, Any],
    trace_mode: str,
) -> dict[str, Any] | None:
    meta = dict(meta)
    compact_q = meta.get("q_values")
    mask_bits = meta.get("mask_bits")
    if compact_q is None or mask_bits is None:
        return None
    mortal_meta: dict[str, Any] = {
        "mask_bits": int(mask_bits),
        "q_values": [float(value) for value in compact_q],
        "eval_time_ns": meta.get("eval_time_ns"),
    }
    if trace_mode == "full":
        expanded_q, action_mask = _expand_compact_meta(mask_bits=int(mask_bits), compact_q=compact_q)
        mortal_meta["expanded_q_values"] = expanded_q
        mortal_meta["action_mask"] = action_mask
    return {
        "event_index": int(event_index),
        "actor": int(actor),
        "event_type": str(event.get("type", "")),
        "chosen_action": None if reaction is None else dict(reaction),
        "mortal_meta": mortal_meta,
    }


def _make_env(*, game_mode: str, seed: int | None):
    if seed is None:
        return RiichiEnv(game_mode=game_mode)
    try:
        return RiichiEnv(game_mode=game_mode, seed=int(seed))
    except TypeError:
        return RiichiEnv(game_mode=game_mode)


def _write_mjson(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def _new_bots(args: argparse.Namespace) -> dict[int, MortalReviewBot]:
    first = MortalReviewBot(
        player_id=0,
        model_path=args.model,
        mortal_root=args.mortal_root,
        device=str(args.device),
        enable_review_log=False,
    )
    bots = {0: first}
    for pid in range(1, 4):
        bots[pid] = MortalReviewBot(
            player_id=pid,
            model_path=args.model,
            mortal_root=args.mortal_root,
            device=str(args.device),
            enable_review_log=False,
            shared_mortal_engine=first._mortal_engine,
            shared_model=first.model,
        )
    return bots


def _expand_compact_meta(*, mask_bits: int, compact_q: Sequence[Any]) -> tuple[list[float], list[bool]]:
    compact_values = [float(value) for value in compact_q]
    q_values = [float("-inf")] * MORTAL_ACTION_SPACE
    action_mask = [False] * MORTAL_ACTION_SPACE
    compact_idx = 0
    for action_id in range(MORTAL_ACTION_SPACE):
        if not (int(mask_bits) & (1 << action_id)):
            continue
        action_mask[action_id] = True
        if compact_idx >= len(compact_values):
            raise RuntimeError("Mortal compact q_values shorter than mask_bits")
        q_values[action_id] = compact_values[compact_idx]
        compact_idx += 1
    if compact_idx != len(compact_values):
        raise RuntimeError("Mortal compact q_values longer than mask_bits")
    return q_values, action_mask


def _legal_action_to_mjai(action: Any) -> dict[str, Any]:
    if hasattr(action, "to_mjai"):
        payload = action.to_mjai()
        if isinstance(payload, str):
            return json.loads(payload)
        return dict(payload)
    return dict(action)


def run_game(args: argparse.Namespace, *, game_id: int) -> dict[str, Any]:
    game_seed = None if args.seed is None else int(args.seed) + int(game_id)
    env = _make_env(game_mode=str(args.game_mode), seed=game_seed)
    bots = _new_bots(args)
    obs_dict = env.reset()
    event_cursors = {pid: 0 for pid in range(4)}
    sidecar: DefaultDict[str, list[dict[str, Any]]] = defaultdict(list)
    obs_size_histogram: Counter[str] = Counter()
    react_call_count = 0
    actual_decision_count = 0
    fallback_count = 0
    no_action_fallback_count = 0
    action_convert_fail_count = 0
    event_index_mismatch_count = 0
    mortal_eval_time_ns_sum = 0
    step_count = 0
    started = time.perf_counter()

    while not env.done():
        obs_size_histogram[str(len(obs_dict))] += 1
        actions: dict[int, Any] = {}
        for pid_raw, obs in obs_dict.items():
            pid = int(pid_raw)
            raw_events = obs.new_events()
            parsed_events = [json.loads(event) if isinstance(event, str) else dict(event) for event in raw_events]
            event_indices, next_cursor, mismatches = _assign_event_indices(
                events=parsed_events,
                mjai_log=getattr(env, "mjai_log", []),
                cursor=event_cursors.get(pid, 0),
            )
            event_cursors[pid] = next_cursor
            event_index_mismatch_count += mismatches

            mjai_action: dict[str, Any] | None = None
            for event, event_index in zip(parsed_events, event_indices):
                reaction = bots[pid].react(event)
                react_call_count += 1
                if reaction is not None:
                    raw_reaction = dict(reaction)
                    meta = dict(raw_reaction.pop("meta", {}) or {})
                    mjai_action = raw_reaction
                else:
                    meta = {}
                if not meta:
                    continue
                actual_decision_count += 1
                trace_entry = _sidecar_entry(
                    actor=pid,
                    event_index=event_index,
                    event=event,
                    reaction=mjai_action,
                    meta=meta,
                    trace_mode=str(args.trace_mode),
                )
                if trace_entry is not None:
                    sidecar[str(pid)].append(trace_entry)
                    eval_time = trace_entry["mortal_meta"].get("eval_time_ns")
                    if eval_time is not None:
                        mortal_eval_time_ns_sum += int(eval_time)

            legal_actions = obs.legal_actions()
            if not legal_actions:
                continue
            if mjai_action is None:
                actions[pid] = legal_actions[0]
                fallback_count += 1
                no_action_fallback_count += 1
                continue
            selected = obs.select_action_from_mjai(json.dumps(mjai_action, ensure_ascii=False, separators=(",", ":")))
            if selected is None:
                actions[pid] = legal_actions[0]
                fallback_count += 1
                action_convert_fail_count += 1
            else:
                actions[pid] = selected
        if not actions:
            raise RuntimeError("RiichiEnv selfplay stalled without any legal action")
        obs_dict = env.step(actions)
        step_count += 1
        if step_count > int(args.max_steps):
            raise RuntimeError(f"RiichiEnv selfplay exceeded max_steps={args.max_steps}")

    elapsed = time.perf_counter() - started
    replay_events = [dict(event) for event in getattr(env, "mjai_log", [])]
    scores = list(env.scores())
    ranks = list(env.ranks())
    return {
        "game_id": int(game_id),
        "seed": game_seed,
        "scores": scores,
        "ranks": ranks,
        "events": replay_events,
        "sidecar": {"by_actor": dict(sidecar)},
        "summary": {
            "game_id": int(game_id),
            "seed": game_seed,
            "wall_time_sec": elapsed,
            "env_step_count": int(step_count),
            "obs_dict_size_histogram": dict(obs_size_histogram),
            "react_call_count": int(react_call_count),
            "actual_decision_count": int(actual_decision_count),
            "mortal_eval_time_ns_sum": int(mortal_eval_time_ns_sum),
            "mjai_event_count": len(replay_events),
            "hora_count": sum(1 for event in replay_events if event.get("type") == "hora"),
            "ryukyoku_count": sum(1 for event in replay_events if event.get("type") == "ryukyoku"),
            "fallback_count": int(fallback_count),
            "no_action_fallback_count": int(no_action_fallback_count),
            "select_action_from_mjai_fail_count": int(action_convert_fail_count),
            "event_index_mismatch_count": int(event_index_mismatch_count),
            "scores": scores,
            "ranks": ranks,
        },
    }


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir
    replays_dir = output_dir / "replays"
    replays_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    started = time.perf_counter()

    for game_id in range(int(args.games)):
        result = run_game(args, game_id=game_id)
        replay_path = replays_dir / f"game_{game_id:05d}.mjson"
        sidecar_path = replays_dir / f"game_{game_id:05d}.decisions.json"
        meta_path = replays_dir / f"game_{game_id:05d}.json"
        _write_mjson(replay_path, result["events"])
        sidecar_path.write_text(json.dumps(result["sidecar"], ensure_ascii=False, indent=2), encoding="utf-8")
        meta = {
            **result["summary"],
            "mjson": str(replay_path),
            "decision_traces": str(sidecar_path),
        }
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        summaries.append(result["summary"])
        manifest.append(
            {
                "game_id": int(game_id),
                "mjson": str(replay_path),
                "meta": str(meta_path),
                "decision_traces": str(sidecar_path),
                "scores": result["summary"]["scores"],
                "ranks": result["summary"]["ranks"],
            }
        )
        print(
            "riichienv-mortal game="
            f"{game_id + 1}/{int(args.games)} steps={result['summary']['env_step_count']} "
            f"events={result['summary']['mjai_event_count']} fallback={result['summary']['fallback_count']} "
            f"wall={result['summary']['wall_time_sec']:.2f}s",
            flush=True,
        )

    total_wall = time.perf_counter() - started
    summary = {
        "backend": "riichienv",
        "game_mode": str(args.game_mode),
        "games": int(args.games),
        "model": str(args.model),
        "device": str(args.device),
        "wall_time_sec": total_wall,
        "game_summaries": summaries,
        "totals": {
            "env_step_count": sum(int(row["env_step_count"]) for row in summaries),
            "mjai_event_count": sum(int(row["mjai_event_count"]) for row in summaries),
            "react_call_count": sum(int(row["react_call_count"]) for row in summaries),
            "actual_decision_count": sum(int(row["actual_decision_count"]) for row in summaries),
            "mortal_eval_time_ns_sum": sum(int(row["mortal_eval_time_ns_sum"]) for row in summaries),
            "hora_count": sum(int(row["hora_count"]) for row in summaries),
            "ryukyoku_count": sum(int(row["ryukyoku_count"]) for row in summaries),
            "fallback_count": sum(int(row["fallback_count"]) for row in summaries),
            "select_action_from_mjai_fail_count": sum(int(row["select_action_from_mjai_fail_count"]) for row in summaries),
            "event_index_mismatch_count": sum(int(row["event_index_mismatch_count"]) for row in summaries),
        },
    }
    (replays_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
