#!/usr/bin/env python3
"""Run a RiichiEnv A/B match between two Mortal checkpoints."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from inference.mortal_bot import MortalReviewBot
from scripts.mortal.eval_metrics import (
    add_rank_point_args,
    build_metrics_document,
    resolve_rank_points,
    summarize_rank_counts_with_references,
    write_metrics,
)
from scripts.mortal.generate_riichienv_selfplay_replays import _make_env


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run RiichiEnv A/B match for Mortal checkpoints")
    parser.add_argument("--model-a", type=Path, required=True)
    parser.add_argument("--model-b", type=Path, required=True)
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/eval/riichienv_ab_match"))
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--game-mode", default="4p-red-half")
    parser.add_argument("--seat-mode", choices=("one-vs-three", "two-vs-two"), default="one-vs-three")
    parser.add_argument("--seed", type=int, default=10000)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add_rank_point_args(parser)
    return parser.parse_args()


def seat_assignment(game_id: int, *, seat_mode: str) -> list[str]:
    if seat_mode == "one-vs-three":
        seats = ["B", "B", "B", "B"]
        seats[game_id % 4] = "A"
        return seats
    if seat_mode == "two-vs-two":
        return (["A", "B", "A", "B"], ["B", "A", "B", "A"])[game_id % 2]
    raise ValueError(f"unsupported seat_mode: {seat_mode}")


def _new_group_bot(
    *,
    label: str,
    seat: int,
    model_path: Path,
    args: argparse.Namespace,
    shared: Mapping[str, Any] | None,
) -> MortalReviewBot:
    return MortalReviewBot(
        player_id=seat,
        model_path=model_path,
        mortal_root=args.mortal_root,
        device=str(args.device),
        enable_review_log=False,
        shared_mortal_engine=None if shared is None else shared["engine"],
        shared_model=None if shared is None else shared["model"],
        model_version=f"mortal-{label.lower()}",
    )


def _new_bots(args: argparse.Namespace, assignment: Sequence[str]) -> dict[int, MortalReviewBot]:
    shared: dict[str, dict[str, Any]] = {}
    bots: dict[int, MortalReviewBot] = {}
    for seat, label in enumerate(assignment):
        model_path = args.model_a if label == "A" else args.model_b
        bot = _new_group_bot(
            label=label,
            seat=seat,
            model_path=model_path,
            args=args,
            shared=shared.get(label),
        )
        bots[seat] = bot
        shared.setdefault(label, {"engine": bot._mortal_engine, "model": bot.model})
    return bots


def _action_to_mjai(action: Any) -> str:
    payload = action.to_mjai() if hasattr(action, "to_mjai") else action
    if isinstance(payload, str):
        return payload
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _run_game(args: argparse.Namespace, *, game_id: int) -> dict[str, Any]:
    game_seed = int(args.seed) + int(game_id)
    env, seed_info = _make_env(game_mode=str(args.game_mode), seed=game_seed)
    obs_dict = env.reset(seed=game_seed) if seed_info.mode == "reset" else env.reset()
    assignment = seat_assignment(game_id, seat_mode=str(args.seat_mode))
    bots = _new_bots(args, assignment)
    fallback_count = 0
    step_count = 0
    started = time.perf_counter()

    while not env.done():
        actions: dict[int, Any] = {}
        for seat_raw, obs in obs_dict.items():
            seat = int(seat_raw)
            mjai_action = None
            for raw_event in obs.new_events():
                event = json.loads(raw_event) if isinstance(raw_event, str) else dict(raw_event)
                reaction = bots[seat].react(event)
                if reaction is not None:
                    mjai_action = {key: value for key, value in dict(reaction).items() if key != "meta"}
            legal_actions = obs.legal_actions()
            if not legal_actions:
                continue
            if mjai_action is None:
                actions[seat] = legal_actions[0]
                fallback_count += 1
                continue
            selected = obs.select_action_from_mjai(json.dumps(mjai_action, ensure_ascii=False, separators=(",", ":")))
            if selected is None:
                actions[seat] = legal_actions[0]
                fallback_count += 1
            else:
                actions[seat] = selected
        if not actions:
            raise RuntimeError("RiichiEnv A/B match stalled without legal actions")
        obs_dict = env.step(actions)
        step_count += 1
        if step_count > int(args.max_steps):
            raise RuntimeError(f"RiichiEnv A/B match exceeded max_steps={args.max_steps}")

    scores = [float(value) for value in env.scores()]
    ranks = [int(value) for value in env.ranks()]
    replay_events = [dict(event) for event in getattr(env, "mjai_log", [])]
    replay_path = args.output_dir / "replays" / f"game_{game_id:05d}.mjson"
    replay_path.parent.mkdir(parents=True, exist_ok=True)
    replay_path.write_text("\n".join(json.dumps(event, ensure_ascii=False) for event in replay_events) + "\n", encoding="utf-8")
    return {
        "game_id": int(game_id),
        "seed": game_seed,
        "assignment": assignment,
        "scores": scores,
        "ranks": ranks,
        "fallback_count": int(fallback_count),
        "env_step_count": int(step_count),
        "mjai_event_count": len(replay_events),
        "wall_time_sec": time.perf_counter() - started,
        "replay": str(replay_path),
    }


def _summarize_games(
    games: Sequence[Mapping[str, Any]],
    *,
    rank_points: Sequence[int | float] = (90.0, 45.0, 0.0, -135.0),
) -> dict[str, Any]:
    rank_counts = {"A": Counter(), "B": Counter()}
    score_sums = {"A": 0.0, "B": 0.0}
    seat_counts = {"A": 0, "B": 0}
    for game in games:
        for seat, label in enumerate(game["assignment"]):
            rank = _rank_index(int(game["ranks"][seat]))
            rank_counts[label][rank] += 1
            score_sums[label] += float(game["scores"][seat])
            seat_counts[label] += 1

    by_label = {}
    for label in ("A", "B"):
        counts = [rank_counts[label].get(rank, 0) for rank in range(4)]
        summary = summarize_rank_counts_with_references(counts, rank_points=rank_points)
        summary["avg_score"] = None if seat_counts[label] == 0 else score_sums[label] / seat_counts[label]
        summary["seat_count"] = seat_counts[label]
        by_label[label] = summary
    return {
        "by_label": by_label,
        "totals": {
            "games": len(games),
            "env_step_count": sum(int(game["env_step_count"]) for game in games),
            "mjai_event_count": sum(int(game["mjai_event_count"]) for game in games),
            "fallback_count": sum(int(game["fallback_count"]) for game in games),
            "wall_time_sec": sum(float(game["wall_time_sec"]) for game in games),
        },
    }


def _rank_index(rank: int) -> int:
    if 1 <= rank <= 4:
        return rank - 1
    if 0 <= rank <= 3:
        return rank
    raise ValueError(f"unsupported rank value: {rank}")


def run(args: argparse.Namespace) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rank_points_profile, rank_points = resolve_rank_points(
        rank_points=getattr(args, "rank_points", None),
        profile=str(getattr(args, "rank_points_profile", "tenhou_reference")),
    )
    games = [_run_game(args, game_id=game_id) for game_id in range(int(args.games))]
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"games": games}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    document = build_metrics_document(
        run={
            "kind": "riichienv_ab_match",
            "backend": "riichienv",
            "model_a": str(args.model_a),
            "model_b": str(args.model_b),
            "games": int(args.games),
            "game_mode": str(args.game_mode),
            "seat_mode": str(args.seat_mode),
            "seed": int(args.seed),
            "device": str(args.device),
            "rank_points_profile": rank_points_profile,
            "rank_points_values": [float(value) for value in rank_points],
        },
        metrics=_summarize_games(games, rank_points=rank_points),
        artifacts={"manifest": str(manifest_path), "replay_dir": str(args.output_dir / "replays")},
        rank_points_profile=rank_points_profile,
        rank_points_values=rank_points,
    )
    write_metrics(args.output_dir / "metrics.json", document)
    print(json.dumps(document["metrics"], ensure_ascii=False, indent=2), flush=True)
    return document


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
