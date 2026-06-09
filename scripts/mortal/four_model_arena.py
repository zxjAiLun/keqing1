#!/usr/bin/env python3
"""4-Model free-for-all arena: 70k, 80k(game), O3@80000, model_v4."""
from __future__ import annotations

import argparse
from collections import Counter
import json
import random
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "src"))

from inference.mortal_bot import MortalReviewBot
from scripts.mortal.generate_riichienv_selfplay_replays import _make_env, derive_riichienv_game_seed
from scripts.mortal.eval_metrics import (
    add_rank_point_args,
    build_metrics_document,
    resolve_rank_points,
    summarize_rank_counts_with_references,
    write_metrics,
)
from src.tools.mjai_jsonl_to_tenhou6 import convert_mjai_jsonl_to_tenhou6

DEFAULT_MODELS = {
    "70k": "artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth",
    "80k_game": "artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth",
    "O3_80k": "artifacts/experiments/online_phase2_2026_05/O3_70k_online_keep_optimizer_cql5/mortal.pth",
    "v4": "artifacts/model_v4_20240308_best_min.pth",
}

MORTAL_ROOT = Path("third_party/Mortal")
GAME_MODE = "4p-red-half"

BASIC_STAT_KEYS = (
    "seat_games",
    "rounds",
    "win_count",
    "deal_in_count",
    "call_count",
    "fuuro_round_count",
    "riichi_count",
    "ryukyoku_count",
    "tsumo_win_count",
    "ron_win_count",
)

def _random_assignment(seed: int):
    labels = list(MODEL_LABELS)
    random.Random(int(seed)).shuffle(labels)
    return labels


MODELS = dict(DEFAULT_MODELS)
MODEL_LABELS = list(MODELS.keys())


def _parse_model_specs(specs: Sequence[str] | None) -> dict[str, str]:
    if not specs:
        return dict(DEFAULT_MODELS)
    models: dict[str, str] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"--model must be LABEL=PATH, got: {spec}")
        label, path = spec.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"--model must be LABEL=PATH, got: {spec}")
        if label in models:
            raise ValueError(f"duplicate model label: {label}")
        models[label] = path
    if len(models) != 4:
        raise ValueError(f"four_model_arena requires exactly 4 models, got {len(models)}")
    return models


def _set_models(models: Mapping[str, str]) -> None:
    global MODELS, MODEL_LABELS
    MODELS = dict(models)
    MODEL_LABELS = list(MODELS.keys())


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--games", type=int, default=250, help="number of half-games to play")
    p.add_argument("--output-dir", type=Path, default=Path("artifacts/eval/four_model_arena"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed-start", type=int, default=330000)
    p.add_argument("--max-steps", type=int, default=10000)
    p.add_argument("--progress-interval", type=int, default=10)
    p.add_argument("--no-resume", action="store_true", help="ignore existing results.jsonl and start from game 0")
    p.add_argument(
        "--model",
        action="append",
        help="Model spec LABEL=PATH. Repeat exactly four times. Defaults to 70k/80k_game/O3_80k/v4.",
    )
    add_rank_point_args(p)
    return p.parse_args()


def _bot(label, seat, path, device):
    return MortalReviewBot(
        player_id=seat,
        model_path=path,
        mortal_root=MORTAL_ROOT,
        device=device,
        enable_review_log=False,
        model_version=f"mortal-{label}",
    )


def _shared_bot(label: str, seat: int, path: str | Path, device: str, shared: Mapping[str, Any]) -> MortalReviewBot:
    return MortalReviewBot(
        player_id=seat,
        model_path=path,
        mortal_root=MORTAL_ROOT,
        device=device,
        enable_review_log=False,
        model_version=f"mortal-{label}",
        shared_mortal_engine=shared["engine"],
        shared_model=shared["model"],
    )


def _preload_models(device: str) -> dict[str, dict[str, Any]]:
    shared: dict[str, dict[str, Any]] = {}
    for label in MODEL_LABELS:
        started = time.perf_counter()
        bot = _bot(label, 0, MODELS[label], device)
        shared[label] = {
            "engine": bot._mortal_engine,
            "model": bot.model,
            "load_time_sec": time.perf_counter() - started,
        }
        print(f"  loaded {label:<8} in {shared[label]['load_time_sec']:.1f}s", flush=True)
    return shared


def _load_existing_results(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    results: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSON in existing results file {path}:{line_no}") from exc
    return results


def _empty_basic_stats() -> dict[str, dict[str, int]]:
    return {label: {key: 0 for key in BASIC_STAT_KEYS} for label in MODELS}


def _round_count(events: Sequence[Mapping[str, Any]]) -> int:
    return sum(1 for event in events if event.get("type") == "start_kyoku")


def _summarize_events_by_seat(events: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, int]]:
    by_seat: dict[int, dict[str, int]] = {
        seat: {
            "win_count": 0,
            "deal_in_count": 0,
            "call_count": 0,
            "riichi_count": 0,
            "tsumo_win_count": 0,
            "ron_win_count": 0,
        }
        for seat in range(4)
    }
    for event in events:
        event_type = str(event.get("type", ""))
        actor = event.get("actor")
        target = event.get("target")
        if event_type == "hora" and isinstance(actor, int) and 0 <= actor < 4:
            by_seat[actor]["win_count"] += 1
            if isinstance(target, int) and 0 <= target < 4 and target != actor:
                by_seat[actor]["ron_win_count"] += 1
                by_seat[target]["deal_in_count"] += 1
            else:
                by_seat[actor]["tsumo_win_count"] += 1
        elif event_type in {"chi", "pon", "daiminkan"} and isinstance(actor, int) and 0 <= actor < 4:
            by_seat[actor]["call_count"] += 1
        elif event_type == "reach" and isinstance(actor, int) and 0 <= actor < 4:
            by_seat[actor]["riichi_count"] += 1
    return by_seat


def _fuuro_rounds_by_seat(events: Sequence[Mapping[str, Any]]) -> dict[int, int]:
    counts = {seat: 0 for seat in range(4)}
    called = [False, False, False, False]
    in_round = False

    def finish_round() -> None:
        for seat, value in enumerate(called):
            counts[seat] += int(value)

    for event in events:
        event_type = str(event.get("type", ""))
        if event_type == "start_kyoku":
            if in_round:
                finish_round()
            called = [False, False, False, False]
            in_round = True
        elif event_type in {"chi", "pon", "daiminkan"} and in_round:
            actor = event.get("actor")
            if isinstance(actor, int) and 0 <= actor < 4:
                called[actor] = True
        elif event_type == "end_kyoku" and in_round:
            finish_round()
            called = [False, False, False, False]
            in_round = False

    if in_round:
        finish_round()
    return counts


def _basic_stats_for_game(
    *,
    assignment: Sequence[str],
    events: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, int]]:
    rounds = _round_count(events)
    ryukyoku_count = sum(1 for event in events if event.get("type") == "ryukyoku")
    event_counts = _summarize_events_by_seat(events)
    fuuro_round_counts = _fuuro_rounds_by_seat(events)
    by_label = _empty_basic_stats()
    for seat, label in enumerate(assignment):
        bucket = by_label[str(label)]
        bucket["seat_games"] += 1
        bucket["rounds"] += rounds
        bucket["ryukyoku_count"] += ryukyoku_count
        bucket["fuuro_round_count"] += fuuro_round_counts[seat]
        for key, value in event_counts[seat].items():
            bucket[key] += int(value)
    return by_label


def _load_mjson(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _merge_basic_stats(target: dict[str, dict[str, int]], source: Mapping[str, Mapping[str, Any]]) -> None:
    for label, source_bucket in source.items():
        if label not in target:
            continue
        for key in BASIC_STAT_KEYS:
            target[label][key] += int(source_bucket.get(key, 0) or 0)


def _finalize_basic_stats(bucket: Mapping[str, Any]) -> dict[str, Any]:
    rounds = int(bucket.get("rounds", 0) or 0)
    seat_games = int(bucket.get("seat_games", 0) or 0)
    return {
        "seat_games": seat_games,
        "rounds": rounds,
        "win_count": int(bucket.get("win_count", 0) or 0),
        "deal_in_count": int(bucket.get("deal_in_count", 0) or 0),
        "call_count": int(bucket.get("call_count", 0) or 0),
        "fuuro_round_count": int(bucket.get("fuuro_round_count", 0) or 0),
        "riichi_count": int(bucket.get("riichi_count", 0) or 0),
        "ryukyoku_count": int(bucket.get("ryukyoku_count", 0) or 0),
        "tsumo_win_count": int(bucket.get("tsumo_win_count", 0) or 0),
        "ron_win_count": int(bucket.get("ron_win_count", 0) or 0),
        "win_rate": (float(bucket.get("win_count", 0) or 0) / rounds) if rounds else None,
        "deal_in_rate": (float(bucket.get("deal_in_count", 0) or 0) / rounds) if rounds else None,
        "call_events_per_round": (float(bucket.get("call_count", 0) or 0) / rounds) if rounds else None,
        "fuuro_rate": (float(bucket.get("fuuro_round_count", 0) or 0) / rounds) if rounds else None,
        "riichi_rate": (float(bucket.get("riichi_count", 0) or 0) / rounds) if rounds else None,
        "ryukyoku_rate": (float(bucket.get("ryukyoku_count", 0) or 0) / rounds) if rounds else None,
        "tsumo_win_rate": (float(bucket.get("tsumo_win_count", 0) or 0) / rounds) if rounds else None,
        "ron_win_rate": (float(bucket.get("ron_win_count", 0) or 0) / rounds) if rounds else None,
    }


def _rebuild_aggregates(
    results: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Counter], dict[str, float], dict[str, dict[str, int]]]:
    rank_counts: dict[str, Counter] = {label: Counter() for label in MODELS}
    score_sums: dict[str, float] = {label: 0.0 for label in MODELS}
    basic_stats = _empty_basic_stats()
    for result in results:
        for seat, label in enumerate(result["assignment"]):
            rank_counts[str(label)][int(result["ranks"][seat])] += 1
            score_sums[str(label)] += float(result["scores"][seat])
        result_stats = dict(result.get("basic_stats") or {})
        if not any("fuuro_round_count" in dict(bucket) for bucket in result_stats.values()) and result.get("replay"):
            result_stats = _basic_stats_for_game(
                assignment=[str(label) for label in result["assignment"]],
                events=_load_mjson(result["replay"]),
            )
        _merge_basic_stats(basic_stats, result_stats)
    return rank_counts, score_sums, basic_stats


def _timing_totals(results: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    totals: dict[str, float] = Counter()
    for result in results:
        for key, value in dict(result.get("timing_sec") or {}).items():
            totals[key] = float(totals.get(key, 0.0)) + float(value)
    return dict(totals)


def _events_with_player_names(
    events: Sequence[Mapping[str, Any]],
    names: Sequence[str],
) -> list[dict[str, Any]]:
    patched = [dict(event) for event in events]
    if len(names) < 4:
        raise ValueError("tenhou6 player names require four seat labels")
    start_game = {"type": "start_game", "names": [str(name) for name in names[:4]]}
    if patched and patched[0].get("type") == "start_game":
        patched[0] = {**patched[0], "names": start_game["names"]}
    else:
        patched.insert(0, start_game)
    return patched


def _write_mjson(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_tenhou6(path: Path, events: Sequence[Mapping[str, Any]], *, names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tenhou6 = convert_mjai_jsonl_to_tenhou6(_events_with_player_names(events, names))
    path.write_text(json.dumps(tenhou6, ensure_ascii=False, separators=(",", ":")) + "\n", encoding="utf-8")


def _write_outputs(
    *,
    output_dir: Path,
    all_results: Sequence[Mapping[str, Any]],
    total_games: int,
    device: str,
    rank_pts_profile: str,
    rank_pts: Sequence[int | float],
    rank_counts: Mapping[str, Counter],
    score_sums: Mapping[str, float],
    basic_stats: Mapping[str, Mapping[str, Any]],
    model_load_times: Mapping[str, float],
) -> None:
    metrics: dict[str, Any] = {}
    for label in sorted(MODELS):
        counts = [rank_counts[label].get(r, 0) for r in range(1, 5)]
        summary = summarize_rank_counts_with_references(counts, rank_points=rank_pts)
        total = sum(counts)
        summary["avg_score"] = score_sums[label] / total if total else 0.0
        summary["basic_stats"] = _finalize_basic_stats(basic_stats.get(label, {}))
        metrics[label] = summary
    metrics["timing_sec"] = _timing_totals(all_results)

    document = build_metrics_document(
        run={
            "kind": "four_model_arena",
            "backend": "RiichiEnv",
            "games": total_games,
            "completed_games": len(all_results),
            "models": MODELS,
            "device": device,
            "rank_points_profile": rank_pts_profile,
            "rank_points_values": [float(v) for v in rank_pts],
            "model_load_time_sec": {key: float(value) for key, value in model_load_times.items()},
        },
        metrics=metrics,
        artifacts={
            "results": [str(output_dir / "results.jsonl")],
            "mjson_replays": [str(output_dir / "replays")],
            "tenhou6_replays": [str(output_dir / "tenhou6")],
        },
        rank_points_profile=rank_pts_profile,
        rank_points_values=rank_pts,
    )
    write_metrics(output_dir / "metrics.json", document)


def _print_timing_summary(results: Sequence[Mapping[str, Any]]) -> None:
    totals = _timing_totals(results)
    wall = sum(float(result.get("wall_time_sec", 0.0)) for result in results)
    if not totals or wall <= 0.0:
        return
    print("\n=== TIMING ===", flush=True)
    for key, value in sorted(totals.items(), key=lambda item: item[1], reverse=True):
        print(f"  {key:<18}: {value:8.1f}s  {value / wall * 100:5.1f}%", flush=True)


def run(args):
    _set_models(_parse_model_specs(getattr(args, "model", None)))
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "replays").mkdir(exist_ok=True)
    results_path = output_dir / "results.jsonl"

    rank_pts_profile, rank_pts = resolve_rank_points(
        rank_points=getattr(args, "rank_points", None),
        profile=str(getattr(args, "rank_points_profile", "tenhou_reference")),
    )

    device = str(args.device)
    total_games = int(args.games)

    existing_results = [] if args.no_resume else _load_existing_results(results_path)
    completed_ids = {int(result["game_id"]) for result in existing_results}
    rank_counts, score_sums, basic_stats = _rebuild_aggregates(existing_results)
    all_results: list[dict] = list(existing_results)

    if existing_results:
        print(f"resuming from {results_path}: {len(existing_results)} completed games", flush=True)

    print("loading models once...", flush=True)
    shared_models = _preload_models(device)
    model_load_times = {label: float(shared_models[label]["load_time_sec"]) for label in MODEL_LABELS}

    results_mode = "a" if existing_results and not args.no_resume else "w"
    results_handle = results_path.open(results_mode, encoding="utf-8", buffering=1)
    try:
        for game_id in range(total_games):
            if game_id in completed_ids:
                continue
            seed = derive_riichienv_game_seed(int(args.seed_start), game_id)
            assignment = _random_assignment(seed)

            timing: dict[str, float] = Counter()

            started_part = time.perf_counter()
            env, _seed_info = _make_env(game_mode=GAME_MODE, seed=seed)
            obs_dict = env.reset(seed=seed)
            timing["env_reset"] += time.perf_counter() - started_part

            started_part = time.perf_counter()
            bots = {}
            for seat, label in enumerate(assignment):
                bots[seat] = _shared_bot(label, seat, MODELS[label], device, shared_models[label])
            timing["bot_setup"] += time.perf_counter() - started_part

            fallback_count = 0
            step_count = 0
            event_count = 0
            reaction_count = 0
            started = time.perf_counter()

            while not env.done():
                actions = {}
                for seat_raw, obs in obs_dict.items():
                    seat = int(seat_raw)
                    mjai_action = None
                    started_part = time.perf_counter()
                    new_events = list(obs.new_events())
                    timing["new_events"] += time.perf_counter() - started_part
                    for raw_event in new_events:
                        event_count += 1
                        started_part = time.perf_counter()
                        event = json.loads(raw_event) if isinstance(raw_event, str) else dict(raw_event)
                        timing["json_parse"] += time.perf_counter() - started_part
                        started_part = time.perf_counter()
                        reaction = bots[seat].react(event)
                        timing["bot_react"] += time.perf_counter() - started_part
                        if reaction is not None:
                            reaction_count += 1
                            mjai_action = {k: v for k, v in dict(reaction).items() if k != "meta"}
                    started_part = time.perf_counter()
                    legal_actions = obs.legal_actions()
                    timing["legal_actions"] += time.perf_counter() - started_part
                    if not legal_actions:
                        continue
                    if mjai_action is None:
                        actions[seat] = legal_actions[0]
                        fallback_count += 1
                        continue
                    started_part = time.perf_counter()
                    selected = obs.select_action_from_mjai(
                        json.dumps(mjai_action, ensure_ascii=False, separators=(",", ":"))
                    )
                    timing["select_action"] += time.perf_counter() - started_part
                    if selected is None:
                        actions[seat] = legal_actions[0]
                        fallback_count += 1
                    else:
                        actions[seat] = selected
                if not actions:
                    raise RuntimeError("stalled")
                started_part = time.perf_counter()
                obs_dict = env.step(actions)
                timing["env_step"] += time.perf_counter() - started_part
                step_count += 1
                if step_count > args.max_steps:
                    raise RuntimeError(f"max_steps {args.max_steps}")

            scores = [float(v) for v in env.scores()]
            ranks = [int(v) for v in env.ranks()]
            replay_events = [dict(event) for event in getattr(env, "mjai_log", [])]
            replay_path = output_dir / "replays" / f"game_{game_id:05d}.mjson"
            tenhou6_path = output_dir / "tenhou6" / f"game_{game_id:05d}.tenhou6.json"
            _write_mjson(replay_path, _events_with_player_names(replay_events, assignment))
            _write_tenhou6(tenhou6_path, replay_events, names=assignment)
            game_basic_stats = _basic_stats_for_game(assignment=assignment, events=replay_events)
            result = {
                "game_id": game_id,
                "seed": seed,
                "assignment": assignment,
                "scores": scores,
                "ranks": ranks,
                "fallback_count": fallback_count,
                "env_step_count": step_count,
                "mjai_event_count": event_count,
                "reaction_count": reaction_count,
                "wall_time_sec": time.perf_counter() - started,
                "timing_sec": {key: float(value) for key, value in timing.items()},
                "round_count": _round_count(replay_events),
                "basic_stats": game_basic_stats,
                "replay": str(replay_path),
                "tenhou6": str(tenhou6_path),
            }
            all_results.append(result)
            results_handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
            results_handle.flush()

            for seat, label in enumerate(assignment):
                r = int(ranks[seat])
                rank_counts[label][r] += 1
                score_sums[label] += float(scores[seat])
            _merge_basic_stats(basic_stats, game_basic_stats)

            _write_outputs(
                output_dir=output_dir,
                all_results=all_results,
                total_games=total_games,
                device=device,
                rank_pts_profile=rank_pts_profile,
                rank_pts=rank_pts,
                rank_counts=rank_counts,
                score_sums=score_sums,
                basic_stats=basic_stats,
                model_load_times=model_load_times,
            )

            progress_interval = max(1, int(args.progress_interval))
            if (len(all_results) % progress_interval == 0) or (game_id + 1 == total_games):
                elapsed = sum(r["wall_time_sec"] for r in all_results[-progress_interval:])
                print(
                    f"  completed {len(all_results)}/{total_games}, "
                    f"last {min(progress_interval, len(all_results))} avg "
                    f"{elapsed / min(progress_interval, len(all_results)):.1f}s/game",
                    flush=True,
                )
    finally:
        results_handle.close()

    print(f"\n=== RESULTS ({total_games} games) ===", flush=True)
    for label in sorted(MODELS):
        counts = [rank_counts[label].get(r, 0) for r in range(1, 5)]
        total = sum(counts)
        avg_rank = sum(r * c for r, c in enumerate(counts, 1)) / total
        avg_pt = sum(p * c for p, c in zip(rank_pts, counts)) / total
        avg_score = score_sums[label] / total if total else 0
        stats = _finalize_basic_stats(basic_stats.get(label, {}))
        print(
            f"  {label:<12}: ranks {counts}  avg_rank={avg_rank:.4f}  tenhou={avg_pt:+.4f}  "
            f"avg_score={avg_score:+.1f}  win={stats['win_rate'] or 0:.3f}  "
            f"deal_in={stats['deal_in_rate'] or 0:.3f}  fuuro={stats['fuuro_rate'] or 0:.3f}  "
            f"call_events={stats['call_events_per_round'] or 0:.3f}  "
            f"riichi={stats['riichi_rate'] or 0:.3f}",
            flush=True,
        )

    _print_timing_summary(all_results)
    _write_outputs(
        output_dir=output_dir,
        all_results=all_results,
        total_games=total_games,
        device=device,
        rank_pts_profile=rank_pts_profile,
        rank_pts=rank_pts,
        rank_counts=rank_counts,
        score_sums=score_sums,
        basic_stats=basic_stats,
        model_load_times=model_load_times,
    )

    print(f"\nsaved: {output_dir}/metrics.json", flush=True)
    print(f"saved: {output_dir}/results.jsonl", flush=True)
    print(f"saved: {output_dir}/replays/*.mjson", flush=True)
    print(f"saved: {output_dir}/tenhou6/*.tenhou6.json", flush=True)


if __name__ == "__main__":
    run(_parse_args())
