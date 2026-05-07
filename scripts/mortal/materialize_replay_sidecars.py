#!/usr/bin/env python3
"""Materialize Mortal decision sidecars for existing MJAI replays."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import sys
import time
from typing import Any, DefaultDict, Mapping

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from inference.mortal_bot import MortalReviewBot  # noqa: E402
from mahjong_env.replay import read_mjai_jsonl  # noqa: E402


def _parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path)
    config_args, remaining_argv = config_parser.parse_known_args()
    configured_argv: list[str] = []
    if config_args.config is not None:
        configured_argv = _config_mapping_to_argv(_load_json_config(config_args.config))
    parser = argparse.ArgumentParser(description="Write .decisions.json sidecars for MJAI replays using Mortal")
    parser.add_argument("--config", type=Path, default=config_args.config, help="JSON config file. CLI args override config values.")
    parser.add_argument("--replay-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--actors", type=int, nargs="+", default=(0, 1, 2, 3))
    parser.add_argument("--skip", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--recursive", action="store_true", help="Recursively scan --replay-dir for .mjson files.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(configured_argv + remaining_argv)


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
            if key in {"overwrite", "recursive"}:
                if value:
                    argv.append(flag)
                continue
            raise ValueError(f"boolean config key is not a known boolean CLI option: {key}")
        argv.append(flag)
        if isinstance(value, list | tuple):
            argv.extend(str(item) for item in value)
        else:
            argv.append(str(value))
    return argv


def _sidecar_entry(*, actor: int, event_index: int, event: Mapping[str, Any], reaction: Mapping[str, Any], meta: Mapping[str, Any]) -> dict[str, Any] | None:
    if meta.get("mask_bits") is None or meta.get("q_values") is None:
        return None
    return {
        "event_index": int(event_index),
        "actor": int(actor),
        "event_type": str(event.get("type", "")),
        "chosen_action": dict(reaction),
        "mortal_meta": {
            "mask_bits": int(meta.get("mask_bits", 0) or 0),
            "q_values": [float(value) for value in (meta.get("q_values") or [])],
            "is_greedy": meta.get("is_greedy"),
            "shanten": meta.get("shanten"),
            "at_furiten": meta.get("at_furiten"),
            "eval_time_ns": meta.get("eval_time_ns"),
            "batch_size": meta.get("batch_size"),
        },
    }


def materialize_one(path: Path, args: argparse.Namespace) -> dict[str, Any]:
    sidecar_path = path.with_suffix(".decisions.json")
    if sidecar_path.exists() and not bool(args.overwrite):
        return {"path": str(path), "skipped_existing": True, "decision_count": 0, "missing_reaction_count": 0}

    events = read_mjai_jsonl(path)
    sidecar: DefaultDict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_reaction_count = 0
    started = time.perf_counter()
    first_engine = None
    first_model = None
    for actor in [int(value) for value in args.actors]:
        bot = MortalReviewBot(
            player_id=actor,
            model_path=args.model,
            mortal_root=args.mortal_root,
            device=str(args.device),
            enable_review_log=False,
            shared_mortal_engine=first_engine,
            shared_model=first_model,
        )
        if first_engine is None:
            first_engine = bot._mortal_engine
            first_model = bot.model
        for event_index, event in enumerate(events):
            reaction = bot.react(dict(event))
            if reaction is None:
                continue
            raw_reaction = dict(reaction)
            meta = dict(raw_reaction.pop("meta", {}) or {})
            entry = _sidecar_entry(actor=actor, event_index=event_index, event=event, reaction=raw_reaction, meta=meta)
            if entry is None:
                missing_reaction_count += 1
                continue
            sidecar[str(actor)].append(entry)

    payload = {"by_actor": dict(sidecar)}
    sidecar_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "path": str(path),
        "skipped_existing": False,
        "decision_count": sum(len(rows) for rows in sidecar.values()),
        "missing_reaction_count": int(missing_reaction_count),
        "wall_time_sec": time.perf_counter() - started,
    }


def main() -> None:
    args = _parse_args()
    replay_paths = (
        sorted(Path(args.replay_dir).rglob("*.mjson"))
        if bool(args.recursive)
        else sorted(Path(args.replay_dir).glob("*.mjson"))
    )
    if int(args.skip) > 0:
        replay_paths = replay_paths[int(args.skip) :]
    if int(args.limit) > 0:
        replay_paths = replay_paths[: int(args.limit)]
    if not replay_paths:
        raise RuntimeError(f"no .mjson files found under {args.replay_dir}")
    summaries: list[dict[str, Any]] = []
    for index, path in enumerate(replay_paths, start=1):
        summary = materialize_one(path, args)
        summaries.append(summary)
        print(
            f"sidecar {index}/{len(replay_paths)} decisions={summary['decision_count']} "
            f"missing={summary['missing_reaction_count']} wall={float(summary.get('wall_time_sec', 0.0)):.2f}s {path.name}",
            flush=True,
        )
    total = {
        "replay_count": len(replay_paths),
        "decision_count": sum(int(row["decision_count"]) for row in summaries),
        "missing_reaction_count": sum(int(row["missing_reaction_count"]) for row in summaries),
        "summaries": summaries,
    }
    (Path(args.replay_dir) / "sidecar_materialize_summary.json").write_text(
        json.dumps(total, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
