#!/usr/bin/env python3
"""Generate Mortal selfplay replays with RiichiEnv."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import csv
import gzip
from collections import Counter, defaultdict
from dataclasses import dataclass
import inspect
import json
import math
from pathlib import Path
import random
import sys
import time
from typing import Any, DefaultDict, Mapping, Sequence

import torch
from riichienv import RiichiEnv

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from inference.mortal_bot import MortalReviewBot
from inference.mortal_bot import _mortal_action_ids_for_mjai
from scripts.mortal import eval_metrics

MORTAL_ACTION_SPACE = 46
STYLE_BIAS_VERSION = "p1_handcrafted_v1"


@dataclass(frozen=True)
class StyleProfile:
    style_id: str
    style_weights: tuple[float, float, float, float]

    def to_json(self, *, style_alpha: float) -> dict[str, Any]:
        return {
            "style_id": self.style_id,
            "style_weights": [float(value) for value in self.style_weights],
            "style_alpha": float(style_alpha),
            "bias_version": STYLE_BIAS_VERSION,
        }


@dataclass(frozen=True)
class StyleDecision:
    action: dict[str, Any]
    meta: dict[str, Any]


DEFAULT_STYLE_PROFILES: dict[str, StyleProfile] = {
    "base": StyleProfile("base", (0.0, 0.0, 0.0, 0.0)),
    "atk_fuuro": StyleProfile("atk_fuuro", (0.8, 0.0, 0.7, 0.0)),
    "def_menzen": StyleProfile("def_menzen", (0.0, 0.8, 0.0, 0.7)),
}


@dataclass(frozen=True)
class EnvSeedInfo:
    requested_seed: int | None
    applied: bool
    mode: str


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 4-Mortal RiichiEnv selfplay replays")
    parser.add_argument("--checkpoint", "--model", dest="model", type=Path, required=True)
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--games", type=int, default=1)
    parser.add_argument("--game-mode", default="4p-red-half")
    parser.add_argument("--out-dir", "--output-dir", dest="output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed-start", "--seed", dest="seed_start", type=int, default=None)
    parser.add_argument("--seed-key", default="0xd5dfaa4cef265cd7")
    parser.add_argument("--profiles", default="base,atk_fuuro,def_menzen")
    parser.add_argument("--seat-profiles", default=None)
    parser.add_argument("--profile-mode", choices=("rotate", "random"), default="rotate")
    parser.add_argument("--style-alpha", type=float, default=0.25)
    parser.add_argument("--rank-points", default="90,45,0,-135")
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--trace-mode", choices=("compact", "full"), default="compact")
    parser.add_argument(
        "--artifact-mode",
        choices=("full", "audit-gz-only"),
        default="full",
        help="full writes mjson/meta/decision sidecars; audit-gz-only writes only GRP-audit gzip logs and manifests.",
    )
    return parser.parse_args()


def _json_dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def validate_style_profile(profile: StyleProfile) -> StyleProfile:
    if not profile.style_id:
        raise ValueError("style_id must be non-empty")
    if len(profile.style_weights) != 4:
        raise ValueError(f"style_weights must have length 4, got {len(profile.style_weights)}")
    for value in profile.style_weights:
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise ValueError(f"style weight must be a finite number, got {value!r}")
    return profile


def parse_style_profiles(value: str) -> list[StyleProfile]:
    profiles: list[StyleProfile] = []
    for raw_id in value.split(","):
        style_id = raw_id.strip()
        if not style_id:
            continue
        try:
            profile = DEFAULT_STYLE_PROFILES[style_id]
        except KeyError as exc:
            known = ", ".join(sorted(DEFAULT_STYLE_PROFILES))
            raise ValueError(f"unknown style profile {style_id!r}; known profiles: {known}") from exc
        profiles.append(validate_style_profile(profile))
    if not profiles:
        raise ValueError("at least one style profile is required")
    return profiles


def parse_seed_key(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    text = str(value).strip()
    return int(text, 16 if text.lower().startswith("0x") else 10)


def parse_rank_points(value: str | Sequence[int | float]) -> tuple[float, float, float, float]:
    return eval_metrics.parse_rank_points(value)


def assign_style_profiles(
    *,
    game_id: int,
    profiles: Sequence[StyleProfile],
    seed_key: int,
    mode: str,
) -> dict[int, StyleProfile]:
    if not profiles:
        raise ValueError("profiles must not be empty")
    if mode == "rotate":
        return {seat: profiles[(int(game_id) + seat) % len(profiles)] for seat in range(4)}
    if mode == "random":
        rng = random.Random((int(seed_key) & ((1 << 63) - 1)) ^ int(game_id))
        return {seat: rng.choice(list(profiles)) for seat in range(4)}
    raise ValueError(f"unsupported profile mode: {mode}")


def assign_seat_profiles(value: str | None) -> dict[int, StyleProfile] | None:
    if value is None:
        return None
    profiles = parse_style_profiles(value)
    if len(profiles) != 4:
        raise ValueError(f"--seat-profiles must contain exactly 4 profiles, got {len(profiles)}")
    return {seat: profiles[seat] for seat in range(4)}


def seat_profile_schedule(seat_profiles: Mapping[int, StyleProfile], *, style_alpha: float) -> list[dict[str, Any]]:
    return [
        {
            "seat": int(seat),
            **seat_profiles[int(seat)].to_json(style_alpha=float(style_alpha)),
        }
        for seat in sorted(seat_profiles)
    ]


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
    if meta.get("style_policy") is not None:
        mortal_meta["style_policy"] = dict(meta["style_policy"])
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


def _make_env(*, game_mode: str, seed: int | None) -> tuple[Any, EnvSeedInfo]:
    if seed is None:
        return RiichiEnv(game_mode=game_mode), EnvSeedInfo(requested_seed=None, applied=False, mode="none")
    try:
        return (
            RiichiEnv(game_mode=game_mode, seed=int(seed)),
            EnvSeedInfo(requested_seed=int(seed), applied=True, mode="constructor"),
        )
    except TypeError:
        env = RiichiEnv(game_mode=game_mode)
        reset = getattr(env, "reset", None)
        if callable(reset):
            try:
                reset_params = inspect.signature(reset).parameters
                if "seed" in reset_params:
                    return env, EnvSeedInfo(requested_seed=int(seed), applied=True, mode="reset")
            except (TypeError, ValueError):
                pass
        return env, EnvSeedInfo(requested_seed=int(seed), applied=False, mode="fallback_unseeded")


def _write_mjson(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_mjson_gz(path: Path, events: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def _write_manifest_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


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


class MortalStylePolicy:
    """Small handcrafted P1 style bias applied only over RiichiEnv legal actions."""

    def __init__(self, *, profile: StyleProfile, style_alpha: float) -> None:
        self.profile = validate_style_profile(profile)
        self.style_alpha = float(style_alpha)

    @property
    def enabled(self) -> bool:
        return self.style_alpha != 0.0 and any(float(value) != 0.0 for value in self.profile.style_weights)

    def select_action(
        self,
        *,
        base_action: Mapping[str, Any],
        mortal_meta: Mapping[str, Any],
        legal_actions: Sequence[Any],
    ) -> StyleDecision:
        style_meta: dict[str, Any] = {
            **self.profile.to_json(style_alpha=self.style_alpha),
            "applied": False,
            "base_action": dict(base_action),
            "selected_action": dict(base_action),
        }
        if not self.enabled:
            return StyleDecision(action=dict(base_action), meta=style_meta)

        compact_q = mortal_meta.get("q_values")
        mask_bits = mortal_meta.get("mask_bits")
        if compact_q is None or mask_bits is None:
            style_meta["reason"] = "missing_mortal_q"
            return StyleDecision(action=dict(base_action), meta=style_meta)

        q_values, action_mask = _expand_compact_meta(mask_bits=int(mask_bits), compact_q=compact_q)
        base_action_ids = tuple(
            action_id for action_id in _mortal_action_ids_for_mjai(dict(base_action)) if action_mask[action_id]
        )
        base_action_score = _score_action_ids(base_action_ids, q_values=q_values)
        candidates: list[dict[str, Any]] = []
        for legal_action in legal_actions:
            action = _legal_action_to_mjai(legal_action)
            action_ids = tuple(action_id for action_id in _mortal_action_ids_for_mjai(action) if action_mask[action_id])
            if not action_ids:
                continue
            base_score = max(float(q_values[action_id]) for action_id in action_ids)
            if not math.isfinite(base_score):
                continue
            bias = self._bias_for_action(action)
            final_score = base_score + self.style_alpha * bias
            candidates.append(
                {
                    "action": action,
                    "mortal_action_ids": list(action_ids),
                    "base_score": base_score,
                    "style_bias": bias,
                    "final_score": final_score,
                }
            )

        if not candidates:
            style_meta["reason"] = "no_scorable_legal_action"
            return StyleDecision(action=dict(base_action), meta=style_meta)

        candidates.sort(key=lambda item: float(item["final_score"]), reverse=True)
        selected = dict(candidates[0]["action"])
        selected_action_ids = tuple(int(action_id) for action_id in candidates[0]["mortal_action_ids"])
        semantic_changed = actions_semantically_changed(
            base_action=dict(base_action),
            selected_action=selected,
            base_action_ids=base_action_ids,
            selected_action_ids=selected_action_ids,
        )
        q_gap = (
            None
            if base_action_score is None
            else float(base_action_score) - float(candidates[0]["base_score"])
        )
        style_meta.update(
            {
                "applied": True,
                "selected_action": selected,
                "base_action_ids": list(base_action_ids),
                "selected_action_ids": list(selected_action_ids),
                "base_action_type": str(base_action.get("type", "")),
                "selected_action_type": str(selected.get("type", "")),
                "base_action_score": base_action_score,
                "selected_base_score": candidates[0]["base_score"],
                "selected_style_bias": candidates[0]["style_bias"],
                "selected_final_score": candidates[0]["final_score"],
                "changed": semantic_changed,
                "semantic_changed": semantic_changed,
                "base_to_selected_q_gap": q_gap,
                "candidate_count": len(candidates),
            }
        )
        return StyleDecision(action=selected, meta=style_meta)

    def _bias_for_action(self, action: Mapping[str, Any]) -> float:
        atk, defense, fuuro, menzen = (float(value) for value in self.profile.style_weights)
        action_type = str(action.get("type", ""))
        if action_type == "hora":
            return 4.0 * atk + 1.0 * defense
        if action_type == "reach":
            return 0.55 * atk + 0.45 * menzen - 0.10 * defense
        if action_type in {"chi", "pon", "daiminkan"}:
            return 0.35 * atk + 0.80 * fuuro - 0.75 * menzen - 0.10 * defense
        if action_type in {"ankan", "kakan"}:
            return 0.10 * atk + 0.15 * fuuro
        if action_type in {"none", "pass"}:
            return 0.55 * defense + 0.20 * menzen - 0.20 * atk - 0.25 * fuuro
        if action_type == "ryukyoku":
            return 0.35 * defense + 0.10 * menzen
        return 0.0


def _score_action_ids(action_ids: Sequence[int], *, q_values: Sequence[float]) -> float | None:
    scores = [float(q_values[action_id]) for action_id in action_ids]
    scores = [score for score in scores if math.isfinite(score)]
    if not scores:
        return None
    return max(scores)


def actions_semantically_changed(
    *,
    base_action: Mapping[str, Any],
    selected_action: Mapping[str, Any],
    base_action_ids: Sequence[int] = (),
    selected_action_ids: Sequence[int] = (),
) -> bool:
    if base_action_ids and selected_action_ids:
        return set(int(action_id) for action_id in base_action_ids).isdisjoint(
            int(action_id) for action_id in selected_action_ids
        )
    return canonical_action_signature(base_action) != canonical_action_signature(selected_action)


def canonical_action_signature(action: Mapping[str, Any]) -> tuple[Any, ...]:
    consumed = action.get("consumed", [])
    if isinstance(consumed, Sequence) and not isinstance(consumed, str):
        consumed_sig = tuple(sorted(str(item) for item in consumed))
    else:
        consumed_sig = ()
    return (
        action.get("type"),
        action.get("actor"),
        action.get("target"),
        action.get("pai"),
        consumed_sig,
        action.get("tsumogiri"),
    )


def _legal_action_to_mjai(action: Any) -> dict[str, Any]:
    if hasattr(action, "to_mjai"):
        payload = action.to_mjai()
        if isinstance(payload, str):
            return json.loads(payload)
        return dict(payload)
    return dict(action)


def _empty_style_bucket(style_id: str, style_weights: Sequence[float]) -> dict[str, Any]:
    return {
        "style_id": style_id,
        "style_weights": [float(value) for value in style_weights],
        "seat_count": 0,
        "seat_games": 0,
        "unique_hanchans": 0,
        "rounds": 0,
        "rank_counts": [0, 0, 0, 0],
        "score_sum": 0.0,
        "win_count": 0,
        "deal_in_count": 0,
        "call_count": 0,
        "riichi_count": 0,
        "ryukyoku_count": 0,
        "decision_count": 0,
        "style_applied_count": 0,
        "style_changed_count": 0,
        "semantic_changed_count": 0,
        "q_gap_count": 0,
        "q_gap_sum": 0.0,
        "semantic_changed_transitions": {},
    }


def _round_count(events: Sequence[Mapping[str, Any]]) -> int:
    count = sum(1 for event in events if event.get("type") == "start_kyoku")
    return max(count, 1)


def _summarize_events_by_seat(events: Sequence[Mapping[str, Any]]) -> dict[int, dict[str, int]]:
    by_seat: dict[int, dict[str, int]] = {
        seat: {
            "win_count": 0,
            "deal_in_count": 0,
            "call_count": 0,
            "riichi_count": 0,
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
                by_seat[target]["deal_in_count"] += 1
        elif event_type in {"chi", "pon", "daiminkan"} and isinstance(actor, int) and 0 <= actor < 4:
            by_seat[actor]["call_count"] += 1
        elif event_type == "reach" and isinstance(actor, int) and 0 <= actor < 4:
            by_seat[actor]["riichi_count"] += 1
    return by_seat


def _summarize_sidecar_by_seat(sidecar: Mapping[str, Any]) -> dict[int, dict[str, int]]:
    by_seat: dict[int, dict[str, int]] = {
        seat: {
            "decision_count": 0,
            "style_applied_count": 0,
            "style_changed_count": 0,
            "semantic_changed_count": 0,
            "q_gap_count": 0,
            "q_gap_sum": 0.0,
            "semantic_changed_transitions": {},
        }
        for seat in range(4)
    }
    for seat_text, rows in dict(sidecar.get("by_actor", {}) or {}).items():
        seat = int(seat_text)
        for row in rows or []:
            by_seat[seat]["decision_count"] += 1
            style_policy = ((row.get("mortal_meta") or {}).get("style_policy") or {})
            if style_policy.get("applied"):
                by_seat[seat]["style_applied_count"] += 1
            if style_policy.get("changed"):
                by_seat[seat]["style_changed_count"] += 1
            if style_policy.get("semantic_changed"):
                by_seat[seat]["semantic_changed_count"] += 1
                transition_key = (
                    f"{style_policy.get('base_action_type', '')}"
                    f"->{style_policy.get('selected_action_type', '')}"
                )
                transitions = by_seat[seat]["semantic_changed_transitions"]
                transition = transitions.setdefault(
                    transition_key,
                    {
                        "base_action_type": str(style_policy.get("base_action_type", "")),
                        "selected_action_type": str(style_policy.get("selected_action_type", "")),
                        "count": 0,
                        "q_gaps": [],
                    },
                )
                transition["count"] += 1
                if style_policy.get("base_to_selected_q_gap") is not None:
                    transition["q_gaps"].append(float(style_policy["base_to_selected_q_gap"]))
            q_gap = style_policy.get("base_to_selected_q_gap")
            if q_gap is not None:
                by_seat[seat]["q_gap_count"] += 1
                by_seat[seat]["q_gap_sum"] += float(q_gap)
    return by_seat


def build_style_metrics(
    games: Sequence[Mapping[str, Any]],
    *,
    rank_points: Sequence[int | float] = eval_metrics.TENHOU_RANK_POINTS,
) -> dict[str, Any]:
    buckets: dict[str, dict[str, Any]] = {}
    for game in games:
        summary = dict(game["summary"])
        events = list(game["events"])
        sidecar = dict(game.get("sidecar") or {})
        rounds = _round_count(events)
        ryukyoku_count = sum(1 for event in events if event.get("type") == "ryukyoku")
        event_counts = _summarize_events_by_seat(events)
        decision_counts = _summarize_sidecar_by_seat(sidecar)
        ranks = [int(rank) for rank in summary.get("ranks", [])]
        scores = [float(score) for score in summary.get("scores", [])]
        styles_seen_in_game: set[str] = set()

        for player in summary.get("players", []):
            seat = int(player["seat"])
            style_id = str(player["style_id"])
            styles_seen_in_game.add(style_id)
            bucket = buckets.setdefault(
                style_id,
                _empty_style_bucket(style_id, player.get("style_weights", [])),
            )
            bucket["seat_count"] += 1
            bucket["seat_games"] += 1
            bucket["rounds"] += rounds
            if seat < len(ranks) and 1 <= ranks[seat] <= 4:
                bucket["rank_counts"][ranks[seat] - 1] += 1
            if seat < len(scores):
                bucket["score_sum"] += scores[seat]
            for key, value in event_counts[seat].items():
                bucket[key] += int(value)
            bucket["ryukyoku_count"] += ryukyoku_count
            for key, value in decision_counts[seat].items():
                if key == "q_gap_sum":
                    bucket[key] += float(value)
                elif key == "semantic_changed_transitions":
                    merge_transition_buckets(bucket[key], value)
                else:
                    bucket[key] += int(value)
        for style_id in styles_seen_in_game:
            buckets[style_id]["unique_hanchans"] += 1

    by_style: dict[str, Any] = {}
    for style_id, bucket in sorted(buckets.items()):
        rank_summary = eval_metrics.summarize_rank_counts(bucket["rank_counts"], rank_points=rank_points)
        rounds = int(bucket["rounds"])
        seat_count = int(bucket["seat_count"])
        decision_count = int(bucket["decision_count"])
        q_gap_count = int(bucket["q_gap_count"])
        by_style[style_id] = {
            **bucket,
            "games": int(bucket["unique_hanchans"]),
            "rank_sample_count": rank_summary["games"],
            "rank_counts": rank_summary["rank_counts"],
            "avg_rank": rank_summary["avg_rank"],
            "avg_rank_pt": rank_summary["avg_rank_pt"],
            "avg_score": (float(bucket["score_sum"]) / seat_count) if seat_count else None,
            "win_rate": (bucket["win_count"] / rounds) if rounds else None,
            "deal_in_rate": (bucket["deal_in_count"] / rounds) if rounds else None,
            "call_rate": (bucket["call_count"] / rounds) if rounds else None,
            "riichi_rate": (bucket["riichi_count"] / rounds) if rounds else None,
            "ryukyoku_rate": (bucket["ryukyoku_count"] / rounds) if rounds else None,
            "style_applied_rate": (bucket["style_applied_count"] / decision_count) if decision_count else None,
            "style_changed_rate": (bucket["style_changed_count"] / decision_count) if decision_count else None,
            "semantic_changed_rate": (bucket["semantic_changed_count"] / decision_count) if decision_count else None,
            "mean_base_to_selected_q_gap": (bucket["q_gap_sum"] / q_gap_count) if q_gap_count else None,
            "semantic_changed_transition_breakdown": finalize_transition_breakdown(
                bucket["semantic_changed_transitions"],
                denominator=int(bucket["semantic_changed_count"]),
            ),
        }
    return {"by_style": by_style, "delta_vs_base": build_delta_vs_base(by_style)}


def merge_transition_buckets(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for transition_key, source_row in source.items():
        target_row = target.setdefault(
            transition_key,
            {
                "base_action_type": str(source_row.get("base_action_type", "")),
                "selected_action_type": str(source_row.get("selected_action_type", "")),
                "count": 0,
                "q_gaps": [],
            },
        )
        target_row["count"] += int(source_row.get("count", 0))
        target_row["q_gaps"].extend(float(value) for value in source_row.get("q_gaps", []))


def finalize_transition_breakdown(transitions: Mapping[str, Any], *, denominator: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for transition_key, row in transitions.items():
        q_gaps = sorted(float(value) for value in row.get("q_gaps", []))
        count = int(row.get("count", 0))
        rows.append(
            {
                "transition": transition_key,
                "base_action_type": str(row.get("base_action_type", "")),
                "selected_action_type": str(row.get("selected_action_type", "")),
                "changed_count": count,
                "changed_rate": (count / denominator) if denominator else None,
                "mean_q_gap": (sum(q_gaps) / len(q_gaps)) if q_gaps else None,
                "p50_q_gap": percentile(q_gaps, 0.50),
                "p90_q_gap": percentile(q_gaps, 0.90),
                "p99_q_gap": percentile(q_gaps, 0.99),
            }
        )
    rows.sort(key=lambda item: (-int(item["changed_count"]), str(item["transition"])))
    return rows


def percentile(sorted_values: Sequence[float], q: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = max(0.0, min(1.0, float(q))) * (len(sorted_values) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def build_delta_vs_base(by_style: Mapping[str, Mapping[str, Any]], *, baseline_style: str = "base") -> dict[str, Any]:
    baseline = by_style.get(baseline_style)
    if baseline is None:
        return {}
    delta_keys = [
        "avg_rank",
        "avg_rank_pt",
        "avg_score",
        "win_rate",
        "deal_in_rate",
        "call_rate",
        "riichi_rate",
        "ryukyoku_rate",
        "style_changed_rate",
        "semantic_changed_rate",
        "style_applied_rate",
        "mean_base_to_selected_q_gap",
    ]
    deltas: dict[str, Any] = {}
    for style_id, style_metrics in by_style.items():
        if style_id == baseline_style:
            continue
        row: dict[str, Any] = {}
        for key in delta_keys:
            left = style_metrics.get(key)
            right = baseline.get(key)
            row[key] = None if left is None or right is None else float(left) - float(right)
        deltas[style_id] = row
    return deltas


def format_style_markdown(metrics: Mapping[str, Any]) -> str:
    rows = [
        "| Metric | " + " | ".join(metrics.get("by_style", {}).keys()) + " |",
        "|---|" + "|".join("---" for _ in metrics.get("by_style", {})) + "|",
    ]
    metric_keys = [
        ("Unique hanchans", "unique_hanchans"),
        ("Seat games", "seat_games"),
        ("Rounds", "rounds"),
        ("1st", ("rank_counts", 0)),
        ("2nd", ("rank_counts", 1)),
        ("3rd", ("rank_counts", 2)),
        ("4th", ("rank_counts", 3)),
        ("Avg rank", "avg_rank"),
        ("Avg rank pt", "avg_rank_pt"),
        ("Avg score", "avg_score"),
        ("Win rate", "win_rate"),
        ("Deal-in rate", "deal_in_rate"),
        ("Call rate", "call_rate"),
        ("Riichi rate", "riichi_rate"),
        ("Ryukyoku rate", "ryukyoku_rate"),
        ("Style applied rate", "style_applied_rate"),
        ("Style changed rate", "style_changed_rate"),
        ("Semantic changed rate", "semantic_changed_rate"),
        ("Mean base-to-selected Q gap", "mean_base_to_selected_q_gap"),
    ]
    styles = list(metrics.get("by_style", {}).values())
    for label, key in metric_keys:
        values: list[str] = []
        for style in styles:
            value: Any
            if isinstance(key, tuple):
                value = style[key[0]][key[1]]
            else:
                value = style.get(key)
            values.append(_format_metric_cell(value))
        rows.append("| " + label + " | " + " | ".join(values) + " |")
    if metrics.get("delta_vs_base"):
        rows.extend(["", "## Delta vs base", "", "| Metric | " + " | ".join(metrics["delta_vs_base"].keys()) + " |"])
        rows.append("|---|" + "|".join("---" for _ in metrics["delta_vs_base"]) + "|")
        delta_keys = [
            ("Avg rank", "avg_rank"),
            ("Avg rank pt", "avg_rank_pt"),
            ("Avg score", "avg_score"),
            ("Win rate", "win_rate"),
            ("Deal-in rate", "deal_in_rate"),
            ("Call rate", "call_rate"),
            ("Riichi rate", "riichi_rate"),
            ("Ryukyoku rate", "ryukyoku_rate"),
            ("Style applied rate", "style_applied_rate"),
            ("Style changed rate", "style_changed_rate"),
            ("Semantic changed rate", "semantic_changed_rate"),
            ("Mean base-to-selected Q gap", "mean_base_to_selected_q_gap"),
        ]
        delta_styles = list(metrics["delta_vs_base"].values())
        for label, key in delta_keys:
            rows.append(
                "| "
                + label
                + " | "
                + " | ".join(_format_metric_cell(style.get(key)) for style in delta_styles)
                + " |"
            )
    rows.extend(["", "## Semantic Changed Transition Breakdown", ""])
    for style_id, style_metrics in metrics.get("by_style", {}).items():
        breakdown = style_metrics.get("semantic_changed_transition_breakdown") or []
        if not breakdown:
            continue
        rows.extend(
            [
                f"### {style_id}",
                "",
                "| transition | changed_count | changed_rate | mean_q_gap | p50_q_gap | p90_q_gap | p99_q_gap |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for transition in breakdown:
            rows.append(
                "| "
                + " | ".join(
                    [
                        str(transition.get("transition", "")),
                        _format_metric_cell(transition.get("changed_count")),
                        _format_metric_cell(transition.get("changed_rate")),
                        _format_metric_cell(transition.get("mean_q_gap")),
                        _format_metric_cell(transition.get("p50_q_gap")),
                        _format_metric_cell(transition.get("p90_q_gap")),
                        _format_metric_cell(transition.get("p99_q_gap")),
                    ]
                )
                + " |"
            )
        rows.append("")
    return "\n".join(rows) + "\n"


def _format_metric_cell(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_style_metrics_csv(path: Path, metrics: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "style_id",
        "games",
        "unique_hanchans",
        "seat_games",
        "seat_count",
        "rank_sample_count",
        "rounds",
        "avg_rank",
        "avg_rank_pt",
        "avg_score",
        "win_rate",
        "deal_in_rate",
        "call_rate",
        "riichi_rate",
        "ryukyoku_rate",
        "style_applied_rate",
        "style_changed_rate",
        "semantic_changed_rate",
        "mean_base_to_selected_q_gap",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for style_id, style_metrics in metrics.get("by_style", {}).items():
            writer.writerow({key: style_metrics.get(key) for key in fieldnames} | {"style_id": style_id})


def run_game(args: argparse.Namespace, *, game_id: int) -> dict[str, Any]:
    profile_pool = parse_style_profiles(str(args.profiles))
    seed_key = parse_seed_key(getattr(args, "seed_key", 0))
    explicit_seat_profiles = assign_seat_profiles(getattr(args, "seat_profiles", None))
    seat_profiles = explicit_seat_profiles or assign_style_profiles(
        game_id=int(game_id),
        profiles=profile_pool,
        seed_key=seed_key,
        mode=str(getattr(args, "profile_mode", "rotate")),
    )
    schedule = seat_profile_schedule(seat_profiles, style_alpha=float(args.style_alpha))
    style_policies = {
        seat: MortalStylePolicy(profile=profile, style_alpha=float(args.style_alpha))
        for seat, profile in seat_profiles.items()
    }
    collect_decision_traces = str(getattr(args, "artifact_mode", "full")) == "full"
    game_seed = None if args.seed_start is None else int(args.seed_start) + int(game_id)
    env, seed_info = _make_env(game_mode=str(args.game_mode), seed=game_seed)
    bots = _new_bots(args)
    if seed_info.mode == "reset":
        obs_dict = env.reset(seed=game_seed)
    else:
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
            legal_actions = obs.legal_actions()
            raw_events = obs.new_events()
            parsed_events = [json.loads(event) if isinstance(event, str) else dict(event) for event in raw_events]
            if collect_decision_traces:
                event_indices, next_cursor, mismatches = _assign_event_indices(
                    events=parsed_events,
                    mjai_log=getattr(env, "mjai_log", []),
                    cursor=event_cursors.get(pid, 0),
                )
                event_cursors[pid] = next_cursor
                event_index_mismatch_count += mismatches
            else:
                event_indices = [-1] * len(parsed_events)

            mjai_action: dict[str, Any] | None = None
            for event, event_index in zip(parsed_events, event_indices):
                reaction = bots[pid].react(event)
                react_call_count += 1
                if reaction is not None:
                    raw_reaction = dict(reaction)
                    meta = dict(raw_reaction.pop("meta", {}) or {})
                    if style_policies[pid].enabled or collect_decision_traces:
                        style_decision = style_policies[pid].select_action(
                            base_action=raw_reaction,
                            mortal_meta=meta,
                            legal_actions=legal_actions,
                        )
                        meta["style_policy"] = style_decision.meta
                        mjai_action = style_decision.action
                    else:
                        mjai_action = raw_reaction
                else:
                    meta = {}
                if not collect_decision_traces or not meta:
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
        "seed_key": hex(seed_key),
        "profile_mode": "explicit" if explicit_seat_profiles is not None else str(getattr(args, "profile_mode", "rotate")),
        "profile_pool": [profile.to_json(style_alpha=float(args.style_alpha)) for profile in profile_pool],
        "seat_profile_schedule": schedule,
        "players": [
            {
                "seat": seat,
                "checkpoint": str(args.model),
                **seat_profiles[seat].to_json(style_alpha=float(args.style_alpha)),
            }
            for seat in range(4)
        ],
        "env_seed_applied": bool(seed_info.applied),
        "env_seed_mode": seed_info.mode,
        "scores": scores,
        "ranks": ranks,
        "events": replay_events,
        "sidecar": {"by_actor": dict(sidecar)},
        "summary": {
            "game_id": int(game_id),
            "seed": game_seed,
            "seed_key": hex(seed_key),
            "profile_mode": "explicit" if explicit_seat_profiles is not None else str(getattr(args, "profile_mode", "rotate")),
            "profile_pool": [profile.to_json(style_alpha=float(args.style_alpha)) for profile in profile_pool],
            "seat_profile_schedule": schedule,
            "players": [
                {
                    "seat": seat,
                    "checkpoint": str(args.model),
                    **seat_profiles[seat].to_json(style_alpha=float(args.style_alpha)),
                }
                for seat in range(4)
            ],
            "env_seed_applied": bool(seed_info.applied),
            "env_seed_mode": seed_info.mode,
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
    profile_pool = parse_style_profiles(str(args.profiles))
    seed_key = parse_seed_key(getattr(args, "seed_key", 0))
    rank_points = parse_rank_points(args.rank_points)
    explicit_seat_profiles = assign_seat_profiles(getattr(args, "seat_profiles", None))
    profile_mode = "explicit" if explicit_seat_profiles is not None else str(args.profile_mode)
    artifact_mode = str(getattr(args, "artifact_mode", "full"))
    game_results: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    manifest: list[dict[str, Any]] = []
    started = time.perf_counter()

    for game_id in range(int(args.games)):
        result = run_game(args, game_id=game_id)
        replay_path = replays_dir / f"game_{game_id:05d}.mjson"
        replay_gz_path = replays_dir / f"game_{game_id:05d}.json.gz"
        sidecar_path = replays_dir / f"game_{game_id:05d}.decisions.json"
        meta_path = replays_dir / f"game_{game_id:05d}.json"
        _write_mjson_gz(replay_gz_path, result["events"])
        if artifact_mode == "full":
            _write_mjson(replay_path, result["events"])
            sidecar_path.write_text(json.dumps(result["sidecar"], ensure_ascii=False, indent=2), encoding="utf-8")
        meta = {
            **result["summary"],
            "json_gz": str(replay_gz_path),
        }
        if artifact_mode == "full":
            meta["mjson"] = str(replay_path)
            meta["decision_traces"] = str(sidecar_path)
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        game_results.append(result)
        summaries.append(result["summary"])
        manifest_row = {
            "game_id": int(game_id),
            "seed": result["summary"]["seed"],
            "seed_key": result["summary"]["seed_key"],
            "profile_mode": result["summary"]["profile_mode"],
            "profile_pool": result["summary"]["profile_pool"],
            "seat_profile_schedule": result["summary"]["seat_profile_schedule"],
            "players": result["summary"]["players"],
            "json_gz": str(replay_gz_path),
            "scores": result["summary"]["scores"],
            "ranks": result["summary"]["ranks"],
        }
        if artifact_mode == "full":
            manifest_row.update(
                {
                    "mjson": str(replay_path),
                    "meta": str(meta_path),
                    "decision_traces": str(sidecar_path),
                }
            )
        manifest.append(manifest_row)
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
        "artifact_mode": artifact_mode,
        "seed_start": None if args.seed_start is None else int(args.seed_start),
        "seed_key": hex(seed_key),
        "rank_points": [float(value) for value in rank_points],
        "profiles": [profile.to_json(style_alpha=float(args.style_alpha)) for profile in profile_pool],
        "profile_mode": profile_mode,
        "seat_profiles": (
            None
            if explicit_seat_profiles is None
            else seat_profile_schedule(explicit_seat_profiles, style_alpha=float(args.style_alpha))
        ),
        "env_seed_applied_all": all(bool(row["env_seed_applied"]) for row in summaries),
        "env_seed_modes": sorted({str(row["env_seed_mode"]) for row in summaries}),
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
    style_metrics = build_style_metrics(game_results, rank_points=rank_points)
    metrics_document = eval_metrics.build_metrics_document(
        run={
            "backend": "riichienv",
            "checkpoint": str(args.model),
            "game_mode": str(args.game_mode),
            "games": int(args.games),
            "seed_start": None if args.seed_start is None else int(args.seed_start),
            "seed_key": hex(seed_key),
            "rank_points": [float(value) for value in rank_points],
            "artifact_mode": artifact_mode,
            "profiles": [profile.to_json(style_alpha=float(args.style_alpha)) for profile in profile_pool],
            "profile_mode": profile_mode,
            "seat_profiles": (
                None
                if explicit_seat_profiles is None
                else seat_profile_schedule(explicit_seat_profiles, style_alpha=float(args.style_alpha))
            ),
        },
        metrics=style_metrics,
        artifacts={
            "summary": str(output_dir / "summary.json"),
            "manifest_json": str(replays_dir / "manifest.json"),
            "manifest_jsonl": str(replays_dir / "manifest.jsonl"),
            "detailed_stats_md": str(output_dir / "detailed_stats.md"),
            "metrics_csv": str(output_dir / "metrics.csv"),
        },
        rank_points_profile="custom",
        rank_points_values=rank_points,
    )
    (replays_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    _write_manifest_jsonl(replays_dir / "manifest.jsonl", manifest)
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    eval_metrics.write_metrics(output_dir / "metrics.json", metrics_document)
    if artifact_mode == "full":
        (output_dir / "detailed_stats.md").write_text(format_style_markdown(style_metrics), encoding="utf-8")
        write_style_metrics_csv(output_dir / "metrics.csv", style_metrics)


if __name__ == "__main__":
    main()
