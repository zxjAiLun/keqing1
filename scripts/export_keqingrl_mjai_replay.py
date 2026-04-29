#!/usr/bin/env python3
"""Replay a KeqingRL rollout seed and export MJAI plus a readable trace."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import copy
import csv
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from keqingrl import DiscardOnlyMahjongEnv
from keqingrl.actions import ActionSpec, ActionType
from keqingrl.selfplay import collect_selfplay_episode
from mahjong_env.action_space import IDX_TO_TILE_NAME
from scripts.probe_keqingrl_sampling_diversity import (
    TemperaturePolicy,
    _load_candidates,
    _load_policy,
    _opponent_pool,
)
from scripts.run_keqingrl_temperature_pilot import _seed_torch_sampling
from scripts.run_keqingrl_tempered_ratio_pilot import DeltaSupportProjectionPolicy, _action_type_tuple


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    candidate = _load_candidates(args)[0]
    source_policy = _load_policy(candidate, device)
    base_policy = copy.deepcopy(source_policy).to(device)
    base_policy.rule_score_scale = float(args.rule_score_scale)
    policy = DeltaSupportProjectionPolicy(
        base_policy,
        support_mode=str(args.delta_support_mode),
        topk=int(args.delta_support_topk),
        margin_threshold=float(args.delta_support_margin_threshold),
        outside_support_delta_mode=str(args.outside_support_delta_mode),
        support_policy_mode=str(args.support_policy_mode),
    ).to(device)
    behavior_policy = TemperaturePolicy(policy, temperature=float(args.temperature)).to(device)
    opponent_pool = _opponent_pool(str(candidate["opponent_mode"]))
    env = DiscardOnlyMahjongEnv(
        max_kyokus=int(args.max_kyokus),
        self_turn_action_types=_action_type_tuple(args.self_turn_action_types),
        response_action_types=_action_type_tuple(args.response_action_types),
        forced_autopilot_action_types=_action_type_tuple(args.forced_autopilot_action_types),
    )

    _seed_torch_sampling(int(args.torch_seed))
    selected_episode = None
    selected_events: list[dict[str, Any]] | None = None
    selected_seed = None
    for episode_idx in range(int(args.episode_index) + 1):
        episode_seed = int(args.seed_base) + episode_idx * int(args.seed_stride)
        episode = collect_selfplay_episode(
            env,
            behavior_policy,
            opponent_pool=opponent_pool,
            learner_seats=tuple(int(seat) for seat in args.learner_seats),
            seed=episode_seed,
            greedy=False,
            max_steps=int(args.max_steps),
            device=device,
        )
        if episode_idx == int(args.episode_index):
            selected_episode = episode
            selected_events = [dict(event) for event in env._require_room().events]
            selected_seed = episode_seed

    if selected_episode is None or selected_events is None or selected_seed is None:
        raise RuntimeError("failed to collect selected episode")

    stem = f"episode_{int(args.episode_index):03d}_seed_{int(selected_seed)}"
    mjai_path = args.output_dir / f"{stem}.mjai.jsonl"
    readable_path = args.output_dir / f"{stem}.readable.md"
    decision_path = args.output_dir / f"{stem}.decisions.csv"
    mjai_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False, sort_keys=True) for event in selected_events) + "\n",
        encoding="utf-8",
    )
    readable_path.write_text(
        _readable_markdown(
            selected_events,
            selected_episode.steps,
            episode_index=int(args.episode_index),
            seed=int(selected_seed),
            candidate=candidate,
            action_scope={
                "self_turn": tuple(args.self_turn_action_types),
                "response": tuple(args.response_action_types),
                "forced_autopilot": tuple(args.forced_autopilot_action_types),
            },
        ),
        encoding="utf-8",
    )
    _write_decisions_csv(decision_path, selected_episode.steps)
    print(f"wrote {mjai_path}")
    print(f"wrote {readable_path}")
    print(f"wrote {decision_path}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a KeqingRL replay as MJAI JSONL and readable markdown")
    parser.add_argument("--candidate-summary", type=Path, required=True)
    parser.add_argument("--source-config-ids", type=int, nargs="+", default=(93,))
    parser.add_argument("--rerun-config-ids", type=int, nargs="+", default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--seed-base", type=int, default=202604300000)
    parser.add_argument("--seed-stride", type=int, default=1)
    parser.add_argument("--torch-seed", type=int, default=202604300000)
    parser.add_argument("--learner-seats", type=int, nargs="+", default=(0,))
    parser.add_argument("--temperature", type=float, default=1.25)
    parser.add_argument("--rule-score-scale", type=float, default=0.25)
    parser.add_argument("--support-policy-mode", default="support-only-topk")
    parser.add_argument("--delta-support-mode", default="topk")
    parser.add_argument("--delta-support-topk", type=int, default=3)
    parser.add_argument("--delta-support-margin-threshold", type=float, default=0.75)
    parser.add_argument("--outside-support-delta-mode", default="zero")
    parser.add_argument("--self-turn-action-types", nargs="+", default=("DISCARD", "REACH_DISCARD", "TSUMO", "RYUKYOKU"))
    parser.add_argument("--response-action-types", nargs="*", default=())
    parser.add_argument("--forced-autopilot-action-types", nargs="*", default=("TSUMO", "RON", "RYUKYOKU"))
    parser.add_argument("--max-kyokus", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=512)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _readable_markdown(
    events: Sequence[dict[str, Any]],
    steps: Sequence[Any],
    *,
    episode_index: int,
    seed: int,
    candidate: dict[str, Any],
    action_scope: dict[str, tuple[str, ...]],
) -> str:
    lines = [
        f"# KeqingRL Replay episode={episode_index} seed={seed}",
        "",
        f"- source_config_id: `{candidate['source_config_id']}`",
        f"- checkpoint: `{candidate['checkpoint_path']}`",
        f"- action_scope: `{action_scope}`",
        "",
        "## MJAI Events",
        "",
    ]
    hands: list[Counter[str]] = [Counter() for _ in range(4)]
    discards: list[list[str]] = [[] for _ in range(4)]
    for index, event in enumerate(events):
        text = _format_event(index, event, hands=hands, discards=discards)
        if text:
            lines.append(text)

    lines.extend(["", "## Decisions", ""])
    for index, step in enumerate(steps):
        lines.append(
            f"- d{index:03d} step={step.step_id} P{step.actor} "
            f"{_format_action_spec(step.action_spec)} | legal: "
            f"{'; '.join(_format_action_spec(action) for action in (step.legal_actions or (step.action_spec,)))}"
        )
    return "\n".join(lines) + "\n"


def _format_event(
    index: int,
    event: dict[str, Any],
    *,
    hands: list[Counter[str]],
    discards: list[list[str]],
) -> str:
    event_type = str(event.get("type", ""))
    prefix = f"- e{index:03d} "
    if event_type == "start_game":
        return prefix + f"开局 names={event.get('names')}"
    if event_type == "start_kyoku":
        tehais = event.get("tehais") or [[], [], [], []]
        for seat, tiles in enumerate(tehais):
            hands[seat] = Counter(str(tile) for tile in tiles)
            discards[seat] = []
        lines = [
            prefix
            + (
                f"{_wind(event.get('bakaze'))}{int(event.get('kyoku', 1))}局 "
                f"{event.get('honba', 0)}本场 供托{event.get('kyotaku', 0)} "
                f"庄=P{event.get('oya')} 宝牌指示={_tile(event.get('dora_marker'))} "
                f"点数={event.get('scores')}"
            )
        ]
        for seat in range(4):
            lines.append(f"  - P{seat} 手牌: {_tiles(hands[seat])}")
        return "\n".join(lines)
    if event_type == "tsumo":
        actor = int(event["actor"])
        pai = str(event["pai"])
        hands[actor][pai] += 1
        rinshan = " 岭上" if event.get("rinshan") else ""
        return prefix + f"P{actor} 摸{rinshan} {_tile(pai)} | 手牌 {_tiles(hands[actor])}"
    if event_type == "dahai":
        actor = int(event["actor"])
        pai = str(event["pai"])
        _remove_tile(hands[actor], pai)
        discards[actor].append(pai)
        cut_type = "摸切" if event.get("tsumogiri") else "手切"
        return prefix + f"P{actor} 打 {_tile(pai)} ({cut_type}) | 手牌 {_tiles(hands[actor])} | 河 {_tiles(discards[actor])}"
    if event_type == "reach":
        return prefix + f"P{event.get('actor')} 宣言立直"
    if event_type == "reach_accepted":
        return prefix + f"P{event.get('actor')} 立直成立 | 点数={event.get('scores')} 供托={event.get('kyotaku')}"
    if event_type == "none":
        actor = event.get("actor")
        return prefix + ("全员无动作/响应窗口结束" if actor is None else f"P{actor} 过")
    if event_type in {"chi", "pon", "daiminkan", "ankan", "kakan", "kakan_accepted"}:
        actor = int(event["actor"])
        for tile in event.get("consumed") or []:
            _remove_tile(hands[actor], str(tile))
        target = event.get("target")
        target_text = "" if target is None else f" from P{target}"
        return (
            prefix
            + f"P{actor} {event_type}{target_text} {_tile(event.get('pai'))} "
            f"consumed={_tiles(Counter(str(tile) for tile in event.get('consumed') or []))} | 手牌 {_tiles(hands[actor])}"
        )
    if event_type == "dora":
        return prefix + f"新宝牌指示 {_tile(event.get('dora_marker'))}"
    if event_type == "hora":
        actor = event.get("actor")
        target = event.get("target")
        win_type = "自摸" if event.get("is_tsumo") else f"荣和 P{target}"
        yaku = ",".join(str(item) for item in event.get("yaku") or [])
        return (
            prefix
            + f"P{actor} {win_type} {_tile(event.get('pai'))} "
            f"{event.get('han')}翻{event.get('fu')}符 cost={event.get('cost')} "
            f"役=[{yaku}] deltas={event.get('deltas')} scores={event.get('scores')}"
        )
    if event_type == "ryukyoku":
        return (
            prefix
            + f"流局 tenpai={event.get('tenpai_players')} deltas={event.get('deltas')} "
            f"scores={event.get('scores')}"
        )
    if event_type == "end_kyoku":
        return prefix + "本局结束"
    if event_type == "end_game":
        return prefix + f"半庄结束 scores={event.get('scores')}"
    return prefix + json.dumps(event, ensure_ascii=False, sort_keys=True)


def _write_decisions_csv(path: Path, steps: Sequence[Any]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=("decision", "step_id", "actor", "selected", "legal_actions"),
        )
        writer.writeheader()
        for index, step in enumerate(steps):
            writer.writerow(
                {
                    "decision": index,
                    "step_id": step.step_id,
                    "actor": step.actor,
                    "selected": _format_action_spec(step.action_spec),
                    "legal_actions": " | ".join(
                        _format_action_spec(action) for action in (step.legal_actions or (step.action_spec,))
                    ),
                }
            )


def _format_action_spec(action: ActionSpec) -> str:
    action_type = action.action_type
    if action_type == ActionType.DISCARD:
        return f"打{_tile_id(action.tile)}"
    if action_type == ActionType.REACH_DISCARD:
        return f"立直打{_tile_id(action.tile)}"
    if action_type == ActionType.TSUMO:
        return "自摸"
    if action_type == ActionType.RON:
        return f"荣和(from P{action.from_who})"
    if action_type == ActionType.PASS:
        return "过"
    if action_type in {ActionType.CHI, ActionType.PON, ActionType.DAIMINKAN, ActionType.ANKAN, ActionType.KAKAN}:
        consumed = ",".join(_tile_id(tile) for tile in action.consumed)
        return f"{action_type.name} {_tile_id(action.tile)} consumed=[{consumed}] from P{action.from_who}"
    if action_type == ActionType.RYUKYOKU:
        return "流局"
    return action_type.name


def _remove_tile(hand: Counter[str], tile: str) -> None:
    candidates = [tile]
    if len(tile) == 2 and tile[0] == "5" and tile[1] in "mps":
        candidates.append(tile + "r")
    if tile.endswith("r"):
        candidates.append(tile[:2])
    for candidate in candidates:
        if hand.get(candidate, 0) > 0:
            hand[candidate] -= 1
            if hand[candidate] <= 0:
                del hand[candidate]
            return


def _tiles(counter: Counter[str] | Sequence[str]) -> str:
    tiles = list(counter.elements()) if isinstance(counter, Counter) else [str(tile) for tile in counter]
    return " ".join(_tile(tile) for tile in sorted(tiles, key=_tile_sort_key)) or "-"


def _tile(value: Any) -> str:
    if value is None:
        return "-"
    tile = str(value)
    if tile.endswith("r") and len(tile) == 3:
        return f"{tile[:2]}赤"
    return {
        "E": "东",
        "S": "南",
        "W": "西",
        "N": "北",
        "P": "白",
        "F": "发",
        "C": "中",
    }.get(tile, tile)


def _tile_id(tile_id: int | None) -> str:
    if tile_id is None:
        return "-"
    return _tile(IDX_TO_TILE_NAME[int(tile_id)])


def _wind(value: Any) -> str:
    return {"E": "东", "S": "南", "W": "西", "N": "北"}.get(str(value), str(value))


def _tile_sort_key(tile: str) -> tuple[int, int, int]:
    honors = {"E": 1, "S": 2, "W": 3, "N": 4, "P": 5, "F": 6, "C": 7}
    if tile in honors:
        return (3, honors[tile], 0)
    base = tile[:2] if tile.endswith("r") else tile
    if len(base) >= 2 and base[0].isdigit() and base[1] in "mps":
        return ("mps".index(base[1]), int(base[0]), 1 if tile.endswith("r") else 0)
    return (9, 9, 9)


if __name__ == "__main__":
    main()
