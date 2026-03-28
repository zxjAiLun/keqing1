from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set

from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.state import GameState, apply_event
from mahjong_env.tiles import AKA_DORA_TILES
from mahjong_env.types import MjaiEvent
from mahjong_env.types import Action


def _normalize_or_keep_aka(tile: str) -> str:
    if tile in AKA_DORA_TILES:
        return tile
    if tile.endswith("r"):
        return tile[:-1]
    return tile


@dataclass
class ReplaySample:
    state: Dict
    actor: int
    actor_name: str
    label_action: Dict
    legal_actions: List[Dict]
    value_target: float


ACTION_TYPES_FOR_LABEL = {
    "dahai",
    "chi",
    "pon",
    "daiminkan",
    "ankan",
    "kakan",
    "reach",
    "hora",
    "ryukyoku",
}


def read_mjai_jsonl(path: str | Path) -> List[MjaiEvent]:
    events: List[MjaiEvent] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            events.append(json.loads(line))
    return events


def extract_actor_names(events: Sequence[MjaiEvent]) -> List[str]:
    for e in events:
        if e.get("type") == "start_game" and isinstance(e.get("names"), list) and len(e["names"]) == 4:
            return [str(x) for x in e["names"]]
    return ["p0", "p1", "p2", "p3"]


def build_supervised_samples(
    events: List[MjaiEvent],
    actor_filter: Optional[Set[int]] = None,
    actor_name_filter: Optional[Set[str]] = None,
) -> List[ReplaySample]:
    state = GameState()
    samples: List[ReplaySample] = []
    actor_names = extract_actor_names(events)

    # Use libriichi's oracle state to provide a local EV proxy based on:
    # - shanten improvement
    # - tenpai/ukeire proxy via number of winning tiles (waits set size)
    # - for calls (chi/pon/kan), a small penalty when shanten/waits don't improve
    #   (chi/pon consumes concealed tiles; if it doesn't improve, it's likely wrong).
    try:
        import riichi  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"riichi is required for shanten/waits value proxy: {e}") from e

    riichi_states = [riichi.state.PlayerState(pid) for pid in range(4)]

    W_SHANTEN = 1.0
    W_WAITS = 0.05
    W_CALL_TEHAI = 0.03
    CALL_ACTIONS = {"chi", "pon", "daiminkan", "ankan", "kakan"}

    for event in events:
        et = event["type"]
        actor = event.get("actor")

        collect_sample = (
            actor is not None
            and et in ACTION_TYPES_FOR_LABEL
            and state.in_game
            and (actor_filter is None or actor in actor_filter)
        )
        if collect_sample and actor_name_filter is not None:
            actor_name = actor_names[actor] if 0 <= actor < len(actor_names) else f"p{actor}"
            collect_sample = actor_name in actor_name_filter

        # Compute local EV proxy for labeled actions (before->after riichi update).
        value_target_local: Optional[float] = None
        shanten_before: Optional[int] = None
        waits_before_cnt: Optional[int] = None
        tehai_before_cnt: Optional[int] = None
        if collect_sample:
            # riichi.state.PlayerState is already updated by previous events.
            ps = riichi_states[actor]  # type: ignore[index]
            shanten_before = int(ps.shanten)
            waits_before_cnt = int(sum(ps.waits))
            tehai_before_cnt = int(sum(ps.tehai))

        payload = json.dumps(event, ensure_ascii=False)
        # Terminal events (hora/ryukyoku) and end_* events in our exported mjai logs
        # might omit `actor`, while libriichi's python bindings expect it.
        # We only need shanten/waits for decision-time actions, so skip them.
        if et not in {"hora", "ryukyoku", "end_kyoku", "end_game"}:
            for s in riichi_states:
                s.update(payload)

        if collect_sample:
            ps_after = riichi_states[actor]  # type: ignore[index]
            shanten_after = int(ps_after.shanten)
            waits_after_cnt = int(sum(ps_after.waits))
            tehai_after_cnt = int(sum(ps_after.tehai))

            delta_shanten = shanten_before - shanten_after  # type: ignore[operator]
            delta_waits = waits_after_cnt - waits_before_cnt  # type: ignore[operator]
            delta_tehai = tehai_after_cnt - tehai_before_cnt  # type: ignore[operator]

            # Terminal override: winning should reflect final score deltas.
            if et == "hora":
                sd = event.get("score_delta")
                if isinstance(sd, list) and len(sd) == 4:
                    value_target_local = float(sd[actor]) / 12000.0
                else:
                    value_target_local = float(W_SHANTEN * delta_shanten + W_WAITS * delta_waits)
            else:
                value_target_local = float(W_SHANTEN * delta_shanten + W_WAITS * delta_waits)
                if et in CALL_ACTIONS:
                    value_target_local += float(W_CALL_TEHAI * delta_tehai)
                    # clip to stabilize advantage weights
                    value_target_local = max(min(value_target_local, 10.0), -10.0)

        if collect_sample:
            # Keep local value proxy computed above.
            actor_name = actor_names[actor] if 0 <= actor < len(actor_names) else f"p{actor}"
            snap = state.snapshot(actor)
            # Only collect supervised samples when the actor hand is visible.
            if snap["hand"]:
                # Attach shanten/waits features for the policy/value model.
                snap["shanten"] = shanten_before if shanten_before is not None else 0
                # waits_after_cnt reflects waits after this action is applied:
                # - for dahai: waits of the resulting tenpai shape after discarding label tile
                # - for chi/pon/kan: waits after the call (0 if not tenpai after call)
                snap["waits_count"] = waits_after_cnt if waits_after_cnt is not None else 0
                snap["waits_tiles"] = list(ps.waits)  # length-34 bool list, before action

                legal = enumerate_legal_actions(snap, actor)
                label = dict(event)
                if "pai" in label:
                    label["pai"] = _normalize_or_keep_aka(label["pai"])
                legal_dicts = [a.to_mjai() for a in legal]
                # Our state reconstruction is intentionally lightweight.
                # If the labeled action is not in enumerated legal set,
                # inject it so supervised learning can proceed.
                if label["type"] == "dahai":
                    found = any(
                        x.get("type") == "dahai" and x.get("pai") == label.get("pai")
                        for x in legal_dicts
                    )
                    if not found and "pai" in label:
                        legal_dicts.append(
                            Action(
                                type="dahai",
                                actor=actor,
                                pai=label["pai"],
                                tsumogiri=bool(label.get("tsumogiri", False)),
                            ).to_mjai()
                        )
                elif label["type"] == "reach":
                    found = any(x.get("type") == "reach" for x in legal_dicts)
                    if not found:
                        legal_dicts.append(Action(type="reach", actor=actor).to_mjai())
                elif label["type"] in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
                    # 对于吃碰杠，如果不在 legal 中，注入它
                    found = any(x.get("type") == label["type"] for x in legal_dicts)
                    if not found:
                        # 创建对应的 Action
                        action_kwargs = {"type": label["type"], "actor": actor}
                        if "consumed" in label:
                            action_kwargs["consumed"] = label["consumed"]
                        if "pai" in label:
                            action_kwargs["pai"] = label["pai"]
                        legal_dicts.append(Action(**action_kwargs).to_mjai())
                elif label["type"] == "hora":
                    found = any(x.get("type") == "hora" for x in legal_dicts)
                    if not found:
                        action_kwargs = {"type": "hora", "actor": actor}
                        if "pai" in label:
                            action_kwargs["pai"] = label["pai"]
                        legal_dicts.append(Action(**action_kwargs).to_mjai())

                samples.append(
                    ReplaySample(
                        state=snap,
                        actor=actor,
                        actor_name=actor_name,
                        label_action=label,
                        legal_actions=legal_dicts,
                        value_target=float(value_target_local if value_target_local is not None else 0.0),
                    )
                )

        apply_event(state, event)
    return samples


def replay_validate_label_legal(samples: List[ReplaySample]) -> List[str]:
    errors: List[str] = []
    for idx, sample in enumerate(samples):
        label = sample.label_action
        legal = sample.legal_actions
        if label["type"] == "dahai":
            found = any(a["type"] == "dahai" and a.get("pai") == label.get("pai") for a in legal)
        else:
            found = any(a["type"] == label["type"] for a in legal)
        if not found:
            errors.append(f"sample#{idx}: label {label} not in legal set")
    return errors

