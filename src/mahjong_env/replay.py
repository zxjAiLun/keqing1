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

# ---------------------------------------------------------------------------
# riichienv-based shanten/waits helpers (replaces riichi.state.PlayerState)
# ---------------------------------------------------------------------------
_STR_TO_136_CACHE: Dict[str, int] = {}

def _build_str_to_136() -> None:
    from mahjong.tile import TilesConverter, FIVE_RED_MAN, FIVE_RED_PIN, FIVE_RED_SOU
    for suit, kw in (('m', 'man'), ('p', 'pin'), ('s', 'sou')):
        for n in range(1, 10):
            t = TilesConverter.string_to_136_array(**{kw: str(n)}, has_aka_dora=True)
            _STR_TO_136_CACHE[f'{n}{suit}'] = t[0]
    _STR_TO_136_CACHE['5mr'] = FIVE_RED_MAN
    _STR_TO_136_CACHE['5pr'] = FIVE_RED_PIN
    _STR_TO_136_CACHE['5sr'] = FIVE_RED_SOU
    for name, z in (('E','1'),('S','2'),('W','3'),('N','4'),('P','5'),('F','6'),('C','7')):
        _STR_TO_136_CACHE[name] = TilesConverter.string_to_136_array(honors=z)[0]

_build_str_to_136()

def _to_136(tile: str) -> int:
    return _STR_TO_136_CACHE.get(tile, -1)

def _snap_melds_to_riichienv(melds: List[dict]):
    from riichienv import Meld as RiichiMeld, MeldType
    _TYPE_MAP = {'chi': MeldType.Chi, 'pon': MeldType.Pon,
                 'daiminkan': MeldType.Daiminkan, 'ankan': MeldType.Ankan, 'kakan': MeldType.Kakan}
    result = []
    for m in melds:
        mt = _TYPE_MAP.get(m.get('type', ''))
        if mt is None:
            continue
        tiles136 = [_to_136(t) for t in m.get('consumed', []) if _to_136(t) >= 0]
        pai = m.get('pai')
        if pai and _to_136(pai) >= 0:
            tiles136.append(_to_136(pai))
        result.append(RiichiMeld(mt, tiles136, m.get('type') != 'ankan'))
    return result

def _calc_shanten_waits(hand: List[str], melds: List[dict]):
    """返回 (shanten, waits_count, waits_tile34_bools, tehai_count)。"""
    from riichienv import calculate_shanten, HandEvaluator
    hand_ids = [_to_136(t) for t in hand if _to_136(t) >= 0]
    if not hand_ids:
        return 8, 0, [False] * 34, 0
    try:
        rmelds = _snap_melds_to_riichienv(melds) or None
        he = HandEvaluator(hand_ids, rmelds)
        shanten = int(calculate_shanten(hand_ids)) if not he.is_tenpai() else 0
        waits136 = he.get_waits() if he.is_tenpai() else []
        waits34 = [False] * 34
        for w in waits136:
            idx = w // 4
            if 0 <= idx < 34:
                waits34[idx] = True
        waits_count = sum(waits34)
        tehai_count = len(hand_ids)
        return shanten, waits_count, waits34, tehai_count
    except Exception:
        return 8, 0, [False] * 34, len(hand_ids)


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
            snap_before = state.snapshot(actor)
            hand_before = snap_before.get("hand", [])
            melds_before = (snap_before.get("melds") or [[],[],[],[]])[actor]
            shanten_before, waits_before_cnt, _waits_before_bools, tehai_before_cnt = \
                _calc_shanten_waits(hand_before, melds_before)

        if collect_sample:
            # Apply event first so we can compute after-state
            apply_event(state, event)
            snap_after = state.snapshot(actor)
            hand_after = snap_after.get("hand", [])
            melds_after = (snap_after.get("melds") or [[],[],[],[]])[actor]
            shanten_after, waits_after_cnt, _waits_after_bools, tehai_after_cnt = \
                _calc_shanten_waits(hand_after, melds_after)
        if collect_sample:
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

            # Keep local value proxy computed above.
            actor_name = actor_names[actor] if 0 <= actor < len(actor_names) else f"p{actor}"
            snap = snap_before
            # Only collect supervised samples when the actor hand is visible.
            if snap["hand"]:
                # Attach shanten/waits features for the policy/value model.
                snap["shanten"] = shanten_before if shanten_before is not None else 0
                # waits_after_cnt reflects waits after this action is applied:
                # - for dahai: waits of the resulting tenpai shape after discarding label tile
                # - for chi/pon/kan: waits after the call (0 if not tenpai after call)
                snap["waits_count"] = waits_after_cnt if waits_after_cnt is not None else 0
                snap["waits_tiles"] = _waits_before_bools  # length-34 bool list, before action

                legal = enumerate_legal_actions(snap, actor)
                label = dict(event)
                if "pai" in label:
                    label["pai"] = _normalize_or_keep_aka(label["pai"])
                if "consumed" in label:
                    label["consumed"] = [_normalize_or_keep_aka(t) for t in label["consumed"]]
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

        if not collect_sample:
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

