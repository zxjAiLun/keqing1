"""Helpers for xmodel1 non-discard action reconstruction.

This module is the first implementation slice of the `xmodel1_discard_v4`
refactor. It keeps the action-candidate semantics explicit instead of routing
everything through the legacy special-slot compression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from mahjong_env.action_space import action_to_idx
from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.state import GameState, PlayerState, apply_event

_RESPONSE_TYPES = frozenset(
    {
        "reach",
        "chi",
        "pon",
        "daiminkan",
        "ankan",
        "kakan",
        "hora",
        "ryukyoku",
        "none",
    }
)
_POST_DISCARD_RESPONSE_TYPES = frozenset({"reach", "chi", "pon", "daiminkan"})


@dataclass
class ResponseActionState:
    action: dict
    action_idx: int
    requires_post_discard: bool
    after_snapshot: dict | None
    post_discard_actions: tuple[dict, ...]


def is_response_action(action: dict) -> bool:
    return action.get("type") in _RESPONSE_TYPES


def is_off_turn_response_action(action: dict) -> bool:
    return is_response_action(action) and action.get("type") in {
        "hora",
        "chi",
        "pon",
        "daiminkan",
        "none",
    }


def requires_post_discard(action: dict) -> bool:
    return action.get("type") in _POST_DISCARD_RESPONSE_TYPES


def iter_response_actions(legal_actions: Iterable[dict]) -> list[dict]:
    actions = [dict(action) for action in legal_actions if is_response_action(action)]
    actions.sort(
        key=lambda action: (
            action_to_idx(action),
            action.get("pai", ""),
            tuple(action.get("consumed") or ()),
        )
    )
    return actions


def iter_off_turn_response_actions(legal_actions: Iterable[dict]) -> list[dict]:
    return [action for action in iter_response_actions(legal_actions) if is_off_turn_response_action(action)]


def _state_from_snapshot(snapshot: dict, *, actor: int) -> GameState:
    state = GameState()
    state.bakaze = snapshot.get("bakaze", "E")
    state.kyoku = int(snapshot.get("kyoku", 1))
    state.honba = int(snapshot.get("honba", 0))
    state.kyotaku = int(snapshot.get("kyotaku", 0))
    state.oya = int(snapshot.get("oya", 0))
    state.dora_markers = list(snapshot.get("dora_markers", []))
    state.ura_dora_markers = list(snapshot.get("ura_dora_markers", []))
    state.scores = list(snapshot.get("scores", [25000, 25000, 25000, 25000]))
    state.actor_to_move = snapshot.get("actor_to_move")
    state.last_discard = dict(snapshot["last_discard"]) if snapshot.get("last_discard") is not None else None
    state.last_kakan = dict(snapshot["last_kakan"]) if snapshot.get("last_kakan") is not None else None
    state.last_tsumo = list(snapshot.get("last_tsumo", [None, None, None, None]))
    state.last_tsumo_raw = list(snapshot.get("last_tsumo_raw", [None, None, None, None]))
    state.remaining_wall = snapshot.get("remaining_wall")
    state.pending_rinshan_actor = snapshot.get("pending_rinshan_actor")
    state.ryukyoku_tenpai_players = list(snapshot.get("ryukyoku_tenpai_players", []))
    state.players = [PlayerState() for _ in range(4)]
    hand_owner = actor
    discards = snapshot.get("discards") or [[], [], [], []]
    melds = snapshot.get("melds") or [[], [], [], []]
    reached = snapshot.get("reached") or [False] * 4
    pending_reach = snapshot.get("pending_reach") or [False] * 4
    furiten = snapshot.get("furiten") or [False] * 4
    sutehai_furiten = snapshot.get("sutehai_furiten") or [False] * 4
    riichi_furiten = snapshot.get("riichi_furiten") or [False] * 4
    doujun_furiten = snapshot.get("doujun_furiten") or [False] * 4
    ippatsu_eligible = snapshot.get("ippatsu_eligible") or [False] * 4
    rinshan_tsumo = snapshot.get("rinshan_tsumo") or [False] * 4
    for pid in range(4):
        player = state.players[pid]
        hand = snapshot.get("hand", []) if pid == hand_owner else []
        player.hand.update(hand)
        player.discards = list(discards[pid])
        player.melds = list(melds[pid])
        player.reached = bool(reached[pid])
        player.pending_reach = bool(pending_reach[pid])
        player.furiten = bool(furiten[pid])
        player.sutehai_furiten = bool(sutehai_furiten[pid])
        player.riichi_furiten = bool(riichi_furiten[pid])
        player.doujun_furiten = bool(doujun_furiten[pid])
        player.ippatsu_eligible = bool(ippatsu_eligible[pid])
        player.rinshan_tsumo = bool(rinshan_tsumo[pid])
    return state


def build_post_response_snapshot(snapshot: dict, actor: int, action: dict) -> dict | None:
    if not requires_post_discard(action):
        return None
    event = dict(action)
    event.setdefault("actor", actor)
    if "pai" in event:
        event.setdefault("pai_raw", event["pai"])
    state = _state_from_snapshot(snapshot, actor=actor)
    apply_event(state, event)
    return state.snapshot(actor)


def build_response_action_states(
    snapshot: dict,
    actor: int,
    legal_actions: Iterable[dict] | None = None,
) -> tuple[ResponseActionState, ...]:
    if legal_actions is None:
        resolved_legal = [action.to_mjai() for action in enumerate_legal_actions(snapshot, actor)]
    else:
        resolved_legal = [dict(action) for action in legal_actions]
    states: list[ResponseActionState] = []
    for action in iter_response_actions(resolved_legal):
        after_snapshot = build_post_response_snapshot(snapshot, actor, action)
        if action.get("type") == "reach":
            post_discard_actions = tuple(
                dict(item)
                for item in resolved_legal
                if item.get("type") == "dahai"
            )
        elif after_snapshot is None:
            post_discard_actions: tuple[dict, ...] = ()
        else:
            post_discard_actions = tuple(
                dict(item.to_mjai())
                for item in enumerate_legal_actions(after_snapshot, actor)
                if item.type == "dahai"
            )
        states.append(
            ResponseActionState(
                action=action,
                action_idx=action_to_idx(action),
                requires_post_discard=requires_post_discard(action),
                after_snapshot=after_snapshot,
                post_discard_actions=post_discard_actions,
            )
        )
    return tuple(states)


__all__ = [
    "ResponseActionState",
    "build_post_response_snapshot",
    "build_response_action_states",
    "is_response_action",
    "is_off_turn_response_action",
    "iter_response_actions",
    "iter_off_turn_response_actions",
    "requires_post_discard",
]
