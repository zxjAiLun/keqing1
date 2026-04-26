from __future__ import annotations

from mahjong_env.state import GameState, PlayerState
from xmodel1.call_response import build_response_action_states


def _chi_response_snapshot() -> tuple[dict, int]:
    player0 = PlayerState()
    player0.hand.update(["2m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p", "5p", "6p"])
    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [PlayerState(), PlayerState(), player0, PlayerState()]
    gs.last_discard = {"actor": 1, "pai": "3m", "pai_raw": "3m"}
    return gs.snapshot(actor=2), 2


def _pon_response_snapshot() -> tuple[dict, int]:
    player = PlayerState()
    player.hand.update(["1p", "1p", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1s", "2s", "3s"])
    gs = GameState()
    gs.bakaze = "E"
    gs.kyoku = 1
    gs.honba = 0
    gs.players = [PlayerState(), player, PlayerState(), PlayerState()]
    gs.last_discard = {"actor": 0, "pai": "1p", "pai_raw": "1p"}
    return gs.snapshot(actor=1), 1


def test_build_response_action_states_reconstructs_post_chi_discards() -> None:
    snap, actor = _chi_response_snapshot()
    states = build_response_action_states(snap, actor)

    chi_states = [item for item in states if item.action["type"] == "chi"]
    none_states = [item for item in states if item.action["type"] == "none"]

    assert chi_states
    assert len(none_states) == 1
    for item in chi_states:
        assert item.requires_post_discard is True
        assert item.after_snapshot is not None
        assert item.after_snapshot["last_discard"] is None
        assert item.post_discard_actions
        assert all(action["type"] == "dahai" for action in item.post_discard_actions)

    assert none_states[0].requires_post_discard is False
    assert none_states[0].after_snapshot is None
    assert none_states[0].post_discard_actions == ()


def test_build_response_action_states_reconstructs_post_pon_discards() -> None:
    snap, actor = _pon_response_snapshot()
    states = build_response_action_states(snap, actor)

    pon_state = next(item for item in states if item.action["type"] == "pon")

    assert pon_state.requires_post_discard is True
    assert pon_state.after_snapshot is not None
    assert pon_state.after_snapshot["last_discard"] is None
    assert pon_state.post_discard_actions
    assert all(action["type"] == "dahai" for action in pon_state.post_discard_actions)


def test_build_response_action_states_keeps_hora_without_post_discard() -> None:
    snap, actor = _chi_response_snapshot()
    legal_actions = [
        {"type": "hora", "actor": actor, "target": 1, "pai": "3m"},
        {"type": "none"},
    ]

    states = build_response_action_states(snap, actor, legal_actions)

    assert [item.action["type"] for item in states] == ["hora", "none"]
    for item in states:
        assert item.requires_post_discard is False
        assert item.after_snapshot is None
        assert item.post_discard_actions == ()


def test_build_response_action_states_reconstructs_reach_followup_discards() -> None:
    snap, actor = _chi_response_snapshot()
    snap["hand"] = ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"]
    snap["last_discard"] = None
    snap["tsumo_pai"] = "4p"
    legal_actions = [
        {"type": "reach", "actor": actor, "pai": "4p"},
        {"type": "dahai", "actor": actor, "pai": "4p", "tsumogiri": True},
    ]

    states = build_response_action_states(snap, actor, legal_actions)

    reach_state = next(item for item in states if item.action["type"] == "reach")
    assert reach_state.requires_post_discard is True
    assert reach_state.after_snapshot is not None
    assert reach_state.post_discard_actions
    assert all(action["type"] == "dahai" for action in reach_state.post_discard_actions)
