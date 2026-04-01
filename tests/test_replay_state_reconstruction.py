import pytest

from mahjong_env.replay_normalizer import normalize_replay_events
from mahjong_env.replay import IllegalLabelActionError, build_supervised_samples
from mahjong_env.state import GameState, PlayerState, apply_event
from mahjong_env.types import ActionSpec


def test_build_supervised_samples_raises_on_illegal_label_action(monkeypatch):
    events = [
        {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
                ["1m"] * 13,
                ["2m"] * 13,
                ["3m"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "5p"},
        {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
    ]

    monkeypatch.setattr(
        "mahjong_env.replay.enumerate_legal_action_specs",
        lambda snap, actor: [ActionSpec(type="none")],
    )

    with pytest.raises(IllegalLabelActionError):
        build_supervised_samples(events)


def test_build_supervised_samples_allows_double_ron_from_same_discard(monkeypatch):
    events = [
        {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m"] * 13,
                ["2m"] * 13,
                ["3m"] * 13,
                ["4m"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 1, "pai": "5m"},
        {"type": "dahai", "actor": 1, "pai": "5m", "tsumogiri": True},
        {"type": "hora", "actor": 2, "target": 1, "deltas": [0, -8000, 8000, 0]},
        {"type": "hora", "actor": 3, "target": 1, "deltas": [0, -3900, 0, 3900]},
        {"type": "end_kyoku"},
    ]

    def fake_legal_specs(snap, actor):
        hand = snap.get("hand", [])
        if actor == 1 and len(hand) % 3 == 2:
            return [ActionSpec(type="dahai", actor=actor, pai="5m", tsumogiri=True)]
        if snap.get("last_discard") is not None and actor in {2, 3}:
            return [
                ActionSpec(
                    type="hora",
                    actor=actor,
                    target=snap["last_discard"]["actor"],
                    pai=snap["last_discard"]["pai"],
                )
            ]
        return [ActionSpec(type="none")]

    monkeypatch.setattr("mahjong_env.replay.enumerate_legal_action_specs", fake_legal_specs)

    samples = build_supervised_samples(events)
    hora_samples = [s for s in samples if s.label_action["type"] == "hora"]
    assert len(hora_samples) == 2


def test_build_supervised_samples_auto_accepts_legacy_kakan_without_explicit_kakan_accepted(monkeypatch):
    events = [
        {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m"] * 13,
                ["W", "W", "W", "5m", "5p", "5p", "6m", "7m", "4p", "4p", "7s", "8s", "9s"],
                ["W"] + ["2m"] * 12,
                ["3m"] * 13,
            ],
        },
        {"type": "dahai", "actor": 2, "pai": "W", "tsumogiri": False},
        {"type": "pon", "actor": 1, "target": 2, "pai": "W", "consumed": ["W", "W"]},
        {"type": "dahai", "actor": 1, "pai": "5m", "tsumogiri": False},
        {"type": "tsumo", "actor": 1, "pai": "W"},
        {"type": "kakan", "actor": 1, "pai": "W", "consumed": ["W", "W", "W"]},
        {"type": "dora", "dora_marker": "9s"},
        {"type": "tsumo", "actor": 3, "pai": "4p"},
        {"type": "dahai", "actor": 3, "pai": "4p", "tsumogiri": True},
        {"type": "hora", "actor": 1, "target": 3, "deltas": [0, 3200, 0, -3200], "ura_markers": []},
        {"type": "end_kyoku"},
    ]

    def fake_legal_specs(snap, actor):
        actor_melds = (snap.get("melds") or [[], [], [], []])[actor]
        if snap.get("last_discard") is not None and actor == 1:
            if snap["last_discard"]["pai"] == "W":
                return [
                    ActionSpec(
                        type="pon",
                        actor=1,
                        target=snap["last_discard"]["actor"],
                        pai="W",
                        consumed=("W", "W"),
                    )
                ]
            return [
                ActionSpec(
                    type="hora",
                    actor=1,
                    target=snap["last_discard"]["actor"],
                    pai=snap["last_discard"]["pai"],
                )
            ]
        if (
            actor == 1
            and any(m.get("type") == "pon" and m.get("pai") == "W" for m in actor_melds)
            and snap.get("hand", []).count("W") >= 2
        ):
            return [ActionSpec(type="kakan", actor=1, pai="W", consumed=("W", "W", "W"))]
        if snap.get("tsumo_pai") is not None:
            return [ActionSpec(type="dahai", actor=actor, pai=snap["tsumo_pai"], tsumogiri=True)]
        if actor == 2 and "W" in snap.get("hand", []):
            return [ActionSpec(type="dahai", actor=2, pai="W", tsumogiri=False)]
        if actor == 1 and "5m" in snap.get("hand", []):
            return [ActionSpec(type="dahai", actor=1, pai="5m", tsumogiri=False)]
        if actor == 3 and "4p" in snap.get("hand", []):
            return [ActionSpec(type="dahai", actor=3, pai="4p", tsumogiri=True)]
        return [ActionSpec(type="none")]

    monkeypatch.setattr("mahjong_env.replay.enumerate_legal_action_specs", fake_legal_specs)

    samples = build_supervised_samples(events)
    hora_samples = [s for s in samples if s.label_action["type"] == "hora"]
    assert len(hora_samples) == 1


def test_normalize_replay_events_injects_legacy_kakan_accepted():
    events = [
        {"type": "start_game", "names": ["p0", "p1", "p2", "p3"]},
        {"type": "kakan", "actor": 1, "pai": "W", "consumed": ["W", "W", "W"]},
        {"type": "dora", "dora_marker": "9s"},
        {"type": "tsumo", "actor": 3, "pai": "4p"},
    ]

    normalized = normalize_replay_events(events)
    assert normalized[1]["type"] == "kakan"
    assert normalized[2]["type"] == "kakan_accepted"
    assert normalized[3]["type"] == "dora"


def test_apply_event_kakan_accepted_upgrades_existing_pon_with_aka_family():
    player = PlayerState()
    player.hand.update(["5pr"])
    player.melds = [
        {"type": "pon", "pai": "5p", "pai_raw": "5p", "consumed": ["5p", "5p"], "target": 0},
    ]

    gs = GameState()
    gs.players = [PlayerState(), PlayerState(), player, PlayerState()]
    gs.last_kakan = {
        "actor": 2,
        "pai": "5pr",
        "pai_raw": "5pr",
        "consumed": ["5p", "5p", "5p"],
        "target": 0,
    }

    apply_event(
        gs,
        {
            "type": "kakan_accepted",
            "actor": 2,
            "pai": "5pr",
            "pai_raw": "5pr",
            "consumed": ["5p", "5p", "5p"],
            "target": 0,
        },
    )

    melds = gs.players[2].melds
    assert len(melds) == 1
    assert melds[0]["type"] == "kakan"
    assert sorted(melds[0]["consumed"]) == ["5p", "5p", "5p", "5pr"]
