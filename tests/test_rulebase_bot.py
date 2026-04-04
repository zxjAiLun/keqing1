from __future__ import annotations

from inference.contracts import DecisionContext
from inference.rulebase_bot import RulebaseBot


def _ctx(*, legal_actions: list[dict], runtime_snap: dict | None = None, model_snap: dict | None = None) -> DecisionContext:
    snap = {
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "hand": ["1m"] * 14,
        "discards": [[], [], [], []],
        "melds": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "actor_to_move": 0,
        "last_discard": None,
        "last_tsumo": ["5m", None, None, None],
        "shanten": 0,
        "waits_count": 4,
        "waits_tiles": [False] * 34,
    }
    runtime = {**snap, **(runtime_snap or {})}
    model = {**runtime, **(model_snap or {})}
    return DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "5m"},
        runtime_snap=runtime,
        model_snap=model,
        legal_actions=legal_actions,
    )


def test_rulebase_bot_prioritizes_hora(monkeypatch):
    bot = RulebaseBot(player_id=0)
    ctx = _ctx(
        legal_actions=[
            {"type": "hora", "actor": 0, "pai": "5m", "target": 0},
            {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True},
        ]
    )
    monkeypatch.setattr(bot._context_builder, "build", lambda state, actor, event: ctx)

    chosen = bot.react({"type": "tsumo", "actor": 0, "pai": "5m"})
    assert chosen is not None
    assert chosen["type"] == "hora"


def test_rulebase_bot_prioritizes_reach_over_discard(monkeypatch):
    bot = RulebaseBot(player_id=0)
    ctx = _ctx(
        legal_actions=[
            {"type": "reach", "actor": 0},
            {"type": "dahai", "actor": 0, "pai": "7p", "tsumogiri": True},
            {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": False},
        ]
    )
    monkeypatch.setattr(bot._context_builder, "build", lambda state, actor, event: ctx)

    chosen = bot.react({"type": "tsumo", "actor": 0, "pai": "7p"})
    assert chosen is not None
    assert chosen["type"] == "reach"


def test_rulebase_bot_tsumogiris_after_riichi(monkeypatch):
    bot = RulebaseBot(player_id=0)
    ctx = _ctx(
        legal_actions=[
            {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True},
            {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
        ],
        runtime_snap={
            "reached": [True, False, False, False],
            "last_tsumo": ["5m", None, None, None],
        },
    )
    monkeypatch.setattr(bot._context_builder, "build", lambda state, actor, event: ctx)

    chosen = bot.react({"type": "tsumo", "actor": 0, "pai": "5m"})
    assert chosen is not None
    assert chosen["type"] == "dahai"
    assert chosen["pai"] == "5m"
    assert chosen["tsumogiri"] is True

