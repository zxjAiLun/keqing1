from mahjong_env.scoring import HoraResult
from replay.server import _normalize_replay_events


def test_normalize_replay_events_recomputes_stale_hora_with_ippatsu(monkeypatch):
    def fake_score_hora(state, *, actor, target, pai, is_tsumo, ura_dora_markers=None, **kwargs):
        assert actor == 1
        assert target == 0
        assert pai == "2m"
        assert is_tsumo is False
        assert state.players[1].ippatsu_eligible is True
        return HoraResult(
            han=2,
            fu=30,
            yaku=["Riichi", "Ippatsu"],
            yaku_details=[
                {"key": "Riichi", "name": "Riichi", "han": 1},
                {"key": "Ippatsu", "name": "Ippatsu", "han": 1},
            ],
            is_open_hand=False,
            cost={
                "main": 2000,
                "main_bonus": 0,
                "additional": 0,
                "additional_bonus": 0,
                "kyoutaku_bonus": 1000,
                "total": 3000,
                "yaku_level": "",
            },
            deltas=[-3000, 3000, 0, 0],
        )

    monkeypatch.setattr("mahjong_env.scoring.score_hora", fake_score_hora)

    events = [
        {"type": "start_game", "names": ["P0", "P1", "P2", "P3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "names": ["P0", "P1", "P2", "P3"],
            "dora_marker": "1m",
            "tehais": [
                ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "1s", "1s", "2m", "9s"],
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "2p", "2p", "2s", "3s", "9m"],
                ["E"] * 13,
                ["S"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 1, "pai": "9p", "rinshan": False},
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "9m", "tsumogiri": False},
        {"type": "reach_accepted", "actor": 1, "scores": [25000, 24000, 25000, 25000], "kyotaku": 1},
        {"type": "tsumo", "actor": 2, "pai": "W", "rinshan": False},
        {"type": "dahai", "actor": 2, "pai": "E", "tsumogiri": False},
        {"type": "tsumo", "actor": 3, "pai": "N", "rinshan": False},
        {"type": "dahai", "actor": 3, "pai": "S", "tsumogiri": False},
        {"type": "tsumo", "actor": 0, "pai": "3m", "rinshan": False},
        {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": False},
        {
            "type": "hora",
            "actor": 1,
            "target": 0,
            "pai": "2m",
            "is_tsumo": False,
            "han": 1,
            "fu": 30,
            "yaku": ["Riichi"],
            "yaku_details": [{"key": "Riichi", "name": "Riichi", "han": 1}],
            "deltas": [-2000, 2000, 0, 0],
            "scores": [23000, 26000, 25000, 25000],
            "cost": {"main": 1000, "kyoutaku_bonus": 1000, "total": 2000},
            "honba": 0,
            "kyotaku": 1,
            "ura_dora_markers": [],
        },
    ]

    normalized = _normalize_replay_events(events)
    hora = normalized[-1]

    assert "Ippatsu" in hora["yaku"]
    assert any(detail["key"] == "Ippatsu" for detail in hora["yaku_details"])
    assert hora["han"] >= 2
