from __future__ import annotations

from mahjong_env.scoring import HoraResult
from gateway.battle import BattleConfig, BattleManager


def _make_room() -> tuple[BattleManager, object]:
    manager = BattleManager()
    room = manager.create_room(
        BattleConfig(
            player_count=4,
            players=[{"id": i, "name": f"P{i}", "type": "bot"} for i in range(4)],
        )
    )
    room.state.bakaze = "E"
    room.state.kyoku = 1
    room.state.honba = 0
    room.state.oya = 0
    room.state.scores = [25000, 25000, 25000, 25000]
    room.state.kyotaku = 0
    return manager, room


def test_start_kyoku_reserves_dead_wall_and_indicators():
    manager, room = _make_room()

    manager.start_kyoku(room, seed=7)

    assert len(room.dead_wall) == 14
    assert len(room.rinshan_tiles) == 4
    assert len(room.dora_indicator_tiles) >= 1
    assert len(room.ura_indicator_tiles) >= 1
    assert room.state.dora_markers == [room.dora_indicator_tiles[0]]
    assert sum(sum(player.hand.values()) for player in room.state.players) == 52
    assert room.remaining_wall() == 70


def test_ryukyoku_tenpai_payment_and_dealer_renchan():
    manager, room = _make_room()
    room.state.oya = 0
    room.state.honba = 2
    room.state.scores = [25000, 25000, 25000, 25000]

    manager.ryukyoku(room, tenpai=[0, 1])

    assert room.state.scores == [26500, 26500, 23500, 23500]
    assert room.state.honba == 3
    assert room.state.oya == 0
    assert room.events[-2]["type"] == "ryukyoku"
    assert room.events[-2]["tenpai_players"] == [0, 1]


def test_ryukyoku_noten_dealer_passes_oya():
    manager, room = _make_room()
    room.state.oya = 0

    manager.ryukyoku(room, tenpai=[1, 2])

    assert room.state.oya == 1
    assert room.state.scores == [23500, 26500, 26500, 23500]


def test_hora_clears_kyotaku_and_advances_oya_on_non_dealer_win(monkeypatch):
    manager, room = _make_room()
    room.state.oya = 0
    room.state.honba = 2
    room.state.kyotaku = 3
    room.state.scores = [25000, 25000, 25000, 25000]

    def fake_score_hora(*args, **kwargs):
        return HoraResult(
            han=3,
            fu=40,
            yaku=["Riichi"],
            yaku_details=[{"key": "Riichi", "name": "Riichi", "han": 1}],
            is_open_hand=False,
            cost={"main": 7700, "kyoutaku_bonus": 3000, "total": 10700},
            deltas=[-10700, 10700, 0, 0],
        )

    monkeypatch.setattr("gateway.battle.score_hora", fake_score_hora)

    result = manager.hora(room, actor=1, target=0, pai="3m", is_tsumo=False)

    assert result["honba"] == 2
    assert result["kyotaku"] == 3
    assert room.state.kyotaku == 0
    assert room.state.honba == 0
    assert room.state.oya == 1
    assert room.state.scores == [14300, 35700, 25000, 25000]


def test_next_kyoku_enters_west_round_when_south_four_not_enough_points():
    manager, room = _make_room()
    room.state.bakaze = "S"
    room.state.kyoku = 4
    room.state.oya = 0
    room.state.scores = [29500, 28000, 22000, 20500]

    advanced = manager.next_kyoku(room)

    assert advanced is True
    assert room.state.bakaze == "W"
    assert room.state.kyoku == 1


def test_is_game_ended_allows_agari_yame_for_dealer_top():
    manager, room = _make_room()
    room.state.bakaze = "S"
    room.state.kyoku = 4
    room.state.oya = 3
    room.state.scores = [24000, 25000, 18000, 33000]

    assert manager.is_game_ended(room) is True
