from __future__ import annotations

from pathlib import Path

import gateway.tenhou_bot_client as tbc


class DummyBot:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.events = []
        self.reset_count = 0

    def reset(self) -> None:
        self.reset_count += 1

    def react(self, event: dict):
        self.events.append(event)
        if event.get("type") == "start_game":
            return None
        return {"type": "none", "actor": self.player_id}


def test_handle_hello_returns_gateway_handshake() -> None:
    client = tbc.GatewayBotClient(
        tbc.BotClientConfig(name="NoName", room="L2147", project_root=Path('.'))
    )

    assert client.handle_message({"type": "hello"}) == {
        "type": "hello",
        "protocol": "mjsonp",
        "protocol_version": 3,
        "name": "NoName",
        "room": "L2147_9",
    }


def test_start_game_creates_runtime_bot_and_defaults_to_none(monkeypatch, tmp_path) -> None:
    created = {}
    fake_ckpt = tmp_path / "best.pth"
    fake_ckpt.write_bytes(b"")

    def fake_create_runtime_bot(**kwargs):
        created.update(kwargs)
        return DummyBot(kwargs["player_id"])

    monkeypatch.setattr(tbc, "create_runtime_bot_for_gateway", fake_create_runtime_bot)
    client = tbc.GatewayBotClient(
        tbc.BotClientConfig(
            name="bot-b",
            room="2147_0",
            bot_name="keqingv2",
            project_root=Path('/tmp/project'),
            model_path=fake_ckpt,
        )
    )

    response = client.handle_message({"type": "start_game", "id": 2, "names": []})

    assert created["player_id"] == 2
    assert created["bot_name"] == "keqingv2"
    assert response == {"type": "none", "actor": 2}


def test_same_seat_start_game_resets_existing_bot(monkeypatch) -> None:
    bot = DummyBot(1)

    def fake_create_runtime_bot(**kwargs):
        return bot

    monkeypatch.setattr(tbc, "create_runtime_bot_for_gateway", fake_create_runtime_bot)
    client = tbc.GatewayBotClient(
        tbc.BotClientConfig(name="bot-c", room="2147_0", project_root=Path('.'))
    )

    client.handle_message({"type": "start_game", "id": 1, "names": []})
    client.handle_message({"type": "start_game", "id": 1, "names": []})

    assert bot.reset_count == 1


def test_hidden_opponent_tsumo_is_buffered_until_discard(monkeypatch, tmp_path) -> None:
    fake_ckpt = tmp_path / "best.pth"
    fake_ckpt.write_bytes(b"")
    bot = DummyBot(0)

    def fake_create_runtime_bot(**kwargs):
        return bot

    monkeypatch.setattr(tbc, "create_runtime_bot_for_gateway", fake_create_runtime_bot)
    client = tbc.GatewayBotClient(
        tbc.BotClientConfig(name="NoName", room="L2147", model_path=fake_ckpt)
    )

    client.handle_message({"type": "start_game", "id": 0, "names": []})
    bot.events.clear()

    assert client.handle_message({"type": "tsumo", "actor": 1, "pai": "?"}) == {"type": "none"}
    assert bot.events == []

    response = client.handle_message(
        {"type": "dahai", "actor": 1, "pai": "4m", "tsumogiri": False}
    )

    assert [event["type"] for event in bot.events] == ["tsumo", "dahai"]
    assert bot.events[0]["pai"] == "4m"
    assert response == {"type": "none", "actor": 0}


def test_public_opponent_meld_marks_skip_hand_update(monkeypatch, tmp_path) -> None:
    fake_ckpt = tmp_path / "best.pth"
    fake_ckpt.write_bytes(b"")
    bot = DummyBot(0)

    def fake_create_runtime_bot(**kwargs):
        return bot

    monkeypatch.setattr(tbc, "create_runtime_bot_for_gateway", fake_create_runtime_bot)
    client = tbc.GatewayBotClient(
        tbc.BotClientConfig(name="NoName", room="L2147", model_path=fake_ckpt)
    )

    client.handle_message({"type": "start_game", "id": 0, "names": []})
    bot.events.clear()
    client.handle_message(
        {"type": "chi", "actor": 1, "target": 0, "pai": "6m", "consumed": ["5m", "7m"]}
    )

    assert bot.events[-1]["skip_hand_update"] is True


def test_public_opponent_post_meld_discard_marks_skip_hand_update(monkeypatch, tmp_path) -> None:
    fake_ckpt = tmp_path / "best.pth"
    fake_ckpt.write_bytes(b"")
    bot = DummyBot(0)

    def fake_create_runtime_bot(**kwargs):
        return bot

    monkeypatch.setattr(tbc, "create_runtime_bot_for_gateway", fake_create_runtime_bot)
    client = tbc.GatewayBotClient(
        tbc.BotClientConfig(name="NoName", room="L2147", model_path=fake_ckpt)
    )

    client.handle_message({"type": "start_game", "id": 0, "names": []})
    bot.events.clear()
    client.handle_message(
        {"type": "pon", "actor": 1, "target": 0, "pai": "8p", "consumed": ["8p", "8p"]}
    )
    client.handle_message({"type": "dahai", "actor": 1, "pai": "6m", "tsumogiri": False})

    assert bot.events[-1]["type"] == "dahai"
    assert bot.events[-1]["skip_hand_update"] is True
