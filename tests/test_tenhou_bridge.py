from __future__ import annotations

import json

import pytest

from gateway.tenhou_bridge import (
    TENHOU_ROOM_PATTERN,
    TenhouBridgeConfig,
    is_valid_tenhou_room,
    normalize_tenhou_room,
)


def test_room_validation_accepts_numeric_and_lobby_rooms() -> None:
    assert is_valid_tenhou_room("2147_0")
    assert is_valid_tenhou_room("2147_9")
    assert is_valid_tenhou_room("2147_41")
    assert is_valid_tenhou_room("L1000_0")
    assert is_valid_tenhou_room("l1000_9")
    assert not is_valid_tenhou_room("2147")
    assert not is_valid_tenhou_room("L1000")
    assert not is_valid_tenhou_room("ROOM_0")
    assert is_valid_tenhou_room("2147_2")
    assert TENHOU_ROOM_PATTERN.match("L1000_0")


def test_normalize_tenhou_room_adds_default_suffix() -> None:
    assert normalize_tenhou_room("L2147") == "L2147_9"
    assert normalize_tenhou_room("2147") == "2147_9"
    assert normalize_tenhou_room("l2147_9") == "L2147_9"
    assert normalize_tenhou_room("L2147", default_suffix="1") == "L2147_1"


def test_build_helo_merges_extra_fields() -> None:
    cfg = TenhouBridgeConfig(sex="F", helo_extra={"tid": "abc", "auth": 1})

    assert cfg.build_helo(name="bot") == {
        "tag": "HELO",
        "name": "bot",
        "sx": "F",
        "tid": "abc",
        "auth": 1,
    }


def test_from_env_rejects_non_object_helo_json(monkeypatch) -> None:
    monkeypatch.setenv("TENHOU_HELO_JSON", json.dumps(["bad"]))

    with pytest.raises(ValueError):
        TenhouBridgeConfig.from_env()
