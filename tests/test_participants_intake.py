# -*- coding: utf-8 -*-
"""R10-D：外部别名 + 天凤统一摄入（preview → 身份解析 → 落账 + artifact + 防重）。"""
from __future__ import annotations

import pytest

from participants import aliases, intake, ledger, registry
from participants.schemas import AccountCreate, ExternalAliasCreate


@pytest.fixture(autouse=True)
def participants_root(tmp_path, monkeypatch):
    root = tmp_path / "participants"
    monkeypatch.setenv("KEQING_PARTICIPANT_DATA_ROOT", str(root))
    return root


FAKE_LOG_ID = "20260804gm-0009-2147-32af115e"


def _kyoku(round_index, scores, result):
    return [
        [round_index, 0, 0],  # 0 meta
        scores,               # 1 scores
        [],                   # 2 dora（空 = 无宝牌指示）
        [],                   # 3
        [], [], [],           # 4,5,6 seat0
        [], [], [],           # 7,8,9 seat1
        [], [], [],           # 10,11,12 seat2
        [], [], [],           # 13,14,15 seat3
        result,               # 16 result
    ]


FAKE_TENHOU6 = {
    "name": ["Nick", "NoName-1", "NoName-2", "FriendID"],
    "rule": {"aka": True},
    "log": [
        _kyoku(0, [25000, 25000, 25000, 25000], ["和了", [5000, -5000, 0, 0], [0, 1]]),
        _kyoku(1, [30000, 20000, 25000, 25000], ["流局", [0, 0, 0, 0]]),
        _kyoku(2, [30000, 20000, 25000, 25000], ["流局", [0, 0, 0, 0]]),
        _kyoku(3, [30000, 20000, 25000, 25000], ["流局", [0, 0, 0, 0]]),
    ],
}


@pytest.fixture
def fake_download(monkeypatch):
    monkeypatch.setattr(intake, "download_tenhou6", lambda log_id: FAKE_TENHOU6)


def test_parse_tenhou_url():
    parsed = intake.parse_tenhou_url(f"https://tenhou.net/3/?log={FAKE_LOG_ID}&tw=2")
    assert parsed["provider"] == "tenhou"
    assert parsed["log_id"] == FAKE_LOG_ID
    assert parsed["tw"] == 2
    with pytest.raises(ValueError):
        intake.parse_tenhou_url("https://example.com/foo")


def test_build_preview_does_not_persist(fake_download, participants_root):
    preview = intake.build_preview(f"https://tenhou.net/3/?log={FAKE_LOG_ID}")
    assert preview["raw_player_names"] == ["Nick", "NoName-1", "NoName-2", "FriendID"]
    assert preview["game_length"] == "tonpu"
    assert preview["final_scores"] == [30000, 20000, 25000, 25000]
    assert preview["hand_count"] == 4
    assert preview["data_completeness"] == "full_replay"
    assert preview["duplicate_match_id"] is None
    # 未落账：无 match，无 artifact
    assert ledger.list_matches().total == 0
    assert intake.read_replay_artifact(FAKE_LOG_ID) is None


def test_hand_summaries_extract_winners():
    events = intake.tenhou6_events(FAKE_TENHOU6)
    hands = intake.hand_summaries(events)
    assert len(hands) == 4
    assert hands[0]["winners"][0]["actor"] == 0
    assert hands[0]["winners"][0]["win_type"] == "ron"
    assert hands[1]["ryukyoku"] is not None


def test_resolve_and_create_match(fake_download, participants_root):
    registry.create_account(AccountCreate(account_id="nick@01", display_name="Nick", account_type="human"))
    resolutions = [
        {"seat": 0, "action": "assign", "account_id": "nick@01", "alias_scope": "global"},
        {"seat": 1, "action": "create", "display_name": "Bot 70k", "account_type": "managed_bot", "alias_scope": "session"},
        {"seat": 2, "action": "create", "display_name": "Bot V3", "account_type": "managed_bot", "alias_scope": "session"},
        {"seat": 3, "action": "create", "display_name": "Friend", "account_type": "human", "alias_scope": "global"},
    ]
    result = intake.resolve_and_create_match(log_id=FAKE_LOG_ID, resolutions=resolutions, session_id="s1")
    match = ledger.get_match(result["match_id"])
    assert match is not None
    assert match.provider == "tenhou"
    assert match.external_match_id == FAKE_LOG_ID
    assert match.data_completeness == "full_replay"
    assert match.raw_player_names == ["Nick", "NoName-1", "NoName-2", "FriendID"]
    assert match.resolution["0"]["account_id"] == "nick@01"
    # artifact 持久化
    artifact = intake.read_replay_artifact(FAKE_LOG_ID)
    assert artifact is not None
    assert artifact["has_events"] is True
    assert len(artifact["hands"]) == 4
    # 别名注册：global + session（session 别名只在同一会话内解析）
    nick_alias = aliases.resolve_candidates("tenhou", "Nick")
    assert any(a.account_id == "nick@01" and a.scope == "global" for a in nick_alias)
    session_alias = aliases.resolve_candidates("tenhou", "NoName-1", session_id="s1")
    assert any(a.scope == "session" for a in session_alias)
    assert aliases.resolve_candidates("tenhou", "NoName-1") == []
    # 重复导入被拒绝
    with pytest.raises(ValueError, match="已导入"):
        intake.resolve_and_create_match(log_id=FAKE_LOG_ID, resolutions=resolutions)
    assert ledger.list_matches().total == 1


def test_preview_auto_resolves_confirmed_global_alias(fake_download, participants_root):
    registry.create_account(AccountCreate(account_id="nick@01", display_name="Nick", account_type="human"))
    aliases.register_alias(
        ExternalAliasCreate(provider="tenhou", external_id="Nick", account_id="nick@01", scope="global")
    )
    preview = intake.build_preview(f"https://tenhou.net/3/?log={FAKE_LOG_ID}")
    assert preview["seats"][0]["auto_account_id"] == "nick@01"
    # 未知 NoName：绝不自动猜
    assert preview["seats"][1]["auto_account_id"] is None
    assert preview["seats"][1]["candidates"] == []


def test_alias_register_and_resolve(participants_root):
    registry.create_account(AccountCreate(account_id="nick@01", display_name="Nick", account_type="human"))
    registry.create_account(AccountCreate(account_id="70k@01", display_name="70k", account_type="managed_bot"))
    aliases.register_alias(
        ExternalAliasCreate(provider="tenhou", external_id="keqing1", account_id="nick@01", scope="global")
    )
    aliases.register_alias(
        ExternalAliasCreate(provider="tenhou", external_id="NoName-1", account_id="70k@01", scope="session", session_id="s1")
    )
    # session-scoped 绑定优先
    cands = aliases.resolve_candidates("tenhou", "NoName-1", session_id="s1")
    assert cands and cands[0].account_id == "70k@01"
    # 无 session 时不命中 session 别名
    cands_none = aliases.resolve_candidates("tenhou", "NoName-1")
    assert all(a.scope != "session" for a in cands_none)
    # global 稳定别名
    cands2 = aliases.resolve_candidates("tenhou", "keqing1")
    assert cands2 and cands2[0].account_id == "nick@01"
