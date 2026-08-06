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


# ---------------------------------------------------------------------------
# R10-D Repair：match alias 隔离 / 原子事务 / session 模型信息 / 时间规则
# ---------------------------------------------------------------------------

LOG_B = "20260807gm-0009-2147-32af115e"


def _nick_resolutions(bot_a_name="Bot 70k"):
    return [
        {"seat": 0, "action": "assign", "account_id": "nick@01", "alias_scope": "global"},
        {"seat": 1, "action": "create", "display_name": bot_a_name, "account_type": "managed_bot", "alias_scope": "match"},
        {"seat": 2, "action": "create", "display_name": "Bot V3", "account_type": "managed_bot", "alias_scope": "match"},
        {"seat": 3, "action": "create", "display_name": "Friend", "account_type": "human", "alias_scope": "match"},
    ]


def test_match_alias_isolated_between_matches(fake_download, participants_root):
    """P1-1：match scope 别名只作用于同一 external_match_id，不串到其他牌谱。"""
    registry.create_account(AccountCreate(account_id="nick@01", display_name="Nick", account_type="human"))
    intake.resolve_and_create_match(log_id=FAKE_LOG_ID, resolutions=_nick_resolutions())
    # 同一 NoName 的 match 别名只属于 FAKE_LOG_ID
    preview_b = intake.build_preview(f"https://tenhou.net/3/?log={LOG_B}")
    no_name_candidates = preview_b["seats"][1]["candidates"]
    assert all(c.get("external_match_id") != FAKE_LOG_ID or c["scope"] != "match" for c in no_name_candidates)
    assert preview_b["seats"][1]["auto_account_id"] is None
    # 导入 B：同名 NoName 指向另一账号，不覆盖 A 的别名 → 两条独立记录
    intake.resolve_and_create_match(log_id=LOG_B, resolutions=_nick_resolutions(bot_a_name="Bot V4"))
    match_aliases = [a for a in aliases.list_aliases() if a.external_id == "NoName-1" and a.scope == "match"]
    assert {a.external_match_id for a in match_aliases} == {FAKE_LOG_ID, LOG_B}
    assert len(match_aliases) == 2


def test_confirm_failure_mid_commit_is_recoverable(fake_download, participants_root, monkeypatch):
    """P1-2：revision append 故障 → 无半成品；pending 恢复后恰好一场。"""
    registry.create_account(AccountCreate(account_id="nick@01", display_name="Nick", account_type="human"))

    original_append = ledger._append_revision

    def failing_append(row, *, fsync=False):
        raise RuntimeError("injected revision append failure")

    monkeypatch.setattr(ledger, "_append_revision", failing_append)
    with pytest.raises(RuntimeError):
        intake.resolve_and_create_match(log_id=FAKE_LOG_ID, resolutions=_nick_resolutions(), session_id="s1")
    # 仅恢复 _append_revision（不能 undo()，否则 fixture 的 env 也被回滚）
    monkeypatch.setattr(ledger, "_append_revision", original_append)

    # 故障后：pending 已写、match/revision 未提交 → 恢复（public 入口模拟重启）
    assert ledger.pending_transaction_path().exists()
    assert ledger.recover_pending_transaction() is True
    assert not ledger.pending_transaction_path().exists()
    match = ledger.find_match_by_external("tenhou", FAKE_LOG_ID)
    assert match is not None
    assert ledger.list_matches().total == 1
    assert len(ledger.list_revisions(match.match_id)) == 1
    artifact = intake.read_replay_artifact(FAKE_LOG_ID)
    assert artifact is not None
    assert artifact["match_id"] == match.match_id
    # 账号/别名都已被事务引用（无孤立残骸：全部账号都被 match 引用）
    for seat in match.seats:
        assert registry.get_account(seat.account_id) is not None


def test_concurrent_confirm_creates_single_match(fake_download, participants_root):
    """P1-2：并发 confirm → 恰好一场 match / 一个 revision / 一个 artifact。"""
    import threading

    registry.create_account(AccountCreate(account_id="nick@01", display_name="Nick", account_type="human"))
    results: list[str] = []
    errors: list[str] = []

    def run():
        try:
            r = intake.resolve_and_create_match(log_id=FAKE_LOG_ID, resolutions=_nick_resolutions(), session_id="s1")
            results.append(r["match_id"])
        except ValueError as exc:
            errors.append(str(exc))

    threads = [threading.Thread(target=run) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 1
    assert len(errors) == 1
    assert ledger.list_matches().total == 1
    match = ledger.find_match_by_external("tenhou", FAKE_LOG_ID)
    assert match is not None
    assert len(ledger.list_revisions(match.match_id)) == 1


def test_session_alias_model_info_enters_match(fake_download, participants_root):
    """P1-3：session 别名的 model_identity/artifact 进入 MatchSeat / resolution / revision / summary。"""
    from participants.schemas import ModelIdentityCreate

    registry.create_account(AccountCreate(account_id="70k@01", display_name="70k", account_type="managed_bot"))
    registry.create_account(AccountCreate(account_id="nick@01", display_name="Nick", account_type="human"))
    registry.create_model_identity(
        ModelIdentityCreate(model_identity_id="70k", label="70k", kind="local_model", artifact_path="ckpt.pth")
    )
    identity = registry.get_model_identity("70k")
    artifact = identity.artifacts[0]
    alias = aliases.register_alias(
        ExternalAliasCreate(
            provider="tenhou", external_id="NoName-1", account_id="70k@01",
            model_identity_id="70k", model_artifact_id=artifact.model_artifact_id,
            scope="session", session_id="s1",
        )
    )
    resolutions = [
        {"seat": 0, "action": "assign", "account_id": "nick@01", "alias_scope": "none"},
        {"seat": 1, "action": "assign", "alias_id": alias.alias_id, "alias_scope": "none"},
        {"seat": 2, "action": "create", "display_name": "Bot V3", "account_type": "managed_bot", "alias_scope": "none"},
        {"seat": 3, "action": "create", "display_name": "Friend", "account_type": "human", "alias_scope": "none"},
    ]
    result = intake.resolve_and_create_match(log_id=FAKE_LOG_ID, resolutions=resolutions, session_id="s1")
    match = ledger.get_match(result["match_id"])
    seat1 = next(s for s in match.seats if s.seat == 1)
    assert seat1.account_id == "70k@01"
    assert seat1.model_identity_id == "70k"
    assert seat1.model_artifact_id == artifact.model_artifact_id
    assert match.resolution["1"]["model_identity_id"] == "70k"
    # revision after 快照含模型信息
    rev = ledger.list_revisions(match.match_id)[0]
    assert rev["after"]["seats"][1]["model_identity_id"] == "70k"
    # replay summary 含 resolution
    replay = intake.read_replay_artifact(FAKE_LOG_ID)
    assert replay["resolution"]["1"]["model_identity_id"] == "70k"
    # 身份/产物归属校验：错误产物被拒（不落账）
    alias_bad = aliases.register_alias(
        ExternalAliasCreate(
            provider="tenhou", external_id="NoName-2", account_id="70k@01",
            model_identity_id="70k", model_artifact_id="art_does_not_exist",
            scope="session", session_id="s1",
        )
    )
    bad_resolutions = [
        {"seat": 0, "action": "assign", "account_id": "nick@01", "alias_scope": "none"},
        {"seat": 1, "action": "assign", "alias_id": alias_bad.alias_id, "alias_scope": "none"},
        {"seat": 2, "action": "create", "display_name": "Bot V3", "account_type": "managed_bot", "alias_scope": "none"},
        {"seat": 3, "action": "create", "display_name": "Friend", "account_type": "human", "alias_scope": "none"},
    ]
    with pytest.raises(ValueError, match="不属于"):
        intake.resolve_and_create_match(log_id=LOG_B, resolutions=bad_resolutions, session_id="s1")


def test_occurred_at_keeps_hour():
    assert intake._occurred_at_from_log_id("2026080700gm-0009-2147-32af115e") == "2026-08-07T00:00:00+08:00"
    assert intake._occurred_at_from_log_id("2026080418gm-0009-2147-32af115e") == "2026-08-04T18:00:00+08:00"
    assert intake._occurred_at_from_log_id("20260804gm-0009-2147-32af115e") == "2026-08-04T00:00:00+08:00"


def test_game_length_prefers_rule_metadata(fake_download):
    assert intake._game_length_from_rule({"tonnan": 1}) == "hanchan"
    assert intake._game_length_from_rule({"tonnan": 0}) == "tonpu"
    assert intake._game_length_from_rule({}) is None
    # 无规则元数据时按局数/风向回退
    events = intake.tenhou6_events(FAKE_TENHOU6)
    hands = intake.hand_summaries(events)
    assert intake._game_length_from_hands(hands) == "tonpu"
