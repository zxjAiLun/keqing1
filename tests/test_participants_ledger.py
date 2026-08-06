# -*- coding: utf-8 -*-
"""participants ledger：对局创建/修订/作废、排名计算、revision 审计。"""
from __future__ import annotations

import json
import pytest

from mahjong_env.final_rank import final_ranks

from participants import ledger, registry
from participants.schemas import AccountCreate, MatchCreate, MatchRevise, MatchSeat, MatchVoid
from participants.ledger import ValidationError


@pytest.fixture(autouse=True)
def participants_root(tmp_path, monkeypatch):
    root = tmp_path / "participants"
    monkeypatch.setenv("KEQING_PARTICIPANT_DATA_ROOT", str(root))
    return root


@pytest.fixture
def four_accounts():
    registry.create_account(AccountCreate(account_id="nick@01", display_name="Nick", account_type="human"))
    registry.create_account(AccountCreate(account_id="friend@01", display_name="Friend", account_type="human", default_controller="manual_only"))
    registry.create_account(AccountCreate(account_id="70k@01", display_name="70k", account_type="managed_bot"))
    registry.create_account(AccountCreate(account_id="mortal41b", display_name="Mortal 4.1b", account_type="external_bot"))
    return ["nick@01", "friend@01", "70k@01", "mortal41b"]


def _create_payload(scores, account_ids):
    return MatchCreate(
        occurred_at="2026-08-06T12:00:00+08:00",
        game_length="hanchan",
        seats=[MatchSeat(seat=i, account_id=account_ids[i]) for i in range(4)],
        final_scores=scores,
    )


def test_create_match_computes_ranks(four_accounts):
    match = ledger.create_match(_create_payload([42300, 28100, 19400, 10200], four_accounts), registry)
    assert list(match.ranks) == list(final_ranks([42300, 28100, 19400, 10200], initial_oya=0))
    assert match.revision == 1
    assert match.status == "active"
    assert len(match.seats) == 4


def test_revision_increments_and_audit_appended(four_accounts):
    match = ledger.create_match(_create_payload([25000] * 4, four_accounts), registry)
    revised = ledger.revise_match(
        match.match_id,
        MatchRevise(final_scores=[30000, 25000, 25000, 20000]),
        registry,
    )
    assert revised.revision == 2
    assert revised.final_scores == [30000, 25000, 25000, 20000]
    # matches.jsonl 当前态 = 修订后（JSONL 每行一局）
    with open(ledger._matches_path(), encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    assert rows[0]["revision"] == 2
    # 审计日志有 2 条（create + revise）
    revisions = ledger.list_revisions(match.match_id)
    assert len(revisions) == 2
    assert revisions[-1]["action"] == "revise"
    assert revisions[-1]["before"]["revision"] == 1
    assert revisions[-1]["after"]["revision"] == 2


def test_void_match(four_accounts):
    match = ledger.create_match(_create_payload([25000] * 4, four_accounts), registry)
    voided = ledger.void_match(match.match_id, MatchVoid(reason="中途结束"))
    assert voided.status == "void"
    assert voided.void_reason == "中途结束"
    assert voided.revision == 2
    revisions = ledger.list_revisions(match.match_id)
    assert revisions[-1]["action"] == "void"
    # 已作废不能再修订
    with pytest.raises(ValueError, match="作废"):
        ledger.revise_match(match.match_id, MatchRevise(final_scores=[1, 2, 3, 4]), registry)


def test_list_matches_filters(four_accounts):
    ledger.create_match(_create_payload([25000] * 4, four_accounts), registry)
    ledger.create_match(_create_payload([25000] * 4, four_accounts), registry)
    all_matches = ledger.list_matches()
    assert all_matches.total == 2
    by_account = ledger.list_matches(account_id="nick@01")
    assert by_account.total == 2
    # 作废一局后 status 过滤
    target = all_matches.matches[0]
    ledger.void_match(target.match_id, MatchVoid(reason="x"))
    active = ledger.list_matches(status="active")
    assert active.total == 1
    voided = ledger.list_matches(status="void")
    assert voided.total == 1


def test_round_trip_persists(four_accounts, participants_root):
    ledger.create_match(_create_payload([42300, 28100, 19400, 10200], four_accounts), registry)
    # 重新加载（新进程语义：直接再读盘）
    rows = ledger.list_matches()
    assert rows.matches[0].final_scores == [42300, 28100, 19400, 10200]
    assert (participants_root / "matches.jsonl").exists()
    assert (participants_root / "match_revisions.jsonl").exists()


def test_force_save_bypasses_total_mismatch(four_accounts):
    payload = _create_payload([42300, 28100, 19400, 10201], four_accounts)  # 总分 100001
    payload.force = True
    payload.reason = "外部结算"
    match = ledger.create_match(payload, registry)
    assert match.final_scores == [42300, 28100, 19400, 10201]
    revisions = ledger.list_revisions(match.match_id)
    assert revisions[-1]["force"] is True
    assert revisions[-1]["reason"] == "外部结算"
    # force+reason 放行 score_total_mismatch：passed=True，但 issue 仍在审计中
    assert revisions[-1]["validation"]["passed"] is True
    assert "score_total_mismatch" in {i["code"] for i in revisions[-1]["validation"]["issues"]}


# ---------------------------------------------------------------------------
# P1-3：pending transaction 恢复（revision append / matches replace / 清理三处故障注入）
# ---------------------------------------------------------------------------

def _build_rev2_tx(match):
    """构造 rev-2 修订状态 + revision 行。"""
    rev2 = match.model_copy(deep=True, update={
        "final_scores": [30000, 25000, 25000, 20000],
        "revision": 2,
        "latest_revision_id": f"rev_{match.match_id}_2",
    })
    revision_row = {
        "schema": "keqing.participant.match_revision.v1",
        "revision_id": rev2.latest_revision_id,
        "match_id": match.match_id,
        "revision": 2,
        "action": "revise",
        "created_at": rev2.updated_at,
        "by": "manual-entry",
        "force": False,
        "reason": None,
        "validation": {"passed": True, "issues": []},
        "before": match.model_dump(by_alias=True),
        "after": rev2.model_dump(by_alias=True),
    }
    return rev2, revision_row


def test_recovery_appends_missing_revision(four_accounts):
    """故障 1：pending 已写、revision 未落盘 → 下次写入恢复补 append + 更新 match。"""
    match = ledger.create_match(_create_payload([25000] * 4, four_accounts), registry)
    rev2, revision_row = _build_rev2_tx(match)
    ledger._write_pending_transaction(rev2, revision_row)  # 模拟 append 前崩溃
    ledger.void_match(match.match_id, MatchVoid(reason="测试"))
    assert ledger.get_match(match.match_id).revision == 3  # 恢复 rev2 + void rev3
    assert not ledger._pending_tx_path().exists()
    revisions = ledger.list_revisions(match.match_id)
    assert any(r["revision"] == 2 for r in revisions)


def test_recovery_rewrites_match_when_revision_present(four_accounts):
    """故障 2：revision 已 append、matches 未替换 → 恢复补替换并删 pending。"""
    match = ledger.create_match(_create_payload([25000] * 4, four_accounts), registry)
    rev2, revision_row = _build_rev2_tx(match)
    ledger._write_pending_transaction(rev2, revision_row)
    ledger._append_revision(revision_row)  # 模拟 matches replace 前崩溃
    ledger.void_match(match.match_id, MatchVoid(reason="测试"))
    assert ledger.get_match(match.match_id).revision == 3
    assert not ledger._pending_tx_path().exists()
    assert len(ledger.list_revisions(match.match_id)) == 3


def test_recovery_cleanup_only_when_complete(four_accounts):
    """故障 3：revision 与 match 都已落盘、仅删 pending 失败 → 恢复只清理。"""
    match = ledger.create_match(_create_payload([25000] * 4, four_accounts), registry)
    rev2, revision_row = _build_rev2_tx(match)
    ledger._write_pending_transaction(rev2, revision_row)
    ledger._append_revision(revision_row)
    ledger._rewrite_match(rev2)
    ledger.void_match(match.match_id, MatchVoid(reason="测试"))
    assert ledger.get_match(match.match_id).revision == 3
    assert not ledger._pending_tx_path().exists()
    assert len(ledger.list_revisions(match.match_id)) == 3


def test_match_references_account_includes_void(four_accounts):
    match = ledger.create_match(_create_payload([25000] * 4, four_accounts), registry)
    ledger.void_match(match.match_id, MatchVoid(reason="中途结束"))
    assert ledger.match_references_account("nick@01") is True, "void 局仍应阻止硬删除账号"
