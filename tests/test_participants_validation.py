# -*- coding: utf-8 -*-
"""participants validation：4 座不变量、总分校验、force-save 路径。"""
from __future__ import annotations

import pytest

from participants import ledger, registry
from participants.schemas import AccountCreate, AccountUpdate, MatchCreate, MatchSeat
from participants.ledger import ValidationError


@pytest.fixture(autouse=True)
def participants_root(tmp_path, monkeypatch):
    root = tmp_path / "participants"
    monkeypatch.setenv("KEQING_PARTICIPANT_DATA_ROOT", str(root))
    return root


@pytest.fixture
def four_accounts():
    for aid in ("nick@01", "friend@01", "70k@01", "mortal41b"):
        registry.create_account(AccountCreate(account_id=aid, display_name=aid, account_type="human"))
    return ["nick@01", "friend@01", "70k@01", "mortal41b"]


def _payload(scores, account_ids, **kw):
    return MatchCreate(
        occurred_at="2026-08-06T12:00:00+08:00",
        game_length="hanchan",
        seats=[MatchSeat(seat=i, account_id=account_ids[i]) for i in range(4)],
        final_scores=scores,
        **kw,
    )


def test_four_seat_invariant(four_accounts):
    with pytest.raises(ValidationError) as exc:
        ledger.create_match(
            MatchCreate(
                occurred_at="2026-08-06T12:00:00+08:00",
                game_length="hanchan",
                seats=[MatchSeat(seat=0, account_id=four_accounts[0])],
                final_scores=[25000, 25000, 25000, 25000],
            ),
            registry,
        )
    codes = {i.code for i in exc.value.issues}
    assert "seat_count" in codes


def test_duplicate_account_rejected(four_accounts):
    seats = [
        MatchSeat(seat=0, account_id="nick@01"),
        MatchSeat(seat=1, account_id="nick@01"),
        MatchSeat(seat=2, account_id="70k@01"),
        MatchSeat(seat=3, account_id="mortal41b"),
    ]
    with pytest.raises(ValidationError) as exc:
        ledger.create_match(
            MatchCreate(
                occurred_at="2026-08-06T12:00:00+08:00",
                game_length="hanchan",
                seats=seats,
                final_scores=[25000] * 4,
            ),
            registry,
        )
    assert "account_duplicate" in {i.code for i in exc.value.issues}


def test_score_total_mismatch_rejected_without_force(four_accounts):
    with pytest.raises(ValidationError) as exc:
        ledger.create_match(_payload([42300, 28100, 19400, 10201], four_accounts), registry)
    assert exc.value.score_mismatch is True
    assert "score_total_mismatch" in {i.code for i in exc.value.issues}


def test_force_without_reason_rejected(four_accounts):
    payload = _payload([42300, 28100, 19400, 10201], four_accounts, force=True, reason=None)
    with pytest.raises(ValidationError) as exc:
        ledger.create_match(payload, registry)
    assert "force_reason_required" in {i.code for i in exc.value.issues}


def test_force_with_reason_allowed(four_accounts):
    payload = _payload([42300, 28100, 19400, 10201], four_accounts, force=True, reason="罚符")
    match = ledger.create_match(payload, registry)
    assert match.final_scores == [42300, 28100, 19400, 10201]


def test_disabled_account_blocked_without_force(four_accounts):
    registry.update_account("nick@01", AccountUpdate(enabled=False))
    with pytest.raises(ValidationError) as exc:
        ledger.create_match(_payload([25000] * 4, four_accounts), registry)
    assert "account_disabled" in {i.code for i in exc.value.issues}
    # force + reason 放行
    payload = _payload([25000] * 4, four_accounts, force=True, reason="人工修正")
    match = ledger.create_match(payload, registry)
    assert match.status == "active"


def test_unknown_account_rejected(four_accounts):
    seats = [
        MatchSeat(seat=0, account_id="ghost@01"),
        MatchSeat(seat=1, account_id=four_accounts[1]),
        MatchSeat(seat=2, account_id=four_accounts[2]),
        MatchSeat(seat=3, account_id=four_accounts[3]),
    ]
    with pytest.raises(ValidationError) as exc:
        ledger.create_match(
            MatchCreate(
                occurred_at="2026-08-06T12:00:00+08:00",
                game_length="hanchan",
                seats=seats,
                final_scores=[25000] * 4,
            ),
            registry,
        )
    assert "account_unknown" in {i.code for i in exc.value.issues}


def test_duplicate_seat_number_rejected(four_accounts):
    seats = [
        MatchSeat(seat=0, account_id=four_accounts[0]),
        MatchSeat(seat=0, account_id=four_accounts[1]),
        MatchSeat(seat=2, account_id=four_accounts[2]),
        MatchSeat(seat=3, account_id=four_accounts[3]),
    ]
    with pytest.raises(ValidationError) as exc:
        ledger.create_match(
            MatchCreate(
                occurred_at="2026-08-06T12:00:00+08:00",
                game_length="hanchan",
                seats=seats,
                final_scores=[25000] * 4,
            ),
            registry,
        )
    assert "seat_duplicate" in {i.code for i in exc.value.issues}
