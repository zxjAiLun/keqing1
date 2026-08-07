# -*- coding: utf-8 -*-
"""R10-G：账号详细统计——ledger 基础指标 + full replay 牌谱屋式指标（completeness-aware）。"""
from __future__ import annotations

import pytest

from participants import intake, ledger, registry, stats
from participants.schemas import AccountCreate, MatchCreate, MatchSeat


@pytest.fixture(autouse=True)
def participants_root(tmp_path, monkeypatch):
    root = tmp_path / "participants"
    monkeypatch.setenv("KEQING_PARTICIPANT_DATA_ROOT", str(root))
    return root


def _accounts():
    for account_id, account_type in (
        ("nick@01", "human"),
        ("70k@01", "managed_bot"),
        ("70k@02", "managed_bot"),
        ("70k@03", "managed_bot"),
    ):
        registry.create_account(AccountCreate(account_id=account_id, display_name=account_id, account_type=account_type))


def _seats():
    return [
        MatchSeat(seat=0, account_id="nick@01"),
        MatchSeat(seat=1, account_id="70k@01"),
        MatchSeat(seat=2, account_id="70k@02"),
        MatchSeat(seat=3, account_id="70k@03"),
    ]


def _full_replay_match(log_id="log1", occurred_at="2026-08-04T10:00:00+08:00"):
    return ledger.create_match(
        MatchCreate(
            occurred_at=occurred_at,
            game_length="hanchan",
            source="imported",
            source_ref=log_id,
            provider="tenhou",
            external_match_id=log_id,
            replay_id=log_id,
            data_completeness="full_replay",
            seats=_seats(),
            final_scores=[30000, 20000, 25000, 25000],
        ),
        registry,
    )


def _result_only_match(occurred_at="2026-08-05T10:00:00+08:00"):
    return ledger.create_match(
        MatchCreate(
            occurred_at=occurred_at,
            game_length="hanchan",
            source="manual",
            data_completeness="result_only",
            seats=_seats(),
            final_scores=[42000, 18000, 25000, 15000],
        ),
        registry,
    )


def _rich_hands():
    """三个事件局：nick@01(seat0) 荣和、他人自摸、流局听牌。"""
    return [
        {
            "bakaze": "E", "kyoku": 1, "honba": 0, "oya": 0,
            "scores_before": [25000, 25000, 25000, 25000],
            "winners": [{"actor": 0, "win_type": "ron", "target": 1, "deltas": [5000, -5000, 0, 0]}],
            "ryukyoku": None, "riichi": [0], "calls": [1, 0, 0, 0],
        },
        {
            "bakaze": "E", "kyoku": 2, "honba": 0, "oya": 1,
            "scores_before": [30000, 20000, 25000, 25000],
            "winners": [{"actor": 1, "win_type": "tsumo", "target": 1, "deltas": [1000, 3000, -2000, -2000]}],
            "ryukyoku": None, "riichi": [1], "calls": [0, 0, 0, 0],
        },
        {
            "bakaze": "E", "kyoku": 3, "honba": 0, "oya": 2,
            "scores_before": [31000, 23000, 23000, 23000],
            "winners": [],
            "ryukyoku": {"reason": "ryukyoku", "deltas": [0, 0, 0, 0], "tenpai": [True, False, False, False]},
            "riichi": [], "calls": [0, 0, 0, 0],
        },
    ]


def _persist(log_id, hands):
    intake._write_artifact_files(
        intake.artifact_dir(log_id),
        tenhou6={},
        events=[],
        hands=hands,
        summary={"log_id": log_id, "names": ["Nick", "A", "B", "C"]},
    )


def test_stats_placement_from_all_matches(participants_root):
    _accounts()
    _full_replay_match()
    _result_only_match()
    result = stats.compute_account_stats("nick@01", registry, ledger)
    assert result["schema"] == stats.STATS_CONTRACT_VERSION
    assert result["coverage"]["total_matches"] == 2
    assert result["coverage"]["matches_with_results"] == 2
    assert result["coverage"]["matches_with_hands"] == 1
    assert result["coverage"]["matches_with_full_replay"] == 1
    assert result["placement"]["match_count"] == 2
    # 第一场 nick 顺位 1（30000 最高），第二场 nick 顺位 1（42000 最高）
    assert result["placement"]["first_rate"] == 1.0
    assert result["placement"]["second_rate"] == 0.0
    assert result["placement"]["avg_rank"] == 1.0
    assert result["placement"]["avg_final_score"] == (30000 + 42000) / 2


def test_stats_detailed_from_full_replay(participants_root):
    _accounts()
    _persist("log1", _rich_hands())
    _full_replay_match()
    _result_only_match()

    result = stats.compute_account_stats("nick@01", registry, ledger)
    detailed = result["detailed"]
    # nick@01 (seat0)：3 局，1 次荣和（5000），0 次放铳，1 次立直，1 局副露，流局听牌 1/1
    assert detailed["hands_played"] == 3
    assert detailed["wins"] == 1
    assert detailed["dealins"] == 0
    assert detailed["win_rate"] == round(1 / 3, 4)
    assert detailed["dealin_rate"] == 0.0
    assert detailed["tsumo_share"] == 0.0  # 1 次和牌是荣和
    assert detailed["riichi_rate"] == round(1 / 3, 4)
    assert detailed["call_rate"] == round(1 / 3, 4)
    assert detailed["tenpai_rate"] == 1.0
    assert detailed["avg_win_points"] == 5000.0
    assert detailed["oya_win_rate"] == 1.0  # 第 1 局 oya=0（nick）
    assert detailed["koshu_win_rate"] == 0.0  # 第 2/3 局非庄，未和牌


def test_stats_detailed_handles_dealin(participants_root):
    _accounts()
    # nick@01 放铳给 70k@01
    hands = [
        {
            "bakaze": "E", "kyoku": 1, "honba": 0, "oya": 0,
            "scores_before": [25000, 25000, 25000, 25000],
            "winners": [{"actor": 1, "win_type": "ron", "target": 0, "deltas": [-8000, 8000, 0, 0]}],
            "ryukyoku": None, "riichi": [], "calls": [0, 0, 0, 0],
        }
    ]
    _persist("log1", hands)
    _full_replay_match()
    result = stats.compute_account_stats("nick@01", registry, ledger)
    assert result["detailed"]["dealins"] == 1
    assert result["detailed"]["avg_dealin_points"] == 8000.0
    assert result["detailed"]["win_rate"] == 0.0


def test_stats_result_only_no_detailed(participants_root):
    """result_only 比赛不得进入详细指标分母。"""
    _accounts()
    _result_only_match()
    result = stats.compute_account_stats("nick@01", registry, ledger)
    assert result["coverage"]["matches_with_full_replay"] == 0
    assert result["detailed"]["hands_played"] == 0
    assert result["detailed"]["win_rate"] is None
    assert result["detailed"]["riichi_rate"] is None


def test_hand_summaries_capture_riichi_calls_tenpai():
    """扩展后的逐局摘要含 riichi/calls/tenpai（R10-G 输入）。"""
    events = [
        {"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0, "oya": 0, "scores": [25000] * 4},
        {"type": "reach", "actor": 0},
        {"type": "pon", "actor": 1, "target": 0},
        {"type": "hora", "actor": 0, "target": 1, "deltas": [5000, -5000, 0, 0]},
        {"type": "start_kyoku", "bakaze": "E", "kyoku": 2, "honba": 0, "oya": 1, "scores": [30000, 20000, 25000, 25000]},
        {"type": "ryukyoku", "reason": "ryukyoku", "deltas": [0, 0, 0, 0], "tenpai": [True, False, False, False]},
    ]
    hands = intake.hand_summaries(events)
    assert hands[0]["riichi"] == [0]
    assert hands[0]["calls"] == [0, 1, 0, 0]
    assert hands[1]["ryukyoku"]["tenpai"] == [True, False, False, False]
