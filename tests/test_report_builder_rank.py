"""Integration tests for the versioned scoring engine inside the report builder.

These tests exercise ``build_platform_account_report.build_report`` with the
Tenhou ranked profile using synthetic mjai-style logs.  ``build_stat_report``
(Mortal libriichi parsing) is stubbed out so the tests only cover the
rank/rating engine pipeline, not Mortal's detailed stats.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import pytest

from scripts.mortal import build_platform_account_report as account_report

TENHOU_SCORING = {
    "system": "tenhou_4p_ranked",
    "version": "2026-08-04",
    "game_length": "hanchan",
    "room_policy": "highest_common_eligible",
    "membership": "premium",
    "initial_rank": "newcomer",
    "initial_rating": 1500,
}

LEGACY_SCORING = {
    "system": "tenhou_houou_7dan_fixed",
    "version": "2026-07-01",
    "game_length": "hanchan",
    "room": "houou",
    "initial_rank": "7dan",
    "initial_pt": 1400,
    "initial_rating": 1500,
    "rank_points": [90, 45, 0, -135],
    "target_pt": 2800,
}


def _write_game(log_dir: Path, name: str, scores: list[int]) -> None:
    """Write a minimal playable log: start_game -> start_kyoku -> ryukyoku.

    The final scores are provided directly in start_kyoku and confirmed by a
    zero-delta ryukyoku, giving placements 1..4 by descending score (ties by
    seat order).
    """
    events = [
        {"type": "start_game", "names": ["a", "b", "c", "d"]},
        {"type": "start_kyoku", "scores": scores},
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
    ]
    path = log_dir / f"{name}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def _build(
    tmp_path: Path,
    scoring_config: dict,
    game_scores: list[list[int]],
    monkeypatch: pytest.MonkeyPatch,
) -> dict:
    def _stub(**kwargs):
        return {
            "players": {},
            "backend": "stub",
            "log_dir": str(kwargs.get("log_dir") or ""),
            "rank_points_profile": str(kwargs.get("rank_points_profile") or "custom"),
            "rank_pts": [float(value) for value in kwargs.get("rank_pts") or [0, 0, 0, 0]],
        }

    monkeypatch.setattr(account_report, "build_stat_report", _stub)
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    for index, scores in enumerate(game_scores):
        _write_game(log_dir, f"game_{index:03d}", scores)
    return account_report.build_report(
        log_dirs=[log_dir],
        output_dir=tmp_path / "out",
        mortal_root=Path("third_party/Mortal"),
        platform_model_label="testmodel",
        rank_points=(90.0, 45.0, 0.0, -135.0),
        scoring_config=scoring_config,
    )


def _ledger_rows(tmp_path: Path) -> list[dict]:
    ledger_path = tmp_path / "out" / "account_ledger.jsonl"
    return [
        json.loads(line)
        for line in ledger_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _by_account(rows: list[dict], account_id: str) -> list[dict]:
    return [row for row in rows if row["account_id"] == account_id]


def test_tenhou_engine_progresses_newcomer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # a wins 1st (+90): newcomer(0) -> 9kyu(0), then 4th (-135): 9kyu floored at 0.
    report = _build(
        tmp_path,
        TENHOU_SCORING,
        [
            [40000, 25000, 20000, 15000],  # a=1st, b=2nd, c=3rd, d=4th
            [15000, 25000, 20000, 40000],  # d=1st, b=2nd, c=3rd, a=4th
        ],
        monkeypatch,
    )
    a = next(row for row in report["accounts"] if row["account_id"] == "testmodel@01")
    assert a["rank_id"] == "9kyu"
    assert a["rank_ordinal"] == 2
    assert a["pt_current"] == 0
    assert a["pt_target"] == 20
    assert a["games"] == 2
    assert a["tenhou_reached"] is False


def test_ledger_records_transitions_and_room(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _build(
        tmp_path,
        TENHOU_SCORING,
        [
            [40000, 25000, 20000, 15000],
            [40000, 25000, 20000, 15000],
        ],
        monkeypatch,
    )
    rows = _ledger_rows(tmp_path)
    a_rows = _by_account(rows, "testmodel@01")
    assert len(a_rows) == 2
    first = a_rows[0]
    assert first["table_room"] == "ippan"
    assert first["game_length"] == "hanchan"
    assert first["rank_before"] == "newcomer"
    assert first["rank_after"] == "9kyu"
    assert first["transition"] == "promotion"
    assert first["pt_delta"] == 30
    second = a_rows[1]
    assert second["rank_before"] == "9kyu"
    assert second["rank_after"] == "8kyu"
    assert second["transition"] == "promotion"


def test_all_players_use_pre_match_rating_snapshot(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Two games: after game 1 every player's rating changed. Game 2's table
    # average must be computed from the four PRE-game ratings, not from a
    # partially-updated mix.
    _build(
        tmp_path,
        TENHOU_SCORING,
        [
            [40000, 25000, 20000, 15000],
            [40000, 25000, 20000, 15000],
        ],
        monkeypatch,
    )
    rows = _ledger_rows(tmp_path)
    game1 = [row for row in rows if row["game_index"] == 0]
    game2 = [row for row in rows if row["game_index"] == 1]
    avg1 = {row["account_id"]: row["table_avg_rating_before"] for row in game1}
    avg2 = {row["account_id"]: row["table_avg_rating_before"] for row in game2}
    # Table average is written per-row; all four rows of a game must agree.
    assert len(set(avg1.values())) == 1
    assert len(set(avg2.values())) == 1
    # Game 1: all four at 1500 -> avg exactly 1500.
    assert abs(avg1["testmodel@01"] - 1500.0) < 1e-9
    # No pollution: every player's game-2 rating_before must equal their
    # game-1 rating_after (the pre-match snapshot, not a mid-loop update).
    by_acct1 = {row["account_id"]: row for row in game1}
    by_acct2 = {row["account_id"]: row for row in game2}
    assert set(by_acct1) == set(by_acct2) == {"testmodel@01", "testmodel@02", "testmodel@03", "testmodel@04"}
    for account_id, row1 in by_acct1.items():
        assert by_acct2[account_id]["rating_before"] == row1["rating_after"]


def test_legacy_fixed_profile_report_keeps_7dan_semantics(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _build(
        tmp_path,
        LEGACY_SCORING,
        [
            [40000, 25000, 20000, 15000],
        ],
        monkeypatch,
    )
    assert report["schema"] == "keqing.mortal.platform_account_report.v2"
    a = next(row for row in report["accounts"] if row["account_id"] == "testmodel@01")
    assert a["rank_id"] == "7dan"
    assert a["rank_name"] == "七段"
    assert a["rank_ordinal"] == 17
    assert a["pt_current"] == 1490
    assert a["pt_target"] == 2800
    assert a["promotions"] == 0
    assert a["demotions"] == 0
    assert a["total_pt_delta"] == 90
    assert a["avg_pt_delta"] == 90.0
    assert report["scoring"]["system"] == "tenhou_houou_7dan_fixed"


def test_report_scoring_block_describes_tenhou(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _build(
        tmp_path,
        TENHOU_SCORING,
        [[40000, 25000, 20000, 15000]],
        monkeypatch,
    )
    scoring = report["scoring"]
    assert scoring["system"] == "tenhou_4p_ranked"
    assert scoring["version"] == "2026-08-04"
    assert scoring["room_policy"] == "highest_common_eligible"
    assert scoring["membership"] == "premium"
    assert scoring["initial_rank"] == "newcomer"
    assert scoring["initial_rating"] == 1500.0
