"""Lock R9 multi-source ingest contract (LadderMatch / adapters / merge / replay).

R9-1 identity & event contract:
- LadderMatch / LadderMatchPlayer are frozen; players carry official account_id
- adapters map raw names -> account_id; replay never guesses identity
- merge dedups by match_id (same content no-op, different content conflict)
- stable ordering by occurred_at + match_id

R9-2 deterministic replay:
- full replay from 新人/R1500 over the merged stream
- ledger carries per-player pt_tier / positive_pt; no shared room semantics
- snapshot contract: account_summary.json + account_ledger.jsonl + rating_curve.csv
"""

from __future__ import annotations

import gzip
import json
from datetime import datetime
from pathlib import Path

import pytest

from replay import ladder
from replay.ladder_ingest import (
    LadderIngestError,
    LadderMatch,
    LadderMatchPlayer,
    ManualTenhouAdapter,
    NativeLogAdapter,
    PlayWithYouResultAdapter,
    build_ingest_report,
    merge_ladder_matches,
    validate_match,
)

ACCOUNTS = {"nick@01", "70k@01", "70k@02", "70k@03", "70k@04"}


def _player(account_id: str, seat: int, final_score: int) -> LadderMatchPlayer:
    return LadderMatchPlayer(account_id=account_id, seat=seat, final_score=final_score)


def _match(
    match_id: str,
    *,
    occurred: str = "2026-08-04T07:00:00Z",
    game_length: str = "hanchan",
    scores: tuple[int, int, int, int] = (42100, 28300, 18100, 11500),
) -> LadderMatch:
    return LadderMatch(
        match_id=match_id,
        occurred_at=datetime.fromisoformat(occurred.replace("Z", "+00:00")),
        game_length=game_length,
        players=tuple(
            _player(account_id, seat, score)
            for account_id, seat, score in zip(
                ("nick@01", "70k@01", "70k@02", "70k@03"), range(4), scores, strict=True
            )
        ),
        source_type="playwithyou",
        source_ref=f"fixture:{match_id}",
    )


# --- Validation -------------------------------------------------------------

def test_validate_match_ok() -> None:
    validate_match(_match("m1"), ACCOUNTS)


def test_validate_match_requires_four_unique_registered_accounts() -> None:
    duplicate = LadderMatch(
        match_id="dup",
        occurred_at=datetime.fromisoformat("2026-08-04T07:00:00+00:00"),
        game_length="hanchan",
        players=tuple(
            _player("nick@01", seat, score) for seat, score in enumerate((1, 2, 3, 4))
        ),
        source_type="t",
    )
    with pytest.raises(LadderIngestError, match="重复"):
        validate_match(duplicate, ACCOUNTS)

    unregistered = LadderMatch(
        match_id="ghost",
        occurred_at=datetime.fromisoformat("2026-08-04T07:00:00+00:00"),
        game_length="hanchan",
        players=tuple(
            _player(account_id, seat, score)
            for account_id, seat, score in zip(
                ("nick@01", "70k@01", "70k@02", "ghost@01"), (0, 1, 2, 3), (1, 2, 3, 4), strict=True
            )
        ),
        source_type="t",
    )
    with pytest.raises(LadderIngestError, match="未在正式赛季注册"):
        validate_match(unregistered, ACCOUNTS)


def test_validate_match_requires_exact_seats() -> None:
    bad = LadderMatch(
        match_id="seats",
        occurred_at=datetime.fromisoformat("2026-08-04T07:00:00+00:00"),
        game_length="hanchan",
        players=tuple(
            _player(account_id, seat, score)
            for account_id, seat, score in zip(
                ("nick@01", "70k@01", "70k@02", "70k@03"), (0, 0, 1, 2), (1, 2, 3, 4), strict=True
            )
        ),
        source_type="t",
    )
    with pytest.raises(LadderIngestError, match="seat"):
        validate_match(bad, ACCOUNTS)


# --- Adapters ---------------------------------------------------------------

def _write_mjai_log(log_dir: Path, name: str, scores: list[int], names: list[str]) -> None:
    events = [
        {"type": "start_game", "names": names},
        {"type": "start_kyoku", "scores": scores},
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
    ]
    path = log_dir / f"{name}.json.gz"
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def test_native_log_adapter_maps_names_to_accounts(tmp_path: Path) -> None:
    log_dir = tmp_path / "native"
    log_dir.mkdir(parents=True)
    _write_mjai_log(log_dir, "g000", [40000, 25000, 20000, 15000], ["Nick", "70k1", "70k2", "70k3"])
    adapter = NativeLogAdapter(
        {"Nick": "nick@01", "70k1": "70k@01", "70k2": "70k@02", "70k3": "70k@03"}
    )
    matches = list(adapter.iter_matches(log_dir))
    assert len(matches) == 1
    match = matches[0]
    assert match.match_id == "g000"
    assert match.source_type == "native"
    assert [p.account_id for p in match.players] == ["nick@01", "70k@01", "70k@02", "70k@03"]
    assert [p.final_score for p in match.players] == [40000, 25000, 20000, 15000]


def test_native_log_adapter_rejects_unknown_name(tmp_path: Path) -> None:
    log_dir = tmp_path / "native"
    log_dir.mkdir(parents=True)
    _write_mjai_log(log_dir, "g001", [40000, 25000, 20000, 15000], ["Nick", "ghost", "70k2", "70k3"])
    adapter = NativeLogAdapter({"Nick": "nick@01", "70k2": "70k@02", "70k3": "70k@03"})
    with pytest.raises(LadderIngestError, match="未在 name_to_account"):
        list(adapter.iter_matches(log_dir))


def _write_jsonl(source_dir: Path, records: list[dict]) -> None:
    source_dir.mkdir(parents=True, exist_ok=True)
    with (source_dir / "capture.jsonl").open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def test_playwithyou_and_manual_adapters_parse_jsonl(tmp_path: Path) -> None:
    record = {
        "match_id": "tenhou:log-123",
        "occurred_at": "2026-08-04T07:05:00Z",
        "game_length": "hanchan",
        "players": [
            {"account_id": "nick@01", "seat": 0, "final_score": 42100},
            {"account_id": "70k@01", "seat": 1, "final_score": 28300},
            {"account_id": "70k@02", "seat": 2, "final_score": 18100},
            {"account_id": "70k@03", "seat": 3, "final_score": 11500},
        ],
    }
    pwy_dir = tmp_path / "pwy"
    _write_jsonl(pwy_dir, [record])
    assert list(PlayWithYouResultAdapter().iter_matches(pwy_dir))[0].source_type == "playwithyou"
    manual_dir = tmp_path / "manual"
    _write_jsonl(manual_dir, [record])
    manual = list(ManualTenhouAdapter().iter_matches(manual_dir))[0]
    assert manual.source_type == "manual_tenhou"
    assert manual.match_id == "tenhou:log-123"


# --- Merge ------------------------------------------------------------------

def test_merge_dedups_identical_match_id_noop(tmp_path: Path) -> None:
    record = {
        "match_id": "m-dup",
        "occurred_at": "2026-08-04T07:00:00Z",
        "game_length": "hanchan",
        "players": [
            {"account_id": "nick@01", "seat": 0, "final_score": 42100},
            {"account_id": "70k@01", "seat": 1, "final_score": 28300},
            {"account_id": "70k@02", "seat": 2, "final_score": 18100},
            {"account_id": "70k@03", "seat": 3, "final_score": 11500},
        ],
    }
    pwy_dir = tmp_path / "pwy"
    _write_jsonl(pwy_dir, [record, record])  # 同 match_id 同内容出现两次
    merged = merge_ladder_matches([(PlayWithYouResultAdapter(), pwy_dir)], allowed_accounts=ACCOUNTS)
    assert len(merged) == 1


def test_merge_rejects_conflicting_match_id(tmp_path: Path) -> None:
    base = {
        "match_id": "m-conflict",
        "occurred_at": "2026-08-04T07:00:00Z",
        "game_length": "hanchan",
    }
    record_a = {
        **base,
        "players": [
            {"account_id": "nick@01", "seat": 0, "final_score": 42100},
            {"account_id": "70k@01", "seat": 1, "final_score": 28300},
            {"account_id": "70k@02", "seat": 2, "final_score": 18100},
            {"account_id": "70k@03", "seat": 3, "final_score": 11500},
        ],
    }
    record_b = {
        **base,
        "players": [
            {"account_id": "nick@01", "seat": 0, "final_score": 99999},
            {"account_id": "70k@01", "seat": 1, "final_score": 28300},
            {"account_id": "70k@02", "seat": 2, "final_score": 18100},
            {"account_id": "70k@03", "seat": 3, "final_score": 11500},
        ],
    }
    pwy_dir = tmp_path / "pwy"
    _write_jsonl(pwy_dir, [record_a])
    manual_dir = tmp_path / "manual"
    _write_jsonl(manual_dir, [record_b])
    with pytest.raises(LadderIngestError, match="match_id 冲突"):
        merge_ladder_matches(
            [(PlayWithYouResultAdapter(), pwy_dir), (ManualTenhouAdapter(), manual_dir)],
            allowed_accounts=ACCOUNTS,
        )


def test_merge_sorts_by_occurred_at_then_match_id(tmp_path: Path) -> None:
    records = []
    for idx, timestamp in enumerate(("2026-08-04T08:00:00Z", "2026-08-04T06:00:00Z")):
        records.append(
            {
                "match_id": f"m-{idx}",
                "occurred_at": timestamp,
                "game_length": "hanchan",
                "players": [
                    {"account_id": "nick@01", "seat": 0, "final_score": 42100},
                    {"account_id": "70k@01", "seat": 1, "final_score": 28300},
                    {"account_id": "70k@02", "seat": 2, "final_score": 18100},
                    {"account_id": "70k@03", "seat": 3, "final_score": 11500},
                ],
            }
        )
    pwy_dir = tmp_path / "pwy"
    _write_jsonl(pwy_dir, records)  # 乱序写入
    merged = merge_ladder_matches([(PlayWithYouResultAdapter(), pwy_dir)], allowed_accounts=ACCOUNTS)
    assert [m.match_id for m in merged] == ["m-1", "m-0"]


# --- Replay -----------------------------------------------------------------

def _replay_report(tmp_path: Path, matches: list[LadderMatch]) -> dict:
    from replay.ladder_ingest import replay_ladder_matches, write_ingest_outputs

    result = replay_ladder_matches(
        matches,
        scoring_config={
            "system": "tenhou_rank_progression",
            "version": "v1",
            "game_length": "hanchan",
        },
    )
    write_ingest_outputs(tmp_path / "out", result)
    return result["report"]


def test_replay_starts_all_from_newcomer_and_ranks(tmp_path: Path) -> None:
    report = _replay_report(tmp_path, [_match("m1")])
    assert report["schema"] == "keqing.mortal.platform_account_report.v2"
    assert report["games"] == 1
    by_id = {row["account_id"]: row for row in report["accounts"]}
    # 从新人/R1500 开始：一位 nick 升到 9kyu(0)，其他仍新人
    assert by_id["nick@01"]["rank_id"] == "9kyu"
    assert by_id["70k@01"]["rank_id"] == "newcomer"
    assert by_id["nick@01"]["pt_current"] == 0
    # 首场晋升：历史最高更新为 9kyu（初始段位已作为种子，正常晋升覆盖）
    assert by_id["nick@01"]["highest_rank_id"] == "9kyu"
    assert by_id["70k@01"]["tenhou_reached"] is False


def test_replay_ledger_records_pt_tier(tmp_path: Path) -> None:
    _replay_report(tmp_path, [_match("m1")])
    ledger_path = tmp_path / "out" / "account_ledger.jsonl"
    rows = [json.loads(line) for line in ledger_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 4
    nick = next(row for row in rows if row["account_id"] == "nick@01")
    assert nick["pt_tier"] == "ippan-equivalent"  # 新人 -> 一般档
    assert nick["positive_pt"] == [30, 15, 0]
    assert "table_room" not in nick
    assert nick["game_length"] == "hanchan"


def test_replay_mixed_ranks_never_rejected(tmp_path: Path) -> None:
    # 12 局重放：全量从新人开始，任何段位分化都不得失败
    matches = [_match(f"m{i}") for i in range(12)]
    report = _replay_report(tmp_path, matches)
    assert report["games"] == 12
    assert len(report["accounts"]) == 4


def test_replay_writes_snapshot_contract(tmp_path: Path) -> None:
    _replay_report(tmp_path, [_match("m1")])
    assert (tmp_path / "out" / "account_summary.json").is_file()
    assert (tmp_path / "out" / "account_ledger.jsonl").is_file()
    assert (tmp_path / "out" / "rating_curve.csv").is_file()


# --- Season-level ingest ----------------------------------------------------

def _season(ingest: dict) -> dict:
    return {
        "schema": ladder.SEASON_SCHEMA,
        "season_id": "official-ladder-v1",
        "status": "running",
        "default": False,
        "scoring": {
            "system": "tenhou_rank_progression",
            "version": "v1",
            "game_length": "hanchan",
        },
        "ingest": ingest,
        "models": [
            {
                "model_id": "human",
                "accounts": [{"account_id": "nick@01", "display_name": "Nick"}],
            },
            {
                "model_id": "70k",
                "accounts": [
                    {"account_id": "70k@01", "display_name": "70k-1"},
                    {"account_id": "70k@02", "display_name": "70k-2"},
                    {"account_id": "70k@03", "display_name": "70k-3"},
                ],
            },
        ],
    }


def test_build_ingest_report_merges_all_sources(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    _write_jsonl(
        sources / "playwithyou",
        [
            {
                "match_id": "pwy:1",
                "occurred_at": "2026-08-04T07:00:00Z",
                "game_length": "hanchan",
                "players": [
                    {"account_id": "nick@01", "seat": 0, "final_score": 42100},
                    {"account_id": "70k@01", "seat": 1, "final_score": 28300},
                    {"account_id": "70k@02", "seat": 2, "final_score": 18100},
                    {"account_id": "70k@03", "seat": 3, "final_score": 11500},
                ],
            }
        ],
    )
    _write_jsonl(
        sources / "manual_tenhou",
        [
            {
                "match_id": "manual:1",
                "occurred_at": "2026-08-04T06:00:00Z",
                "game_length": "hanchan",
                "players": [
                    {"account_id": "nick@01", "seat": 0, "final_score": 40000},
                    {"account_id": "70k@01", "seat": 1, "final_score": 30000},
                    {"account_id": "70k@02", "seat": 2, "final_score": 20000},
                    {"account_id": "70k@03", "seat": 3, "final_score": 10000},
                ],
            }
        ],
    )
    season = _season({"sources_root": str(sources)})
    report = build_ingest_report(season=season, sources_root=sources, output_dir=tmp_path / "out")
    assert report["games"] == 2  # 两个来源各一局
    assert report["scoring"]["system"] == "tenhou_rank_progression"
    assert "room_policy" not in report["scoring"]
    # 三件套就位
    assert (tmp_path / "out" / "account_summary.json").is_file()
    assert (tmp_path / "out" / "account_ledger.jsonl").is_file()
    assert (tmp_path / "out" / "rating_curve.csv").is_file()


def test_build_ingest_report_rejects_unregistered_account(tmp_path: Path) -> None:
    sources = tmp_path / "sources"
    _write_jsonl(
        sources / "playwithyou",
        [
            {
                "match_id": "pwy:bad",
                "occurred_at": "2026-08-04T07:00:00Z",
                "game_length": "hanchan",
                "players": [
                    {"account_id": "nick@01", "seat": 0, "final_score": 42100},
                    {"account_id": "70k@01", "seat": 1, "final_score": 28300},
                    {"account_id": "70k@02", "seat": 2, "final_score": 18100},
                    {"account_id": "ghost@01", "seat": 3, "final_score": 11500},
                ],
            }
        ],
    )
    season = _season({"sources_root": str(sources)})
    with pytest.raises(LadderIngestError, match="未在正式赛季注册"):
        build_ingest_report(season=season, sources_root=sources, output_dir=tmp_path / "out")
