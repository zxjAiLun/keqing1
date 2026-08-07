# -*- coding: utf-8 -*-
"""R10-F Repair：fingerprint、generation CAS、season move、exclusive、自动消费。"""
from __future__ import annotations

import json

import pytest

from participants import ledger, projection, registry
from participants.schemas import AccountCreate, MatchCreate, MatchRevise, MatchSeat
from replay import ladder_ingest

SEASON = "official-ladder-v1"


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


def _match_create(season_id=SEASON, rating_eligible=True, occurred_at="2026-08-04T10:00:00+08:00"):
    return MatchCreate(
        occurred_at=occurred_at,
        game_length="hanchan",
        season_id=season_id,
        rating_eligible=rating_eligible,
        seats=[
            MatchSeat(seat=0, account_id="nick@01"),
            MatchSeat(seat=1, account_id="70k@01"),
            MatchSeat(seat=2, account_id="70k@02"),
            MatchSeat(seat=3, account_id="70k@03"),
        ],
        final_scores=[30000, 20000, 25000, 25000],
        source="manual",
    )


def _season_config(tmp_path, *, exclusive=True) -> dict:
    return {
        "schema": "keqing.ladder.season.v1",
        "season_id": SEASON,
        "report_dir": str(tmp_path / "reports" / SEASON),
        "status": "running",
        "scoring": {
            "system": "tenhou_rank_progression",
            "version": "test-v1",
        },
        "ingest": {
            "sources_root": str(tmp_path / "sources"),
            "participants": {"enabled": True, "exclusive": exclusive},
        },
        "models": [
            {
                "model_id": "70k",
                "accounts": [
                    {"account_id": "nick@01"},
                    {"account_id": "70k@01"},
                    {"account_id": "70k@02"},
                    {"account_id": "70k@03"},
                ],
            }
        ],
    }


def _write_season_config(tmp_path, monkeypatch, *, exclusive=True) -> dict:
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    season_cfg = _season_config(tmp_path, exclusive=exclusive)
    (configs / f"{SEASON}.json").write_text(json.dumps(season_cfg, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("KEQING_LADDER_CONFIG_DIR", str(configs))
    return season_cfg


def _mock_publish(monkeypatch, tmp_path):
    from scripts.mortal import publish_ladder_snapshot

    def _publish(**kw):
        return {"snapshot_dir": str(tmp_path / "snap"), "games": 1}

    monkeypatch.setattr(publish_ladder_snapshot, "publish_snapshot", _publish)


# ---------------------------------------------------------------------------
# P1-1：participants ledger 纳入 publisher fingerprint
# ---------------------------------------------------------------------------

def test_participants_projection_fingerprint_changes(participants_root):
    _accounts()
    match = ledger.create_match(_match_create(), registry)
    fp1 = ladder_ingest.participants_projection_fingerprint(SEASON)
    # 新增一场 → 指纹变化
    ledger.create_match(_match_create(occurred_at="2026-08-05T10:00:00+08:00"), registry)
    fp2 = ladder_ingest.participants_projection_fingerprint(SEASON)
    assert fp1 != fp2
    # 只改 note（不影响天梯）→ 指纹不变
    ledger.revise_match(match.match_id, MatchRevise(note="只是备注"), registry)
    assert ladder_ingest.participants_projection_fingerprint(SEASON) == fp2


def test_publish_uses_ledger_extra_fingerprint(participants_root, monkeypatch, tmp_path):
    from scripts.mortal import publish_ladder_snapshot as pls

    _accounts()
    ledger.create_match(_match_create(), registry)

    season = _season_config(tmp_path, exclusive=True)
    extra = pls._participants_extra_fingerprint(season)
    assert extra == ladder_ingest.participants_projection_fingerprint(SEASON)
    # 未启用 → None（普通 ingest season 不受影响）
    season_no = dict(season)
    season_no["ingest"] = {"sources_root": str(tmp_path / "sources")}
    assert pls._participants_extra_fingerprint(season_no) is None
    # compute_source_fingerprint 合并 extra 后变化
    fp_a = pls.compute_source_fingerprint([], ingest_root=tmp_path / "sources", extra_fingerprint="aaa")
    fp_b = pls.compute_source_fingerprint([], ingest_root=tmp_path / "sources", extra_fingerprint="bbb")
    assert fp_a != fp_b


# ---------------------------------------------------------------------------
# P1-2：generation CAS（lost-update 防护）
# ---------------------------------------------------------------------------

def test_project_season_lost_update_keeps_dirty(participants_root, monkeypatch, tmp_path):
    from scripts.mortal import publish_ladder_snapshot

    _accounts()
    ledger.create_match(_match_create(), registry)
    _write_season_config(tmp_path, monkeypatch)

    # 发布期间发生新的 ledger 写入（generation 变）
    def _publish_with_concurrent_write(**kw):
        ledger.create_match(_match_create(occurred_at="2026-08-06T10:00:00+08:00"), registry)
        return {"snapshot_dir": str(tmp_path / "snap"), "games": 1}

    monkeypatch.setattr(publish_ladder_snapshot, "publish_snapshot", _publish_with_concurrent_write)
    result = projection.project_season(SEASON)
    assert result["state"] == "needs_rebuild"
    # dirty 保留 + 新比赛仍 pending
    assert ledger.ladder_dirty_path(SEASON).exists()
    matches = ledger.list_matches(status="active").matches
    assert all(m.ladder_projection_state == "pending" for m in matches if m.season_id == SEASON)
    # 第二次投影（无并发写入）→ ready + 清 dirty
    _mock_publish(monkeypatch, tmp_path)
    result2 = projection.project_season(SEASON)
    assert result2["state"] == "ready"
    assert not ledger.ladder_dirty_path(SEASON).exists()
    assert all(m.ladder_projection_state == "ready" for m in ledger.list_matches(status="active").matches if m.season_id == SEASON)


# ---------------------------------------------------------------------------
# P1-3：season move → 双 dirty / 显式清空
# ---------------------------------------------------------------------------

def test_revise_season_move_dirty_both(participants_root):
    _accounts()
    match = ledger.create_match(_match_create(season_id="season-a"), registry)
    ledger.clear_ladder_dirty("season-a")
    assert not ledger.ladder_dirty_path("season-b").exists()

    # A → B：A、B 都 dirty
    ledger.revise_match(match.match_id, MatchRevise(season_id="season-b", rating_eligible=True), registry)
    assert ledger.ladder_dirty_path("season-a").exists()
    assert ledger.ladder_dirty_path("season-b").exists()
    assert ledger.get_match(match.match_id).season_id == "season-b"

    # B → null（显式）：B dirty，season 清空
    ledger.clear_ladder_dirty("season-b")
    ledger.revise_match(match.match_id, MatchRevise(season_id=None, rating_eligible=False), registry)
    assert ledger.ladder_dirty_path("season-b").exists()
    updated = ledger.get_match(match.match_id)
    assert updated.season_id is None
    assert updated.ladder_projection_state == "not_applicable"

    # 省略字段（不传 season_id）→ 不清空、不标 dirty
    ledger.clear_ladder_dirty("season-b")
    ledger.revise_match(match.match_id, MatchRevise(note="仅备注"), registry)
    assert ledger.get_match(match.match_id).season_id is None
    assert not ledger.ladder_dirty_path("season-b").exists()


# ---------------------------------------------------------------------------
# P1-4：participants exclusive 排他
# ---------------------------------------------------------------------------

def test_exclusive_mode_skips_legacy_sources(participants_root, tmp_path):
    _accounts()
    sources = tmp_path / "sources"
    manual_dir = sources / "manual_tenhou"
    manual_dir.mkdir(parents=True)
    # 无效 legacy 文件——若被读取会抛 LadderIngestError
    (manual_dir / "bad.jsonl").write_text("{invalid json\n", encoding="utf-8")
    ledger.create_match(_match_create(), registry)

    season = _season_config(tmp_path, exclusive=True)
    output = tmp_path / "out"
    # exclusive：legacy source 不被读取 → 不因坏文件抛错
    result = ladder_ingest.build_ingest_report(season=season, sources_root=sources, output_dir=output)
    assert result is not None


# ---------------------------------------------------------------------------
# P1-6：自动 dirty consumer
# ---------------------------------------------------------------------------

def test_run_dirty_projection_auto_consumes(participants_root, monkeypatch, tmp_path):
    _accounts()
    ledger.create_match(_match_create(), registry)
    _write_season_config(tmp_path, monkeypatch)
    _mock_publish(monkeypatch, tmp_path)
    assert ledger.ladder_dirty_path(SEASON).exists()

    results = projection.run_dirty_projection()
    assert results and results[0]["state"] == "ready"
    assert not ledger.ladder_dirty_path(SEASON).exists()
    assert all(m.ladder_projection_state == "ready" for m in ledger.list_matches(status="active").matches if m.season_id == SEASON)


def test_projection_gate_requires_participants_enabled(participants_root, monkeypatch, tmp_path):
    _accounts()
    ledger.create_match(_match_create(), registry)
    # 普通 ingest season（无 participants.enabled）
    season_cfg = _season_config(tmp_path, exclusive=True)
    season_cfg["ingest"] = {"sources_root": str(tmp_path / "sources")}
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    (configs / f"{SEASON}.json").write_text(json.dumps(season_cfg, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("KEQING_LADDER_CONFIG_DIR", str(configs))
    _mock_publish(monkeypatch, tmp_path)

    result = projection.project_season(SEASON)
    assert result["state"] == "error"
    assert "未启用 participants 投影" in result.get("reason", "")
    # dirty 保留、不误清
    assert ledger.ladder_dirty_path(SEASON).exists()


# ---------------------------------------------------------------------------
# R10-F Repair 2：begin barrier / single-flight / failure CAS / wake
# ---------------------------------------------------------------------------

def test_begin_barrier_blocks_mid_mutation(participants_root, monkeypatch):
    """P1-1：dirty 已写但 Match 事务未完成时，投影 begin barrier 必须阻塞到提交完成。"""
    import threading
    import time

    _accounts()
    started = threading.Event()
    release = threading.Event()
    original = ledger.mark_ladder_dirty

    def _slow_mark(season_id):
        original(season_id)
        started.set()
        release.wait(5)  # 模拟 dirty 后、pending/match 提交前被挂起

    monkeypatch.setattr(ledger, "mark_ladder_dirty", _slow_mark)

    def _mutation():
        ledger.create_match(_match_create(), registry)

    t = threading.Thread(target=_mutation)
    t.start()
    assert started.wait(5), "mutation 未开始"

    begin_gen: dict = {}

    def _begin():
        begin_gen["gen"] = ledger.begin_ladder_projection(SEASON)

    b = threading.Thread(target=_begin)
    b.start()
    time.sleep(0.3)
    assert "gen" not in begin_gen, "begin barrier 未阻塞在未提交事务上"
    release.set()
    t.join()
    b.join()
    assert "gen" in begin_gen
    assert begin_gen["gen"] == ledger.read_ladder_generation(SEASON)
    # 投影此时读到的是已提交的 11 局
    assert len(ledger.list_matches(status="active").matches) == 1


def test_single_flight_prevents_concurrent_publishers(participants_root, monkeypatch, tmp_path):
    """P1-2：worker + manual 同时 project 同一 generation → 只有一个进入 publisher。"""
    import threading

    from scripts.mortal import publish_ladder_snapshot

    _accounts()
    ledger.create_match(_match_create(), registry)
    _write_season_config(tmp_path, monkeypatch)
    entered = threading.Event()
    release = threading.Event()
    publish_calls = {"n": 0}

    def _slow_publish(**kw):
        publish_calls["n"] += 1
        entered.set()
        release.wait(5)
        return {"snapshot_dir": str(tmp_path / "snap"), "games": 1}

    monkeypatch.setattr(publish_ladder_snapshot, "publish_snapshot", _slow_publish)

    first: dict = {}

    def _run_first():
        first["r"] = projection.project_season(SEASON)

    t = threading.Thread(target=_run_first)
    t.start()
    assert entered.wait(5), "第一个 publisher 未进入"

    # 第二个调用不进入 publisher → already_running
    r2 = projection.project_season(SEASON)
    assert r2["state"] == "already_running"
    assert publish_calls["n"] == 1

    release.set()
    t.join()
    assert first["r"]["state"] == "ready"
    assert not ledger.ladder_dirty_path(SEASON).exists()
    assert all(m.ladder_projection_state == "ready" for m in ledger.list_matches(status="active").matches if m.season_id == SEASON)


def test_stale_failure_does_not_overwrite_ready(participants_root, monkeypatch, tmp_path):
    """P1-2：过期失败回写（旧 generation）不能覆盖已成功的 ready。"""
    _accounts()
    ledger.create_match(_match_create(), registry)
    _write_season_config(tmp_path, monkeypatch)
    _mock_publish(monkeypatch, tmp_path)

    result = projection.project_season(SEASON)
    assert result["state"] == "ready"
    assert not ledger.ladder_dirty_path(SEASON).exists()

    # 用旧 generation 回写 error → CAS 拒绝，状态保持 ready
    assert ledger.mark_season_projection_error(SEASON, "stale-generation") is False
    matches = ledger.list_matches(status="active").matches
    assert all(m.ladder_projection_state == "ready" for m in matches if m.season_id == SEASON)


def test_gate_requires_exclusive(participants_root, monkeypatch, tmp_path):
    """配置建议：projection 门禁要求 enabled + exclusive（缺 exclusive 拒绝）。"""
    _accounts()
    ledger.create_match(_match_create(), registry)
    season_cfg = _season_config(tmp_path, exclusive=True)
    season_cfg["ingest"]["participants"] = {"enabled": True, "exclusive": False}
    configs = tmp_path / "configs"
    configs.mkdir(exist_ok=True)
    (configs / f"{SEASON}.json").write_text(json.dumps(season_cfg, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("KEQING_LADDER_CONFIG_DIR", str(configs))
    _mock_publish(monkeypatch, tmp_path)

    result = projection.project_season(SEASON)
    assert result["state"] == "error"
    assert "exclusive" in result.get("reason", "")
    assert ledger.ladder_dirty_path(SEASON).exists()


def test_request_projection_none_wakes(participants_root):
    """P2-2：A→null 后 request_projection(None) 也唤醒 worker（扫描全部 dirty marker）。"""
    projection._wake.clear()
    projection.request_projection(None)
    assert projection._wake.is_set()
