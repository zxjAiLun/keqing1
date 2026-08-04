"""Lock R9-3 Play-with-you ladder capture semantics (fixtures; no live Tenhou).

Collector gates C4-C9 and API gates C1-C3 / C10-C14.  C15 (players order ->
seat canonical replay) and C16 (cross-source dedup) are covered by
test_ladder_ingest.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(_ROOT), str(_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from gateway.api import playwithyou as pwy  # noqa: E402
from gateway.playwithyou_capture import (  # noqa: E402
    CaptureBinding,
    PlayWithYouCaptureCollector,
    capture_dir_for_session,
    extract_tenhou_match_id,
)


def _binding(session_id: str = "abc123") -> CaptureBinding:
    return CaptureBinding(
        session_id=session_id,
        season_id="official-ladder-v1",
        human_account_id="nick@01",
        bot_account_ids=("70k@01", "70k@02", "70k@03"),
        mode="confirm",
    )


def _start_game(seat: int, log: str = "https://tenhou.net/3/?log=20260804gm-abc-xyz&tw=2") -> dict:
    return {
        "type": "start_game",
        "id": seat,
        "names": ["NoName-1", "Nick", "NoName-2", "NoName-3"],
        "log": log,
    }


def _end_game(scores: list[int]) -> dict:
    return {"type": "end_game", "scores": scores}


def _collector(tmp_path: Path) -> PlayWithYouCaptureCollector:
    capture_dir = capture_dir_for_session(tmp_path / "data", "abc123")
    return PlayWithYouCaptureCollector(binding=_binding(), capture_dir=capture_dir)


def _read_pending(capture_dir: Path) -> list[dict]:
    pending = capture_dir / "pending"
    if not pending.is_dir():
        return []
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(pending.glob("*.json"))
    ]


def _read_state(capture_dir: Path) -> dict:
    return json.loads((capture_dir / "state.json").read_text(encoding="utf-8"))


# --- Collector: identity mapping -------------------------------------------

def test_extract_tenhou_match_id_from_url() -> None:
    assert extract_tenhou_match_id("https://tenhou.net/3/?log=20260804gm-abc-xyz&tw=2") == (
        "tenhou:20260804gm-abc-xyz"
    )
    assert extract_tenhou_match_id(None) is None
    assert extract_tenhou_match_id("https://example.com/nolog") is None


def test_collector_derives_human_seat(tmp_path: Path) -> None:
    """C4：三个 observer 分别在不同 seat，剩余 seat 推导为人类。"""
    collector = _collector(tmp_path)
    collector.observe("70k@01", _start_game(2))
    collector.observe("70k@02", _start_game(0))
    collector.observe("70k@03", _start_game(3))
    assert collector._human_seat() == 1  # noqa: SLF001
    assert collector._account_for_seat(1) == "nick@01"  # noqa: SLF001


def test_collector_rotated_perspectives_still_canonical(tmp_path: Path) -> None:
    """C5+C6：observer 旋转视角最终规范结果一致；三份相同 end_game 只生成一个 pending。"""
    collector = _collector(tmp_path)
    collector.observe("70k@02", _start_game(0))
    collector.observe("70k@01", _start_game(2))
    collector.observe("70k@03", _start_game(3))
    collector.observe("70k@01", _end_game([42100, 28300, 18100, 11500]))
    collector.observe("70k@02", _end_game([42100, 28300, 18100, 11500]))
    collector.observe("70k@03", _end_game([42100, 28300, 18100, 11500]))
    collector.finalize()

    pending = _read_pending(collector.capture_dir)
    assert len(pending) == 1
    players = pending[0]["match"]["players"]
    by_seat = {p["seat"]: p["account_id"] for p in players}
    assert by_seat[0] == "70k@02"
    assert by_seat[1] == "nick@01"
    assert by_seat[2] == "70k@01"
    assert by_seat[3] == "70k@03"


def test_collector_one_observer_enough_when_other_disconnects(tmp_path: Path) -> None:
    """C7：一个 observer 掉线，另一个完整 end_game 仍可形成 pending。"""
    collector = _collector(tmp_path)
    collector.observe("70k@01", _start_game(2))
    collector.observe("70k@02", _start_game(0))
    collector.observe("70k@03", _start_game(3))
    collector.observe("70k@01", _end_game([42100, 28300, 18100, 11500]))
    collector.finalize()
    pending = _read_pending(collector.capture_dir)
    assert len(pending) == 1
    assert pending[0]["score_observers"] == ["70k@01"]


def test_collector_score_conflict_rejected(tmp_path: Path) -> None:
    """C8：分数冲突进入 conflict，不得确认。"""
    collector = _collector(tmp_path)
    collector.observe("70k@01", _start_game(2))
    collector.observe("70k@02", _start_game(0))
    collector.observe("70k@03", _start_game(3))
    collector.observe("70k@01", _end_game([42100, 28300, 18100, 11500]))
    collector.observe("70k@02", _end_game([40000, 30000, 20000, 10000]))
    collector.finalize()
    assert _read_state(collector.capture_dir)["state"] == "conflict"
    assert _read_pending(collector.capture_dir) == []


def test_collector_interrupted_game_no_pending(tmp_path: Path) -> None:
    """C9：没有 end_game 的中断局不计分。"""
    collector = _collector(tmp_path)
    collector.observe("70k@01", _start_game(2))
    collector.observe("70k@02", _start_game(0))
    collector.observe("70k@03", _start_game(3))
    collector.finalize()
    assert _read_state(collector.capture_dir)["state"] == "incomplete"
    assert _read_pending(collector.capture_dir) == []


# --- API: bindings & confirm ------------------------------------------------

@pytest.fixture()
def season_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    config_dir = tmp_path / "configs"
    config_dir.mkdir(parents=True)
    sources_root = tmp_path / "data" / "sources"
    season = {
        "schema": "keqing.ladder.season.v1",
        "season_id": "official-ladder-v1",
        "status": "running",
        "default": False,
        "report_dir": "seasons/official-ladder-v1/snapshots",
        "scoring": {
            "system": "tenhou_rank_progression",
            "version": "v1",
            "game_length": "hanchan",
        },
        "ingest": {"sources_root": str(sources_root)},
        "models": [
            {"model_id": "human", "accounts": [{"account_id": "nick@01", "display_name": "Nick"}]},
            {"model_id": "70k", "accounts": [
                {"account_id": "70k@01", "display_name": "70k-1"},
                {"account_id": "70k@02", "display_name": "70k-2"},
                {"account_id": "70k@03", "display_name": "70k-3"},
            ]},
        ],
    }
    (config_dir / "official-ladder-v1.json").write_text(json.dumps(season, ensure_ascii=False), encoding="utf-8")
    monkeypatch.setenv("KEQING_LADDER_CONFIG_DIR", str(config_dir))
    monkeypatch.setenv("KEQING_LADDER_DATA_ROOT", str(tmp_path / "data"))
    return {"root": tmp_path, "configs": config_dir, "season": season}


def _capture_request(**overrides) -> pwy.LadderCaptureRequest:
    defaults = {
        "enabled": True,
        "season_id": "official-ladder-v1",
        "human_account_id": "nick@01",
        "bot_account_ids": ["70k@01", "70k@02", "70k@03"],
        "mode": "confirm",
    }
    defaults.update(overrides)
    return pwy.LadderCaptureRequest(**defaults)


def test_capture_disabled_flow_unchanged() -> None:
    """C1：capture disabled 时原流程完全不变。"""
    assert pwy._validate_ladder_capture(
        pwy.LadderCaptureRequest(enabled=False), ["70k", "70k", "70k"]
    ) is None


def test_invalid_season_or_account_rejected_before_start(season_env) -> None:
    """C2：非法 season/account/checkpoint 在启动前拒绝。"""
    with pytest.raises(ValueError, match="season_id 不能为空"):
        pwy._validate_ladder_capture(_capture_request(season_id=""), ["70k", "70k", "70k"])
    with pytest.raises(ValueError, match="赛季不存在"):
        pwy._validate_ladder_capture(_capture_request(season_id="missing"), ["70k", "70k", "70k"])
    with pytest.raises(ValueError, match="未在赛季"):
        pwy._validate_ladder_capture(
            _capture_request(bot_account_ids=["ghost@01", "70k@02", "70k@03"]),
            ["70k", "70k", "70k"],
        )
    # bot 模型与 spec 不一致：绑定已注册的 70k 账号却选 ext_mortal spec
    with pytest.raises(ValueError, match="不能绑定 spec"):
        pwy._validate_ladder_capture(
            _capture_request(),
            ["ext_mortal", "ext_mortal", "ext_mortal"],
        )


def test_capture_dir_lives_outside_sources_root(season_env) -> None:
    """C12：capture 目录不在 sources_root 内，pending 文件不改变 ingest 指纹。"""
    from scripts.mortal import publish_ladder_snapshot as publisher

    capture_dir = capture_dir_for_session(Path(season_env["root"]) / "data", "abc123")
    pending_dir = capture_dir / "pending"
    pending_dir.mkdir(parents=True)
    (pending_dir / "m.json").write_text(json.dumps({"match_id": "m"}), encoding="utf-8")
    sources_root = Path(season_env["season"]["ingest"]["sources_root"])
    assert not str(sources_root.resolve()).startswith(str(capture_dir.resolve()))
    assert publisher._ordered_source_stats(sources_root) == []  # pending 不影响指纹


def test_confirm_writes_single_jsonl_and_publishes(tmp_path: Path, season_env) -> None:
    """C10+C11：confirm 原子写一局一个 JSONL；重复 confirm 幂等。"""
    from gateway.api import playwithyou as pwy_api
    from gateway.playwithyou_capture import CAPTURE_SCHEMA

    capture_dir = capture_dir_for_session(Path(season_env["root"]) / "data", "abc123")
    pending_dir = capture_dir / "pending"
    pending_dir.mkdir(parents=True)
    payload = {
        "schema": CAPTURE_SCHEMA,
        "capture_id": "abc123:tenhou:20260804gm-abc-xyz",
        "session_id": "abc123",
        "state": "pending_confirmation",
        "season_id": "official-ladder-v1",
        "match": {
            "match_id": "tenhou:20260804gm-abc-xyz",
            "occurred_at": "2026-08-04T07:00:00Z",
            "game_length": "hanchan",
            "players": [
                {"account_id": "nick@01", "seat": 1, "final_score": 42100},
                {"account_id": "70k@01", "seat": 2, "final_score": 28300},
                {"account_id": "70k@02", "seat": 0, "final_score": 18100},
                {"account_id": "70k@03", "seat": 3, "final_score": 11500},
            ],
        },
        "tenhou_log_url": "https://tenhou.net/3/?log=20260804gm-abc-xyz&tw=2",
        "observer_accounts": ["70k@01", "70k@02", "70k@03"],
        "score_observers": ["70k@01", "70k@02"],
    }
    (pending_dir / "tenhou_20260804gm-abc-xyz.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )

    result = pwy_api._confirm_capture("abc123:tenhou:20260804gm-abc-xyz")
    assert result["state"] == "published"
    assert result["games"] == 1

    sources_root = Path(season_env["season"]["ingest"]["sources_root"])
    source_files = list((sources_root / "playwithyou").glob("*.jsonl"))
    assert len(source_files) == 1
    lines = source_files[0].read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1  # 一局一行
    row = json.loads(lines[0])
    assert row["match_id"] == "tenhou:20260804gm-abc-xyz"
    assert len(row["players"]) == 4
    assert "table_room" not in row

    # C11：重复 confirm 幂等，不产生第二局
    pwy_api._confirm_capture("abc123:tenhou:20260804gm-abc-xyz")
    assert len(list((sources_root / "playwithyou").glob("*.jsonl"))) == 1


def test_publish_failure_keeps_accepted_source_and_can_retry(
    tmp_path: Path, season_env, monkeypatch
) -> None:
    """C13：publish 失败保留 accepted source，可 retry-publish。"""
    import scripts.mortal.publish_ladder_snapshot as publisher_mod
    from gateway.api import playwithyou as pwy_api
    from gateway.playwithyou_capture import CAPTURE_SCHEMA

    capture_dir = capture_dir_for_session(Path(season_env["root"]) / "data", "abc123")
    pending_dir = capture_dir / "pending"
    pending_dir.mkdir(parents=True)
    payload = {
        "schema": CAPTURE_SCHEMA,
        "capture_id": "abc123:tenhou:fail",
        "session_id": "abc123",
        "state": "pending_confirmation",
        "season_id": "official-ladder-v1",
        "match": {
            "match_id": "tenhou:fail",
            "occurred_at": "2026-08-04T07:00:00Z",
            "game_length": "hanchan",
            "players": [
                {"account_id": "nick@01", "seat": 0, "final_score": 42100},
                {"account_id": "70k@01", "seat": 1, "final_score": 28300},
                {"account_id": "70k@02", "seat": 2, "final_score": 18100},
                {"account_id": "70k@03", "seat": 3, "final_score": 11500},
            ],
        },
        "tenhou_log_url": "x",
        "observer_accounts": [],
        "score_observers": ["70k@01"],
    }
    (pending_dir / "tenhou_fail.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def _boom(**kwargs):
        raise RuntimeError("simulated publish failure")

    # 仅在此 context 内替换 publisher；退出后恢复真实实现且 env 保持不变
    with monkeypatch.context() as context:
        context.setattr(publisher_mod, "publish_snapshot", _boom)
        with pytest.raises(Exception, match="simulated publish failure"):
            pwy_api._confirm_capture("abc123:tenhou:fail")

    sources_root = Path(season_env["season"]["ingest"]["sources_root"])
    assert len(list((sources_root / "playwithyou").glob("*.jsonl"))) == 1  # accepted source 保留
    state_file = capture_dir / "pending" / "tenhou_fail.json"
    assert json.loads(state_file.read_text(encoding="utf-8"))["state"] == "accepted_publish_failed"

    # retry-publish：真实发布成功
    result = pwy_api._confirm_capture("abc123:tenhou:fail")
    assert result["state"] == "published"


def test_rediscover_pending_after_restart(tmp_path: Path, season_env) -> None:
    """C14：后端重启后从磁盘重发现 pending（不依赖 SESSIONS 内存）。"""
    from gateway.playwithyou_capture import CAPTURE_SCHEMA

    capture_dir = capture_dir_for_session(Path(season_env["root"]) / "data", "abc123")
    pending_dir = capture_dir / "pending"
    pending_dir.mkdir(parents=True)
    (pending_dir / "m.json").write_text(
        json.dumps(
            {
                "schema": CAPTURE_SCHEMA,
                "capture_id": "abc123:tenhou:m",
                "session_id": "abc123",
                "state": "pending_confirmation",
                "season_id": "official-ladder-v1",
                "match": {"match_id": "tenhou:m", "players": []},
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    captures = pwy._discover_captures()
    assert "abc123:tenhou:m" in [entry["capture_id"] for entry in captures]
