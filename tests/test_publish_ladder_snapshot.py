from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Callable

import pytest

_ROOT = Path(__file__).resolve().parents[1]
for entry in (str(_ROOT), str(_ROOT / "src")):
    if entry not in sys.path:
        sys.path.insert(0, entry)

from scripts.mortal import publish_ladder_snapshot as publisher  # noqa: E402
from replay import ladder  # noqa: E402


def _running_season(report_dir: str = "artifacts/old-snapshot") -> dict:
    return {
        "schema": ladder.SEASON_SCHEMA,
        "season_id": "dev-live",
        "status": "running",
        "report_dir": report_dir,
        "models": [{"model_id": "m1", "checkpoint": "artifacts/m1.pth", "accounts": [{"account_id": "m1@01"}]}],
    }


def _row(account_id: str = "m1@01", model_label: str = "m1") -> dict:
    return {
        "account_id": account_id,
        "model_label": model_label,
        "games": 1,
        "rank_name": "七段",
        "pt_current": 1500.0,
        "pt_target": 2800.0,
        "rating": 1500.0,
        "rank_1": 1,
        "rank_2": 0,
        "rank_3": 0,
        "rank_4": 0,
        "avg_rank": 1.0,
    }


def _write_registry(tmp_path: Path, season: dict) -> Path:
    registry_dir = tmp_path / "registries"
    registry_dir.mkdir(parents=True)
    registry_path = registry_dir / "dev-live.json"
    registry_path.write_text(json.dumps(season), encoding="utf-8")
    return registry_path


def _fake_build(accounts: list[dict], games: int, omit: tuple[str, ...] = ()) -> Callable[..., dict[str, Any]]:
    def build(**kwargs: Any) -> dict[str, Any]:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        if "account_summary.json" not in omit:
            (output_dir / "account_summary.json").write_text(
                json.dumps({"schema": ladder.REPORT_SCHEMA, "games": games, "accounts": accounts}),
                encoding="utf-8",
            )
        if "account_ledger.jsonl" not in omit:
            (output_dir / "account_ledger.jsonl").write_text("", encoding="utf-8")
        if "rating_curve.csv" not in omit:
            (output_dir / "rating_curve.csv").write_text(
                "game_index,account_id,model_label,rating,pt,rank_name,games\n", encoding="utf-8")
        return {"games": games, "accounts": accounts}
    return build


def test_conditional_switch_registry_is_atomic_and_readable(tmp_path: Path):
    season = _running_season(report_dir="artifacts/old-snapshot")
    registry_path = _write_registry(tmp_path, season)
    snapshot_dir = tmp_path / "snapshots" / "20260801-120000"
    snapshot_dir.mkdir(parents=True)

    publisher.conditional_switch_registry(
        registry_path,
        expected_season=season,
        expected_report_dir="artifacts/old-snapshot",
        new_report_dir=snapshot_dir,
    )

    updated = ladder.read_registry(registry_path)
    assert updated["report_dir"] == str(snapshot_dir)
    assert not list((tmp_path / "registries").glob("*.tmp"))
    assert not (tmp_path / "registries" / "dev-live.json.lock").exists()


def test_publish_dry_run_builds_and_validates_without_switching(tmp_path: Path):
    registry_path = _write_registry(tmp_path, _running_season())
    result = publisher.publish_snapshot(
        registry_path=registry_path,
        log_dirs=[tmp_path / "logs"],
        snapshot_root=tmp_path / "snapshots",
        dry_run=True,
        build_report=_fake_build([_row()], games=1),
    )
    assert result["dry_run"] is True
    assert result["registry_switched"] is False
    snapshot_dir = Path(result["snapshot_dir"])
    assert snapshot_dir.exists()
    assert (snapshot_dir / "account_summary.json").exists()
    assert (snapshot_dir / "manifest.json").exists()
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema"] == publisher.MANIFEST_SCHEMA
    assert manifest["season_id"] == "dev-live"
    assert ladder.read_registry(registry_path)["report_dir"] == "artifacts/old-snapshot"


def test_publish_switches_registry(tmp_path: Path):
    registry_path = _write_registry(tmp_path, _running_season())
    result = publisher.publish_snapshot(
        registry_path=registry_path,
        log_dirs=[tmp_path / "logs"],
        snapshot_root=tmp_path / "snapshots",
        dry_run=False,
        build_report=_fake_build([_row()], games=1),
    )
    assert result["registry_switched"] is True
    assert ladder.read_registry(registry_path)["report_dir"] == str(Path(result["snapshot_dir"]))


def test_publish_rejects_completed_season(tmp_path: Path):
    season = _running_season()
    season["status"] = "completed"
    registry_path = _write_registry(tmp_path, season)
    with pytest.raises(publisher.PublishError, match="completed"):
        publisher.publish_snapshot(
            registry_path=registry_path,
            log_dirs=[tmp_path / "logs"],
            snapshot_root=tmp_path / "snapshots",
            build_report=_fake_build([_row()], games=1),
        )
    assert ladder.read_registry(registry_path)["report_dir"] == "artifacts/old-snapshot"


def test_publish_removes_staging_on_invalid_snapshot(tmp_path: Path):
    registry_path = _write_registry(tmp_path, _running_season())
    bad_rows = [_row(), _row(account_id="ghost@01")]
    with pytest.raises(ladder.SeasonDataError, match="未在注册表声明"):
        publisher.publish_snapshot(
            registry_path=registry_path,
            log_dirs=[tmp_path / "logs"],
            snapshot_root=tmp_path / "snapshots",
            build_report=_fake_build(bad_rows, games=2),
        )
    snapshots = tmp_path / "snapshots"
    assert not list(snapshots.glob("*")) if snapshots.exists() else True
    assert ladder.read_registry(registry_path)["report_dir"] == "artifacts/old-snapshot"


def test_publish_keeps_previous_snapshot_available(tmp_path: Path):
    registry_path = _write_registry(tmp_path, _running_season())
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "a.json.gz").write_text("", encoding="utf-8")
    first = publisher.publish_snapshot(
        registry_path=registry_path,
        log_dirs=[log_dir],
        snapshot_root=tmp_path / "snapshots",
        build_report=_fake_build([_row()], games=1),
    )
    # 新增日志使指纹变化，触发真实二次发布
    (log_dir / "b.json.gz").write_text("", encoding="utf-8")
    second = publisher.publish_snapshot(
        registry_path=registry_path,
        log_dirs=[log_dir],
        snapshot_root=tmp_path / "snapshots",
        build_report=_fake_build([_row()], games=2),
    )
    assert Path(first["snapshot_dir"]).exists()
    assert Path(second["snapshot_dir"]).exists()
    assert first["snapshot_dir"] != second["snapshot_dir"]
    assert second["registry_switched"] is True
    assert ladder.read_registry(registry_path)["report_dir"] == str(Path(second["snapshot_dir"]))


def test_stale_publisher_cannot_overwrite_advanced_report_dir(tmp_path: Path):
    season = _running_season(report_dir="artifacts/old-snapshot")
    registry_path = _write_registry(tmp_path, season)
    snapshot_dir = tmp_path / "snapshots" / "20260801-120000"
    snapshot_dir.mkdir(parents=True)
    # 构建期间另一个发布已把 report_dir 推进到 newer
    advanced = dict(season)
    advanced["report_dir"] = "artifacts/newer-snapshot"
    registry_path.write_text(json.dumps(advanced), encoding="utf-8")

    with pytest.raises(publisher.PublishError, match="已推进"):
        publisher.conditional_switch_registry(
            registry_path,
            expected_season=season,
            expected_report_dir="artifacts/old-snapshot",
            new_report_dir=snapshot_dir,
        )
    assert ladder.read_registry(registry_path)["report_dir"] == "artifacts/newer-snapshot"


def test_registry_contract_change_rejects_stale_publish(tmp_path: Path):
    season = _running_season()
    registry_path = _write_registry(tmp_path, season)
    snapshot_dir = tmp_path / "snapshots" / "x"
    snapshot_dir.mkdir(parents=True)
    # 构建期间训练线新增了账号：契约变化
    changed = _running_season()
    changed["models"][0]["accounts"].append({"account_id": "m1@02"})
    registry_path.write_text(json.dumps(changed), encoding="utf-8")

    with pytest.raises(publisher.PublishError, match="契约"):
        publisher.conditional_switch_registry(
            registry_path,
            expected_season=season,
            expected_report_dir="artifacts/old-snapshot",
            new_report_dir=snapshot_dir,
        )
    assert len(ladder.read_registry(registry_path)["models"][0]["accounts"]) == 2


def test_tmp_paths_are_unique(tmp_path: Path):
    registry_path = tmp_path / "registries" / "dev-live.json"
    first = publisher._tmp_path_for(registry_path)
    second = publisher._tmp_path_for(registry_path)
    assert first != second
    assert first.name.endswith(".tmp") and second.name.endswith(".tmp")
    assert ".lock" not in first.name


def test_lock_conflict_preserves_registry(tmp_path: Path):
    season = _running_season()
    registry_path = _write_registry(tmp_path, season)
    lock_path = tmp_path / "registries" / "dev-live.json.lock"
    lock_path.write_text("99999\n", encoding="ascii")
    snapshot_dir = tmp_path / "snapshots" / "x"
    snapshot_dir.mkdir(parents=True)

    with pytest.raises(publisher.PublishError, match="锁获取超时"):
        publisher.conditional_switch_registry(
            registry_path,
            expected_season=season,
            expected_report_dir="artifacts/old-snapshot",
            new_report_dir=snapshot_dir,
            timeout=0.5,
        )
    assert ladder.read_registry(registry_path)["report_dir"] == "artifacts/old-snapshot"
    assert lock_path.exists()  # 锁文件由持锁方负责删除


def test_conflicted_publish_cleans_staging(tmp_path: Path):
    registry_path = _write_registry(tmp_path, _running_season())

    def build_advances_registry(**kwargs: Any) -> dict[str, Any]:
        # 模拟并发发布在构建期间推进了 registry
        advanced = _running_season()
        advanced["report_dir"] = "artifacts/other-snapshot"
        registry_path.write_text(json.dumps(advanced), encoding="utf-8")
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "account_summary.json").write_text(
            json.dumps({"schema": ladder.REPORT_SCHEMA, "games": 1, "accounts": [_row()]}),
            encoding="utf-8")
        (output_dir / "account_ledger.jsonl").write_text("", encoding="utf-8")
        (output_dir / "rating_curve.csv").write_text(
            "game_index,account_id,model_label,rating,pt,rank_name,games\n", encoding="utf-8")
        return {"games": 1, "accounts": [_row()]}

    with pytest.raises(publisher.PublishError, match="已推进"):
        publisher.publish_snapshot(
            registry_path=registry_path,
            log_dirs=[tmp_path / "logs"],
            snapshot_root=tmp_path / "snapshots",
            build_report=build_advances_registry,
        )
    assert ladder.read_registry(registry_path)["report_dir"] == "artifacts/other-snapshot"
    assert not list((tmp_path / "snapshots").glob("*"))


def test_missing_rating_curve_rejected(tmp_path: Path):
    registry_path = _write_registry(tmp_path, _running_season())
    with pytest.raises(ladder.SeasonDataError, match="rating_curve.csv"):
        publisher.publish_snapshot(
            registry_path=registry_path,
            log_dirs=[tmp_path / "logs"],
            snapshot_root=tmp_path / "snapshots",
            build_report=_fake_build([_row()], games=1, omit=("rating_curve.csv",)),
        )
    assert ladder.read_registry(registry_path)["report_dir"] == "artifacts/old-snapshot"
    assert not list((tmp_path / "snapshots").glob("*"))


def test_missing_ledger_rejected(tmp_path: Path):
    registry_path = _write_registry(tmp_path, _running_season())
    with pytest.raises(ladder.SeasonDataError, match="account_ledger.jsonl"):
        publisher.publish_snapshot(
            registry_path=registry_path,
            log_dirs=[tmp_path / "logs"],
            snapshot_root=tmp_path / "snapshots",
            build_report=_fake_build([_row()], games=1, omit=("account_ledger.jsonl",)),
        )
    assert ladder.read_registry(registry_path)["report_dir"] == "artifacts/old-snapshot"
    assert not list((tmp_path / "snapshots").glob("*"))


def test_rank_points_invalid_rejected(tmp_path: Path):
    registry_path = _write_registry(tmp_path, _running_season())
    with pytest.raises(ValueError, match="four numbers"):
        publisher.publish_snapshot(
            registry_path=registry_path,
            log_dirs=[tmp_path / "logs"],
            snapshot_root=tmp_path / "snapshots",
            rank_points="1,2,3",
            build_report=_fake_build([_row()], games=1),
        )
    assert ladder.read_registry(registry_path)["report_dir"] == "artifacts/old-snapshot"


# ---------------------------------------------------------------------------
# Round 5: compact publication, retention, telemetry
# ---------------------------------------------------------------------------

def _fake_build_with_account_logs(accounts: list[dict], games: int) -> Callable[..., dict[str, Any]]:
    def build(**kwargs: Any) -> dict[str, Any]:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "account_summary.json").write_text(
            json.dumps({"schema": ladder.REPORT_SCHEMA, "games": games, "accounts": accounts}),
            encoding="utf-8")
        (output_dir / "account_ledger.jsonl").write_text("", encoding="utf-8")
        (output_dir / "rating_curve.csv").write_text(
            "game_index,account_id,model_label,rating,pt,rank_name,games\n", encoding="utf-8")
        account_logs = output_dir / "account_logs"
        account_logs.mkdir(parents=True)
        (account_logs / "x.json.gz").write_text("payload", encoding="utf-8")
        return {"games": games, "accounts": accounts}
    return build


def _seed_snapshot(root: Path, name: str, *, created_at: str, season_id: str = "dev-live") -> Path:
    snapshot = root / name
    snapshot.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": publisher.MANIFEST_SCHEMA,
        "season_id": season_id,
        "snapshot_id": name,
        "created_at": created_at,
        "games": 1,
        "accounts": 1,
    }
    (snapshot / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return snapshot


def test_compact_snapshot_excludes_account_logs_by_default(tmp_path: Path):
    registry_path = _write_registry(tmp_path, _running_season())
    result = publisher.publish_snapshot(
        registry_path=registry_path,
        log_dirs=[tmp_path / "logs"],
        snapshot_root=tmp_path / "snapshots",
        build_report=_fake_build_with_account_logs([_row()], games=1),
    )
    snapshot_dir = Path(result["snapshot_dir"])
    assert not (snapshot_dir / "account_logs").exists()
    for name in publisher.SNAPSHOT_REQUIRED_FILES:
        assert (snapshot_dir / name).is_file()


def test_keep_account_logs_preserves_derived_logs(tmp_path: Path):
    registry_path = _write_registry(tmp_path, _running_season())
    result = publisher.publish_snapshot(
        registry_path=registry_path,
        log_dirs=[tmp_path / "logs"],
        snapshot_root=tmp_path / "snapshots",
        keep_account_logs=True,
        build_report=_fake_build_with_account_logs([_row()], games=1),
    )
    assert (Path(result["snapshot_dir"]) / "account_logs" / "x.json.gz").is_file()


def test_build_failure_cleans_hidden_staging_only(tmp_path: Path):
    registry_path = _write_registry(tmp_path, _running_season())

    def failing_build(**kwargs: Any) -> dict[str, Any]:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "partial.json").write_text("x", encoding="utf-8")
        raise RuntimeError("build exploded")

    with pytest.raises(RuntimeError, match="build exploded"):
        publisher.publish_snapshot(
            registry_path=registry_path,
            log_dirs=[tmp_path / "logs"],
            snapshot_root=tmp_path / "snapshots",
            build_report=failing_build,
        )
    snapshots = tmp_path / "snapshots"
    assert not list(snapshots.glob("*"))
    assert not list(snapshots.glob(".*.staging"))
    assert ladder.read_registry(registry_path)["report_dir"] == "artifacts/old-snapshot"


def test_validation_failure_does_not_materialize_snapshot(tmp_path: Path):
    registry_path = _write_registry(tmp_path, _running_season())
    with pytest.raises(ladder.SeasonDataError, match="未在注册表声明"):
        publisher.publish_snapshot(
            registry_path=registry_path,
            log_dirs=[tmp_path / "logs"],
            snapshot_root=tmp_path / "snapshots",
            build_report=_fake_build([_row(), _row(account_id="ghost@01")], games=2),
        )
    snapshots = tmp_path / "snapshots"
    assert not list(snapshots.glob("*"))
    assert not list(snapshots.glob(".*.staging"))


def test_registry_conflict_removes_orphan_snapshot(tmp_path: Path):
    registry_path = _write_registry(tmp_path, _running_season())

    def build_advances_registry(**kwargs: Any) -> dict[str, Any]:
        advanced = _running_season()
        advanced["report_dir"] = "artifacts/other-snapshot"
        registry_path.write_text(json.dumps(advanced), encoding="utf-8")
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "account_summary.json").write_text(
            json.dumps({"schema": ladder.REPORT_SCHEMA, "games": 1, "accounts": [_row()]}),
            encoding="utf-8")
        (output_dir / "account_ledger.jsonl").write_text("", encoding="utf-8")
        (output_dir / "rating_curve.csv").write_text(
            "game_index,account_id,model_label,rating,pt,rank_name,games\n", encoding="utf-8")
        return {"games": 1, "accounts": [_row()]}

    with pytest.raises(publisher.PublishError, match="已推进"):
        publisher.publish_snapshot(
            registry_path=registry_path,
            log_dirs=[tmp_path / "logs"],
            snapshot_root=tmp_path / "snapshots",
            build_report=build_advances_registry,
        )
    assert ladder.read_registry(registry_path)["report_dir"] == "artifacts/other-snapshot"
    assert not list((tmp_path / "snapshots").glob("*"))


def test_retention_keeps_only_latest_n(tmp_path: Path):
    root = tmp_path / "snapshots"
    root.mkdir(parents=True)
    for index in range(4):
        _seed_snapshot(root, f"s{index}", created_at=f"2000-01-01T00:00:0{index}Z")
    result = publisher.enforce_retention(
        root,
        season_id="dev-live",
        retain=2,
        current_snapshot_dir=root / "s3",
        previous_report_dir=None,
        protected_snapshots=set(),
    )
    assert sorted(Path(name).name for name in result["retention_deleted"]) == ["s0", "s1"]
    assert (root / "s2").exists() and (root / "s3").exists()


def test_current_snapshot_never_deleted(tmp_path: Path):
    root = tmp_path / "snapshots"
    root.mkdir(parents=True)
    for index in range(3):
        _seed_snapshot(root, f"s{index}", created_at=f"2000-01-01T00:00:0{index}Z")
    result = publisher.enforce_retention(
        root,
        season_id="dev-live",
        retain=1,
        current_snapshot_dir=root / "s1",
        previous_report_dir=None,
        protected_snapshots=set(),
    )
    # s1 虽非最新，但作为当前 registry 指向必须保留；s2 为最新正常保留；仅最旧 s0 被清理
    assert (root / "s1").exists()
    assert (root / "s2").exists()
    assert not (root / "s0").exists()


def test_previous_snapshot_kept_as_rollback_point(tmp_path: Path):
    root = tmp_path / "snapshots"
    root.mkdir(parents=True)
    for index in range(3):
        _seed_snapshot(root, f"s{index}", created_at=f"2000-01-01T00:00:0{index}Z")
    result = publisher.enforce_retention(
        root,
        season_id="dev-live",
        retain=1,
        current_snapshot_dir=root / "s2",
        previous_report_dir=str(root / "s0"),
        protected_snapshots=set(),
    )
    assert (root / "s2").exists()
    assert (root / "s0").exists()
    assert not (root / "s1").exists()


def test_retention_ignores_foreign_dirs_and_manifests(tmp_path: Path):
    root = tmp_path / "snapshots"
    root.mkdir(parents=True)
    _seed_snapshot(root, "s0", created_at="2000-01-01T00:00:00Z")
    _seed_snapshot(root, "s1", created_at="2000-01-01T00:00:01Z", season_id="other-season")
    (root / "no_manifest").mkdir()
    (root / "notes.txt").write_text("keep me", encoding="utf-8")
    (root / ".hidden-staging").mkdir()
    result = publisher.enforce_retention(
        root,
        season_id="dev-live",
        retain=1,
        current_snapshot_dir=root / "s0",
        previous_report_dir=None,
        protected_snapshots=set(),
    )
    assert not result["retention_deleted"]
    assert (root / "s0").exists()
    assert (root / "s1").exists()
    assert (root / "no_manifest").exists()
    assert (root / "notes.txt").exists()
    assert (root / ".hidden-staging").exists()


def test_dry_run_does_not_trigger_retention(tmp_path: Path):
    registry_path = _write_registry(tmp_path, _running_season())
    root = tmp_path / "snapshots"
    root.mkdir(parents=True)
    _seed_snapshot(root, "old0", created_at="2000-01-01T00:00:00Z")
    _seed_snapshot(root, "old1", created_at="2000-01-01T00:00:01Z")
    result = publisher.publish_snapshot(
        registry_path=registry_path,
        log_dirs=[tmp_path / "logs"],
        snapshot_root=root,
        dry_run=True,
        retain_snapshots=1,
        build_report=_fake_build([_row()], games=1),
    )
    assert result["dry_run"] is True
    assert result["registry_switched"] is False
    assert result["retention_deleted"] == []
    assert ladder.read_registry(registry_path)["report_dir"] == "artifacts/old-snapshot"
    assert (root / "old0").exists() and (root / "old1").exists()


def test_same_fingerprint_second_publish_skipped(tmp_path: Path):
    registry_path = _write_registry(tmp_path, _running_season())
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "a.json.gz").write_text("", encoding="utf-8")
    first = publisher.publish_snapshot(
        registry_path=registry_path,
        log_dirs=[log_dir],
        snapshot_root=tmp_path / "snapshots",
        build_report=_fake_build([_row()], games=1),
    )
    second = publisher.publish_snapshot(
        registry_path=registry_path,
        log_dirs=[log_dir],
        snapshot_root=tmp_path / "snapshots",
        build_report=_fake_build([_row()], games=1),
    )
    assert first["registry_switched"] is True
    assert second["skipped_unchanged"] is True
    assert second["registry_switched"] is False
    assert second["snapshot_dir"] == first["snapshot_dir"]


def test_fingerprint_change_triggers_new_snapshot(tmp_path: Path, monkeypatch):
    registry_path = _write_registry(tmp_path, _running_season())
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    log_file = log_dir / "a.json.gz"
    log_file.write_text("v1", encoding="utf-8")
    first = publisher.publish_snapshot(
        registry_path=registry_path,
        log_dirs=[log_dir],
        snapshot_root=tmp_path / "snapshots",
        build_report=_fake_build([_row()], games=1),
    )

    # 大小变化
    log_file.write_text("v1-longer", encoding="utf-8")
    second = publisher.publish_snapshot(
        registry_path=registry_path,
        log_dirs=[log_dir],
        snapshot_root=tmp_path / "snapshots",
        build_report=_fake_build([_row()], games=2),
    )
    assert second["registry_switched"] is True
    assert second["snapshot_dir"] != first["snapshot_dir"]

    # 仅 mtime 变化
    import os
    os.utime(log_file, (1_600_000_000, 1_600_000_001))
    third = publisher.publish_snapshot(
        registry_path=registry_path,
        log_dirs=[log_dir],
        snapshot_root=tmp_path / "snapshots",
        build_report=_fake_build([_row()], games=3),
    )
    assert third["registry_switched"] is True
    assert third["snapshot_dir"] != second["snapshot_dir"]


def test_retention_failure_does_not_fail_switch(tmp_path: Path, monkeypatch):
    import shutil as shutil_module

    registry_path = _write_registry(tmp_path, _running_season())
    root = tmp_path / "snapshots"
    root.mkdir(parents=True)
    _seed_snapshot(root, "old0", created_at="2000-01-01T00:00:00Z")
    _seed_snapshot(root, "old1", created_at="2000-01-01T00:00:01Z")

    real_rmtree = shutil_module.rmtree  # 在 patch 前绑定真实函数

    def flaky_rmtree(path, **kwargs):
        if Path(path).name == "old0":
            raise OSError("denied")
        return real_rmtree(path, **kwargs)

    monkeypatch.setattr(shutil_module, "rmtree", flaky_rmtree)
    result = publisher.publish_snapshot(
        registry_path=registry_path,
        log_dirs=[tmp_path / "logs"],
        snapshot_root=root,
        retain_snapshots=1,
        build_report=_fake_build([_row()], games=1),
    )
    assert result["registry_switched"] is True
    assert result["retention_errors"]
    # 本次新快照为最新被保留；old1 为可清理项且删除成功 → 已删；old0 删除失败 → 仍在
    assert not (root / "old1").exists()
    assert (root / "old0").exists()


def test_manifest_telemetry_matches_actual_snapshot(tmp_path: Path):
    registry_path = _write_registry(tmp_path, _running_season())
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True)
    (log_dir / "a.json.gz").write_text("x" * 100, encoding="utf-8")
    result = publisher.publish_snapshot(
        registry_path=registry_path,
        log_dirs=[log_dir],
        snapshot_root=tmp_path / "snapshots",
        build_report=_fake_build_with_account_logs([_row()], games=1),
    )
    snapshot_dir = Path(result["snapshot_dir"])
    manifest = json.loads((snapshot_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["compact"] is True
    assert manifest["kept_account_logs"] is False
    assert manifest["source_file_count"] == 1
    assert manifest["source_total_bytes"] == 100
    expected_bytes = sum(
        (snapshot_dir / name).stat().st_size
        for name in publisher.SNAPSHOT_REQUIRED_FILES + publisher.SNAPSHOT_RECOMMENDED_FILES
        if (snapshot_dir / name).is_file()
    )
    assert manifest["snapshot_total_bytes"] == expected_bytes
    assert manifest["source_fingerprint"]
    assert manifest["registry_contract"]
    assert manifest["build_duration_seconds"] >= 0
    assert manifest["materialize_duration_seconds"] >= 0
    assert not (snapshot_dir / "account_logs").exists()
