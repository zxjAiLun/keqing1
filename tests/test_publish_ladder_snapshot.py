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
    first = publisher.publish_snapshot(
        registry_path=registry_path,
        log_dirs=[tmp_path / "logs"],
        snapshot_root=tmp_path / "snapshots",
        build_report=_fake_build([_row()], games=1),
    )
    second = publisher.publish_snapshot(
        registry_path=registry_path,
        log_dirs=[tmp_path / "logs"],
        snapshot_root=tmp_path / "snapshots",
        build_report=_fake_build([_row()], games=1),
    )
    assert Path(first["snapshot_dir"]).exists()
    assert Path(second["snapshot_dir"]).exists()
    assert first["snapshot_dir"] != second["snapshot_dir"]
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
