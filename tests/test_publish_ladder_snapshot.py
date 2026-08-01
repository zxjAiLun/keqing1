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


def _fake_build(accounts: list[dict], games: int) -> Callable[..., dict[str, Any]]:
    def build(**kwargs: Any) -> dict[str, Any]:
        output_dir = Path(kwargs["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "account_summary.json").write_text(
            json.dumps({"schema": ladder.REPORT_SCHEMA, "games": games, "accounts": accounts}),
            encoding="utf-8",
        )
        return {"games": games, "accounts": accounts}
    return build


def test_switch_registry_is_atomic_and_readable(tmp_path: Path):
    season = _running_season(report_dir="artifacts/old-snapshot")
    registry_path = _write_registry(tmp_path, season)
    snapshot_dir = tmp_path / "snapshots" / "20260801-120000"
    snapshot_dir.mkdir(parents=True)

    publisher.switch_registry(registry_path, season, snapshot_dir)

    updated = ladder.read_registry(registry_path)
    assert updated["report_dir"] == str(snapshot_dir)
    assert not (tmp_path / "registries" / "dev-live.json.tmp").exists()


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
