# -*- coding: utf-8 -*-
"""R10-F：天梯投影自动消费 worker（P1-6）与 generation CAS（P1-2）。

- ``project_season``：P2-1 门禁（season 必须启用 ingest.participants.enabled）→
  冻结 dirty generation → 全量重建/发布 → CAS（generation 未变才清 dirty + ready，
  变了保持 dirty + pending 返回 needs_rebuild）。
- ``run_dirty_projection``：扫描全部 dirty marker，逐赛季投影（coalesce 多次 dirty）。
- worker：服务启动时扫描 dirty + Match mutation 后唤醒 + 定时兜底。
"""
from __future__ import annotations

import threading
from pathlib import Path

from . import ledger
from .paths import data_root

PROJECT_ROOT = Path(__file__).resolve().parents[2]

_wake = threading.Event()
_thread: threading.Thread | None = None
_thread_lock = threading.Lock()


def _dirty_markers() -> list[Path]:
    root = data_root()
    if not root.is_dir():
        return []
    return sorted(root.glob("ladder_dirty_*.json"))


def project_season(season_id: str) -> dict:
    """投影单个赛季。返回 {season_id, state: ready|error|needs_rebuild, ...}。"""
    from replay import ladder as ladder_data

    configs_dir = ladder_data.resolve_config_dir(PROJECT_ROOT)
    registry_path = configs_dir / f"{season_id}.json"
    if not registry_path.is_file():
        ledger.set_season_projection_state(season_id, "error")
        return {"season_id": season_id, "state": "error", "reason": f"赛季配置不存在: {season_id}"}
    try:
        season = ladder_data.get_season_config(configs_dir, season_id)
    except ladder_data.SeasonNotFoundError as exc:
        ledger.set_season_projection_state(season_id, "error")
        return {"season_id": season_id, "state": "error", "reason": str(exc)}
    ingest = season.get("ingest") if isinstance(season.get("ingest"), dict) else {}
    participants_cfg = ingest.get("participants")
    # P2-1：fail-fast 门禁——普通 ingest season 不允许清 dirty/标 ready
    if not (isinstance(participants_cfg, dict) and participants_cfg.get("enabled")):
        ledger.set_season_projection_state(season_id, "error")
        return {
            "season_id": season_id,
            "state": "error",
            "reason": "赛季未启用 participants 投影（ingest.participants.enabled）",
        }

    # P1-2：发布前冻结 generation
    start_generation = ledger.read_ladder_generation(season_id)
    try:
        from scripts.mortal.publish_ladder_snapshot import publish_snapshot

        result = publish_snapshot(registry_path=registry_path, log_dirs=[])
    except Exception as exc:  # noqa: BLE001
        ledger.set_season_projection_state(season_id, "error")
        return {"season_id": season_id, "state": "error", "reason": str(exc)}

    # P1-2 CAS：发布期间有新写入（generation 变）→ 保留 dirty + pending
    if ledger.complete_ladder_projection(season_id, start_generation):
        return {
            "season_id": season_id,
            "state": "ready",
            "snapshot_dir": result.get("snapshot_dir"),
            "games": result.get("games"),
        }
    return {
        "season_id": season_id,
        "state": "needs_rebuild",
        "reason": "发布期间有新的账本写入，dirty 已保留",
    }


def run_dirty_projection() -> list[dict]:
    """扫描并投影全部 dirty 赛季（worker / 启动恢复 / 手动重试共用）。"""
    results: list[dict] = []
    for marker in _dirty_markers():
        season_id = marker.name[len("ladder_dirty_"):-len(".json")]
        results.append(project_season(season_id))
    return results


def request_projection(season_id: str | None) -> None:
    """Match mutation 后唤醒 worker（dirty 已由 ledger 写入）。"""
    if not season_id:
        return
    _wake.set()


def start_worker() -> None:
    """启动后台 worker（幂等）：扫描现有 dirty + 唤醒后投影 + 10s 兜底轮询。"""
    global _thread
    with _thread_lock:
        if _thread is not None and _thread.is_alive():
            return

        def _loop() -> None:
            while True:
                try:
                    run_dirty_projection()
                except Exception:  # noqa: BLE001
                    pass
                _wake.wait(timeout=10)
                _wake.clear()

        _thread = threading.Thread(target=_loop, name="ladder-projection", daemon=True)
        _thread.start()


__all__ = [
    "project_season",
    "run_dirty_projection",
    "request_projection",
    "start_worker",
]
