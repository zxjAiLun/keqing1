#!/usr/bin/env python3
"""Publish an atomic Ladder snapshot for a dynamic (dev) season.

staging build -> validation -> manifest -> atomic registry switch.

- 每次发布生成一个全新的不可变快照目录，绝不原地覆写在线目录；
- 校验通过后通过 ``.tmp + os.replace`` 原子切换 registry 的 ``report_dir``；
- 校验失败时删除本次 staging 目录，旧快照与旧 registry 保持可用；
- API 保持纯只读，本脚本由训练线在固定间隔触发。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from replay import ladder  # noqa: E402

MANIFEST_SCHEMA = "keqing.ladder.snapshot.v1"


class PublishError(Exception):
    """发布流程错误（completed 赛季、目录冲突等）。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True,
                        help="动态赛季注册表 JSON 路径（如 keqing-data/ladder/registries/dev-live.json）")
    parser.add_argument("--log-dir", action="append", type=Path, required=True,
                        help="输入 mjai 日志目录，可重复")
    parser.add_argument("--snapshot-root", type=Path, default=None,
                        help="快照根目录；默认 <KEQING_LADDER_DATA_ROOT>/seasons/<season_id>/snapshots")
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--platform-model-label", default=None,
                        help="强制所有座位为 MODEL@01-04（临时演示用）")
    parser.add_argument("--rank-points", default="90,45,0,-135")
    parser.add_argument("--preserve-log-dir-order", action="store_true")
    parser.add_argument("--interleave-log-dirs", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="只构建并校验，不切换 registry")
    return parser.parse_args()


def load_registry(registry_path: Path) -> dict[str, Any]:
    try:
        return ladder.read_registry(registry_path)
    except ladder.SeasonRegistryError as exc:
        raise PublishError(str(exc)) from exc


def default_snapshot_root(data_root: Path | None, season_id: str) -> Path:
    base = data_root or Path.cwd()
    return base / "seasons" / season_id / "snapshots"


def build_snapshot(
    *,
    log_dirs: list[Path],
    snapshot_dir: Path,
    mortal_root: Path,
    platform_model_label: str | None,
    rank_points: str,
    preserve_log_dir_order: bool,
    interleave_log_dirs: bool,
    build_report: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """在全新快照目录构建完整 report（staging）。"""
    from scripts.mortal import build_platform_account_report as account_report

    builder = build_report or account_report.build_report
    report = builder(
        log_dirs=log_dirs,
        output_dir=snapshot_dir,
        mortal_root=mortal_root,
        platform_model_label=platform_model_label,
        rank_points=tuple(float(part) for part in rank_points.split(",") if part.strip()),
        preserve_log_dir_order=preserve_log_dir_order,
        interleave_log_dirs=interleave_log_dirs,
    )
    if not isinstance(report, dict):
        raise PublishError("构建脚本未返回 report 字典")
    return report


def write_manifest(
    snapshot_dir: Path,
    *,
    season_id: str,
    snapshot_id: str,
    report: dict[str, Any],
    registry_path: Path,
    previous_report_dir: str | None,
) -> Path:
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "season_id": season_id,
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "games": report.get("games"),
        "accounts": len(report.get("accounts") or []),
        "source_registry": str(registry_path),
        "previous_report_dir": previous_report_dir,
    }
    manifest_path = snapshot_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest_path


def switch_registry(registry_path: Path, season: dict[str, Any], new_report_dir: Path) -> None:
    """原子切换 registry 的 report_dir：写 .tmp -> 结构校验 -> os.replace。"""
    updated = dict(season)
    updated["report_dir"] = str(new_report_dir)
    tmp_path = registry_path.with_name(registry_path.name + ".tmp")
    tmp_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    try:
        ladder.read_registry(tmp_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise
    os.replace(tmp_path, registry_path)


def publish_snapshot(
    *,
    registry_path: Path,
    log_dirs: list[Path],
    snapshot_root: Path | None = None,
    mortal_root: Path = Path("third_party/Mortal"),
    platform_model_label: str | None = None,
    rank_points: str = "90,45,0,-135",
    preserve_log_dir_order: bool = False,
    interleave_log_dirs: bool = False,
    dry_run: bool = False,
    build_report: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """构建 -> 校验 -> manifest -> 原子切换 registry（staging 流程）。

    返回 dict：{snapshot_dir, games, dry_run, registry_switched}。
    校验失败时删除本次 staging 目录并抛出原异常，旧快照与旧 registry 不受影响。
    """
    season = load_registry(registry_path)
    season_id = str(season["season_id"])
    if str(season.get("status") or "") == "completed":
        raise PublishError(f"season {season_id}: completed 赛季不可发布快照")

    data_root = os.environ.get("KEQING_LADDER_DATA_ROOT", "").strip()
    root = snapshot_root or default_snapshot_root(Path(data_root) if data_root else None, season_id)
    # 秒级时间戳命名；同一秒内多次发布时追加序号，避免目录冲突
    base_name = datetime.now().strftime("%Y%m%d-%H%M%S")
    snapshot_dir = root / base_name
    attempt = 1
    while snapshot_dir.exists():
        attempt += 1
        snapshot_dir = root / f"{base_name}-{attempt}"
    snapshot_dir.mkdir(parents=True)

    previous_report_dir = season.get("report_dir")
    try:
        report = build_snapshot(
            log_dirs=log_dirs,
            snapshot_dir=snapshot_dir,
            mortal_root=mortal_root,
            platform_model_label=platform_model_label,
            rank_points=rank_points,
            preserve_log_dir_order=preserve_log_dir_order,
            interleave_log_dirs=interleave_log_dirs,
            build_report=build_report,
        )
        ladder.validate_snapshot(season, snapshot_dir)
        write_manifest(
            snapshot_dir,
            season_id=season_id,
            snapshot_id=snapshot_dir.name,
            report=report,
            registry_path=registry_path,
            previous_report_dir=str(previous_report_dir) if previous_report_dir else None,
        )
        if dry_run:
            return {
                "snapshot_dir": str(snapshot_dir),
                "games": report.get("games"),
                "dry_run": True,
                "registry_switched": False,
            }
        switch_registry(registry_path, season, snapshot_dir)
        return {
            "snapshot_dir": str(snapshot_dir),
            "games": report.get("games"),
            "dry_run": False,
            "registry_switched": True,
        }
    except Exception:
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        raise


def main() -> None:
    args = parse_args()
    result = publish_snapshot(
        registry_path=args.registry,
        log_dirs=args.log_dir,
        snapshot_root=args.snapshot_root,
        mortal_root=args.mortal_root,
        platform_model_label=args.platform_model_label,
        rank_points=args.rank_points,
        preserve_log_dir_order=args.preserve_log_dir_order,
        interleave_log_dirs=args.interleave_log_dirs,
        dry_run=args.dry_run,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

