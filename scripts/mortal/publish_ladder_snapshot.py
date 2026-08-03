#!/usr/bin/env python3
"""Publish an atomic Ladder snapshot for a dynamic (dev) season.

hidden staging build -> compact snapshot -> validation -> manifest ->
atomic materialize -> atomic registry switch.

- 每次发布生成一个全新的不可变快照目录，绝不原地覆写在线目录；
- 完整报告先在隐藏 staging/build 构建，再从 build 提取线上所需产物，
  在 staging/snapshot 形成紧凑快照（默认不含 account_logs 派生副本）；
- 校验通过后 ``os.replace`` 将 snapshot staging 原子改名为最终快照；
- registry 通过 ``.tmp + os.replace`` 原子切换 ``report_dir``；
- registry 切换失败时删除本次已 materialize 的孤儿快照；
- API 保持纯只读，本脚本由训练线在固定间隔触发。
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from replay import ladder  # noqa: E402

MANIFEST_SCHEMA = "keqing.ladder.snapshot.v1"

# 线上 runtime/UI 实际消费、快照必须保留的产物
SNAPSHOT_REQUIRED_FILES = (
    "account_summary.json",
    "account_ledger.jsonl",
    "rating_curve.csv",
)
# 建议保留的派生报告（便于人工查看，runtime 不依赖）
SNAPSHOT_RECOMMENDED_FILES = (
    "account_summary.csv",
    "account_summary.md",
    "per_game_results.csv",
    "detailed_stats.json",
    "detailed_stats.md",
)


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
    parser.add_argument("--keep-account-logs", action="store_true",
                        help="调试用：在快照中保留 account_logs/ 派生副本（默认丢弃）")
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
        # 复用仓库严格解析：恰好四个顺位值，错误立即拒绝
        rank_points=account_report.parse_rank_points(rank_points),
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


def _lock_path(registry_path: Path) -> Path:
    return registry_path.with_name(registry_path.name + ".lock")


def _tmp_path_for(registry_path: Path) -> Path:
    """唯一临时文件路径（含 PID + 随机后缀），避免多个发布共享同一 tmp。"""
    return registry_path.with_name(f"{registry_path.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")


def _registry_contract(season: dict[str, Any]) -> str:
    """除 report_dir 外的注册表契约（模型/账号/状态/checkpoint 等）。"""
    clone = dict(season)
    clone.pop("report_dir", None)
    return json.dumps(clone, sort_keys=True, ensure_ascii=False)


@contextmanager
def _registry_lock(registry_path: Path, *, timeout: float = 30.0) -> Iterator[None]:
    """per-registry 排他锁：O_CREAT|O_EXCL 原子获取，超时抛 PublishError。"""
    lock_path = _lock_path(registry_path)
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode("ascii"))
            os.close(fd)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise PublishError(f"registry 锁获取超时: {lock_path}（可能有其他发布进行中）")
            time.sleep(0.2)
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def conditional_switch_registry(
    registry_path: Path,
    *,
    expected_season: dict[str, Any],
    expected_report_dir: str | None,
    new_report_dir: Path,
    timeout: float = 30.0,
) -> None:
    """加锁下的条件原子切换。

    - 切换前确认当前 report_dir 仍等于发布启动时看到的 previous_report_dir；
    - 除 report_dir 外的注册表契约（模型/账号/状态/checkpoint）发生变化则拒绝；
    - 使用唯一 tmp 文件，写盘 -> 结构校验 -> os.replace；
    - 冲突时抛 PublishError，由调用方清理本次 staging，线上 registry 与快照保持不变。
    """
    with _registry_lock(registry_path, timeout=timeout):
        current = ladder.read_registry(registry_path)
        current_report = str(current.get("report_dir") or "")
        if str(expected_report_dir or "") != current_report:
            raise PublishError(
                f"registry 已推进（当前 report_dir={current_report!r}，期望 {expected_report_dir!r}），拒绝旧发布覆盖"
            )
        if _registry_contract(current) != _registry_contract(expected_season):
            raise PublishError("registry 契约（模型/账号/状态/checkpoint）在构建期间发生变化，拒绝旧发布覆盖")

        updated = dict(current)
        updated["report_dir"] = str(new_report_dir)
        tmp_path = _tmp_path_for(registry_path)
        tmp_path.write_text(json.dumps(updated, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        try:
            ladder.read_registry(tmp_path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        os.replace(tmp_path, registry_path)


def _materialize_compact_snapshot(
    build_dir: Path,
    snapshot_stage: Path,
    *,
    keep_account_logs: bool,
) -> Path:
    """从完整 build 目录提取线上所需产物到紧凑 snapshot staging。

    - 必需产物（runtime/API 消费）必须存在；
    - 建议产物存在则复制，便于人工查看；
    - account_logs/ 是 build 阶段为计算详细统计而生成的派生副本，
      原始 mjai 日志归训练线所有，默认不写入快照。
    """
    snapshot_stage.mkdir(parents=True)
    for name in SNAPSHOT_REQUIRED_FILES:
        src = build_dir / name
        if not src.is_file():
            raise ladder.SeasonDataError(f"快照缺少必需文件: {name}")
        shutil.copy2(src, snapshot_stage / name)
    for name in SNAPSHOT_RECOMMENDED_FILES:
        src = build_dir / name
        if src.is_file():
            shutil.copy2(src, snapshot_stage / name)
    if keep_account_logs:
        logs_src = build_dir / "account_logs"
        if logs_src.is_dir():
            shutil.copytree(logs_src, snapshot_stage / "account_logs")
    return snapshot_stage


def _staging_root_for(snapshot_root: Path, snapshot_name: str) -> Path:
    """隐藏 staging 根目录：完整 build + 紧凑 snapshot staging 都放其下。"""
    return snapshot_root / f".{snapshot_name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.staging"


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
    keep_account_logs: bool = False,
    build_report: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """隐藏 staging 构建 -> 紧凑快照 -> 校验 -> manifest -> 原子发布（staging 流程）。

    返回 dict：{snapshot_dir, games, dry_run, registry_switched}。
    校验失败时删除本次 staging 目录并抛出原异常，旧快照与旧 registry 不受影响；
    registry 切换失败时删除本次已 materialize 的孤儿快照，旧 registry 保持可用。
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

    staging_root = _staging_root_for(root, snapshot_dir.name)
    build_dir = staging_root / "build"
    snapshot_stage = staging_root / "snapshot"
    staging_root.mkdir(parents=True)

    previous_report_dir = season.get("report_dir")
    try:
        report = build_snapshot(
            log_dirs=log_dirs,
            snapshot_dir=build_dir,
            mortal_root=mortal_root,
            platform_model_label=platform_model_label,
            rank_points=rank_points,
            preserve_log_dir_order=preserve_log_dir_order,
            interleave_log_dirs=interleave_log_dirs,
            build_report=build_report,
        )
        _materialize_compact_snapshot(
            build_dir=build_dir,
            snapshot_stage=snapshot_stage,
            keep_account_logs=keep_account_logs,
        )
        ladder.validate_snapshot(season, snapshot_stage)
        write_manifest(
            snapshot_stage,
            season_id=season_id,
            snapshot_id=snapshot_dir.name,
            report=report,
            registry_path=registry_path,
            previous_report_dir=str(previous_report_dir) if previous_report_dir else None,
        )
        # 只有完整构建与校验通过后，才把 snapshot staging 原子改名为最终快照
        os.replace(snapshot_stage, snapshot_dir)
        if dry_run:
            return {
                "snapshot_dir": str(snapshot_dir),
                "games": report.get("games"),
                "dry_run": True,
                "registry_switched": False,
            }
        conditional_switch_registry(
            registry_path,
            expected_season=season,
            expected_report_dir=str(previous_report_dir) if previous_report_dir else None,
            new_report_dir=snapshot_dir,
        )
        return {
            "snapshot_dir": str(snapshot_dir),
            "games": report.get("games"),
            "dry_run": False,
            "registry_switched": True,
        }
    except Exception:
        # registry 切换失败时，清理本次已 materialize 的孤儿快照
        shutil.rmtree(snapshot_dir, ignore_errors=True)
        raise
    finally:
        # 无论成败都清理完整 build staging（含 account_logs 派生副本）
        shutil.rmtree(staging_root, ignore_errors=True)


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
        keep_account_logs=args.keep_account_logs,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

