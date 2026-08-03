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
import hashlib
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

# 默认保留的最近有效快照数；0 表示禁用自动清理
DEFAULT_RETAIN_SNAPSHOTS = 24


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
    parser.add_argument("--retain-snapshots", type=int, default=DEFAULT_RETAIN_SNAPSHOTS,
                        help=f"成功发布后保留的最近有效快照数（默认 {DEFAULT_RETAIN_SNAPSHOTS}；0 表示禁用自动清理）")
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
    keep_account_logs: bool = False,
    source_fingerprint: str = "",
    registry_contract: str = "",
    source_log_dirs: list[Path] | None = None,
    source_file_count: int = 0,
    source_total_bytes: int = 0,
    build_duration_seconds: float = 0.0,
    materialize_duration_seconds: float = 0.0,
    snapshot_total_bytes: int = 0,
    rank_points: str = "90,45,0,-135",
    platform_model_label: str | None = None,
    preserve_log_dir_order: bool = False,
    interleave_log_dirs: bool = False,
) -> Path:
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "season_id": season_id,
        "snapshot_id": snapshot_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "games": report.get("games"),
        "accounts": len(report.get("accounts") or []),
        "compact": True,
        "kept_account_logs": bool(keep_account_logs),
        "rank_points": rank_points,
        "platform_model_label": platform_model_label,
        "preserve_log_dir_order": bool(preserve_log_dir_order),
        "interleave_log_dirs": bool(interleave_log_dirs),
        "source_fingerprint": source_fingerprint,
        "registry_contract": registry_contract,
        "source_log_dirs": [str(path) for path in (source_log_dirs or [])],
        "source_file_count": source_file_count,
        "source_total_bytes": source_total_bytes,
        "build_duration_seconds": build_duration_seconds,
        "materialize_duration_seconds": materialize_duration_seconds,
        "snapshot_total_bytes": snapshot_total_bytes,
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


def _read_manifest(snapshot_dir: Path) -> dict[str, Any] | None:
    """读取快照目录的 manifest；无 manifest 或 schema 不合法时返回 None。"""
    manifest_path = snapshot_dir / "manifest.json"
    if not manifest_path.is_file():
        return None
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(manifest, dict) or manifest.get("schema") != MANIFEST_SCHEMA:
        return None
    return manifest


def _log_file_stats(log_dirs: list[Path]) -> list[tuple[str, int, int]]:
    """收集排序后的源日志指纹元组：(绝对路径, size, mtime_ns)。"""
    stats: list[tuple[str, int, int]] = []
    for log_dir in log_dirs:
        for path in sorted(log_dir.glob("*.json.gz")):
            try:
                st = path.stat()
            except OSError:
                continue
            stats.append((str(path.resolve()), st.st_size, st.st_mtime_ns))
    stats.sort(key=lambda item: item[0])
    return stats


def compute_source_fingerprint(
    log_dirs: list[Path],
    *,
    preserve_log_dir_order: bool = False,
    interleave_log_dirs: bool = False,
) -> str:
    """源日志输入指纹：覆盖排序后的路径、大小、mtime_ns。

    相同的日志文件集合（含 ordering/interleave 参数）产生相同指纹；
    用于避免人工重复执行、resume 无新增日志、调度器重复触发时的重复重建。
    """
    stats = _log_file_stats(log_dirs)
    hasher = hashlib.sha256()
    hasher.update(("preserve=%s;interleave=%s;" % (preserve_log_dir_order, interleave_log_dirs)).encode("ascii"))
    for path, size, mtime_ns in stats:
        hasher.update(f"{path}\0{size}\0{mtime_ns}\n".encode("utf-8"))
    return hasher.hexdigest()


def _should_skip_unchanged(
    season: dict[str, Any],
    manifest: dict[str, Any],
    *,
    source_fingerprint: str,
    rank_points: str,
    platform_model_label: str | None,
    preserve_log_dir_order: bool,
    interleave_log_dirs: bool,
) -> bool:
    """当前 snapshot manifest 与本次发布条件一致时返回 True（跳过重建）。

    需要 season contract、source_fingerprint、rank_points、
    platform_model_label、log ordering/interleave 参数全部一致。
    """
    if manifest.get("season_id") != season.get("season_id"):
        return False
    if _registry_contract(season) != _registry_contract_from_manifest(manifest):
        return False
    if manifest.get("source_fingerprint") != source_fingerprint:
        return False
    if manifest.get("rank_points") != rank_points:
        return False
    if manifest.get("platform_model_label") != (platform_model_label or None):
        return False
    if bool(manifest.get("preserve_log_dir_order")) != preserve_log_dir_order:
        return False
    if bool(manifest.get("interleave_log_dirs")) != interleave_log_dirs:
        return False
    return True


def _registry_contract_from_manifest(manifest: dict[str, Any]) -> str:
    raw = manifest.get("registry_contract")
    return raw if isinstance(raw, str) else ""


def enforce_retention(
    snapshot_root: Path,
    *,
    season_id: str,
    retain: int,
    current_snapshot_dir: Path,
    previous_report_dir: str | None,
    protected_snapshots: set[Path],
) -> dict[str, list[str]]:
    """清理 snapshots 根目录下超出保留上限的旧快照。

    只在 registry 成功切换之后调用；任何清理失败都记录 warning 并返回
    ``retention_errors``，绝不把已成功发布的切换标记为失败。

    永不删除：
    - 当前 registry 指向的快照；
    - manifest 中记录的 previous snapshot（保留回滚点）；
    - 本次发布保护集（如 dry-run 快照）；
    - 没有合法 ``keqing.ladder.snapshot.v1`` manifest 的目录；
    - season_id 不一致的目录；
    - 隐藏 staging 目录；
    - snapshots 根目录中的其他人工文件。

    返回 ``{retention_deleted: [...], retention_errors: [...]}``。
    """
    result: dict[str, list[str]] = {"retention_deleted": [], "retention_errors": []}
    if retain <= 0 or not snapshot_root.exists():
        return result

    protected: set[Path] = set(protected_snapshots)
    protected.add(current_snapshot_dir.resolve())
    if previous_report_dir:
        protected.add(Path(previous_report_dir).resolve())

    # 收集本 season 的合法快照（带 manifest、season_id 一致、非隐藏）
    candidates: list[tuple[str, Path]] = []
    for entry in snapshot_root.iterdir():
        if not entry.is_dir():
            continue
        if entry.name.startswith("."):
            continue
        manifest = _read_manifest(entry)
        if manifest is None:
            continue
        if manifest.get("season_id") != season_id:
            continue
        candidates.append((str(manifest.get("created_at") or entry.name), entry))

    # 按 created_at 排序（desc），保留最近 retain 个
    candidates.sort(key=lambda item: (item[0], item[1].name), reverse=True)
    keep = candidates[:retain]
    keep_paths = {path.resolve() for _created, path in keep}
    for _created, entry in candidates[retain:]:
        resolved = entry.resolve()
        if resolved in protected or resolved in keep_paths:
            continue
        try:
            shutil.rmtree(entry)
        except OSError as exc:
            result["retention_errors"].append(f"{entry}: {exc}")
        else:
            result["retention_deleted"].append(str(entry))
    return result


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
    retain_snapshots: int = DEFAULT_RETAIN_SNAPSHOTS,
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

    previous_report_dir = season.get("report_dir")
    previous_snapshot: Path | None = None
    if previous_report_dir:
        previous_snapshot = Path(str(previous_report_dir))
        if not previous_snapshot.is_absolute():
            data_root_env = Path(data_root) if data_root else None
            base = data_root_env or _REPO_ROOT
            previous_snapshot = (base / str(previous_report_dir)).resolve()

    source_fingerprint = compute_source_fingerprint(
        log_dirs,
        preserve_log_dir_order=preserve_log_dir_order,
        interleave_log_dirs=interleave_log_dirs,
    )
    if previous_snapshot is not None and previous_snapshot.is_dir():
        prev_manifest = _read_manifest(previous_snapshot)
        if prev_manifest is not None and _should_skip_unchanged(
            season,
            prev_manifest,
            source_fingerprint=source_fingerprint,
            rank_points=rank_points,
            platform_model_label=platform_model_label,
            preserve_log_dir_order=preserve_log_dir_order,
            interleave_log_dirs=interleave_log_dirs,
        ):
            return {
                "snapshot_dir": str(previous_snapshot),
                "games": prev_manifest.get("games"),
                "dry_run": dry_run,
                "registry_switched": False,
                "skipped_unchanged": True,
                "retention_deleted": [],
                "retention_errors": [],
            }

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

    try:
        build_started = time.monotonic()
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
        build_duration = time.monotonic() - build_started

        materialize_started = time.monotonic()
        _materialize_compact_snapshot(
            build_dir=build_dir,
            snapshot_stage=snapshot_stage,
            keep_account_logs=keep_account_logs,
        )
        ladder.validate_snapshot(season, snapshot_stage)
        materialize_duration = time.monotonic() - materialize_started

        source_stats = _log_file_stats(log_dirs)
        snapshot_total_bytes = sum(
            (snapshot_stage / name).stat().st_size
            for name in SNAPSHOT_REQUIRED_FILES + SNAPSHOT_RECOMMENDED_FILES
            if (snapshot_stage / name).is_file()
        )
        write_manifest(
            snapshot_stage,
            season_id=season_id,
            snapshot_id=snapshot_dir.name,
            report=report,
            registry_path=registry_path,
            previous_report_dir=str(previous_report_dir) if previous_report_dir else None,
            keep_account_logs=keep_account_logs,
            source_fingerprint=source_fingerprint,
            registry_contract=_registry_contract(season),
            source_log_dirs=log_dirs,
            source_file_count=len(source_stats),
            source_total_bytes=sum(int(size) for _path, size, _mtime in source_stats),
            build_duration_seconds=round(build_duration, 4),
            materialize_duration_seconds=round(materialize_duration, 4),
            snapshot_total_bytes=snapshot_total_bytes,
            rank_points=rank_points,
            platform_model_label=platform_model_label,
            preserve_log_dir_order=preserve_log_dir_order,
            interleave_log_dirs=interleave_log_dirs,
        )
        # 只有完整构建与校验通过后，才把 snapshot staging 原子改名为最终快照
        os.replace(snapshot_stage, snapshot_dir)
        if dry_run:
            return {
                "snapshot_dir": str(snapshot_dir),
                "games": report.get("games"),
                "dry_run": True,
                "registry_switched": False,
                "skipped_unchanged": False,
                "retention_deleted": [],
                "retention_errors": [],
            }
        conditional_switch_registry(
            registry_path,
            expected_season=season,
            expected_report_dir=str(previous_report_dir) if previous_report_dir else None,
            new_report_dir=snapshot_dir,
        )
        retention = enforce_retention(
            root,
            season_id=season_id,
            retain=retain_snapshots,
            current_snapshot_dir=snapshot_dir,
            previous_report_dir=str(previous_report_dir) if previous_report_dir else None,
            protected_snapshots=set(),
        )
        return {
            "snapshot_dir": str(snapshot_dir),
            "games": report.get("games"),
            "dry_run": False,
            "registry_switched": True,
            "skipped_unchanged": False,
            **retention,
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
        retain_snapshots=args.retain_snapshots,
    )
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

