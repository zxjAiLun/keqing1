# -*- coding: utf-8 -*-
"""participants 数据平面：数据根解析、原子写、写锁与时间戳工具。

沿用 ladder 数据纪律：纯文件 JSON/JSONL + 临时文件 + ``os.replace`` 原子切换；
跨进程写操作由 ``O_CREAT|O_EXCL`` 锁文件保护（参考 ``scripts/mortal/publish_ladder_snapshot.py``）。
"""
from __future__ import annotations

import contextlib
import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

# src/participants/paths.py -> src -> repo root
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DATA_ROOT = REPO_ROOT / "artifacts" / "participants"

TZ_OFFSET = timezone(timedelta(hours=8))  # 默认 +08:00


def data_root() -> Path:
    env = os.environ.get("KEQING_PARTICIPANT_DATA_ROOT")
    if env:
        return Path(env)
    return DEFAULT_DATA_ROOT


def ensure_data_root() -> Path:
    root = data_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def now_iso() -> str:
    """当前时间 ISO-8601，带 +08:00 偏移。"""
    return datetime.now(TZ_OFFSET).isoformat(timespec="seconds")


def atomic_write_text(path: Path, text: str) -> None:
    """临时文件 + ``os.replace`` 原子写入。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(f"{path.suffix}.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.replace(tmp, path)


def read_json(path: Path, default):
    """读取 JSON；文件不存在返回 default，损坏时抛 ValueError。"""
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, schema: str, payload) -> None:
    """写入带 schema/updated_at 的版本化 JSON。"""
    text = json.dumps(
        {"schema": schema, "updated_at": now_iso(), **payload},
        ensure_ascii=False,
        indent=2,
    )
    atomic_write_text(path, text)


@contextlib.contextmanager
def file_lock(lock_path: Path, timeout: float = 8.0, stale_after: float = 30.0):
    """跨进程写锁：``O_CREAT|O_EXCL`` 独占创建锁文件。

    - 等不到锁且超过 ``timeout`` 秒时抛 ``TimeoutError``；
    - 锁文件 mtime 早于 ``stale_after`` 秒视为 stale，删除后重试。
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout
    acquired = False
    while not acquired:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
            os.close(fd)
            acquired = True
        except FileExistsError:
            if time.monotonic() > deadline:
                raise TimeoutError(f"lock not acquired in {timeout}s: {lock_path}")
            # stale 判定：st_mtime 是 epoch 时间戳，必须用 time.time()（而非 monotonic）比较
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > stale_after:
                try:
                    lock_path.unlink()
                except FileNotFoundError:
                    pass
            time.sleep(0.05)
    try:
        yield
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


@contextlib.contextmanager
def data_lock():
    """整个 participants 数据根的单写锁（registry/ledger 共享）。"""
    lock_path = ensure_data_root() / "participants.lock"
    with file_lock(lock_path):
        yield


@contextlib.contextmanager
def try_file_lock(lock_path: Path, stale_after: float = 30.0):
    """非阻塞跨进程锁（single-flight）：已占用立即 yield False，不会等待。

    - 成功获取：yield True，退出时删除锁文件；
    - 已被占用：yield False；锁文件过期则清除并重试一次。
    """
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    acquired = False
    for _attempt in range(2):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("ascii"))
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            try:
                if time.time() - lock_path.stat().st_mtime <= stale_after:
                    break  # 被活跃持有者占用
                lock_path.unlink()  # stale：清除后重试
            except FileNotFoundError:
                break
    if not acquired:
        yield False
        return
    try:
        yield True
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass
