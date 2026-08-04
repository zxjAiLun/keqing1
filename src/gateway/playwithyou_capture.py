"""Play-with-you ladder capture: shared collector in the launcher process.

The three ``GatewayBotClient`` instances run in ONE launcher process (separate
threads) and share a single :class:`PlayWithYouCaptureCollector`.  Each bot
observes ``start_game`` (binding its own seat from ``message["id"]``) and
``end_game`` (final scores).  Identity mapping is seat-based, never name-based,
so duplicate NoName display names are irrelevant.

Progress is persisted progressively (after every start_game / end_game), so a
forced Windows process-tree kill of the launcher does not lose a finished game:
the collector writes observation state and, once a complete consistent result
is available, a provisional pending file immediately.

The capture directory lives OUTSIDE the season ``sources_root`` on purpose:
unconfirmed pending/ignored files must not perturb the ingest fingerprint.
"""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

CAPTURE_SCHEMA = "keqing.playwithyou.capture.v1"

CAPTURE_STATES = (
    "waiting_start",
    "in_game",
    "pending_confirmation",
    "published",
    "ignored",
    "incomplete",
    "conflict",
    "accepted_publish_failed",
)

# 各子目录允许的状态（discovery 校验用）。
STATE_BY_SUBDIR = {
    "pending": {"pending_confirmation", "published", "accepted_publish_failed"},
    "ignored": {"ignored"},
    "errors": {"incomplete", "conflict"},
}


class LadderCaptureError(ValueError):
    """Capture validation / conflict failure."""


@dataclass(frozen=True)
class CaptureBinding:
    """Frozen at game start; never modified mid-game."""

    session_id: str
    season_id: str
    human_account_id: str
    bot_account_ids: tuple[str, ...]
    mode: str  # only "confirm" this round


class PlayWithYouCaptureSink(Protocol):
    """Per-bot observation sink handed to each GatewayBotClient."""

    def observe(self, observer_account_id: str, message: Mapping[str, Any]) -> None: ...


def extract_tenhou_match_id(log_url: str | None) -> str | None:
    """从 Tenhou log URL 提取规范 match_id（忽略 ``tw``）。

    ``https://tenhou.net/3/?log=20260804gm-xxxx-xxxx&tw=2`` -> ``tenhou:20260804gm-xxxx-xxxx``
    """
    if not log_url:
        return None
    match = re.search(r"[?&]log=([0-9a-zA-Z-]+)", log_url)
    if not match:
        return None
    return f"tenhou:{match.group(1)}"


def capture_dir_for_session(data_root: Path, session_id: str) -> Path:
    """每个 session 一个捕获目录（不在 sources_root 内）。"""
    return data_root / "captures" / "playwithyou" / session_id


def _atomic_write(target: Path, payload: Mapping[str, Any]) -> None:
    """原子写 JSON（.tmp -> os.replace），绝不留下半行。"""
    tmp = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, target)


def _safe_slug(value: str) -> str:
    return re.sub(r"[^0-9a-zA-Z-]+", "_", value or "match")


@dataclass
class PlayWithYouCaptureCollector:
    """共享 collector：三名 bot observer 的消息累积 + 渐进持久化。"""

    binding: CaptureBinding
    capture_dir: Path
    game_length: str = "hanchan"

    _seat_of: dict[str, int] = field(default_factory=dict)
    _log_id_by_observer: dict[str, str] = field(default_factory=dict)
    _log_url_by_observer: dict[str, str] = field(default_factory=dict)
    _end_scores: list[tuple[str, list[int]]] = field(default_factory=list)
    _terminal: bool = False
    _state: str = "waiting_start"
    _conflict_reason: str | None = None
    _finalized: bool = False

    # --- observation --------------------------------------------------------

    def observe(self, observer_account_id: str, message: Mapping[str, Any]) -> None:
        mtype = message.get("type")
        if mtype == "start_game":
            self._on_start_game(observer_account_id, message)
        elif mtype == "end_game":
            self._on_end_game(observer_account_id, message)
        elif mtype == "error":
            self._terminal = True
        # 渐进持久化：每次观察都更新 state；完整且一致时立即写 provisional pending。
        self._try_complete()

    def _on_start_game(self, observer: str, message: Mapping[str, Any]) -> None:
        try:
            seat = int(message["id"])
        except (KeyError, TypeError, ValueError):
            return
        if seat not in (0, 1, 2, 3):
            return
        if observer in self._seat_of and self._seat_of[observer] != seat:
            self._set_conflict(f"observer {observer} 重复上报不同 seat: {self._seat_of[observer]} vs {seat}")
            return
        self._seat_of[observer] = seat

        log_url = message.get("log")
        match_id = extract_tenhou_match_id(log_url) if isinstance(log_url, str) else None
        if match_id is None:
            return
        if observer in self._log_id_by_observer and self._log_id_by_observer[observer] != match_id:
            self._set_conflict(
                f"observer {observer} 重复上报不同 log: {self._log_id_by_observer[observer]} vs {match_id}"
            )
            return
        # 所有拥有 log ID 的 start_game 必须指向同一个规范 match_id。
        for other, other_id in self._log_id_by_observer.items():
            if other != observer and other_id != match_id:
                self._set_conflict(
                    f"observer {observer} 的 log {match_id} 与 {other} 的 {other_id} 不一致"
                )
                return
        self._log_id_by_observer[observer] = match_id
        self._log_url_by_observer[observer] = log_url if isinstance(log_url, str) else ""
        if self._state not in ("conflict", "in_game"):
            self._state = "in_game"

    def _on_end_game(self, observer: str, message: Mapping[str, Any]) -> None:
        raw_scores = message.get("scores")
        if not isinstance(raw_scores, list) or len(raw_scores) != 4:
            return
        try:
            scores = [int(value) for value in raw_scores]
        except (TypeError, ValueError):
            return
        for other, other_scores in self._end_scores:
            if other_scores != scores:
                self._set_conflict(
                    f"observer {observer} 分数与 {other} 不一致，拒绝确认"
                )
                return
        self._end_scores.append((observer, scores))

    def _set_conflict(self, reason: str) -> None:
        self._state = "conflict"
        self._conflict_reason = reason

    # --- seat resolution -----------------------------------------------------

    def _human_seat(self) -> int | None:
        if len(self._seat_of) < 3:
            return None
        bot_seats = set(self._seat_of.values())
        if len(bot_seats) != len(self._seat_of):
            return None  # 两名 observer 报告了同一 seat，无法建立唯一映射
        missing = [seat for seat in range(4) if seat not in bot_seats]
        return missing[0] if len(missing) == 1 else None

    def _account_for_seat(self, seat: int) -> str | None:
        for account_id, bound_seat in self._seat_of.items():
            if bound_seat == seat:
                return account_id
        if seat == self._human_seat():
            return self.binding.human_account_id
        return None

    def _canonical_log_id(self) -> str | None:
        if not self._log_id_by_observer:
            return None
        return next(iter(self._log_id_by_observer.values()))

    def _canonical_log_url(self) -> str | None:
        if not self._log_url_by_observer:
            return None
        return next(iter(self._log_url_by_observer.values())) or None

    # --- progressive completion ---------------------------------------------

    def _try_complete(self) -> None:
        """每次观察后调用：完整且一致 -> 立即写 provisional pending；冲突 -> 移除 pending。"""
        self.capture_dir.mkdir(parents=True, exist_ok=True)

        if self._state == "conflict":
            self._remove_pending()
            self._write_state()
            return

        human_seat = self._human_seat()
        if self._canonical_log_id() is None or human_seat is None or not self._end_scores:
            self._write_state()
            return

        final_scores = self._end_scores[0][1]
        players = [
            {
                "account_id": account_id,
                "seat": seat,
                "final_score": final_scores[seat],
            }
            for seat in range(4)
            if (account_id := self._account_for_seat(seat)) is not None
        ]
        if len(players) != 4:
            self._write_state()
            return

        self._state = "pending_confirmation"
        self._write_pending(players)
        self._write_state()

    # --- finalize -----------------------------------------------------------

    def finalize(self) -> None:
        """launcher 退出时调用一次：兜底触发 completion / 写 errors 详情。"""
        if self._finalized:
            return
        self._finalized = True
        if self._state == "conflict":
            self._write_error("conflict")
            self._write_state()
            return
        self._try_complete()
        if self._state in ("incomplete", "waiting_start", "in_game"):
            # 没有完整结果：写 errors/ 详情，让 discovery/UI 可见（C22）。
            self._state = "incomplete"
            self._write_error("incomplete")
            self._write_state()

    # --- persistence ---------------------------------------------------------

    def _capture_id(self) -> str:
        return f"{self.binding.session_id}:{self._canonical_log_id()}"

    def _pending_path(self) -> Path | None:
        log_id = self._canonical_log_id()
        if log_id is None:
            return None
        return self.capture_dir / "pending" / f"{_safe_slug(log_id)}.json"

    def _remove_pending(self) -> None:
        path = self._pending_path()
        if path is not None and path.is_file():
            path.unlink(missing_ok=True)

    def _write_pending(self, players: Sequence[Mapping[str, Any]]) -> None:
        target = self._pending_path()
        if target is None:
            return
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": CAPTURE_SCHEMA,
            "capture_id": self._capture_id(),
            "session_id": self.binding.session_id,
            "state": "pending_confirmation",
            "season_id": self.binding.season_id,
            "match": {
                "match_id": self._canonical_log_id(),
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "game_length": self.game_length,
                "players": players,
            },
            "tenhou_log_url": self._canonical_log_url(),
            "observer_accounts": sorted(self._seat_of),
            "score_observers": [observer for observer, _scores in self._end_scores],
        }
        _atomic_write(target, payload)

    def _write_error(self, state: str) -> None:
        errors_dir = self.capture_dir / "errors"
        errors_dir.mkdir(parents=True, exist_ok=True)
        target = errors_dir / f"{_safe_slug(self._canonical_log_id() or self.binding.session_id)}.json"
        payload = {
            "schema": CAPTURE_SCHEMA,
            "capture_id": self._capture_id(),
            "session_id": self.binding.session_id,
            "state": state,
            "season_id": self.binding.season_id,
            "match": {
                "match_id": self._canonical_log_id(),
                "game_length": self.game_length,
                "players": [
                    {
                        "account_id": account_id,
                        "seat": seat,
                        "final_score": None,
                    }
                    for seat in range(4)
                    if (account_id := self._account_for_seat(seat)) is not None
                ],
            },
            "tenhou_log_url": self._canonical_log_url(),
            "observer_accounts": sorted(self._seat_of),
            "score_observers": [observer for observer, _scores in self._end_scores],
            "conflict_reason": self._conflict_reason,
        }
        _atomic_write(target, payload)

    def _write_state(self) -> None:
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "session_id": self.binding.session_id,
            "season_id": self.binding.season_id,
            "state": self._state,
            "match_id": self._canonical_log_id(),
            "seats": dict(sorted(self._seat_of.items())),
            "human_account_id": self.binding.human_account_id,
            "score_observers": [observer for observer, _scores in self._end_scores],
            "conflict_reason": self._conflict_reason,
        }
        _atomic_write(self.capture_dir / "state.json", state)
