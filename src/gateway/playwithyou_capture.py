"""Play-with-you ladder capture: shared collector in the launcher process.

The three ``GatewayBotClient`` instances run in ONE launcher process (separate
threads) and share a single :class:`PlayWithYouCaptureCollector`.  Each bot
observes ``start_game`` (binding its own seat from ``message["id"]``) and
``end_game`` (final scores).  Identity mapping is seat-based, never name-based,
so duplicate NoName display names are irrelevant.

When a complete, conflict-free result is available the collector writes a
pending capture file (atomically) into ``capture_dir/pending/``.  The GUI later
confirms it into the official season source; nothing here writes to the ladder
itself.

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


@dataclass
class PlayWithYouCaptureCollector:
    """共享 collector：积累三名 bot observer 的消息，最终产出 pending 文件。"""

    binding: CaptureBinding
    capture_dir: Path
    game_length: str = "hanchan"

    _seat_of: dict[str, int] = field(default_factory=dict)
    _log_id: str | None = None
    _log_url: str | None = None
    _end_scores: list[tuple[str, list[int]]] = field(default_factory=list)
    _terminal: bool = False
    _state: str = "waiting_start"
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

    def _on_start_game(self, observer: str, message: Mapping[str, Any]) -> None:
        try:
            seat = int(message["id"])
        except (KeyError, TypeError, ValueError):
            seat = -1
        if seat not in (0, 1, 2, 3):
            return
        self._seat_of[observer] = seat
        log_url = message.get("log")
        if isinstance(log_url, str) and self._log_id is None:
            match_id = extract_tenhou_match_id(log_url)
            if match_id:
                self._log_id = match_id
                self._log_url = log_url
        self._state = "in_game"

    def _on_end_game(self, observer: str, message: Mapping[str, Any]) -> None:
        raw_scores = message.get("scores")
        if not isinstance(raw_scores, list) or len(raw_scores) != 4:
            return
        try:
            scores = [int(value) for value in raw_scores]
        except (TypeError, ValueError):
            return
        self._end_scores.append((observer, scores))
        if self._state not in ("conflict", "pending_confirmation"):
            self._state = "pending_confirmation"

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

    # --- finalize ------------------------------------------------------------

    def finalize(self) -> None:
        """launcher 退出时调用一次：按积累状态写 pending / 记录 state。"""
        if self._finalized:
            return
        self._finalized = True
        self.capture_dir.mkdir(parents=True, exist_ok=True)

        human_seat = self._human_seat()
        if self._log_id is None or human_seat is None:
            self._state = "incomplete"
            self._write_state()
            return
        if not self._end_scores:
            self._state = "incomplete"  # 没有 end_game 的中断局不计分
            self._write_state()
            return
        score_sets = {tuple(scores) for _observer, scores in self._end_scores}
        if len(score_sets) > 1:
            self._state = "conflict"  # 多个 observer 分数不一致，不得确认
            self._write_state()
            return

        final_scores = list(next(iter(score_sets)))
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
            self._state = "incomplete"
            self._write_state()
            return

        self._state = "pending_confirmation"
        self._write_pending(players)
        self._write_state()

    # --- persistence ---------------------------------------------------------

    def _capture_id(self) -> str:
        return f"{self.binding.session_id}:{self._log_id}"

    def _write_pending(self, players: Sequence[Mapping[str, Any]]) -> None:
        pending_dir = self.capture_dir / "pending"
        pending_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": CAPTURE_SCHEMA,
            "capture_id": self._capture_id(),
            "session_id": self.binding.session_id,
            "state": "pending_confirmation",
            "season_id": self.binding.season_id,
            "match": {
                "match_id": self._log_id,
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "game_length": self.game_length,
                "players": players,
            },
            "tenhou_log_url": self._log_url,
            "observer_accounts": sorted(self._seat_of),
            "score_observers": [observer for observer, _scores in self._end_scores],
        }
        safe = re.sub(r"[^0-9a-zA-Z-]+", "_", self._log_id or "match")
        target = pending_dir / f"{safe}.json"
        tmp = pending_dir / f".{safe}.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, target)

    def _write_state(self) -> None:
        self.capture_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "session_id": self.binding.session_id,
            "season_id": self.binding.season_id,
            "state": self._state,
            "match_id": self._log_id,
            "seats": dict(sorted(self._seat_of.items())),
            "human_account_id": self.binding.human_account_id,
            "score_observers": [observer for observer, _scores in self._end_scores],
        }
        tmp = self.capture_dir / f".state.{os.getpid()}.{uuid.uuid4().hex[:8]}.tmp"
        tmp.write_text(json.dumps(state, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, self.capture_dir / "state.json")
