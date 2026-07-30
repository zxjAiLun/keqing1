"""Play-with-you backend: summon Mortal-weight bot accounts into a Tenhou
private room (e.g. L2147) as NoName guests, driven from the GUI.

This mirrors mjai.ekyu.moe's "Play with you": the user types a Tenhou lobby
ID, picks a Speed, chooses a Mortal network per AI seat, and clicks a button.
The backend spawns ``scripts/launch_tenhou_bots.py`` as one owned subprocess
(which boots one local mjai-gateway) and streams its logs back to the GUI.

Design notes:
* Only ONE play-with-you session may be active at a time. The launcher always
  starts its own gateway on TCP 11600, so a second concurrent session would
  fail to bind that port. We reject new starts while one is running (the user
  must stop the previous one first).
 * UI sends human-friendly network ids (``mortal`` / ``70k`` / ``ext_mortal`` /
  ``none`` / ``custom``); we translate them into launcher specs
  (named checkpoints or absolute ``.pth`` paths) here, keeping the frontend
  dumb.
"""
from __future__ import annotations

import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/battle/playwithyou")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = PROJECT_ROOT / "scripts" / "launch_tenhou_bots.py"

# "Speed" selector -> per-turn think delay (seconds) handed to the bots.
SPEED_PRESETS: Dict[str, float] = {
    "slow": 2.0,
    "normal": 0.8,
    "fast": 0.3,
    "turbo": 0.0,
}

# UI network id -> launcher spec. Custom paths are resolved to absolute files;
# the rest are named local checkpoints.
NETWORK_TO_SPEC: Dict[str, str] = {
    "mortal": "mortal",
    "70k": "70k",
    "ext_mortal": "ext_mortal",
}

# Hard cap on remembered log lines per session (bound memory; GUI shows a tail).
MAX_LOG_LINES = 4000
PWY_LOG_DIR = PROJECT_ROOT / "logs" / "playwithyou"

_LOCK = threading.Lock()
# session_id -> PWYSession
SESSIONS: Dict[str, "PWYSession"] = {}
# Keep a short history of finished sessions so the GUI can still show the last run.
_HISTORY: List[str] = []
MAX_HISTORY = 5

# Marker arguments make orphan cleanup precise: never kill a gateway a user
# started manually outside this page.
SESSION_TOKEN = "playwithyou-session"
GATEWAY_OWNER_TOKEN = "playwithyou-gateway"


# ---------------------------------------------------------------------------
# Session model
# ---------------------------------------------------------------------------


class PWYSession:
    def __init__(
        self,
        *,
        session_id: str,
        proc: subprocess.Popen[str],
        command: List[str],
        lobby_id: str,
        specs: List[str],
        names: List[str],
        speed: str,
        device: str,
        started_at: float,
    ) -> None:
        self.session_id = session_id
        self.proc = proc
        self.command = command
        self.lobby_id = lobby_id
        self.specs = specs
        self.names = names
        self.speed = speed
        self.device = device
        self.started_at = started_at
        self.log_lines: List[str] = []
        # Keep the original launcher output outside the FastAPI process.  The
        # status endpoint is intentionally in-memory for simplicity, but a
        # server restart or an abrupt launcher exit must not erase the useful
        # evidence needed to diagnose a live Tenhou table.
        PWY_LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = PWY_LOG_DIR / f"{started_at:.0f}-{session_id}.log"
        self._reader = threading.Thread(target=self._pump_logs, daemon=True)
        self._reader.start()

    def _pump_logs(self) -> None:
        assert self.proc.stdout is not None
        try:
            with self.log_path.open("a", encoding="utf-8", buffering=1) as log_file:
                for line in self.proc.stdout:
                    clean_line = line.rstrip("\n")
                    self.log_lines.append(clean_line)
                    log_file.write(clean_line + "\n")
                    if len(self.log_lines) > MAX_LOG_LINES:
                        # Drop oldest in bulk to keep the append cheap.
                        del self.log_lines[: MAX_LOG_LINES // 2]
        except Exception:
            logger.exception("failed to persist play-with-you launcher output")
        rc = self.proc.poll()
        ended_line = (
            f"[session ended with returncode {rc}]"
            if rc is not None
            else "[session ended]"
        )
        self.log_lines.append(ended_line)
        try:
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(ended_line + "\n")
        except Exception:
            logger.exception("failed to persist play-with-you session end")

    @property
    def running(self) -> bool:
        return self.proc.poll() is None

    def tail(self, n: int = 200) -> List[str]:
        return self.log_lines[-n:]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _resolve_spec(network: str, custom_paths: Dict[int, str], slot: int) -> Optional[str]:
    """Map a UI network id to a launcher spec, or None for "none"."""
    network = (network or "none").strip().lower()
    if network in ("none", ""):
        return None
    if network == "custom":
        raw = (custom_paths or {}).get(slot, "")
        if not raw or not raw.strip():
            raise ValueError(f"slot {slot + 1}: custom network selected but no path given")
        path = Path(raw.strip())
        if not path.is_absolute() or not path.exists():
            raise FileNotFoundError(
                f"slot {slot + 1}: custom checkpoint not found: {raw}"
            )
        return str(path.resolve())
    if network in NETWORK_TO_SPEC:
        return NETWORK_TO_SPEC[network]
    # Allow passing a raw launcher spec (named checkpoint or explicit path) too.
    return network


def _current_session() -> Optional[PWYSession]:
    with _LOCK:
        # Prefer a running session, else the most recently created.
        running = [s for s in SESSIONS.values() if s.running]
        if running:
            return max(running, key=lambda s: s.started_at)
        if _HISTORY:
            last_id = _HISTORY[-1]
            return SESSIONS.get(last_id)
    return None


def _kill_tree(pid: int) -> None:
    """Best-effort kill of a process and its descendants."""
    if os.name == "nt":
        subprocess.call(
            ["taskkill", "/F", "/T", "/PID", str(pid)],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except Exception:
        try:
            os.kill(pid, signal.SIGTERM)
        except Exception:
            pass


def _find_owned_pids() -> List[int]:
    """Return only processes owned by this GUI's Play-with-you launcher.

    The command line contains one of our dedicated ownership markers. This is
    intentionally narrower than matching every ``gateway/main.py`` process.
    """
    pids: List[int] = []
    if os.name == "nt":
        try:
            out = subprocess.check_output(
                [
                    "wmic",
                    "process",
                    "where",
                    "commandline like '%playwithyou%'",
                    "get",
                    "CommandLine,ProcessId",
                    "/format:list",
                ],
                stderr=subprocess.DEVNULL,
                text=True,
            )
        except Exception:
            return pids
        command_line = ""
        for line in out.splitlines():
            if line.startswith("CommandLine="):
                command_line = line.removeprefix("CommandLine=")
            elif line.startswith("ProcessId="):
                pid_s = line.removeprefix("ProcessId=")
                if pid_s.isdigit() and _is_owned_command(command_line):
                    pids.append(int(pid_s))
                command_line = ""
        return [pid for pid in pids if _pid_is_alive(pid)]
    # POSIX best-effort: match our marker, which is passed to both launcher and
    # its gateway child.
    for marker in (
        f"--session-token {SESSION_TOKEN}",
        f"--owner-token {GATEWAY_OWNER_TOKEN}",
    ):
        try:
            out = subprocess.check_output(["pgrep", "-f", marker], text=True)
            pids.extend(int(pid_s) for pid_s in out.split())
        except Exception:
            pass
    return sorted(pid for pid in set(pids) if _pid_is_alive(pid))


def _is_owned_command(command_line: str) -> bool:
    """Recognise our exact command-line arguments, not a log/query string."""
    return bool(
        re.search(rf"(?:^|\s)--session-token\s+{re.escape(SESSION_TOKEN)}(?:\s|$)", command_line)
        or re.search(rf"(?:^|\s)--owner-token\s+{re.escape(GATEWAY_OWNER_TOKEN)}(?:\s|$)", command_line)
    )


def _pid_is_alive(pid: int) -> bool:
    """Exclude the short-lived wmic/pgrep query process from its own result."""
    if pid == os.getpid():
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Existing process owned by another Windows user/session.
        return True
    except OSError:
        return False
    return True


def _kill_owned_artifacts() -> None:
    """Stop only launcher/gateway artifacts marked as owned by this GUI."""
    for pid in _find_owned_pids():
        _kill_tree(pid)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class StartPlayWithYouRequest(BaseModel):
    lobby_id: str = "2147"
    speed: str = "normal"
    quantity: int = 1
    # Per-slot network id, length 4. Slots beyond `quantity` are ignored; "none" = no bot.
    networks: List[str] = ["mortal", "none", "none", "none"]
    # Absolute custom checkpoint paths keyed by slot index (0-based) for "custom".
    custom_paths: Dict[int, str] = {}
    device: str = "cuda"
    name_prefix: str = "NoName"
    tenhou_cookie: Optional[str] = None


class PlayWithYouStatus(BaseModel):
    session_id: Optional[str] = None
    running: bool = False
    lobby_id: Optional[str] = None
    speed: Optional[str] = None
    device: Optional[str] = None
    bots: List[Dict[str, str]] = []  # [{name, spec}]
    log_tail: List[str] = []
    started_at: Optional[float] = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.post("/start", response_model=PlayWithYouStatus)
def start_playwithyou(req: StartPlayWithYouRequest) -> PlayWithYouStatus:
    if not LAUNCHER.exists():
        raise HTTPException(status_code=500, detail=f"launcher missing: {LAUNCHER}")

    # Single active session: refuse to start a second one (gateway port clash).
    with _LOCK:
        for existing in SESSIONS.values():
            if existing.running:
                raise HTTPException(
                    status_code=409,
                    detail=(
                        f"已有运行中的会话 ({existing.session_id})，请先停止再呼出新的账号。"
                    ),
                )

    # Do not silently kill a process left by a previous GUI/backend instance.
    # Surface it and let the user choose Stop; this avoids unexpectedly
    # terminating an unrelated manual gateway.
    if _find_owned_pids():
        raise HTTPException(
            status_code=409,
            detail="检测到上一轮天凤呼出的遗留进程，请先点击「停止呼出」再重新启动。",
        )

    lobby_id = (req.lobby_id or "2147").strip().lstrip("Ll").strip() or "2147"
    if not (lobby_id.isdigit() and 1 <= int(lobby_id) <= 9999):
        raise HTTPException(
            status_code=400, detail="Lobby ID 必须是 1-9999 之间的数字（例如 2147）。"
        )

    quantity = max(1, min(4, int(req.quantity)))
    speed = req.speed if req.speed in SPEED_PRESETS else "normal"
    think_delay = SPEED_PRESETS[speed]

    networks = list(req.networks)
    while len(networks) < 4:
        networks.append("none")

    # Resolve the first `quantity` slots into launcher specs.
    specs: List[str] = []
    try:
        for slot in range(quantity):
            spec = _resolve_spec(networks[slot], req.custom_paths, slot)
            if spec is not None:
                specs.append(spec)
    except (ValueError, FileNotFoundError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    if not specs:
        raise HTTPException(
            status_code=400,
            detail="没有选择任何模型（所有 AI 槽位都是 none）。请至少选择一个 Mortal 网络。",
        )
    if len(specs) > 4:
        raise HTTPException(status_code=400, detail="最多呼出 4 个账号。")

    count = len(specs)
    name_prefix = (req.name_prefix or "").strip() or "NoName"
    names = [name_prefix] if count == 1 else [f"{name_prefix}-{i + 1}" for i in range(count)]
    device = req.device if req.device in ("cuda", "cpu") else "cuda"

    command = [
        sys.executable,
        "-u",
        str(LAUNCHER),
        "--room",
        f"L{lobby_id}",
        "--bots",
        *specs,
        "--device",
        device,
        "--start-gateway",
        "--stagger-seconds",
        "1.5",
        "--think-delay",
        str(think_delay),
        "--name-prefix",
        name_prefix,
        "--session-token",
        SESSION_TOKEN,
        "--gateway-owner-token",
        GATEWAY_OWNER_TOKEN,
    ]
    if req.tenhou_cookie:
        command += ["--tenhou-cookie", req.tenhou_cookie]

    logger.info("spawning playwithyou: %s", " ".join(command))

    try:
        popen_kwargs = {
            "cwd": str(PROJECT_ROOT),
            "stdout": subprocess.PIPE,
            "stderr": subprocess.STDOUT,
            "text": True,
            "bufsize": 1,
        }
        if os.name == "nt":
            popen_kwargs["creationflags"] = getattr(
                subprocess, "CREATE_NEW_PROCESS_GROUP", 0
            )
        else:
            popen_kwargs["start_new_session"] = True
        proc = subprocess.Popen(
            command,
            **popen_kwargs,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"启动失败: {exc}")

    session_id = uuid.uuid4().hex[:12]
    session = PWYSession(
        session_id=session_id,
        proc=proc,
        command=command,
        lobby_id=lobby_id,
        specs=specs,
        names=names,
        speed=speed,
        device=device,
        started_at=time.time(),
    )
    with _LOCK:
        SESSIONS[session_id] = session
        _HISTORY.append(session_id)
        if len(_HISTORY) > MAX_HISTORY:
            _HISTORY.pop(0)

    return PlayWithYouStatus(
        session_id=session_id,
        running=True,
        lobby_id=lobby_id,
        speed=speed,
        device=device,
        bots=[{"name": n, "spec": s} for n, s in zip(names, specs)],
        log_tail=session.tail(50),
        started_at=session.started_at,
    )


@router.get("/status", response_model=PlayWithYouStatus)
def playwithyou_status() -> PlayWithYouStatus:
    session = _current_session()
    if session is not None and session.running:
        return PlayWithYouStatus(
            session_id=session.session_id,
            running=True,
            lobby_id=session.lobby_id,
            speed=session.speed,
            device=session.device,
            bots=[{"name": n, "spec": s} for n, s in zip(session.names, session.specs)],
            log_tail=session.tail(200),
            started_at=session.started_at,
        )
    # No in-memory session, but a launcher may still be alive as an orphan
    # (e.g. the backend process was restarted and lost its session record).
    # Surface it so the GUI keeps offering a Stop button that will clean it up.
    if _find_owned_pids():
        return PlayWithYouStatus(
            running=True,
            lobby_id=None,
            speed=None,
            device=None,
            bots=[],
            log_tail=[
                "（检测到遗留的 bot 进程，可能是上次后端重启后残留；"
                "点击「停止呼出」即可清理）"
            ],
            started_at=None,
        )
    return PlayWithYouStatus(running=False)


@router.post("/stop", response_model=PlayWithYouStatus)
def stop_playwithyou() -> PlayWithYouStatus:
    session = _current_session()
    if session is not None and session.running:
        pid = session.proc.pid
        try:
            # Kill the process tree while the launcher is still alive, so its
            # local gateway child cannot escape as an orphan.
            _kill_tree(pid)
            try:
                session.proc.wait(timeout=5)
            except Exception:
                pass
            if session.proc.poll() is None:
                session.proc.kill()
                try:
                    session.proc.wait(timeout=5)
                except Exception:
                    pass
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(status_code=500, detail=f"停止失败: {exc}")
    # Also clean a marked launcher/gateway left after a backend restart. This
    # deliberately excludes any manually launched gateway.
    _kill_owned_artifacts()
    return PlayWithYouStatus(
        session_id=session.session_id if session else None,
        running=False,
        lobby_id=session.lobby_id if session else None,
        speed=session.speed if session else None,
        device=session.device if session else None,
        bots=[{"name": n, "spec": s} for n, s in zip(session.names, session.specs)]
        if session
        else [],
        started_at=session.started_at if session else None,
    )
