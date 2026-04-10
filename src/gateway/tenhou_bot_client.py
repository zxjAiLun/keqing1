from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gateway.tenhou_bridge import normalize_tenhou_room

logger = logging.getLogger(__name__)
SUPPORTED_GATEWAY_BOTS = {"keqingv1", "keqingv2", "keqingv3", "keqingv31", "rulebase"}


def create_runtime_bot_for_gateway(**kwargs):
    bot_name = kwargs["bot_name"]
    if bot_name == "rulebase":
        from inference.rulebase_bot import RulebaseBot

        return RulebaseBot(player_id=kwargs["player_id"], verbose=kwargs.get("verbose", False))
    from inference.runtime_bot import RuntimeBot

    project_root = Path(kwargs["project_root"])
    model_path = kwargs.get("model_path")
    resolved_model_path = (
        Path(model_path)
        if model_path is not None
        else project_root / "artifacts" / "models" / bot_name / "best.pth"
    )
    return RuntimeBot(
        player_id=kwargs["player_id"],
        model_path=resolved_model_path,
        device=kwargs.get("device", "cuda"),
        verbose=kwargs.get("verbose", False),
        model_version=bot_name,
    )


@dataclass(slots=True)
class BotClientConfig:
    host: str = "127.0.0.1"
    port: int = 11600
    room: str = "L2147_9"
    name: str = "NoName"
    bot_name: str = "keqingv2"
    project_root: Path = Path.cwd()
    model_path: Path | None = None
    device: str = "cuda"
    verbose: bool = False

    def resolved_model_path(self) -> Path | None:
        if self.bot_name == "rulebase":
            return None
        if self.model_path is not None:
            return Path(self.model_path)
        return Path(self.project_root) / "artifacts" / "models" / self.bot_name / "best.pth"


class GatewayBotClient:
    def __init__(self, config: BotClientConfig):
        if config.bot_name not in SUPPORTED_GATEWAY_BOTS:
            raise ValueError(f"unsupported bot name: {config.bot_name}")
        config.room = normalize_tenhou_room(config.room)
        model_path = config.resolved_model_path()
        if model_path is not None and not model_path.exists():
            raise FileNotFoundError(f"missing model checkpoint: {model_path}")
        self.config = config
        self._bot: Any | None = None
        self._seat: int | None = None
        self._pending_public_tsumo: dict[int, dict[str, Any]] = {}
        self._pending_public_post_meld_discard: set[int] = set()

    def _ensure_runtime_bot(self, seat: int) -> None:
        if self._bot is not None and self._seat == seat:
            self._bot.reset()
            self._pending_public_tsumo.clear()
            self._pending_public_post_meld_discard.clear()
            return
        self._bot = create_runtime_bot_for_gateway(
            bot_name=self.config.bot_name,
            player_id=seat,
            project_root=self.config.project_root,
            model_path=self.config.resolved_model_path(),
            device=self.config.device,
            verbose=self.config.verbose,
        )
        self._seat = seat
        self._pending_public_tsumo.clear()
        self._pending_public_post_meld_discard.clear()

    def _flush_pending_public_tsumo_for_current_message(self, message: dict[str, Any]) -> None:
        if self._bot is None or self._seat is None:
            return
        actor = message.get("actor")
        if actor is None:
            return
        pending = self._pending_public_tsumo.pop(actor, None)
        if pending is None:
            return
        if message.get("type") == "dahai":
            enriched = dict(pending)
            enriched["pai"] = message.get("pai", "?")
            pending = enriched
        self._bot.react(pending)

    def _annotate_public_opponent_event(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._seat is None:
            return message
        actor = message.get("actor")
        if actor is None or actor == self._seat:
            return message
        if message.get("type") == "dahai" and actor in self._pending_public_post_meld_discard:
            enriched = dict(message)
            enriched["skip_hand_update"] = True
            self._pending_public_post_meld_discard.discard(actor)
            return enriched
        if message.get("type") in {"chi", "pon", "daiminkan", "ankan", "kakan_accepted"}:
            enriched = dict(message)
            enriched["skip_hand_update"] = True
            self._pending_public_post_meld_discard.add(actor)
            return enriched
        return message

    def handle_message(self, message: dict[str, Any]) -> dict[str, Any]:
        mtype = message.get("type")
        if mtype == "hello":
            return {
                "type": "hello",
                "protocol": "mjsonp",
                "protocol_version": 3,
                "name": self.config.name,
                "room": self.config.room,
            }
        if mtype == "error":
            code = message.get("code")
            raw = message.get("raw")
            logger.error(
                "[%s] Tenhou bridge error code=%s message=%s raw=%s",
                self.config.name,
                code,
                message.get("message"),
                raw,
            )
            return {"type": "none"}

        if mtype == "start_game":
            seat = int(message["id"])
            self._ensure_runtime_bot(seat)

        if self._bot is None:
            return {"type": "none"}

        actor = message.get("actor")
        if (
            mtype == "tsumo"
            and actor is not None
            and actor != self._seat
            and message.get("pai") == "?"
        ):
            self._pending_public_tsumo[actor] = dict(message)
            return {"type": "none"}

        self._flush_pending_public_tsumo_for_current_message(message)

        message = self._annotate_public_opponent_event(message)
        action = self._bot.react(message)
        if action is None:
            if self._seat is None:
                return {"type": "none"}
            return {"type": "none", "actor": self._seat}
        return action

    def run(self) -> None:
        logger.info(
            "[%s] connecting to gateway %s:%s room=%s bot=%s",
            self.config.name,
            self.config.host,
            self.config.port,
            self.config.room,
            self.config.bot_name,
        )
        with socket.create_connection((self.config.host, self.config.port)) as sock:
            reader = sock.makefile("rb")
            writer = sock.makefile("wb")
            while True:
                line = reader.readline()
                if not line:
                    logger.info("[%s] gateway closed connection", self.config.name)
                    return
                message = json.loads(line.decode("utf-8"))
                response = self.handle_message(message)
                writer.write((json.dumps(response, ensure_ascii=False) + "\n").encode("utf-8"))
                writer.flush()


def launch_bot_threads(
    configs: list[BotClientConfig], stagger_seconds: float = 0.5
) -> list[threading.Thread]:
    threads: list[threading.Thread] = []
    for cfg in configs:
        client = GatewayBotClient(cfg)
        thread = threading.Thread(target=client.run, daemon=False, name=cfg.name)
        thread.start()
        threads.append(thread)
        if stagger_seconds > 0:
            time.sleep(stagger_seconds)
    return threads


def start_gateway_subprocess(
    *,
    project_root: Path,
    debug: bool = False,
    log_dir: Path | None = None,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[str]:
    log_dir = log_dir or project_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "src/gateway/main.py", "-o", str(log_dir)]
    if debug:
        cmd.append("-d")
    logger.info("starting gateway subprocess: %s", " ".join(cmd))
    env = os.environ.copy()
    if extra_env:
        env.update(extra_env)
    return subprocess.Popen(cmd, cwd=project_root, text=True, env=env)
