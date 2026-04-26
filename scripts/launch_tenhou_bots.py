from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from gateway.tenhou_bridge import normalize_tenhou_room
from gateway.tenhou_bot_client import (
    BotClientConfig,
    launch_bot_threads,
    start_gateway_subprocess,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch one or more runtime bots into a Tenhou gateway room"
    )
    parser.add_argument(
        "--room",
        default="L2147",
        help="Tenhou lobby/room, e.g. L2147, L2147_9, 2147, or 2147_9",
    )
    parser.add_argument(
        "--game-type",
        type=int,
        default=9,
        help="Tenhou queue/game type integer (default 9 = 4p hanchan multiplayer)",
    )
    parser.add_argument("--count", type=int, default=2, help="How many bot clients to start")
    parser.add_argument(
        "--bot",
        default="xmodel1",
        help="Bot type: xmodel1/keqingv4/rulebase",
    )
    parser.add_argument(
        "--name-prefix", default="NoName", help="Prefix for bot display names"
    )
    parser.add_argument("--gateway-host", default="127.0.0.1")
    parser.add_argument("--gateway-port", type=int, default=11600)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-path", default=None, help="Optional explicit checkpoint path")
    parser.add_argument("--stagger-seconds", type=float, default=0.5)
    parser.add_argument(
        "--start-gateway",
        action="store_true",
        help="Start src/gateway/main.py automatically",
    )
    parser.add_argument("--gateway-debug", action="store_true")
    parser.add_argument("--gateway-log-dir", default="logs")
    parser.add_argument("--tenhou-uri", default=None, help="Override TENHOU_URI for gateway")
    parser.add_argument(
        "--tenhou-origin", default=None, help="Override TENHOU_ORIGIN for gateway"
    )
    parser.add_argument(
        "--tenhou-cookie",
        default=None,
        help="Optional Cookie header for Tenhou websocket authentication",
    )
    parser.add_argument(
        "--tenhou-helo-json",
        default=None,
        help="Optional JSON object merged into the HELO payload",
    )
    parser.add_argument("--bot-verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    normalized_room = normalize_tenhou_room(args.room, default_suffix=str(args.game_type))

    gateway_proc = None
    if args.start_gateway:
        extra_env = {}
        if args.tenhou_uri:
            extra_env["TENHOU_URI"] = args.tenhou_uri
        if args.tenhou_origin:
            extra_env["TENHOU_ORIGIN"] = args.tenhou_origin
        if args.tenhou_cookie:
            extra_env["TENHOU_COOKIE"] = args.tenhou_cookie
        if args.tenhou_helo_json:
            extra_env["TENHOU_HELO_JSON"] = args.tenhou_helo_json
        gateway_proc = start_gateway_subprocess(
            project_root=PROJECT_ROOT,
            debug=args.gateway_debug,
            log_dir=(PROJECT_ROOT / args.gateway_log_dir),
            extra_env=extra_env or None,
        )
        time.sleep(1.5)
        if gateway_proc.poll() is not None:
            raise RuntimeError("gateway subprocess exited early; check logs and port availability")

    configs = [
        BotClientConfig(
            host=args.gateway_host,
            port=args.gateway_port,
            room=normalized_room,
            name=(
                args.name_prefix
                if args.name_prefix == "NoName" or args.count == 1
                else f"{args.name_prefix}-{i + 1}"
            ),
            bot_name=args.bot,
            project_root=PROJECT_ROOT,
            model_path=Path(args.model_path) if args.model_path else None,
            device=args.device,
            verbose=args.bot_verbose,
        )
        for i in range(args.count)
    ]

    threads = launch_bot_threads(configs, stagger_seconds=args.stagger_seconds)

    def _shutdown(*_: object) -> None:
        if gateway_proc is not None and gateway_proc.poll() is None:
            gateway_proc.terminate()
        raise SystemExit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        for thread in threads:
            thread.join()
    finally:
        if gateway_proc is not None and gateway_proc.poll() is None:
            gateway_proc.terminate()
            gateway_proc.wait(timeout=5)


if __name__ == "__main__":
    main()
