from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from bot.mjai_bot import MjaiPolicyBot
from mahjong_env.replay import read_mjai_jsonl


def _spawn_bundle_bot(bundle_zip: Path, player_id: int) -> tuple[subprocess.Popen, Path]:
    tmp_dir = Path(tempfile.mkdtemp(prefix="bundle_parity_"))
    with zipfile.ZipFile(bundle_zip, "r") as zf:
        zf.extractall(tmp_dir)
    bot_py = tmp_dir / "bot.py"
    if not bot_py.exists():
        raise RuntimeError(f"bundle missing bot.py: {bundle_zip}")
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(
        [sys.executable, str(bot_py), str(player_id)],
        cwd=str(tmp_dir),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=env,
    )
    return proc, tmp_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle-zip", required=True)
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--player-id", type=int, default=0)
    parser.add_argument("--checkpoint", required=True, help="local checkpoint path for source bot")
    parser.add_argument("--max-events", type=int, default=0, help="0 means all")
    parser.add_argument("--out", default="artifacts/debug/bundle_parity_report.json")
    args = parser.parse_args()

    events = read_mjai_jsonl(args.log_path)
    if args.max_events > 0:
        events = events[: args.max_events]

    local_bot = MjaiPolicyBot(player_id=args.player_id, checkpoint_path=args.checkpoint)
    proc, extracted_dir = _spawn_bundle_bot(Path(args.bundle_zip), args.player_id)
    assert proc.stdin is not None
    assert proc.stdout is not None

    mismatches: List[Dict[str, Any]] = []
    compared = 0
    try:
        for i, ev in enumerate(events):
            req = json.dumps(ev, ensure_ascii=False)
            local_out = local_bot.react(req).strip()

            proc.stdin.write(req + "\n")
            proc.stdin.flush()
            bundle_out = proc.stdout.readline().strip()

            compared += 1
            if local_out != bundle_out:
                mismatches.append(
                    {
                        "event_index": i,
                        "request": ev,
                        "local_out": local_out,
                        "bundle_out": bundle_out,
                    }
                )
                break
    finally:
        try:
            proc.kill()
        except Exception:
            pass

    report = {
        "bundle_zip": args.bundle_zip,
        "log_path": args.log_path,
        "player_id": args.player_id,
        "compared_events": compared,
        "ok": len(mismatches) == 0,
        "first_mismatch": mismatches[0] if mismatches else None,
        "extracted_dir": str(extracted_dir),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
