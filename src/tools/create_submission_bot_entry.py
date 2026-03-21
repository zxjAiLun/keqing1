from __future__ import annotations

import argparse
from pathlib import Path


BOT_TEMPLATE = """#!/usr/bin/env python3
from __future__ import annotations

import sys


def main() -> None:
    # mjai simulator invokes: `python -u bot.py <player_id>`
    if len(sys.argv) < 2:
        raise SystemExit("usage: bot.py <player_id>")
    player_id = int(sys.argv[1])

    from bot.mjai_bot import MjaiPolicyBot

    bot = MjaiPolicyBot(
        player_id=player_id,
        checkpoint_path="{checkpoint_path}",
    )

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        print(bot.react(line), flush=True)


if __name__ == "__main__":
    # Ensure imports work when src/ is bundled next to this file.
    import pathlib
    import sys as _sys

    base = pathlib.Path(__file__).resolve().parent
    _sys.path.insert(0, str(base / "src"))
    main()
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="artifacts/mjai_submission/bot.py")
    parser.add_argument("--checkpoint-path", default="checkpoints/best.npz")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(BOT_TEMPLATE.format(checkpoint_path=args.checkpoint_path), encoding="utf-8")
    print(f"wrote: {out_path}")


if __name__ == "__main__":
    main()
