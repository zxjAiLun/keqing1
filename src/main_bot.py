from __future__ import annotations

import argparse
import sys

from bot.mjai_bot import MjaiPolicyBot


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--player-id", type=int, required=True)
    parser.add_argument("--checkpoint", default="artifacts/sl/best.npz")
    args = parser.parse_args()

    bot = MjaiPolicyBot(player_id=args.player_id, checkpoint_path=args.checkpoint)
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        print(bot.react(line), flush=True)


if __name__ == "__main__":
    main()

