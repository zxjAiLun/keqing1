from __future__ import annotations

import argparse
import json

from bot.mjai_bot import MjaiPolicyBot
from bot.rule_bot import fallback_action
from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.replay import read_mjai_jsonl
from mahjong_env.state import GameState, apply_event


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", default="log.jsonl")
    parser.add_argument("--checkpoint", default="artifacts/sl/best.npz")
    parser.add_argument("--player-id", type=int, default=2)
    args = parser.parse_args()

    bot = MjaiPolicyBot(player_id=args.player_id, checkpoint_path=args.checkpoint)
    state = GameState()
    events = read_mjai_jsonl(args.log_path)
    total = 0
    same_as_fallback = 0
    for e in events:
        apply_event(state, e)
        if state.actor_to_move == args.player_id:
            snap = state.snapshot(args.player_id)
            legal = [a.to_mjai() for a in enumerate_legal_actions(snap, args.player_id)]
            pred = json.loads(bot.react(json.dumps({"type": "none"})))
            base = fallback_action(legal, args.player_id)
            total += 1
            if pred.get("type") == base.get("type") and pred.get("pai") == base.get("pai"):
                same_as_fallback += 1
    print(
        json.dumps(
            {"turns": total, "same_as_fallback_ratio": same_as_fallback / max(total, 1)},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

