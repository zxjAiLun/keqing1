from __future__ import annotations

import argparse
import json

from bot.mjai_bot import MjaiPolicyBot
from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.replay import read_mjai_jsonl
from mahjong_env.state import GameState, apply_event


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True)
    parser.add_argument("--checkpoint", default="artifacts/sl/best.npz")
    parser.add_argument("--player-id", type=int, default=0)
    parser.add_argument("--window", type=int, default=80, help="Lookahead window for matching ground-truth events")
    parser.add_argument("--max-mismatches", type=int, default=30)
    args = parser.parse_args()

    bot = MjaiPolicyBot(player_id=args.player_id, checkpoint_path=args.checkpoint)

    events = read_mjai_jsonl(args.log_path)

    def same_action(action: dict, gt_event: dict) -> bool:
        if action.get("type") != gt_event.get("type"):
            return False
        if action.get("type") in {"dahai", "chi", "pon", "daiminkan", "ankan", "kakan"}:
            if action.get("pai") != gt_event.get("pai"):
                return False
        if action.get("type") == "dahai":
            # tsumogiri is important for legality and red-tile matching.
            if action.get("tsumogiri") != gt_event.get("tsumogiri"):
                return False
        if action.get("type") in {"chi", "pon", "daiminkan", "ankan", "kakan"}:
            if action.get("target") is not None and action.get("target") != gt_event.get("target"):
                return False
            if "consumed" in action and action.get("consumed") != gt_event.get("consumed"):
                return False
        return True

    requested = 0
    matched = 0
    exact = 0
    mismatches: list[dict] = []

    # Note: we feed *ground-truth* incoming events to the bot and let the bot output its chosen reaction.
    # Then we compare the bot's chosen action with the next ground-truth event of the same (type, actor).
    for i, ev in enumerate(events):
        action_str = bot.react(json.dumps(ev))
        action = json.loads(action_str)
        if action.get("type") == "none":
            continue

        requested += 1

        gt: dict | None = None
        gt_match_idx = None
        for j in range(i + 1, min(len(events), i + 1 + args.window)):
            cand = events[j]
            if cand.get("actor") == args.player_id and cand.get("type") == action.get("type"):
                gt = cand
                gt_match_idx = j
                break

        if gt is None:
            mismatches.append(
                {
                    "i": i,
                    "action": action,
                    "reason": "no_ground_truth_match",
                }
            )
            if len(mismatches) >= args.max_mismatches:
                break
            continue

        matched += 1
        if same_action(action, gt):
            exact += 1
        else:
            mismatches.append(
                {
                    "i": i,
                    "gt_i": gt_match_idx,
                    "action": action,
                    "gt": gt,
                }
            )
            if len(mismatches) >= args.max_mismatches:
                break

    report = {
        "log_path": args.log_path,
        "player_id": args.player_id,
        "requested_decisions": requested,
        "ground_truth_matched": matched,
        "exact_match": exact,
        "exact_ratio": (exact / matched) if matched else 0.0,
        "mismatches": mismatches,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

