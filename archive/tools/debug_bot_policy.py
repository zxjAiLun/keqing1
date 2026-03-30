from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from bot.mjai_bot import MjaiPolicyBot
from mahjong_env.legal_actions import enumerate_legal_actions
from mahjong_env.replay import read_mjai_jsonl
from mahjong_env.state import apply_event


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True, help="mjai jsonl path")
    parser.add_argument("--checkpoint", required=True, help="checkpoint path (.npz or .json)")
    parser.add_argument("--player-id", type=int, default=0)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-decisions", type=int, default=0, help="0 means no limit")
    parser.add_argument("--out", default="artifacts/debug/policy_debug_report.json")
    args = parser.parse_args()

    events = read_mjai_jsonl(args.log_path)
    bot = MjaiPolicyBot(player_id=args.player_id, checkpoint_path=args.checkpoint)

    decisions: List[Dict[str, Any]] = []
    route_counter: Counter[str] = Counter()
    total_vocab_miss = 0
    total_legal_count = 0
    total_dahai = 0
    dahai_tsumogiri = 0

    for i, ev in enumerate(events):
        apply_event(bot.state, ev)
        if bot._riichi_states is not None:
            payload = json.dumps(ev, ensure_ascii=False)
            if "?" not in payload:
                try:
                    for ps in bot._riichi_states:
                        ps.update(payload)
                except BaseException:
                    bot._riichi_states = None

        if bot.state.actor_to_move != args.player_id:
            continue

        snap = bot.state.snapshot(args.player_id)
        if bot._riichi_states is not None:
            ps = bot._riichi_states[args.player_id]
            snap["shanten"] = int(ps.shanten)
            snap["waits_count"] = int(sum(ps.waits))
        legal = [a.to_mjai() for a in enumerate_legal_actions(snap, args.player_id)]
        chosen, meta = bot._choose_action_with_meta(legal, snap, topk=args.topk)

        route = str(meta.get("route", "unknown"))
        route_counter[route] += 1
        vocab_miss = int(meta.get("vocab_miss", 0))
        total_vocab_miss += vocab_miss
        total_legal_count += len(legal)

        is_dahai = chosen.get("type") == "dahai"
        if is_dahai:
            total_dahai += 1
            if bool(chosen.get("tsumogiri")):
                dahai_tsumogiri += 1

        decisions.append(
            {
                "event_index": i,
                "route": route,
                "chosen": chosen,
                "is_dahai": is_dahai,
                "vocab_miss": vocab_miss,
                "legal_dahai_count": sum(1 for a in legal if a.get("type") == "dahai"),
                "topk": meta.get("topk", []),
            }
        )
        if args.max_decisions > 0 and len(decisions) >= args.max_decisions:
            break

    report = {
        "log_path": args.log_path,
        "checkpoint": args.checkpoint,
        "player_id": args.player_id,
        "num_decisions": len(decisions),
        "route_counts": dict(route_counter),
        "fallback_rate": (route_counter.get("fallback", 0) / len(decisions)) if decisions else 0.0,
        "vocab_miss_rate": (total_vocab_miss / total_legal_count) if total_legal_count else 0.0,
        "tsumogiri_rate": (dahai_tsumogiri / total_dahai) if total_dahai else 0.0,
        "non_tsumogiri_rate": ((total_dahai - dahai_tsumogiri) / total_dahai) if total_dahai else 0.0,
        "decisions": decisions,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"ok": True, "out": str(out_path), "num_decisions": len(decisions)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
