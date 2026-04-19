#!/usr/bin/env python3
"""Offline evidence harness for keqingv4 special-action calibration on real replay review logs.

Reads replay/review decisions.json logs, extracts entries with special candidates
(reach / hora / ryukyoku), computes keqingv4-style special summaries from the
logged state + legal actions, overlays calibration bonus onto the logged
candidate scores, and reports before/after choice changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate keqingv4 special calibration on real replay decision logs")
    parser.add_argument("--inputs", nargs="+", required=True, help="decisions.json files or replay directories")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--max-per-type", type=int, default=20, help="Maximum cases per special action type")
    return parser.parse_args()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))

    from inference.review import candidate_to_log_dict
    from inference.scoring import _special_action_calibration_bonus, _special_meta_from_summary
    from mahjong_env.action_space import action_to_idx
    from keqingv4.preprocess_features import build_typed_action_summaries

    args = _parse_args()

    input_paths: list[Path] = []
    for raw in args.inputs:
        p = Path(raw)
        if p.is_dir():
            if p.name.startswith("replay_") and (p / "decisions.json").exists():
                input_paths.append(p / "decisions.json")
            else:
                input_paths.extend(sorted(p.glob("*/decisions.json")))
                input_paths.extend(sorted(p.glob("decisions.json")))
        else:
            input_paths.append(p)

    special_types = {"reach": 0, "hora": 1, "ryukyoku": 2}
    kept_per_type: Counter[str] = Counter()
    cases: list[dict] = []

    for path in input_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload["log"] if isinstance(payload, dict) and "log" in payload else payload if isinstance(payload, list) else []
        for entry in rows:
            candidates = entry.get("candidates") or []
            legal_actions = [c.get("action") for c in candidates if c.get("action")]
            if not legal_actions:
                continue
            candidate_specials = [a.get("type") for a in legal_actions if a.get("type") in special_types]
            if not candidate_specials:
                continue

            state = {
                "bakaze": entry.get("bakaze", "E"),
                "kyoku": entry.get("kyoku", 1),
                "honba": entry.get("honba", 0),
                "kyotaku": 0,
                "oya": entry.get("oya", 0),
                "scores": entry.get("scores", [25000, 25000, 25000, 25000]),
                "hand": entry.get("hand", []),
                "discards": entry.get("discards", [[], [], [], []]),
                "melds": entry.get("melds", [[], [], [], []]),
                "dora_markers": entry.get("dora_markers", []),
                "reached": entry.get("reached", [False, False, False, False]),
                "tsumo_pai": entry.get("tsumo_pai"),
            }
            actor = int(entry.get("actor_to_move", 0))
            try:
                _discard_summary, _call_summary, special_summary = build_typed_action_summaries(state, actor, legal_actions)
            except Exception:
                continue

            before_candidates: list[dict] = []
            after_candidates: list[dict] = []
            best_before = None
            best_after = None
            best_before_score = -1e18
            best_after_score = -1e18

            special_present_types: set[str] = set()
            for cand in candidates:
                action = cand.get("action") or {}
                action_type = action.get("type", "")
                score = cand.get("final_score", cand.get("beam_score", cand.get("logit")))
                if score is None:
                    continue
                base_score = float(score)
                calib_bonus = 0.0
                meta = {}
                if action_type in special_types:
                    slot = special_types[action_type]
                    meta = _special_meta_from_summary(action, special_summary[slot])
                    calib_bonus = _special_action_calibration_bonus(meta)
                    special_present_types.add(action_type)
                before_item = {
                    "action": action,
                    "score": base_score,
                }
                after_item = {
                    "action": action,
                    "score": base_score + calib_bonus,
                }
                if meta:
                    after_item["meta"] = meta
                    after_item["calibration_bonus"] = calib_bonus
                before_candidates.append(before_item)
                after_candidates.append(after_item)
                if base_score > best_before_score:
                    best_before_score = base_score
                    best_before = action
                if base_score + calib_bonus > best_after_score:
                    best_after_score = base_score + calib_bonus
                    best_after = action

            for special_type in sorted(special_present_types):
                if kept_per_type[special_type] >= args.max_per_type:
                    continue
                kept_per_type[special_type] += 1
                cases.append(
                    {
                        "source": str(path),
                        "step": entry.get("step"),
                        "special_type": special_type,
                        "before": {"chosen": best_before, "candidates": before_candidates},
                        "after": {"chosen": best_after, "candidates": after_candidates},
                        "flipped": best_before != best_after,
                    }
                )

    output = {
        "summary": {
            "total_cases": len(cases),
            "per_type": dict(kept_per_type),
            "flipped_cases": sum(1 for case in cases if case["flipped"]),
        },
        "cases": cases,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote replay special calibration evidence -> {output_path}")


if __name__ == "__main__":
    main()
