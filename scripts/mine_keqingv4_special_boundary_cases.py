#!/usr/bin/env python3
"""Mine near-boundary special-action cases from real replay/review logs.

Looks for decisions where a special action (reach / hora / ryukyoku) exists in
the candidate list but is not overwhelmingly above/below the chosen action.
This helps find cases where calibration may realistically flip the decision.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mine near-boundary keqingv4 special-action cases from replay decision logs")
    parser.add_argument("--inputs", nargs="+", required=True, help="decisions.json files or replay directories")
    parser.add_argument("--output", required=True, help="JSON output path")
    parser.add_argument("--max-per-type", type=int, default=20)
    parser.add_argument("--margin-threshold", type=float, default=1.5, help="Keep cases where |chosen-special score diff| <= threshold")
    return parser.parse_args()


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    sys.path.insert(0, str(root / "src"))

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

    wanted = {"reach", "hora", "ryukyoku"}
    per_type_candidates: dict[str, list[dict]] = {key: [] for key in wanted}

    for path in input_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        rows = payload["log"] if isinstance(payload, dict) and "log" in payload else payload if isinstance(payload, list) else []
        for entry in rows:
            candidates = entry.get("candidates") or []
            if not candidates:
                continue

            chosen = entry.get("chosen") or {}
            chosen_score = None
            for cand in candidates:
                if cand.get("action") == chosen:
                    chosen_score = cand.get("final_score", cand.get("beam_score", cand.get("logit")))
                    break
            if chosen_score is None:
                continue
            chosen_score = float(chosen_score)

            specials = []
            for cand in candidates:
                action = cand.get("action") or {}
                special_type = action.get("type")
                if special_type not in wanted:
                    continue
                score = cand.get("final_score", cand.get("beam_score", cand.get("logit")))
                if score is None:
                    continue
                margin = chosen_score - float(score)
                if abs(margin) > args.margin_threshold:
                    continue
                specials.append(
                    {
                        "special_type": special_type,
                        "action": action,
                        "score": float(score),
                        "margin_vs_chosen": float(margin),
                        "is_chosen": action == chosen,
                    }
                )

            for item in specials:
                special_type = item["special_type"]
                per_type_candidates[special_type].append(
                    {
                        "source": str(path),
                        "step": entry.get("step"),
                        "special_type": special_type,
                        "chosen": chosen,
                        "chosen_score": chosen_score,
                        "candidate": item["action"],
                        "candidate_score": item["score"],
                        "margin_vs_chosen": item["margin_vs_chosen"],
                        "candidate_is_chosen": item["is_chosen"],
                        "bakaze": entry.get("bakaze"),
                        "kyoku": entry.get("kyoku"),
                        "honba": entry.get("honba"),
                        "hand": entry.get("hand"),
                        "tsumo_pai": entry.get("tsumo_pai"),
                    }
                )

    kept_per_type: dict[str, int] = {}
    cases: list[dict] = []
    for special_type in sorted(wanted):
        items = per_type_candidates[special_type]
        if not items:
            continue
        items.sort(key=lambda item: (item["candidate_is_chosen"], abs(item["margin_vs_chosen"])))
        selected = items[: args.max_per_type]
        kept_per_type[special_type] = len(selected)
        cases.extend(selected)

    output = {
        "summary": {
            "total_cases": len(cases),
            "per_type": {k: v for k, v in kept_per_type.items() if v > 0},
            "margin_threshold": args.margin_threshold,
        },
        "cases": cases,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote special boundary cases -> {output_path}")


if __name__ == "__main__":
    main()
