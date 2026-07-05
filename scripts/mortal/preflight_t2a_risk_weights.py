#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
from collections import Counter
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight T2a risk-gated teacher CE alignment.")
    parser.add_argument("--glob", action="append", default=[], help="Teacher replay glob. Repeatable.")
    parser.add_argument("--path", action="append", default=[], help="Exact teacher replay path. Repeatable.")
    parser.add_argument("--player-name", action="append", default=[])
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-weight", type=float, default=0.05)
    parser.add_argument("--risk-weight", type=float, default=0.0)
    return parser.parse_args()


def expand_inputs(patterns: list[str], paths: list[str]) -> list[str]:
    files: list[str] = []
    for pattern in patterns:
        files.extend(glob.glob(pattern, recursive=True))
    files.extend(paths)
    return sorted(dict.fromkeys(str(Path(path)) for path in files if str(path).endswith(".json.gz")))


def main() -> None:
    args = parse_args()
    player_names = [str(name) for name in args.player_name] or ["challenger"]
    mortal_python_dir = (args.mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))

    from libriichi.dataset import GameplayLoader  # noqa: PLC0415
    from scripts.mortal.risk_gated_mortal_dataloader import (  # noqa: PLC0415
        RiskGateConfig,
        build_teacher_ce_weights_for_game,
    )

    files = expand_inputs(args.glob, args.path)
    if args.limit and args.limit > 0:
        files = files[: int(args.limit)]
    if not files:
        raise FileNotFoundError(f"no json.gz files found for glob={args.glob} path={args.path}")

    gate = RiskGateConfig(
        enabled=True,
        base_weight=float(args.base_weight),
        risk_weight=float(args.risk_weight),
        disable_after_fuuro_discard=True,
        disable_vs_riichi_discard=True,
        disable_after_fuuro_vs_riichi_discard=True,
        disable_dealer_or_leading=True,
        disable_start_rank_1=True,
    )
    loader = GameplayLoader(
        version=int(args.version),
        oracle=False,
        player_names=player_names,
        augmented=False,
    )

    totals = Counter()
    reason_counts: Counter[str] = Counter()
    mismatches: list[dict[str, Any]] = []
    checked_games = 0

    processed = 0
    for start in range(0, len(files), max(1, int(args.batch_size))):
        batch_files = files[start : start + max(1, int(args.batch_size))]
        data = loader.load_gz_log_files(batch_files)
        for file_path, file_data in zip(batch_files, data, strict=True):
            for game in file_data:
                actions = game.take_actions()
                player_id = int(game.take_player_id())
                _, summary = build_teacher_ce_weights_for_game(
                    actions=actions,
                    file_path=file_path,
                    player_id=player_id,
                    player_names=player_names,
                    gate=gate,
                )
                checked_games += 1
                totals["samples"] += int(summary["sample_count"])
                totals["discard_samples"] += int(summary["discard_count"])
                totals["raw_dahai"] += int(summary["raw_dahai_count"])
                totals["raw_explicit_decisions"] += int(summary["raw_explicit_decision_count"])
                totals["matched_explicit_decisions"] += int(summary["matched_explicit_decision_count"])
                totals["extra_loader_decisions"] += int(summary["extra_loader_decision_count"])
                totals["gated_samples"] += int(summary["gated_count"])
                totals["base_samples"] += int(summary["base_count"])
                totals["gated_discard_samples"] += int(summary["gated_discard_count"])
                totals["base_discard_samples"] += int(summary["base_discard_count"])
                totals["active_samples"] += int(summary["active_count"])
                totals["disabled_samples"] += int(summary["disabled_count"])
                totals["disabled_discard_samples"] += int(summary["disabled_discard_count"])
                totals["weight_sum"] += float(summary["weight_mean"]) * int(summary["sample_count"])
                reason_counts.update(summary["reason_counts"])
                for mismatch in summary["mismatches"]:
                    if len(mismatches) < 100:
                        mismatches.append({"file": file_path, "player_id": player_id, **mismatch})
                    totals["mismatches"] += 1
            processed += 1
        if processed % 250 == 0 or processed >= len(files):
            print(
                f"[preflight] files {processed}/{len(files)} samples={totals['samples']} "
                f"mismatches={totals['mismatches']}",
                flush=True,
            )

    samples = int(totals["samples"])
    discards = int(totals["discard_samples"])
    report = {
        "schema": "keqing.mortal.t2a_risk_weight_preflight.v1",
        "files": len(files),
        "checked_games": int(checked_games),
        "player_names": player_names,
        "base_weight": float(args.base_weight),
        "risk_weight": float(args.risk_weight),
        "sample_count": samples,
        "discard_sample_count": discards,
        "raw_dahai_count": int(totals["raw_dahai"]),
        "raw_explicit_decision_count": int(totals["raw_explicit_decisions"]),
        "matched_explicit_decision_count": int(totals["matched_explicit_decisions"]),
        "extra_loader_decision_count": int(totals["extra_loader_decisions"]),
        "gated_sample_count": int(totals["gated_samples"]),
        "base_sample_count": int(totals["base_samples"]),
        "gated_discard_count": int(totals["gated_discard_samples"]),
        "base_discard_count": int(totals["base_discard_samples"]),
        "active_sample_count": int(totals["active_samples"]),
        "disabled_sample_count": int(totals["disabled_samples"]),
        "disabled_discard_count": int(totals["disabled_discard_samples"]),
        "teacher_ce_weight_mean": (float(totals["weight_sum"]) / samples) if samples else None,
        "gated_rate": (float(totals["gated_samples"]) / samples) if samples else None,
        "base_rate": (float(totals["base_samples"]) / samples) if samples else None,
        "gated_discard_rate": (float(totals["gated_discard_samples"]) / discards) if discards else None,
        "active_rate": (float(totals["active_samples"]) / samples) if samples else None,
        "disabled_rate": (float(totals["disabled_samples"]) / samples) if samples else None,
        "disabled_discard_rate": (float(totals["disabled_discard_samples"]) / discards) if discards else None,
        "reason_counts": dict(reason_counts),
        "mismatch_count": int(totals["mismatches"]),
        "mismatches_preview": mismatches,
        "ok": int(totals["mismatches"]) == 0
        and int(totals["matched_explicit_decisions"]) == int(totals["raw_explicit_decisions"]),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
