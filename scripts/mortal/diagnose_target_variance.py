#!/usr/bin/env python3
"""Compare q_target_mc distributions between original training logs and selfplay logs."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_MORTAL_PYTHON = (_REPO_ROOT / "third_party" / "Mortal" / "mortal").resolve()
if str(_MORTAL_PYTHON) not in sys.path:
    sys.path.insert(0, str(_MORTAL_PYTHON))

from libriichi.dataset import GameplayLoader, Grp  # type: ignore[import-untyped]
from model import GRP  # type: ignore[import-untyped]
from reward_calculator import RewardCalculator  # type: ignore[import-untyped]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-glob", default="artifacts/mortal_mjai_gz/train/**/*.json.gz")
    parser.add_argument("--selfplay-glob", default="artifacts/experiments/grp_audit/selfplay_70k_base_500h_arena/logs/**/*.json.gz")
    parser.add_argument("--selfplay-glob-extra", default="artifacts/experiments/selfplay_finetune_2026_05/pools/selfplay_70k_base_extra500_arena/logs/**/*.json.gz")
    parser.add_argument("--grp-checkpoint", type=Path, default=Path("artifacts/mortal_training/grp.pth"))
    parser.add_argument("--grp-hidden-size", type=int, default=64)
    parser.add_argument("--grp-num-layers", type=int, default=2)
    parser.add_argument("--pts", type=float, nargs=4, default=[6.0, 4.0, 2.0, 0.0])
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--max-games", type=int, default=500)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--version", type=int, default=4)
    return parser.parse_args()


def _glob_files(pattern: str) -> list[str]:
    import glob as _glob
    matches = sorted(_glob.glob(str(pattern), recursive=True))
    if not matches:
        raise FileNotFoundError(f"no files matched: {pattern}")
    return matches


def _compute_domain_stats(
    files: list[str],
    reward_calc: RewardCalculator,
    gamma: float,
    max_games: int,
    version: int,
) -> dict:
    loader = GameplayLoader(version=version, oracle=False)
    print(f"    files: {len(files)}, batch size: 20", flush=True)

    all_q_targets: list[float] = []
    all_kyoku_rewards: list[float] = []
    all_steps_to_done: list[int] = []
    pairs_step_reward: list[tuple[int, float]] = []
    binned_var: dict[str, list[float]] = defaultdict(list)

    kyoku_counts: list[int] = []
    game_steps: list[int] = []
    game_count = 0
    skipped = 0

    FILE_BATCH = 20
    for batch_start in range(0, len(files), FILE_BATCH):
        if game_count >= max_games:
            break
        batch_files = files[batch_start:batch_start + FILE_BATCH]
        print(f"    batch {batch_start // FILE_BATCH + 1}: loading {len(batch_files)} files...", flush=True)
        gameplay_files = loader.load_gz_log_files(batch_files)
        for gameplay_file in gameplay_files:
            for game in gameplay_file:
                if game_count >= max_games:
                    break
                obs_list = game.take_obs()
                game_size = len(obs_list)
                if game_size == 0:
                    skipped += 1
                    continue

                at_kyoku = game.take_at_kyoku()
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()
                grp = game.take_grp()
                player_id = game.take_player_id()

                grp_feature = grp.take_feature()
                rank_by_player = grp.take_rank_by_player()

                n_kyokus = grp_feature.shape[0]
                if n_kyokus < 1:
                    skipped += 1
                    continue

                kyoku_rewards = reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)

                steps_to_done = np.zeros(game_size, dtype=np.int64)
                for i in reversed(range(game_size)):
                    if not dones[i]:
                        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                for i in range(game_size):
                    k = int(at_kyoku[i])
                    if k >= len(kyoku_rewards):
                        continue
                    kr = float(kyoku_rewards[k])
                    sd = int(steps_to_done[i])
                    qt = (gamma ** sd) * kr

                    all_q_targets.append(qt)
                    all_kyoku_rewards.append(kr)
                    all_steps_to_done.append(sd)
                    pairs_step_reward.append((sd, kr))

                    bucket = "0" if sd <= 0 else "1-10" if sd <= 10 else "11-30" if sd <= 30 else "31-60" if sd <= 60 else "61-100" if sd <= 100 else "101+"
                    binned_var[bucket].append(qt)

                kyoku_counts.append(n_kyokus)
                game_steps.append(game_size)
                game_count += 1
                if game_count % 50 == 0:
                    print(f"    processed {game_count} games...", flush=True)

            if game_count >= max_games:
                break

    q_targets = np.array(all_q_targets, dtype=np.float64)
    kyoku_rewards_arr = np.array(all_kyoku_rewards, dtype=np.float64)
    steps_to_done_arr = np.array(all_steps_to_done, dtype=np.int64)
    pairs = np.array(pairs_step_reward, dtype=np.float64).reshape(-1, 2)

    corr = float(np.corrcoef(pairs[:, 0], pairs[:, 1])[0, 1]) if len(pairs) > 1 else float("nan")

    def _pct(arr, p):
        return float(np.percentile(arr, p))

    binned_stats = {}
    for bucket, vals in sorted(binned_var.items()):
        arr = np.array(vals, dtype=np.float64)
        binned_stats[bucket] = {
            "count": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "var": float(np.var(arr)),
        }

    return {
        "game_count": game_count,
        "skipped_empty": skipped,
        "total_steps": len(q_targets),
        "kyokus_per_game_mean": float(np.mean(kyoku_counts)),
        "kyokus_per_game_std": float(np.std(kyoku_counts)),
        "steps_per_game_mean": float(np.mean(game_steps)),
        "steps_per_game_std": float(np.std(game_steps)),
        "q_target_mc": {
            "mean": float(np.mean(q_targets)),
            "std": float(np.std(q_targets)),
            "var": float(np.var(q_targets)),
            "p50": _pct(q_targets, 50),
            "p90": _pct(q_targets, 90),
            "p99": _pct(q_targets, 99),
        },
        "kyoku_rewards": {
            "mean": float(np.mean(kyoku_rewards_arr)),
            "std": float(np.std(kyoku_rewards_arr)),
            "var": float(np.var(kyoku_rewards_arr)),
            "p50": _pct(kyoku_rewards_arr, 50),
            "p90": _pct(kyoku_rewards_arr, 90),
            "p99": _pct(kyoku_rewards_arr, 99),
        },
        "steps_to_done": {
            "mean": float(np.mean(steps_to_done_arr)),
            "std": float(np.std(steps_to_done_arr)),
            "var": float(np.var(steps_to_done_arr)),
            "p50": _pct(steps_to_done_arr, 50),
            "p90": _pct(steps_to_done_arr, 90),
            "p99": _pct(steps_to_done_arr, 99),
        },
        "corr_steps_to_done__kyoku_reward": corr,
        "binned_by_steps_to_done": binned_stats,
    }


def main() -> None:
    args = _parse_args()
    pt_values = [float(v) for v in args.pts]

    grp = GRP(hidden_size=args.grp_hidden_size, num_layers=args.grp_num_layers)
    grp_state = torch.load(args.grp_checkpoint, weights_only=True, map_location=torch.device("cpu"))
    grp.load_state_dict(grp_state["model"])
    reward_calc = RewardCalculator(grp, pts=pt_values)

    max_files = max(args.max_games * 2, 30)
    original_files = _glob_files(args.original_glob)[:max_files]
    sp_files_a = _glob_files(args.selfplay_glob)
    sp_files_b = _glob_files(args.selfplay_glob_extra)
    selfplay_files = sorted(set(sp_files_a + sp_files_b))[:max_files]

    print(f"original files: {len(original_files)}", flush=True)
    print(f"selfplay files: {len(selfplay_files)}", flush=True)

    print("\n--- original ---", flush=True)
    original_stats = _compute_domain_stats(
        original_files, reward_calc, float(args.gamma), args.max_games, args.version,
    )

    print("\n--- selfplay ---", flush=True)
    selfplay_stats = _compute_domain_stats(
        selfplay_files, reward_calc, float(args.gamma), args.max_games, args.version,
    )

    report = {
        "schema": "keqing.mortal.target_variance_diagnostic.v1",
        "config": {
            "original_glob": args.original_glob,
            "selfplay_globs": [args.selfplay_glob, args.selfplay_glob_extra],
            "gamma": float(args.gamma),
            "pts": pt_values,
            "max_games": args.max_games,
        },
        "original": original_stats,
        "selfplay": selfplay_stats,
    }

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print("\n" + "=" * 60)
    _print_comparison(report)
    print("=" * 60)


def _print_comparison(report: dict) -> None:
    orig = report["original"]
    sp = report["selfplay"]

    rows = [
        ("games", "d", orig["game_count"], sp["game_count"]),
        ("total steps", "d", orig["total_steps"], sp["total_steps"]),
        ("kyokus/game", ".1f", orig["kyokus_per_game_mean"], sp["kyokus_per_game_mean"]),
        ("steps/game", ".0f", orig["steps_per_game_mean"], sp["steps_per_game_mean"]),
    ]

    for label, key in [("q_target_mc", "q_target_mc"), ("kyoku_rewards", "kyoku_rewards"), ("steps_to_done", "steps_to_done")]:
        for stat in ["mean", "std", "var", "p50", "p90", "p99"]:
            val_o = orig[label][stat]
            val_sp = sp[label][stat]
            if isinstance(val_o, float):
                fmt = ".6f"
            else:
                fmt = ".2f"
            rows.append((f"{label}.{stat}", fmt, val_o, val_sp))

    rows.append(("corr(steps, reward)", ".4f", orig["corr_steps_to_done__kyoku_reward"], sp["corr_steps_to_done__kyoku_reward"]))

    print(f"{'metric':<40} {'original':>16} {'selfplay':>16}")
    print("-" * 72)
    for name, fmt, o, s in rows:
        print(f"{name:<40} {o:>{16}{fmt}} {s:>{16}{fmt}}")

    print("\n--- binned target variance by steps_to_done ---")
    print(f"{'bucket':<12} {'orig_var':>14} {'sp_var':>14} {'ratio':>10}")
    print("-" * 50)
    for bucket in sorted(set(list(orig["binned_by_steps_to_done"].keys()) + list(sp["binned_by_steps_to_done"].keys()))):
        vo = orig["binned_by_steps_to_done"].get(bucket, {}).get("var", float("nan"))
        vs = sp["binned_by_steps_to_done"].get(bucket, {}).get("var", float("nan"))
        ratio = vs / vo if vo and vo > 0 else float("nan")
        print(f"{bucket:<12} {vo:>14.6f} {vs:>14.6f} {ratio:>10.3f}")


if __name__ == "__main__":
    main()
