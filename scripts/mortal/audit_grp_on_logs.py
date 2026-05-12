#!/usr/bin/env python3
"""Audit a fixed Mortal GRP checkpoint on MJAI gzip logs."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
from collections import defaultdict
import glob
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import torch
from torch.nn import functional as F

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.eval_metrics import RANK_POINT_PROFILES, write_metrics

AUDIT_SCHEMA = "keqing.mortal.grp_audit.v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit old GRP calibration on original/style MJAI logs")
    parser.add_argument("--grp-checkpoint", type=Path, default=Path("artifacts/mortal_training/grp.pth"))
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--logs", nargs="+", required=True, help="Replay .json.gz files, directories, or glob patterns")
    parser.add_argument("--output", type=Path, default=Path("artifacts/experiments/grp_audit/metrics.json"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--limit-games", type=int, default=None)
    parser.add_argument("--max-prefixes-per-game", type=int, default=None)
    return parser.parse_args()


def expand_log_paths(values: Sequence[str | Path]) -> list[str]:
    files: list[Path] = []
    for value in values:
        path = Path(str(value))
        if path.is_dir():
            files.extend(sorted(path.glob("**/*.json.gz")))
            continue
        matches = [Path(match) for match in sorted(glob.glob(str(value)))]
        if matches:
            for match in matches:
                if match.is_dir():
                    files.extend(sorted(match.glob("**/*.json.gz")))
                elif match.name.endswith(".json.gz"):
                    files.append(match)
            continue
        if path.name.endswith(".json.gz"):
            files.append(path)
    unique = sorted({str(path) for path in files})
    if not unique:
        raise FileNotFoundError(f"no .json.gz files found from: {list(values)}")
    return unique


def _load_grp_model(*, grp_checkpoint: Path, mortal_root: Path, device: torch.device) -> Any:
    mortal_python_dir = (mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))
    from model import GRP  # noqa: PLC0415

    state = torch.load(grp_checkpoint, weights_only=True, map_location=device)
    model = GRP().to(device).eval()
    model.load_state_dict(state["model"] if isinstance(state, Mapping) and "model" in state else state)
    return model


def _load_grp_games(*, log_files: Sequence[str], mortal_root: Path) -> list[Any]:
    mortal_python_dir = (mortal_root / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))
    from libriichi.dataset import Grp  # noqa: PLC0415

    return list(Grp.load_gz_log_files([str(path) for path in log_files]))


def _prefix_indices(length: int, max_prefixes: int | None) -> list[int]:
    if max_prefixes is None or max_prefixes <= 0 or length <= max_prefixes:
        return list(range(length))
    if max_prefixes == 1:
        return [length - 1]
    step = (length - 1) / (max_prefixes - 1)
    return sorted({min(length - 1, int(round(idx * step))) for idx in range(max_prefixes)})


def audit_grp(
    *,
    grp_checkpoint: str | Path,
    mortal_root: str | Path,
    log_files: Sequence[str | Path],
    device: str = "cpu",
    limit_games: int | None = None,
    max_prefixes_per_game: int | None = None,
) -> dict[str, Any]:
    torch_device = torch.device(device)
    model = _load_grp_model(grp_checkpoint=Path(grp_checkpoint), mortal_root=Path(mortal_root), device=torch_device)
    games = _load_grp_games(log_files=[str(path) for path in log_files], mortal_root=Path(mortal_root))
    if limit_games is not None:
        games = games[: int(limit_games)]

    sample_count = 0
    ce_sum = 0.0
    top1_correct = 0
    calibration = [{"count": 0, "confidence_sum": 0.0, "accuracy_sum": 0.0, "true_prob_sum": 0.0} for _ in range(10)]
    profile_errors = {
        profile: {"count": 0, "abs_error_sum": 0.0, "sq_error_sum": 0.0, "signed_error_sum": 0.0, "reward_deltas": []}
        for profile in RANK_POINT_PROFILES
    }

    with torch.inference_mode():
        for game in games:
            feature = game.take_feature()
            rank_by_player = list(int(value) for value in game.take_rank_by_player())
            indices = _prefix_indices(int(feature.shape[0]), max_prefixes_per_game)
            if not indices:
                continue
            exp_pt_by_profile: dict[str, list[list[float]]] = {profile: [] for profile in profile_errors}
            for idx in indices:
                inputs = [torch.as_tensor(feature[: idx + 1], dtype=torch.float64, device=torch_device)]
                logits = model(inputs)
                labels = model.get_label(torch.tensor([rank_by_player], dtype=torch.int64, device=torch_device))
                loss = F.cross_entropy(logits, labels)
                probs = logits.softmax(-1)
                confidence, pred = probs.max(-1)
                correct = int(pred.item() == labels.item())
                true_prob = float(probs[0, labels.item()].detach().cpu())
                bin_idx = min(9, max(0, int(float(confidence.item()) * 10)))
                calibration[bin_idx]["count"] += 1
                calibration[bin_idx]["confidence_sum"] += float(confidence.item())
                calibration[bin_idx]["accuracy_sum"] += correct
                calibration[bin_idx]["true_prob_sum"] += true_prob
                ce_sum += float(loss.detach().cpu())
                top1_correct += correct
                sample_count += 1

                matrix = model.calc_matrix(logits).detach().cpu()[0]
                for profile, pts in RANK_POINT_PROFILES.items():
                    pts_tensor = torch.tensor(pts, dtype=torch.float64)
                    expected = (matrix @ pts_tensor).tolist()
                    exp_pt_by_profile[profile].append([float(value) for value in expected])
                    for player_id, expected_pt in enumerate(expected):
                        actual_pt = float(pts[rank_by_player[player_id]])
                        error = float(expected_pt) - actual_pt
                        bucket = profile_errors[profile]
                        bucket["count"] += 1
                        bucket["abs_error_sum"] += abs(error)
                        bucket["sq_error_sum"] += error * error
                        bucket["signed_error_sum"] += error

            for profile, seq in exp_pt_by_profile.items():
                pts = RANK_POINT_PROFILES[profile]
                terminal = [float(pts[rank_by_player[player_id]]) for player_id in range(4)]
                full_seq = [*seq, terminal]
                deltas = profile_errors[profile]["reward_deltas"]
                for left, right in zip(full_seq, full_seq[1:]):
                    deltas.extend(float(right[player_id]) - float(left[player_id]) for player_id in range(4))

    return {
        "schema": AUDIT_SCHEMA,
        "grp_checkpoint": str(grp_checkpoint),
        "log_file_count": len(log_files),
        "game_count": len(games),
        "sample_count": sample_count,
        "rank_ce": None if sample_count == 0 else ce_sum / sample_count,
        "top1_rank_accuracy": None if sample_count == 0 else top1_correct / sample_count,
        "calibration": finalize_calibration(calibration),
        "expected_pt_error_by_profile": finalize_profile_errors(profile_errors),
    }


def finalize_calibration(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        count = int(row["count"])
        output.append(
            {
                "bin_start": idx / 10,
                "bin_end": (idx + 1) / 10,
                "count": count,
                "avg_confidence": None if count == 0 else float(row["confidence_sum"]) / count,
                "accuracy": None if count == 0 else float(row["accuracy_sum"]) / count,
                "avg_true_label_probability": None if count == 0 else float(row["true_prob_sum"]) / count,
            }
        )
    return output


def finalize_profile_errors(profile_errors: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for profile, row in profile_errors.items():
        count = int(row["count"])
        deltas = [float(value) for value in row.get("reward_deltas", [])]
        mean_delta = sum(deltas) / len(deltas) if deltas else None
        output[profile] = {
            "rank_points": [float(value) for value in RANK_POINT_PROFILES[profile]],
            "count": count,
            "mean_abs_expected_pt_error": None if count == 0 else float(row["abs_error_sum"]) / count,
            "rmse_expected_pt_error": None if count == 0 else math.sqrt(float(row["sq_error_sum"]) / count),
            "mean_signed_expected_pt_error": None if count == 0 else float(row["signed_error_sum"]) / count,
            "reward_delta_count": len(deltas),
            "old_grp_reward_delta_mean": mean_delta,
            "old_grp_reward_delta_variance": None if not deltas else sum((value - float(mean_delta)) ** 2 for value in deltas) / len(deltas),
        }
    return output


def main() -> None:
    args = _parse_args()
    log_files = expand_log_paths(args.logs)
    report = audit_grp(
        grp_checkpoint=args.grp_checkpoint,
        mortal_root=args.mortal_root,
        log_files=log_files,
        device=str(args.device),
        limit_games=args.limit_games,
        max_prefixes_per_game=args.max_prefixes_per_game,
    )
    document = {
        "schema": AUDIT_SCHEMA,
        "run": {
            "grp_checkpoint": str(args.grp_checkpoint),
            "mortal_root": str(args.mortal_root),
            "logs": [str(value) for value in args.logs],
            "device": str(args.device),
            "limit_games": args.limit_games,
            "max_prefixes_per_game": args.max_prefixes_per_game,
        },
        "metrics": report,
        "artifacts": {},
    }
    write_metrics(args.output, document)
    print(json.dumps(document, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
