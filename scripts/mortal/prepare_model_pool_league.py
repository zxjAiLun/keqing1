#!/usr/bin/env python3
"""Prepare a persistent seven-model, balanced four-player native league."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


DEFAULT_ROOT = Path("artifacts/experiments/model_pool_2026_07")
REGISTRY = (
    ("model_v4", "artifacts/model_v4_20240308_best_min.pth", "external_strong"),
    ("V0a_10000", "artifacts/experiments/v4_synthetic_2026_06/V0a_v4_synthetic_scratch_2026_06/checkpoints/mortal_v0a_10000.pth", "closed_pilot"),
    ("V0b_15000", "artifacts/experiments/v4_synthetic_2026_06/V0b_v4_synthetic_clean_2026_07/checkpoints/mortal_15000.pth", "closed_clean_candidate"),
    ("V1_74000", "artifacts/experiments/v4_synthetic_2026_06/V1_v4_synthetic_warmstart_2026_06/checkpoints/mortal_v1_74000.pth", "synthetic_warmstart"),
    ("T1_71000", "artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth", "transfer_reference"),
    ("70k", "artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth", "balanced_anchor"),
    ("80k_game", "artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth", "aggressive_anchor"),
)

# Complements of the Fano-plane triples. Every account appears in four tables,
# and every pair of accounts co-occurs in exactly two tables.
FANO_COMPLEMENT_LINEUPS = (
    (3, 4, 5, 6),
    (1, 2, 5, 6),
    (1, 2, 3, 4),
    (0, 2, 4, 6),
    (0, 2, 3, 5),
    (0, 1, 4, 5),
    (0, 1, 3, 6),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--games-per-lineup", type=int, default=250)
    parser.add_argument("--seed-start", type=int, default=950000)
    parser.add_argument("--seed-key", type=int, default=8192)
    parser.add_argument("--require-checkpoints", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if int(args.games_per_lineup) <= 0:
        raise ValueError("--games-per-lineup must be positive")
    root = args.output_root.resolve()
    accounts: list[dict[str, Any]] = []
    for label, relative_path, style_tag in REGISTRY:
        checkpoint = Path(relative_path)
        if args.require_checkpoints and not checkpoint.exists():
            raise FileNotFoundError(checkpoint)
        accounts.append(
            {
                "account_id": f"{label}@01",
                "model_label": label,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256_file(checkpoint),
                "style_tag": style_tag,
                "status": "ready" if checkpoint.exists() else "pending_checkpoint",
            }
        )

    lineups: list[dict[str, Any]] = []
    for index, member_indexes in enumerate(FANO_COMPLEMENT_LINEUPS, 1):
        members = [accounts[member_index] for member_index in member_indexes]
        lineups.append(
            {
                "lineup_id": f"L{index:02d}",
                "league_order": index,
                "models": [member["model_label"] for member in members],
                "account_ids": [member["account_id"] for member in members],
                "checkpoint_paths": [member["checkpoint"] for member in members],
                "games": int(args.games_per_lineup),
                "seed_start": int(args.seed_start) + (index - 1) * int(args.games_per_lineup),
                "seed_key": int(args.seed_key),
                "seat_mode": "random",
                "native_batch_games": 100,
                "output_dir": str(root / "lineups" / f"L{index:02d}"),
                "status": "pending",
            }
        )

    manifest = {
        "schema": "keqing.mortal.model_pool_league.v1",
        "league_id": "model_pool_2026_07_balanced_7x4",
        "description": "Seven-model balanced incomplete block league; all seats random.",
        "accounts": accounts,
        "lineups": lineups,
        "design": {
            "accounts": 7,
            "models_per_table": 4,
            "lineups": 7,
            "games_per_lineup": int(args.games_per_lineup),
            "games_per_account": int(args.games_per_lineup) * 4,
            "pair_cooccurrence_games": int(args.games_per_lineup) * 2,
            "rank_points": [90, 45, 0, -135],
            "strength_readout": "persistent Tenhou-style Pt/R plus lineup-conditioned avg rank/pt",
        },
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "model_pool_registry.json").write_text(
        json.dumps({"schema": "keqing.mortal.model_pool_registry.v1", "accounts": accounts}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (root / "league_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["design"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
