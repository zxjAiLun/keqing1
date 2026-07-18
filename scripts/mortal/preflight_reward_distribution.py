#!/usr/bin/env python3
"""Scan a Mortal dataset and report the configured reward distribution."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import json
import logging
from pathlib import Path
import random
import sys
import tomllib

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MORTAL_ROOT = REPO_ROOT / "third_party" / "Mortal"
MORTAL_PYTHON_ROOT = MORTAL_ROOT / "mortal"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(MORTAL_ROOT) not in sys.path:
    sys.path.insert(0, str(MORTAL_ROOT))
if str(MORTAL_PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(MORTAL_PYTHON_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-seed", type=int, default=20260718)
    parser.add_argument("--max-files", type=int, default=0, help="optional bounded smoke limit")
    return parser.parse_args()


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {}
    quantile_values = np.quantile(values, [0.01, 0.05, 0.5, 0.95, 0.99])
    return {
        "q01": float(quantile_values[0]),
        "q05": float(quantile_values[1]),
        "q50": float(quantile_values[2]),
        "q95": float(quantile_values[3]),
        "q99": float(quantile_values[4]),
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config_path = args.config.resolve()
    if not config_path.exists():
        raise FileNotFoundError(config_path)
    import os

    os.environ["MORTAL_CFG"] = str(config_path)
    with config_path.open("rb") as handle:
        config_data = tomllib.load(handle)

    from scripts.mortal.mainline_dataloader import (  # noqa: PLC0415
        FileDatasetsIter,
        reward_contract_from_config,
    )
    from scripts.run_mortal_dqn_offline import (  # noqa: PLC0415
        _load_or_build_file_index,
        _load_player_names,
    )

    file_list = _load_or_build_file_index(config_data)
    if args.max_files > 0:
        file_list = file_list[: args.max_files]
    player_names = _load_player_names(config_data)
    random.seed(int(args.data_seed))
    dataset_config = config_data["dataset"]
    dataset = FileDatasetsIter(
        version=int(config_data["control"]["version"]),
        file_list=list(file_list),
        pts=config_data["env"]["pts"],
        file_batch_size=int(dataset_config["file_batch_size"]),
        reserve_ratio=float(dataset_config["reserve_ratio"]),
        player_names=player_names,
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
    )

    rewards: list[float] = []
    samples = 0
    for entry in dataset:
        rewards.append(float(entry[4]))
        samples += 1
        if samples % 100000 == 0:
            logging.info("scanned samples=%s", samples)

    values = np.asarray(rewards, dtype=np.float64)
    if not len(values):
        raise RuntimeError("reward preflight produced zero samples")
    contract = reward_contract_from_config(config_data)
    report = {
        "schema": "keqing.mortal.reward_distribution_preflight.v1",
        "config": str(config_path),
        "reward_contract": contract,
        "data_seed": int(args.data_seed),
        "file_count": len(file_list),
        "player_names": sorted(player_names),
        "decision_samples": int(samples),
        "reward": {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
            "nonzero_rate": float(np.mean(values != 0)),
            "quantiles": _quantiles(values),
        },
        "passed": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
