"""Local JSONL registry helpers for Mortal model experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

REGISTRY_SCHEMA = "keqing.mortal.model_registry.v1"
DEFAULT_REGISTRY_PATH = Path("artifacts/experiments/mortal_model_registry.jsonl")

REQUIRED_FIELDS = (
    "experiment_id",
    "parent_checkpoint",
    "reward_profile",
    "pt_table",
    "grp_checkpoint",
    "training_data",
    "style_data",
    "train_steps",
    "eval_bundle",
    "notes",
)


def validate_registry_entry(entry: Mapping[str, Any]) -> dict[str, Any]:
    missing = [field for field in REQUIRED_FIELDS if field not in entry]
    if missing:
        raise ValueError(f"registry entry missing required fields: {', '.join(missing)}")
    normalized = dict(entry)
    normalized["schema"] = str(normalized.get("schema") or REGISTRY_SCHEMA)
    normalized["experiment_id"] = str(normalized["experiment_id"])
    normalized["parent_checkpoint"] = str(normalized["parent_checkpoint"])
    normalized["reward_profile"] = str(normalized["reward_profile"])
    normalized["pt_table"] = [float(value) for value in normalized["pt_table"]]
    if len(normalized["pt_table"]) != 4:
        raise ValueError(f"pt_table must have length 4, got {len(normalized['pt_table'])}")
    normalized["grp_checkpoint"] = str(normalized["grp_checkpoint"])
    normalized["training_data"] = str(normalized["training_data"])
    normalized["style_data"] = None if normalized["style_data"] is None else str(normalized["style_data"])
    normalized["train_steps"] = int(normalized["train_steps"])
    normalized["eval_bundle"] = None if normalized["eval_bundle"] is None else str(normalized["eval_bundle"])
    normalized["notes"] = str(normalized["notes"])
    return normalized


def append_registry_entry(path: str | Path, entry: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_registry_entry(entry)
    registry_path = Path(path)
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(normalized, ensure_ascii=False, separators=(",", ":")) + "\n")
    return normalized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Append one Mortal model experiment registry entry")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY_PATH)
    parser.add_argument("--entry-json", required=True, help="Complete JSON object for one registry entry")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    entry = append_registry_entry(args.registry, json.loads(str(args.entry_json)))
    print(json.dumps(entry, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
