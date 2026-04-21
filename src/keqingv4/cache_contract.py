from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from training.cache_schema import (
    BASE_CACHE_FIELDS,
    KEQINGV4_CALL_SUMMARY_SLOTS,
    KEQINGV4_EVENT_HISTORY_DIM,
    KEQINGV4_EVENT_HISTORY_LEN,
    KEQINGV4_OPPORTUNITY_DIM,
    KEQINGV4_SPECIAL_SUMMARY_SLOTS,
    KEQINGV4_SUMMARY_DIM,
)

KEQINGV4_SCHEMA_NAME = "keqingv4"
KEQINGV4_SCHEMA_VERSION = 7
KEQINGV4_EXPORT_MODE = "rust_semantic_core"


def _shape_of(data: Any, key: str) -> tuple[int, ...] | None:
    if key not in data:
        return None
    return tuple(int(v) for v in np.asarray(data[key]).shape)


def validate_keqingv4_npz(
    data: Any,
    *,
    path: Path | None = None,
    require_full_contract: bool = True,
) -> list[str]:
    label = str(path) if path is not None else "<npz>"
    problems: list[str] = []

    missing_base = [key for key in BASE_CACHE_FIELDS if key not in data]
    if missing_base:
        return [f"{label}: missing base fields {missing_base}"]

    tile_shape = _shape_of(data, "tile_feat")
    scalar_shape = _shape_of(data, "scalar")
    mask_shape = _shape_of(data, "mask")
    action_idx_shape = _shape_of(data, "action_idx")
    value_shape = _shape_of(data, "value")
    if tile_shape is None or len(tile_shape) != 3:
        problems.append(f"{label}: tile_feat must be rank-3, got {tile_shape}")
        return problems
    sample_count = tile_shape[0]
    expected_shapes = {
        "scalar": (sample_count, scalar_shape[1]) if scalar_shape and len(scalar_shape) == 2 else None,
        "mask": (sample_count, mask_shape[1]) if mask_shape and len(mask_shape) == 2 else None,
        "action_idx": (sample_count,),
        "value": (sample_count,),
    }
    required_extra_shapes = {
        "score_delta_target": (sample_count,),
        "win_target": (sample_count,),
        "dealin_target": (sample_count,),
        "pts_given_win_target": (sample_count,),
        "pts_given_dealin_target": (sample_count,),
        "opp_tenpai_target": (sample_count, 3),
        "final_rank_target": (sample_count,),
        "final_score_delta_points_target": (sample_count,),
        "event_history": (sample_count, KEQINGV4_EVENT_HISTORY_LEN, KEQINGV4_EVENT_HISTORY_DIM),
        "v4_opportunity": (sample_count, KEQINGV4_OPPORTUNITY_DIM),
        "v4_discard_summary": (sample_count, 34, KEQINGV4_SUMMARY_DIM),
        "v4_call_summary": (sample_count, KEQINGV4_CALL_SUMMARY_SLOTS, KEQINGV4_SUMMARY_DIM),
        "v4_special_summary": (sample_count, KEQINGV4_SPECIAL_SUMMARY_SLOTS, KEQINGV4_SUMMARY_DIM),
    }

    if scalar_shape is None or len(scalar_shape) != 2 or scalar_shape[0] != sample_count:
        problems.append(f"{label}: scalar shape mismatch, got {scalar_shape}")
    if mask_shape is None or len(mask_shape) != 2 or mask_shape[0] != sample_count:
        problems.append(f"{label}: mask shape mismatch, got {mask_shape}")
    if action_idx_shape != (sample_count,):
        problems.append(f"{label}: action_idx shape mismatch, got {action_idx_shape}")
    if value_shape != (sample_count,):
        problems.append(f"{label}: value shape mismatch, got {value_shape}")

    for key, expected in expected_shapes.items():
        if expected is None or key in {"scalar", "mask", "action_idx", "value"}:
            continue
        actual = _shape_of(data, key)
        if actual != expected:
            problems.append(f"{label}: {key} shape mismatch, expected {expected}, got {actual}")
    for key, expected in required_extra_shapes.items():
        actual = _shape_of(data, key)
        if actual is None:
            if require_full_contract:
                problems.append(f"{label}: missing required field {key}")
            continue
        if actual != expected:
            problems.append(f"{label}: {key} shape mismatch, expected {expected}, got {actual}")
    return problems


def load_keqingv4_manifest(path: Path) -> tuple[dict[str, Any], list[str]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {}, [f"{path}: failed to parse manifest: {exc}"]
    return payload, validate_keqingv4_manifest(payload, path=path)


def validate_keqingv4_manifest(manifest: dict[str, Any], *, path: Path | None = None) -> list[str]:
    label = str(path) if path is not None else "<manifest>"
    problems: list[str] = []
    if manifest.get("schema_name") != KEQINGV4_SCHEMA_NAME:
        problems.append(
            f"{label}: schema_name mismatch, expected {KEQINGV4_SCHEMA_NAME}, got {manifest.get('schema_name')}"
        )
    version = manifest.get("schema_version")
    if version is None or int(version) != KEQINGV4_SCHEMA_VERSION:
        problems.append(
            f"{label}: schema_version mismatch, expected {KEQINGV4_SCHEMA_VERSION}, got {version}"
        )
    if int(manifest.get("summary_dim", -1)) != KEQINGV4_SUMMARY_DIM:
        problems.append(
            f"{label}: summary_dim mismatch, expected {KEQINGV4_SUMMARY_DIM}, got {manifest.get('summary_dim')}"
        )
    if int(manifest.get("call_summary_slots", -1)) != KEQINGV4_CALL_SUMMARY_SLOTS:
        problems.append(
            f"{label}: call_summary_slots mismatch, expected {KEQINGV4_CALL_SUMMARY_SLOTS}, got {manifest.get('call_summary_slots')}"
        )
    if int(manifest.get("special_summary_slots", -1)) != KEQINGV4_SPECIAL_SUMMARY_SLOTS:
        problems.append(
            f"{label}: special_summary_slots mismatch, expected {KEQINGV4_SPECIAL_SUMMARY_SLOTS}, got {manifest.get('special_summary_slots')}"
        )
    if int(manifest.get("opportunity_dim", -1)) != KEQINGV4_OPPORTUNITY_DIM:
        problems.append(
            f"{label}: opportunity_dim mismatch, expected {KEQINGV4_OPPORTUNITY_DIM}, got {manifest.get('opportunity_dim')}"
        )
    if manifest.get("export_mode") != KEQINGV4_EXPORT_MODE:
        problems.append(
            f"{label}: export_mode mismatch, expected {KEQINGV4_EXPORT_MODE}, got {manifest.get('export_mode')}"
        )
    return problems


def inspect_keqingv4_contract(file_paths: list[Path], *, max_files: int = 16) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "files_scanned": 0,
        "pts_given_win_files": 0,
        "pts_given_dealin_files": 0,
        "opp_tenpai_files": 0,
        "event_history_files": 0,
        "opportunity_files": 0,
        "summary_dims": set(),
        "call_summary_slots": set(),
        "special_summary_slots": set(),
        "event_history_shapes": set(),
        "opp_tenpai_shapes": set(),
        "opportunity_shapes": set(),
        "npz_problems": [],
        "manifest_problems": [],
        "manifests": {},
    }
    seen_manifests: set[Path] = set()

    for path in file_paths[: min(len(file_paths), max_files)]:
        try:
            with np.load(path, allow_pickle=False) as data:
                summary["files_scanned"] += 1
                if "pts_given_win_target" in data:
                    summary["pts_given_win_files"] += 1
                if "pts_given_dealin_target" in data:
                    summary["pts_given_dealin_files"] += 1
                if "opp_tenpai_target" in data:
                    summary["opp_tenpai_files"] += 1
                if "event_history" in data:
                    summary["event_history_files"] += 1
                if "v4_opportunity" in data:
                    summary["opportunity_files"] += 1
                discard_shape = _shape_of(data, "v4_discard_summary")
                call_shape = _shape_of(data, "v4_call_summary")
                special_shape = _shape_of(data, "v4_special_summary")
                event_shape = _shape_of(data, "event_history")
                opp_shape = _shape_of(data, "opp_tenpai_target")
                opportunity_shape = _shape_of(data, "v4_opportunity")
                if discard_shape and len(discard_shape) == 3:
                    summary["summary_dims"].add(discard_shape[-1])
                if call_shape and len(call_shape) == 3:
                    summary["call_summary_slots"].add(call_shape[1])
                if special_shape and len(special_shape) == 3:
                    summary["special_summary_slots"].add(special_shape[1])
                if event_shape and len(event_shape) == 3:
                    summary["event_history_shapes"].add(event_shape[1:])
                if opp_shape and len(opp_shape) == 2:
                    summary["opp_tenpai_shapes"].add(opp_shape[1:])
                if opportunity_shape and len(opportunity_shape) == 2:
                    summary["opportunity_shapes"].add(opportunity_shape[1:])
                summary["npz_problems"].extend(validate_keqingv4_npz(data, path=path))
        except Exception as exc:
            summary["npz_problems"].append(f"{path}: failed to read npz: {exc}")
            continue

        for parent in (path.parent, *path.parent.parents):
            manifest_path = parent / "keqingv4_export_manifest.json"
            if not manifest_path.exists() or manifest_path in seen_manifests:
                continue
            seen_manifests.add(manifest_path)
            manifest, problems = load_keqingv4_manifest(manifest_path)
            summary["manifests"][str(manifest_path)] = manifest
            summary["manifest_problems"].extend(problems)
            break
    return summary


def assert_keqingv4_contract(
    inspected: dict[str, Any],
    *,
    smoke: bool,
    allow_stale_cache: bool,
) -> None:
    if smoke or allow_stale_cache or inspected["files_scanned"] == 0:
        return

    problems: list[str] = []
    if inspected["npz_problems"]:
        problems.extend(inspected["npz_problems"])
    if inspected["manifest_problems"]:
        problems.extend(inspected["manifest_problems"])
    if not inspected["manifests"]:
        problems.append("no keqingv4_export_manifest.json found near scanned cache files")

    summary_dims = sorted(int(v) for v in inspected["summary_dims"])
    if summary_dims != [KEQINGV4_SUMMARY_DIM]:
        problems.append(
            f"summary_dim mismatch: expected {KEQINGV4_SUMMARY_DIM}, got {summary_dims or ['missing']}"
        )
    call_slots = sorted(int(v) for v in inspected["call_summary_slots"])
    if call_slots != [KEQINGV4_CALL_SUMMARY_SLOTS]:
        problems.append(
            f"call_summary_slots mismatch: expected {KEQINGV4_CALL_SUMMARY_SLOTS}, got {call_slots or ['missing']}"
        )
    special_slots = sorted(int(v) for v in inspected["special_summary_slots"])
    if special_slots != [KEQINGV4_SPECIAL_SUMMARY_SLOTS]:
        problems.append(
            f"special_summary_slots mismatch: expected {KEQINGV4_SPECIAL_SUMMARY_SLOTS}, got {special_slots or ['missing']}"
        )
    event_shapes = sorted(inspected["event_history_shapes"])
    if event_shapes != [(KEQINGV4_EVENT_HISTORY_LEN, KEQINGV4_EVENT_HISTORY_DIM)]:
        problems.append(
            "event_history shape mismatch: expected "
            f"{(KEQINGV4_EVENT_HISTORY_LEN, KEQINGV4_EVENT_HISTORY_DIM)}, got {event_shapes or ['missing']}"
        )
    opp_shapes = sorted(inspected["opp_tenpai_shapes"])
    if opp_shapes != [(3,)]:
        problems.append(f"opp_tenpai_target shape mismatch: expected {(3,)}, got {opp_shapes or ['missing']}")
    opportunity_shapes = sorted(inspected["opportunity_shapes"])
    if opportunity_shapes != [(KEQINGV4_OPPORTUNITY_DIM,)]:
        problems.append(
            "v4_opportunity shape mismatch: expected "
            f"{(KEQINGV4_OPPORTUNITY_DIM,)}, got {opportunity_shapes or ['missing']}"
        )
    if inspected["pts_given_win_files"] == 0 or inspected["pts_given_dealin_files"] == 0:
        problems.append("pts_given_* targets are missing from scanned caches")
    if inspected["opp_tenpai_files"] == 0:
        problems.append("opp_tenpai_target is missing from scanned caches")
    if inspected["event_history_files"] == 0:
        problems.append("event_history is missing from scanned caches")
    if inspected["opportunity_files"] == 0:
        problems.append("v4_opportunity is missing from scanned caches")

    if problems:
        joined = "; ".join(problems)
        raise RuntimeError(
            "keqingv4 cache contract check failed; rerun preprocess_keqingv4 before full training. "
            f"Details: {joined}. Use --allow-stale-cache only if you intentionally want fallback behavior."
        )


__all__ = [
    "KEQINGV4_EXPORT_MODE",
    "KEQINGV4_SCHEMA_NAME",
    "KEQINGV4_SCHEMA_VERSION",
    "assert_keqingv4_contract",
    "inspect_keqingv4_contract",
    "load_keqingv4_manifest",
    "validate_keqingv4_manifest",
    "validate_keqingv4_npz",
]
