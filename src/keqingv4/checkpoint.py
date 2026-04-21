"""Fail-closed checkpoint helpers for keqingv4."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from keqingv4.cache_contract import KEQINGV4_SCHEMA_NAME, KEQINGV4_SCHEMA_VERSION
from training.cache_schema import (
    KEQINGV4_CALL_SUMMARY_SLOTS,
    KEQINGV4_EVENT_HISTORY_DIM,
    KEQINGV4_EVENT_HISTORY_LEN,
    KEQINGV4_OPPORTUNITY_DIM,
    KEQINGV4_SPECIAL_SUMMARY_SLOTS,
    KEQINGV4_SUMMARY_DIM,
)

_REQUIRED_METADATA_KEYS = (
    "model_version",
    "cfg",
    "schema_name",
    "schema_version",
    "summary_dim",
    "call_summary_slots",
    "special_summary_slots",
    "event_history_len",
    "event_history_dim",
    "opportunity_dim",
    "hidden_dim",
    "num_res_blocks",
    "action_embed_dim",
    "context_dim",
    "dropout",
)


def _shape_of(state_dict: Mapping[str, Any], key: str) -> tuple[int, ...] | None:
    value = state_dict.get(key)
    if value is None or getattr(value, "shape", None) is None:
        return None
    return tuple(int(v) for v in value.shape)


def infer_keqingv4_input_dims(state_dict: Mapping[str, Any]) -> dict[str, int]:
    input_proj_shape = _shape_of(state_dict, "input_proj.0.weight")
    scalar_proj_shape = _shape_of(state_dict, "scalar_proj.0.weight")
    if input_proj_shape is None or len(input_proj_shape) != 3:
        raise RuntimeError("keqingv4 checkpoint is missing input_proj.0.weight for input shape inference")
    if scalar_proj_shape is None or len(scalar_proj_shape) != 2:
        raise RuntimeError("keqingv4 checkpoint is missing scalar_proj.0.weight for scalar shape inference")
    return {
        "c_tile": int(input_proj_shape[1]),
        "n_scalar": int(scalar_proj_shape[1]),
    }


def build_keqingv4_checkpoint_metadata(
    *,
    cfg: Mapping[str, Any],
    model,
) -> dict[str, Any]:
    return {
        "model_version": "keqingv4",
        "cfg": dict(cfg),
        "schema_name": KEQINGV4_SCHEMA_NAME,
        "schema_version": KEQINGV4_SCHEMA_VERSION,
        "summary_dim": KEQINGV4_SUMMARY_DIM,
        "call_summary_slots": KEQINGV4_CALL_SUMMARY_SLOTS,
        "special_summary_slots": KEQINGV4_SPECIAL_SUMMARY_SLOTS,
        "event_history_len": KEQINGV4_EVENT_HISTORY_LEN,
        "event_history_dim": KEQINGV4_EVENT_HISTORY_DIM,
        "opportunity_dim": KEQINGV4_OPPORTUNITY_DIM,
        "hidden_dim": int(getattr(model, "hidden_dim")),
        "num_res_blocks": int(len(getattr(model, "res_tower"))),
        "action_embed_dim": int(getattr(model, "action_embed_dim")),
        "context_dim": int(getattr(model, "context_dim")),
        "dropout": float(cfg.get("dropout", 0.1)),
        "state_tile_channels": int(getattr(model.input_proj[0], "in_channels")),
        "state_scalar_dim": int(getattr(model.scalar_proj[0], "in_features")),
    }


def build_keqingv4_checkpoint_payload(
    *,
    base_payload: Mapping[str, Any],
    cfg: Mapping[str, Any],
    model,
    **_: Any,
) -> dict[str, Any]:
    payload = dict(base_payload)
    payload.update(build_keqingv4_checkpoint_metadata(cfg=cfg, model=model))
    return payload


def validate_keqingv4_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> dict[str, int]:
    cfg = checkpoint.get("cfg")
    if not isinstance(cfg, Mapping):
        raise RuntimeError(f"{checkpoint_label} is missing cfg metadata")
    missing = [key for key in _REQUIRED_METADATA_KEYS if checkpoint.get(key) is None]
    if missing:
        raise RuntimeError(
            f"{checkpoint_label} is missing required keqingv4 checkpoint metadata {sorted(set(missing))}"
        )
    if checkpoint.get("model_version") != "keqingv4":
        raise RuntimeError(
            f"{checkpoint_label} has unexpected model_version={checkpoint.get('model_version')!r}"
        )
    if cfg.get("model_name") not in {None, "keqingv4"}:
        raise RuntimeError(f"{checkpoint_label} has unexpected cfg.model_name={cfg.get('model_name')!r}")
    if checkpoint.get("schema_name") != KEQINGV4_SCHEMA_NAME or int(checkpoint.get("schema_version", -1)) != KEQINGV4_SCHEMA_VERSION:
        raise RuntimeError(
            f"{checkpoint_label} targets {checkpoint.get('schema_name')!r}@{checkpoint.get('schema_version')!r}, "
            f"but the current keqingv4 runtime requires {KEQINGV4_SCHEMA_NAME}@{KEQINGV4_SCHEMA_VERSION}. "
            "Old keqingv4 checkpoints are not load-compatible with the active keqingv4 contract cutover."
        )
    if int(checkpoint.get("summary_dim", -1)) != KEQINGV4_SUMMARY_DIM:
        raise RuntimeError(f"{checkpoint_label} summary_dim metadata drifted from the active keqingv4 contract")
    if int(checkpoint.get("call_summary_slots", -1)) != KEQINGV4_CALL_SUMMARY_SLOTS:
        raise RuntimeError(f"{checkpoint_label} call_summary_slots metadata drifted from the active keqingv4 contract")
    if int(checkpoint.get("special_summary_slots", -1)) != KEQINGV4_SPECIAL_SUMMARY_SLOTS:
        raise RuntimeError(f"{checkpoint_label} special_summary_slots metadata drifted from the active keqingv4 contract")
    if int(checkpoint.get("event_history_len", -1)) != KEQINGV4_EVENT_HISTORY_LEN:
        raise RuntimeError(f"{checkpoint_label} event_history_len metadata drifted from the active keqingv4 contract")
    if int(checkpoint.get("event_history_dim", -1)) != KEQINGV4_EVENT_HISTORY_DIM:
        raise RuntimeError(f"{checkpoint_label} event_history_dim metadata drifted from the active keqingv4 contract")
    if int(checkpoint.get("opportunity_dim", -1)) != KEQINGV4_OPPORTUNITY_DIM:
        raise RuntimeError(f"{checkpoint_label} opportunity_dim metadata drifted from the active keqingv4 contract")
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, Mapping):
        raise RuntimeError(f"{checkpoint_label} is missing model weights")
    return infer_keqingv4_input_dims(state_dict)


def load_keqingv4_checkpoint_state(
    model,
    state_dict: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> None:
    try:
        model.load_state_dict(state_dict, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            f"{checkpoint_label} is incompatible with the current keqingv4 model: {exc}"
        ) from exc


def restore_keqingv4_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    model,
    optimizer,
    scheduler,
    checkpoint_label: str,
    weights_only: bool,
) -> tuple[int, int, float]:
    validate_keqingv4_checkpoint_metadata(
        checkpoint,
        checkpoint_label=checkpoint_label,
    )
    state_dict = checkpoint.get("model")
    assert isinstance(state_dict, Mapping)
    load_keqingv4_checkpoint_state(
        model,
        state_dict,
        checkpoint_label=checkpoint_label,
    )
    if weights_only:
        return 0, 0, float("inf")
    if optimizer is None:
        raise RuntimeError(f"{checkpoint_label} cannot restore optimizer state because optimizer is missing")
    optimizer_state = checkpoint.get("optimizer")
    if optimizer_state is None:
        raise RuntimeError(f"{checkpoint_label} is missing optimizer state for resume")
    optimizer.load_state_dict(optimizer_state)
    if scheduler is not None:
        scheduler_state = checkpoint.get("scheduler")
        if scheduler_state is not None:
            scheduler.load_state_dict(scheduler_state)
    return (
        int(checkpoint.get("epoch", 0)),
        int(checkpoint.get("step", 0)),
        float(checkpoint.get("best_val_loss", float("inf"))),
    )


def keqingv4_checkpoint_contract_summary() -> dict[str, int | str]:
    return {
        "schema_name": KEQINGV4_SCHEMA_NAME,
        "schema_version": KEQINGV4_SCHEMA_VERSION,
        "summary_dim": KEQINGV4_SUMMARY_DIM,
        "call_summary_slots": KEQINGV4_CALL_SUMMARY_SLOTS,
        "special_summary_slots": KEQINGV4_SPECIAL_SUMMARY_SLOTS,
        "event_history_len": KEQINGV4_EVENT_HISTORY_LEN,
        "event_history_dim": KEQINGV4_EVENT_HISTORY_DIM,
        "opportunity_dim": KEQINGV4_OPPORTUNITY_DIM,
    }


__all__ = [
    "build_keqingv4_checkpoint_metadata",
    "build_keqingv4_checkpoint_payload",
    "infer_keqingv4_input_dims",
    "keqingv4_checkpoint_contract_summary",
    "load_keqingv4_checkpoint_state",
    "restore_keqingv4_checkpoint",
    "validate_keqingv4_checkpoint_metadata",
]
