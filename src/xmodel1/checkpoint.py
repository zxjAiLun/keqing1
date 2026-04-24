"""Shared Xmodel1 checkpoint and dimension helpers."""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite
from typing import Any

from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_RULE_CONTEXT_DIM,
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
)
from training.state_features import N_SCALAR

_REQUIRED_XMODEL1_METADATA_KEYS = (
    "model_version",
    "cfg",
    "schema_name",
    "schema_version",
    "state_tile_channels",
    "state_scalar_dim",
    "candidate_feature_dim",
    "candidate_flag_dim",
    "hidden_dim",
    "num_res_blocks",
    "dropout",
    "placement_semantics",
    "rule_context_dim",
    "rule_context_semantics",
)

_DEFAULT_XMODEL1_RANK_BONUS = (90.0, 45.0, 0.0, -135.0)


def _normalize_rank_bonus(values: Any) -> tuple[float, float, float, float]:
    if not isinstance(values, (list, tuple)) or len(values) != 4:
        return _DEFAULT_XMODEL1_RANK_BONUS
    normalized = tuple(float(value) for value in values)
    if not all(isfinite(value) for value in normalized):
        raise RuntimeError(f"xmodel1 placement rank_bonus must be finite, got {values!r}")
    return normalized  # type: ignore[return-value]


def _pt_map_scale(rank_bonus: tuple[float, float, float, float]) -> float:
    first_place = abs(float(rank_bonus[0]))
    if first_place > 0.0:
        return first_place
    fallback = max(abs(float(value)) for value in rank_bonus)
    return fallback if fallback > 0.0 else 1.0


def resolve_xmodel1_placement_semantics(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg_dict = dict(cfg or {})
    placement_cfg = cfg_dict.get("placement", {})
    if not isinstance(placement_cfg, Mapping):
        placement_cfg = {}
    rank_bonus = _normalize_rank_bonus(
        placement_cfg.get("rank_bonus", cfg_dict.get("rank_bonus", _DEFAULT_XMODEL1_RANK_BONUS))
    )
    semantics = {
        "rank_loss_weight": float(
            placement_cfg.get("rank_loss_weight", cfg_dict.get("final_rank_loss_weight", 0.05))
        ),
        "final_score_delta_loss_weight": float(
            placement_cfg.get(
                "final_score_delta_loss_weight",
                cfg_dict.get("final_score_delta_loss_weight", 0.05),
            )
        ),
        "rank_pt_value_loss_weight": float(
            placement_cfg.get(
                "rank_pt_value_loss_weight",
                cfg_dict.get("rank_pt_value_loss_weight", 0.01),
            )
        ),
        "rank_bonus": list(rank_bonus),
        "rank_bonus_norm": float(
            placement_cfg.get("rank_bonus_norm", cfg_dict.get("rank_bonus_norm", 90.0))
        ),
        "rank_score_scale": float(
            placement_cfg.get("rank_score_scale", cfg_dict.get("rank_score_scale", 0.0))
        ),
        "score_norm": float(
            placement_cfg.get("score_norm", cfg_dict.get("score_norm", 30000.0))
        ),
    }
    semantics["placement_trained"] = any(
        float(semantics[key]) > 0.0
        for key in (
            "rank_loss_weight",
            "final_score_delta_loss_weight",
            "rank_pt_value_loss_weight",
        )
    )
    return semantics


def resolve_xmodel1_rule_context_semantics(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    placement_semantics = resolve_xmodel1_placement_semantics(cfg)
    rank_bonus = tuple(float(value) for value in placement_semantics["rank_bonus"])
    scale = max(
        abs(float(placement_semantics["rank_bonus_norm"])),
        _pt_map_scale(rank_bonus),
        1e-6,
    )
    vector = [
        float(value) / scale
        for value in rank_bonus
    ]
    vector.append(float(placement_semantics["rank_score_scale"]))
    cfg_dict = dict(cfg or {})
    vector.append(1.0 if bool(cfg_dict.get("is_hanchan", cfg_dict.get("hanchan", True))) else 0.0)
    return {
        "vector": vector,
        "rank_bonus": list(rank_bonus),
        "rank_score_scale": float(placement_semantics["rank_score_scale"]),
        "is_hanchan": bool(cfg_dict.get("is_hanchan", cfg_dict.get("hanchan", True))),
    }


def resolve_xmodel1_response_post_semantics(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg_dict = dict(cfg or {})
    human_ce_loss_weight = float(
        cfg_dict.get(
            "response_post_human_ce_loss_weight",
            cfg_dict.get("response_post_ce_loss_weight", cfg_dict.get("special_rank_loss_weight", 0.25)),
        )
    )
    heuristic_rank_loss_weight = float(cfg_dict.get("response_post_rank_loss_weight", 0.15))
    return {
        "human_ce_loss_weight": human_ce_loss_weight,
        "heuristic_rank_loss_weight": heuristic_rank_loss_weight,
        "human_target_field": "response_human_discard_idx",
        "heuristic_target_field": "response_post_candidate_quality_score",
    }


def canonicalize_xmodel1_checkpoint_cfg(cfg: Mapping[str, Any] | None) -> dict[str, Any]:
    cfg_dict = dict(cfg or {})
    placement_semantics = resolve_xmodel1_placement_semantics(cfg_dict)
    response_post_semantics = resolve_xmodel1_response_post_semantics(cfg_dict)
    cfg_dict["placement"] = {
        "rank_loss_weight": float(placement_semantics["rank_loss_weight"]),
        "final_score_delta_loss_weight": float(placement_semantics["final_score_delta_loss_weight"]),
        "rank_pt_value_loss_weight": float(placement_semantics["rank_pt_value_loss_weight"]),
        "rank_bonus": list(placement_semantics["rank_bonus"]),
        "rank_bonus_norm": float(placement_semantics["rank_bonus_norm"]),
        "rank_score_scale": float(placement_semantics["rank_score_scale"]),
        "score_norm": float(placement_semantics["score_norm"]),
    }
    cfg_dict["response_post_human_ce_loss_weight"] = float(response_post_semantics["human_ce_loss_weight"])
    cfg_dict["response_post_ce_loss_weight"] = float(response_post_semantics["human_ce_loss_weight"])
    cfg_dict["response_post_rank_loss_weight"] = float(response_post_semantics["heuristic_rank_loss_weight"])
    return cfg_dict


def default_xmodel1_state_scalar_dim() -> int:
    return int(N_SCALAR)


def _shape_of(state_dict: Mapping[str, Any], key: str) -> tuple[int, ...] | None:
    value = state_dict.get(key)
    if value is None or getattr(value, "shape", None) is None:
        return None
    return tuple(int(v) for v in value.shape)


def infer_xmodel1_model_dims(
    state_dict: Mapping[str, Any],
    *,
    cfg: Mapping[str, Any] | None = None,
) -> dict[str, int | float]:
    state_proj_shape = _shape_of(state_dict, "state_proj.0.weight")
    scalar_proj_shape = _shape_of(state_dict, "scalar_proj.0.weight")
    candidate_proj_shape = _shape_of(state_dict, "candidate_proj.0.weight")
    if state_proj_shape is None or scalar_proj_shape is None or candidate_proj_shape is None:
        raise RuntimeError(
            "xmodel1 checkpoint is missing required projection weights for legacy shape inference"
        )

    state_tile_channels = int(state_proj_shape[1])
    hidden_dim = int(state_proj_shape[0])
    state_scalar_dim = int(scalar_proj_shape[1])

    candidate_flag_dim = int(
        cfg.get("candidate_flag_dim", XMODEL1_CANDIDATE_FLAG_DIM)
        if cfg is not None
        else XMODEL1_CANDIDATE_FLAG_DIM
    )
    candidate_in_dim = int(candidate_proj_shape[1])
    candidate_feature_dim = candidate_in_dim - 16 - candidate_flag_dim
    if candidate_feature_dim <= 0:
        raise RuntimeError(
            "xmodel1 legacy checkpoint candidate_proj shape is incompatible with "
            f"candidate_flag_dim={candidate_flag_dim}: shape={candidate_proj_shape}"
        )

    num_res_blocks = len(
        {
            key.split(".")[1]
            for key in state_dict
            if key.startswith("state_blocks.") and len(key.split(".")) > 1
        }
    )
    if num_res_blocks <= 0:
        num_res_blocks = int(cfg.get("num_res_blocks", 0) if cfg is not None else 0)
    if num_res_blocks <= 0:
        raise RuntimeError("xmodel1 checkpoint is missing state_blocks weights for depth inference")

    dropout = float(cfg.get("dropout", 0.1) if cfg is not None else 0.1)
    return {
        "state_tile_channels": state_tile_channels,
        "state_scalar_dim": state_scalar_dim,
        "candidate_feature_dim": candidate_feature_dim,
        "candidate_flag_dim": candidate_flag_dim,
        "hidden_dim": hidden_dim,
        "num_res_blocks": num_res_blocks,
        "dropout": dropout,
    }


def resolve_xmodel1_state_scalar_dim(
    cfg: Mapping[str, Any] | None,
    state_dict: Mapping[str, Any] | None = None,
) -> int:
    if cfg is not None:
        value = cfg.get("state_scalar_dim")
        if value is not None:
            return int(value)
    if state_dict is not None:
        weight = state_dict.get("scalar_proj.0.weight")
        if weight is not None and getattr(weight, "shape", None) is not None:
            shape = tuple(int(v) for v in weight.shape)
            if len(shape) >= 2:
                return int(shape[1])
    return default_xmodel1_state_scalar_dim()


def load_xmodel1_checkpoint_state(
    model,
    state_dict: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> None:
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    if not missing and not unexpected:
        return
    detail_parts: list[str] = []
    if missing:
        suffix = "..." if len(missing) > 8 else ""
        detail_parts.append(f"missing_keys={missing[:8]}{suffix}")
    if unexpected:
        suffix = "..." if len(unexpected) > 8 else ""
        detail_parts.append(f"unexpected_keys={unexpected[:8]}{suffix}")
    detail = "; ".join(detail_parts)
    raise RuntimeError(
        f"{checkpoint_label} is incompatible with the current xmodel1 model: {detail}. "
        "Refusing partial load; rebuild or migrate the checkpoint."
    )


def assert_xmodel1_metadata_matches_weights(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_label: str,
) -> dict[str, int | float]:
    state_dict = checkpoint.get("model")
    if not isinstance(state_dict, Mapping):
        raise RuntimeError(f"{checkpoint_label} is missing model weights")
    cfg = checkpoint.get("cfg")
    if cfg is not None and not isinstance(cfg, Mapping):
        raise RuntimeError(f"{checkpoint_label} has invalid cfg metadata")

    inferred = infer_xmodel1_model_dims(state_dict, cfg=cfg if isinstance(cfg, Mapping) else None)
    metadata = {
        "state_tile_channels": checkpoint.get("state_tile_channels", cfg.get("state_tile_channels") if isinstance(cfg, Mapping) else None),
        "state_scalar_dim": checkpoint.get("state_scalar_dim", cfg.get("state_scalar_dim") if isinstance(cfg, Mapping) else None),
        "candidate_feature_dim": checkpoint.get("candidate_feature_dim", cfg.get("candidate_feature_dim") if isinstance(cfg, Mapping) else None),
        "candidate_flag_dim": checkpoint.get("candidate_flag_dim", cfg.get("candidate_flag_dim") if isinstance(cfg, Mapping) else None),
        "hidden_dim": checkpoint.get("hidden_dim", cfg.get("hidden_dim") if isinstance(cfg, Mapping) else None),
        "num_res_blocks": checkpoint.get("num_res_blocks", cfg.get("num_res_blocks") if isinstance(cfg, Mapping) else None),
    }
    for key, value in metadata.items():
        if value is None:
            continue
        if int(value) != int(inferred[key]):
            raise RuntimeError(
                f"{checkpoint_label} metadata mismatch for {key}: metadata={value} weights={inferred[key]}"
            )
    return inferred


def validate_xmodel1_checkpoint_metadata(
    checkpoint: Mapping[str, Any],
    *,
    checkpoint_label: str,
    allow_legacy_inference: bool = False,
    require_complete_metadata: bool = False,
) -> None:
    cfg = checkpoint.get("cfg")
    if cfg is not None and not isinstance(cfg, Mapping):
        raise RuntimeError(f"{checkpoint_label} has invalid cfg metadata")
    normalized_cfg = canonicalize_xmodel1_checkpoint_cfg(cfg if isinstance(cfg, Mapping) else None)
    cfg_schema_name = cfg.get("schema_name") if isinstance(cfg, Mapping) else None
    cfg_schema_version = cfg.get("schema_version") if isinstance(cfg, Mapping) else None
    schema_name = checkpoint.get("schema_name", cfg_schema_name)
    schema_version = checkpoint.get("schema_version", cfg_schema_version)

    if schema_name is None and schema_version is None and allow_legacy_inference:
        assert_xmodel1_metadata_matches_weights(
            checkpoint,
            checkpoint_label=checkpoint_label,
        )
        return

    if schema_name != XMODEL1_SCHEMA_NAME or int(schema_version or -1) != XMODEL1_SCHEMA_VERSION:
        raise RuntimeError(
            f"{checkpoint_label} targets {schema_name!r}@{schema_version!r}, "
            f"but the current xmodel1 runtime requires {XMODEL1_SCHEMA_NAME}@{XMODEL1_SCHEMA_VERSION}. "
            "Old xmodel1 checkpoints are not load-compatible with the placement-auxiliary v6 schema cutover; "
            "rerun train for xmodel1_discard_v6."
        )

    inferred = assert_xmodel1_metadata_matches_weights(
        checkpoint,
        checkpoint_label=checkpoint_label,
    )
    placement_semantics = checkpoint.get("placement_semantics")
    if placement_semantics is not None:
        expected = resolve_xmodel1_placement_semantics(normalized_cfg)
        if placement_semantics != expected:
            raise RuntimeError(
                f"{checkpoint_label} placement_semantics metadata drifted from the training cfg: "
                f"expected {expected}, got {placement_semantics}"
            )
    rule_context_semantics = checkpoint.get("rule_context_semantics")
    if rule_context_semantics is not None:
        expected = resolve_xmodel1_rule_context_semantics(normalized_cfg)
        if rule_context_semantics != expected:
            raise RuntimeError(
                f"{checkpoint_label} rule_context_semantics metadata drifted from the training cfg: "
                f"expected {expected}, got {rule_context_semantics}"
            )
    response_post_semantics = checkpoint.get("response_post_semantics")
    if response_post_semantics is not None:
        expected = resolve_xmodel1_response_post_semantics(normalized_cfg)
        if response_post_semantics != expected:
            raise RuntimeError(
                f"{checkpoint_label} response_post_semantics metadata drifted from the training cfg: "
                f"expected {expected}, got {response_post_semantics}"
            )
    rule_context_dim = checkpoint.get("rule_context_dim")
    if rule_context_dim is not None and int(rule_context_dim) != XMODEL1_RULE_CONTEXT_DIM:
        raise RuntimeError(
            f"{checkpoint_label} rule_context_dim metadata drifted from the active xmodel1 contract"
        )
    if not require_complete_metadata:
        return

    missing = [key for key in _REQUIRED_XMODEL1_METADATA_KEYS if checkpoint.get(key) is None]
    if cfg is None:
        missing.append("cfg")
    if missing:
        raise RuntimeError(
            f"{checkpoint_label} is missing required xmodel1 checkpoint metadata {sorted(set(missing))}"
        )
    if checkpoint.get("model_version") not in {None, "xmodel1"}:
        raise RuntimeError(
            f"{checkpoint_label} has unexpected model_version={checkpoint.get('model_version')!r}"
        )
    if int(checkpoint.get("candidate_feature_dim")) != int(inferred["candidate_feature_dim"]):
        raise RuntimeError(f"{checkpoint_label} candidate_feature_dim metadata drifted from weights")


def build_xmodel1_checkpoint_metadata(
    *,
    cfg: Mapping[str, Any],
    model,
) -> dict[str, Any]:
    normalized_cfg = canonicalize_xmodel1_checkpoint_cfg(cfg)
    return {
        "model_version": "xmodel1",
        "cfg": normalized_cfg,
        "schema_name": XMODEL1_SCHEMA_NAME,
        "schema_version": XMODEL1_SCHEMA_VERSION,
        "state_tile_channels": int(getattr(model, "state_tile_channels")),
        "state_scalar_dim": int(getattr(model, "state_scalar_dim")),
        "candidate_feature_dim": int(getattr(model, "candidate_feature_dim", XMODEL1_CANDIDATE_FEATURE_DIM)),
        "candidate_flag_dim": int(getattr(model, "candidate_flag_dim", XMODEL1_CANDIDATE_FLAG_DIM)),
        "hidden_dim": int(getattr(model, "hidden_dim")),
        "num_res_blocks": int(len(getattr(model, "state_blocks"))),
        "dropout": float(normalized_cfg.get("dropout", 0.1)),
        "placement_semantics": resolve_xmodel1_placement_semantics(normalized_cfg),
        "rule_context_dim": int(getattr(model, "rule_context_dim", XMODEL1_RULE_CONTEXT_DIM)),
        "rule_context_semantics": resolve_xmodel1_rule_context_semantics(normalized_cfg),
        "response_post_semantics": resolve_xmodel1_response_post_semantics(normalized_cfg),
    }


__all__ = [
    "assert_xmodel1_metadata_matches_weights",
    "build_xmodel1_checkpoint_metadata",
    "default_xmodel1_state_scalar_dim",
    "canonicalize_xmodel1_checkpoint_cfg",
    "infer_xmodel1_model_dims",
    "load_xmodel1_checkpoint_state",
    "resolve_xmodel1_placement_semantics",
    "resolve_xmodel1_response_post_semantics",
    "resolve_xmodel1_rule_context_semantics",
    "resolve_xmodel1_state_scalar_dim",
    "validate_xmodel1_checkpoint_metadata",
]
