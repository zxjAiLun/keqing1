"""Contract-version constants for KeqingRL-Lite artifacts."""

from __future__ import annotations

import math

MODEL_FAMILY = "keqingrl_lite"
POLICY_CONTRACT_VERSION = "keqingrl_policy_v2"
ACTION_CONTRACT_VERSION = "keqingrl_action_v2"
NATIVE_SCHEMA_NAME = "keqingrl_native_boundary"
NATIVE_SCHEMA_VERSION = 1
NATIVE_ACTION_IDENTITY_VERSION = 1
NATIVE_LEGAL_ENUMERATION_VERSION = 1
NATIVE_TERMINAL_RESOLVER_VERSION = 1
OBSERVATION_CONTRACT_VERSION = "keqingrl_observation_v1"
ACTION_FEATURE_CONTRACT_VERSION = "keqingrl_action_feature_v1"
ENV_CONTRACT_VERSION = "keqingrl_env_v2"
RULE_SCORE_VERSION = "keqingrl_rule_score_v1"
RULE_SCORE_SCALE_VERSION = "keqingrl_rule_score_scale_v1"
DEFAULT_RULE_SCORE_SCALE = 1.0
RULE_CONTEXT_ENCODING_VERSION = "keqingrl_rule_context_v1"
REWARD_SPEC_VERSION = "keqingrl_reward_spec_v1"
STYLE_CONTEXT_VERSION = "keqingrl_style_context_v1"
STYLE_CONTEXT_DIM = 5

REQUIRED_CHECKPOINT_METADATA_KEYS = (
    "model_family",
    "policy_contract_version",
    "action_contract_version",
    "native_schema_name",
    "native_schema_version",
    "native_action_identity_version",
    "native_legal_enumeration_version",
    "native_terminal_resolver_version",
    "observation_contract_version",
    "action_feature_contract_version",
    "env_contract_version",
    "rule_score_version",
    "rule_context_encoding_version",
    "reward_spec_version",
    "style_context_version",
    "style_context_dim",
    "controlled_action_types",
    "rule_score_clip_min",
    "rule_score_clip_max",
    "prior_temperature",
    "rule_score_scale",
    "rule_score_scale_version",
    "reward_spec",
    "gamma",
    "gae_lambda",
    "ppo_config_hash",
)


def default_checkpoint_metadata(
    *,
    controlled_action_types: tuple[str, ...] = ("DISCARD",),
    rule_score_clip_min: float = -10.0,
    rule_score_clip_max: float = 0.0,
    prior_temperature: float = 1.0,
    rule_score_scale: float = DEFAULT_RULE_SCORE_SCALE,
    reward_spec: dict[str, object] | None = None,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    ppo_config_hash: str = "unset",
) -> dict[str, object]:
    return {
        "model_family": MODEL_FAMILY,
        "policy_contract_version": POLICY_CONTRACT_VERSION,
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "native_schema_name": NATIVE_SCHEMA_NAME,
        "native_schema_version": NATIVE_SCHEMA_VERSION,
        "native_action_identity_version": NATIVE_ACTION_IDENTITY_VERSION,
        "native_legal_enumeration_version": NATIVE_LEGAL_ENUMERATION_VERSION,
        "native_terminal_resolver_version": NATIVE_TERMINAL_RESOLVER_VERSION,
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
        "action_feature_contract_version": ACTION_FEATURE_CONTRACT_VERSION,
        "env_contract_version": ENV_CONTRACT_VERSION,
        "rule_score_version": RULE_SCORE_VERSION,
        "rule_context_encoding_version": RULE_CONTEXT_ENCODING_VERSION,
        "reward_spec_version": REWARD_SPEC_VERSION,
        "style_context_version": STYLE_CONTEXT_VERSION,
        "style_context_dim": STYLE_CONTEXT_DIM,
        "controlled_action_types": list(controlled_action_types),
        "rule_score_clip_min": float(rule_score_clip_min),
        "rule_score_clip_max": float(rule_score_clip_max),
        "prior_temperature": float(prior_temperature),
        "rule_score_scale": float(rule_score_scale),
        "rule_score_scale_version": RULE_SCORE_SCALE_VERSION,
        "reward_spec": {} if reward_spec is None else dict(reward_spec),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "ppo_config_hash": str(ppo_config_hash),
    }


def validate_checkpoint_metadata(
    metadata: dict[str, object],
    *,
    strict_metadata: bool = True,
    expected_rule_score_scale: float | None = None,
) -> None:
    missing = [key for key in REQUIRED_CHECKPOINT_METADATA_KEYS if key not in metadata]
    if not strict_metadata:
        missing = [
            key
            for key in missing
            if key not in {"rule_score_scale", "rule_score_scale_version"}
        ]
    if missing:
        raise ValueError(f"keqingrl checkpoint metadata missing required keys: {missing}")
    if metadata["model_family"] != MODEL_FAMILY:
        raise ValueError(f"unsupported model_family: {metadata['model_family']!r}")
    expected_versions = {
        "policy_contract_version": POLICY_CONTRACT_VERSION,
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "native_schema_name": NATIVE_SCHEMA_NAME,
        "native_schema_version": NATIVE_SCHEMA_VERSION,
        "native_action_identity_version": NATIVE_ACTION_IDENTITY_VERSION,
        "native_legal_enumeration_version": NATIVE_LEGAL_ENUMERATION_VERSION,
        "native_terminal_resolver_version": NATIVE_TERMINAL_RESOLVER_VERSION,
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
        "action_feature_contract_version": ACTION_FEATURE_CONTRACT_VERSION,
        "env_contract_version": ENV_CONTRACT_VERSION,
        "rule_score_version": RULE_SCORE_VERSION,
        "rule_context_encoding_version": RULE_CONTEXT_ENCODING_VERSION,
        "reward_spec_version": REWARD_SPEC_VERSION,
        "style_context_version": STYLE_CONTEXT_VERSION,
        "rule_score_scale_version": RULE_SCORE_SCALE_VERSION,
    }
    for key, expected in expected_versions.items():
        if key not in metadata and not strict_metadata:
            continue
        actual = metadata[key]
        if actual != expected:
            raise ValueError(f"unsupported {key}: {actual!r}; expected {expected!r}")
    if int(metadata["style_context_dim"]) != STYLE_CONTEXT_DIM:
        raise ValueError(
            f"unsupported style_context_dim: {metadata['style_context_dim']!r}; "
            f"expected {STYLE_CONTEXT_DIM}"
        )
    resolve_rule_score_scale_metadata(
        metadata,
        strict_metadata=strict_metadata,
        expected_rule_score_scale=expected_rule_score_scale,
    )


def resolve_rule_score_scale_metadata(
    metadata: dict[str, object],
    *,
    strict_metadata: bool,
    expected_rule_score_scale: float | None = None,
) -> float:
    actual = metadata.get("rule_score_scale")
    if actual is None:
        if strict_metadata:
            raise ValueError("missing rule score scale")
        actual_scale = DEFAULT_RULE_SCORE_SCALE
    else:
        try:
            actual_scale = float(actual)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"invalid rule score scale: {actual!r}") from exc
        if not math.isfinite(actual_scale) or actual_scale < 0.0:
            raise ValueError(f"invalid rule score scale: {actual!r}")

    actual_version = metadata.get("rule_score_scale_version")
    if actual_version is None:
        if strict_metadata:
            raise ValueError("missing rule score scale contract")
    elif actual_version != RULE_SCORE_SCALE_VERSION:
        raise ValueError(
            f"unsupported rule score scale contract: {actual_version!r}; "
            f"expected {RULE_SCORE_SCALE_VERSION!r}"
        )

    if expected_rule_score_scale is not None:
        expected_scale = float(expected_rule_score_scale)
        if not math.isfinite(expected_scale) or expected_scale < 0.0:
            raise ValueError(f"invalid expected rule score scale: {expected_rule_score_scale!r}")
        if not math.isclose(actual_scale, expected_scale, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError(
                "rule score scale mismatch: "
                f"expected {expected_scale!r}, got {actual_scale!r}"
            )
    return actual_scale


__all__ = [
    "ACTION_CONTRACT_VERSION",
    "ACTION_FEATURE_CONTRACT_VERSION",
    "DEFAULT_RULE_SCORE_SCALE",
    "ENV_CONTRACT_VERSION",
    "MODEL_FAMILY",
    "NATIVE_ACTION_IDENTITY_VERSION",
    "NATIVE_LEGAL_ENUMERATION_VERSION",
    "NATIVE_SCHEMA_NAME",
    "NATIVE_SCHEMA_VERSION",
    "NATIVE_TERMINAL_RESOLVER_VERSION",
    "OBSERVATION_CONTRACT_VERSION",
    "POLICY_CONTRACT_VERSION",
    "REQUIRED_CHECKPOINT_METADATA_KEYS",
    "REWARD_SPEC_VERSION",
    "RULE_CONTEXT_ENCODING_VERSION",
    "RULE_SCORE_SCALE_VERSION",
    "RULE_SCORE_VERSION",
    "STYLE_CONTEXT_DIM",
    "STYLE_CONTEXT_VERSION",
    "default_checkpoint_metadata",
    "resolve_rule_score_scale_metadata",
    "validate_checkpoint_metadata",
]
