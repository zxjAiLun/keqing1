"""Contract-version constants for KeqingRL-Lite artifacts."""

from __future__ import annotations

MODEL_FAMILY = "keqingrl_lite"
POLICY_CONTRACT_VERSION = "keqingrl_policy_v2"
ACTION_CONTRACT_VERSION = "keqingrl_action_v2"
OBSERVATION_CONTRACT_VERSION = "keqingrl_observation_v1"
ACTION_FEATURE_CONTRACT_VERSION = "keqingrl_action_feature_v1"
ENV_CONTRACT_VERSION = "keqingrl_env_v2"
RULE_SCORE_VERSION = "keqingrl_rule_score_v1"
RULE_CONTEXT_ENCODING_VERSION = "keqingrl_rule_context_v1"
STYLE_CONTEXT_DIM = 5

REQUIRED_CHECKPOINT_METADATA_KEYS = (
    "model_family",
    "policy_contract_version",
    "action_contract_version",
    "observation_contract_version",
    "action_feature_contract_version",
    "env_contract_version",
    "rule_score_version",
    "rule_context_encoding_version",
    "style_context_dim",
    "controlled_action_types",
    "rule_score_clip_min",
    "rule_score_clip_max",
    "prior_temperature",
    "rule_score_scale",
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
    rule_score_scale: float = 1.0,
    reward_spec: dict[str, object] | None = None,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    ppo_config_hash: str = "unset",
) -> dict[str, object]:
    return {
        "model_family": MODEL_FAMILY,
        "policy_contract_version": POLICY_CONTRACT_VERSION,
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
        "action_feature_contract_version": ACTION_FEATURE_CONTRACT_VERSION,
        "env_contract_version": ENV_CONTRACT_VERSION,
        "rule_score_version": RULE_SCORE_VERSION,
        "rule_context_encoding_version": RULE_CONTEXT_ENCODING_VERSION,
        "style_context_dim": STYLE_CONTEXT_DIM,
        "controlled_action_types": list(controlled_action_types),
        "rule_score_clip_min": float(rule_score_clip_min),
        "rule_score_clip_max": float(rule_score_clip_max),
        "prior_temperature": float(prior_temperature),
        "rule_score_scale": float(rule_score_scale),
        "reward_spec": {} if reward_spec is None else dict(reward_spec),
        "gamma": float(gamma),
        "gae_lambda": float(gae_lambda),
        "ppo_config_hash": str(ppo_config_hash),
    }


def validate_checkpoint_metadata(metadata: dict[str, object]) -> None:
    missing = [key for key in REQUIRED_CHECKPOINT_METADATA_KEYS if key not in metadata]
    if missing:
        raise ValueError(f"keqingrl checkpoint metadata missing required keys: {missing}")
    if metadata["model_family"] != MODEL_FAMILY:
        raise ValueError(f"unsupported model_family: {metadata['model_family']!r}")
    expected_versions = {
        "policy_contract_version": POLICY_CONTRACT_VERSION,
        "action_contract_version": ACTION_CONTRACT_VERSION,
        "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
        "action_feature_contract_version": ACTION_FEATURE_CONTRACT_VERSION,
        "env_contract_version": ENV_CONTRACT_VERSION,
        "rule_score_version": RULE_SCORE_VERSION,
        "rule_context_encoding_version": RULE_CONTEXT_ENCODING_VERSION,
    }
    for key, expected in expected_versions.items():
        actual = metadata[key]
        if actual != expected:
            raise ValueError(f"unsupported {key}: {actual!r}; expected {expected!r}")
    if int(metadata["style_context_dim"]) != STYLE_CONTEXT_DIM:
        raise ValueError(
            f"unsupported style_context_dim: {metadata['style_context_dim']!r}; "
            f"expected {STYLE_CONTEXT_DIM}"
        )


__all__ = [
    "ACTION_CONTRACT_VERSION",
    "ACTION_FEATURE_CONTRACT_VERSION",
    "ENV_CONTRACT_VERSION",
    "MODEL_FAMILY",
    "OBSERVATION_CONTRACT_VERSION",
    "POLICY_CONTRACT_VERSION",
    "REQUIRED_CHECKPOINT_METADATA_KEYS",
    "RULE_CONTEXT_ENCODING_VERSION",
    "RULE_SCORE_VERSION",
    "STYLE_CONTEXT_DIM",
    "default_checkpoint_metadata",
    "validate_checkpoint_metadata",
]
