"""Rollout collation and return/advantage utilities for keqingrl."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from keqingrl.contracts import ObsTensorBatch, PolicyInput
from keqingrl.metadata import (
    ACTION_FEATURE_CONTRACT_VERSION,
    ENV_CONTRACT_VERSION,
    NATIVE_ACTION_IDENTITY_VERSION,
    NATIVE_LEGAL_ENUMERATION_VERSION,
    NATIVE_SCHEMA_NAME,
    NATIVE_SCHEMA_VERSION,
    NATIVE_TERMINAL_RESOLVER_VERSION,
    OBSERVATION_CONTRACT_VERSION,
    REWARD_SPEC_VERSION,
    RULE_SCORE_SCALE_VERSION,
    RULE_SCORE_VERSION,
    STYLE_CONTEXT_VERSION,
    resolve_rule_score_scale_metadata,
)
from keqingrl.rollout import RolloutStep


@dataclass(frozen=True)
class PPOBatch:
    policy_input: PolicyInput
    action_index: torch.LongTensor
    old_log_prob: torch.Tensor
    old_value: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    final_rank_target: torch.LongTensor | None = None

    def to(self, device: torch.device | str) -> "PPOBatch":
        moved_policy_input = PolicyInput(
            obs=ObsTensorBatch(
                tile_obs=self.policy_input.obs.tile_obs.to(device),
                scalar_obs=self.policy_input.obs.scalar_obs.to(device),
                history_obs=None
                if self.policy_input.obs.history_obs is None
                else self.policy_input.obs.history_obs.to(device),
                extras={key: value.to(device) for key, value in self.policy_input.obs.extras.items()},
            ),
            legal_action_ids=self.policy_input.legal_action_ids.to(device),
            legal_action_features=self.policy_input.legal_action_features.to(device),
            legal_action_mask=self.policy_input.legal_action_mask.to(device),
            rule_context=self.policy_input.rule_context.to(device),
            raw_rule_scores=None
            if self.policy_input.raw_rule_scores is None
            else self.policy_input.raw_rule_scores.to(device),
            prior_logits=None
            if self.policy_input.prior_logits is None
            else self.policy_input.prior_logits.to(device),
            style_context=None
            if self.policy_input.style_context is None
            else self.policy_input.style_context.to(device),
            legal_actions=self.policy_input.legal_actions,
            recurrent_state=self.policy_input.recurrent_state,
            metadata=self.policy_input.metadata,
        )
        return PPOBatch(
            policy_input=moved_policy_input,
            action_index=self.action_index.to(device),
            old_log_prob=self.old_log_prob.to(device),
            old_value=self.old_value.to(device),
            advantages=self.advantages.to(device),
            returns=self.returns.to(device),
            final_rank_target=None
            if self.final_rank_target is None
            else self.final_rank_target.to(device),
        )


def compute_returns_and_advantages(
    rewards: torch.Tensor | list[float],
    values: torch.Tensor | list[float],
    dones: torch.Tensor | list[bool],
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    last_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    rewards_t = torch.as_tensor(rewards, dtype=torch.float32)
    values_t = torch.as_tensor(values, dtype=torch.float32)
    dones_t = torch.as_tensor(dones, dtype=torch.bool)

    if rewards_t.ndim != 1 or values_t.ndim != 1 or dones_t.ndim != 1:
        raise ValueError("rewards, values, and dones must be 1-D")
    if not (len(rewards_t) == len(values_t) == len(dones_t)):
        raise ValueError("rewards, values, and dones must have the same length")

    advantages = torch.zeros_like(rewards_t)
    next_value = float(last_value)
    next_advantage = 0.0

    for idx in range(len(rewards_t) - 1, -1, -1):
        if dones_t[idx]:
            next_value = 0.0
            next_advantage = 0.0
        delta = rewards_t[idx] + gamma * next_value - values_t[idx]
        next_advantage = float(delta) + gamma * gae_lambda * next_advantage
        advantages[idx] = next_advantage
        next_value = float(values_t[idx])

    returns = advantages + values_t
    return advantages, returns


def build_ppo_batch(
    steps: list[RolloutStep],
    advantages: torch.Tensor | list[float],
    returns: torch.Tensor | list[float],
    *,
    final_rank_target: torch.Tensor | list[int] | None = None,
    strict_metadata: bool = False,
) -> PPOBatch:
    if not steps:
        raise ValueError("steps must not be empty")
    invalid_policy_steps = [
        index
        for index, step in enumerate(steps)
        if step.is_autopilot or not step.is_learner_controlled
    ]
    if invalid_policy_steps:
        raise ValueError(
            "PPO batch only accepts learner-controlled policy steps; "
            f"invalid indices={invalid_policy_steps}"
        )

    advantages_t = torch.as_tensor(advantages, dtype=torch.float32)
    returns_t = torch.as_tensor(returns, dtype=torch.float32)
    if len(steps) != len(advantages_t) or len(steps) != len(returns_t):
        raise ValueError("steps, advantages, and returns must have the same length")

    obs = stack_obs_batches([step.obs for step in steps])
    legal_action_ids, legal_action_features, legal_action_mask = pad_legal_action_tensors(steps)
    _assert_rollout_action_order(steps, strict_metadata=strict_metadata)
    raw_rule_scores = pad_optional_legal_action_values(
        [step.raw_rule_scores for step in steps],
        legal_action_mask.shape[-1],
        field_name="raw_rule_scores",
    )
    prior_logits = pad_optional_legal_action_values(
        [step.prior_logits for step in steps],
        legal_action_mask.shape[-1],
        field_name="prior_logits",
    )
    style_context = stack_optional_contexts([step.style_context for step in steps], field_name="style_context")
    legal_actions = None
    if all(step.legal_actions is not None for step in steps):
        legal_actions = tuple(step.legal_actions for step in steps)  # type: ignore[misc]
    policy_input = PolicyInput(
        obs=obs,
        legal_action_ids=legal_action_ids,
        legal_action_features=legal_action_features,
        legal_action_mask=legal_action_mask,
        rule_context=torch.stack([step.rule_context.float() for step in steps], dim=0),
        raw_rule_scores=raw_rule_scores,
        prior_logits=prior_logits,
        style_context=style_context,
        legal_actions=legal_actions,
        metadata={
            "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
            "action_feature_contract_version": ACTION_FEATURE_CONTRACT_VERSION,
            "env_contract_version": ENV_CONTRACT_VERSION,
            "native_schema_name": NATIVE_SCHEMA_NAME,
            "native_schema_version": NATIVE_SCHEMA_VERSION,
            "native_action_identity_version": NATIVE_ACTION_IDENTITY_VERSION,
            "native_legal_enumeration_version": NATIVE_LEGAL_ENUMERATION_VERSION,
            "native_terminal_resolver_version": NATIVE_TERMINAL_RESOLVER_VERSION,
            "rule_score_version": RULE_SCORE_VERSION,
            "rule_score_scale": _resolve_batch_rule_score_scale(
                steps,
                strict_metadata=strict_metadata,
            ),
            "rule_score_scale_version": RULE_SCORE_SCALE_VERSION,
            "reward_spec_version": REWARD_SPEC_VERSION,
            "style_context_version": STYLE_CONTEXT_VERSION,
        },
    )
    rank_target = None if final_rank_target is None else torch.as_tensor(final_rank_target, dtype=torch.long)
    return PPOBatch(
        policy_input=policy_input,
        action_index=torch.tensor([step.action_index for step in steps], dtype=torch.long),
        old_log_prob=torch.tensor([step.log_prob for step in steps], dtype=torch.float32),
        old_value=torch.tensor([step.value for step in steps], dtype=torch.float32),
        advantages=advantages_t,
        returns=returns_t,
        final_rank_target=rank_target,
    )


def pad_legal_action_tensors(
    steps: list[RolloutStep],
) -> tuple[torch.LongTensor, torch.Tensor, torch.BoolTensor]:
    if not steps:
        raise ValueError("steps must not be empty")

    max_actions = max(int(step.legal_action_mask.numel()) for step in steps)
    feature_dim = int(steps[0].legal_action_features.shape[-1])

    legal_action_ids = torch.zeros((len(steps), max_actions), dtype=torch.long)
    legal_action_features = torch.zeros((len(steps), max_actions, feature_dim), dtype=torch.float32)
    legal_action_mask = torch.zeros((len(steps), max_actions), dtype=torch.bool)

    for row, step in enumerate(steps):
        if step.legal_action_ids.ndim != 1:
            raise ValueError("step.legal_action_ids must be 1-D")
        if step.legal_action_features.ndim != 2:
            raise ValueError("step.legal_action_features must be 2-D")
        if step.legal_action_mask.ndim != 1:
            raise ValueError("step.legal_action_mask must be 1-D")
        if int(step.legal_action_features.shape[-1]) != feature_dim:
            raise ValueError("legal_action_features width must match across all steps")

        action_count = int(step.legal_action_ids.numel())
        if action_count != int(step.legal_action_features.shape[0]) or action_count != int(step.legal_action_mask.numel()):
            raise ValueError("legal action ids/features/mask length must match for every step")

        legal_action_ids[row, :action_count] = step.legal_action_ids.long()
        legal_action_features[row, :action_count] = step.legal_action_features.float()
        legal_action_mask[row, :action_count] = step.legal_action_mask.bool()

    return legal_action_ids, legal_action_features, legal_action_mask


def pad_optional_legal_action_values(
    values: list[torch.Tensor | None],
    max_actions: int,
    *,
    field_name: str,
) -> torch.Tensor | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"{field_name} presence must match across PPO steps")

    out = torch.zeros((len(values), max_actions), dtype=torch.float32)
    for row, value in enumerate(values):
        assert value is not None
        if value.ndim != 1:
            raise ValueError(f"{field_name} must be 1-D per rollout step")
        if value.numel() > max_actions:
            raise ValueError(f"{field_name} length exceeds padded legal-action width")
        out[row, : value.numel()] = value.float()
    return out


def stack_optional_contexts(
    values: list[torch.Tensor | None],
    *,
    field_name: str,
) -> torch.Tensor | None:
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(f"{field_name} presence must match across PPO steps")
    return torch.stack([value.float() for value in values if value is not None], dim=0)


def _assert_rollout_action_order(steps: list[RolloutStep], *, strict_metadata: bool = False) -> None:
    for step_idx, step in enumerate(steps):
        if step.legal_actions is None:
            continue
        action_index = int(step.action_index)
        if action_index < 0 or action_index >= len(step.legal_actions):
            raise IndexError(f"rollout step {step_idx} action_index is outside legal_actions")
        actual_key = step.legal_actions[action_index].canonical_key
        expected_key = step.chosen_action_canonical_key
        if actual_key != expected_key:
            raise ValueError(
                "rollout action-order contract violation: "
                f"step={step_idx} index={action_index} actual={actual_key!r} expected={expected_key!r}"
            )
        _assert_contract_version(
            step.observation_contract_version,
            OBSERVATION_CONTRACT_VERSION,
            "observation contract",
            strict_metadata=strict_metadata,
        )
        _assert_contract_version(
            step.action_feature_contract_version,
            ACTION_FEATURE_CONTRACT_VERSION,
            "action feature contract",
            strict_metadata=strict_metadata,
        )
        _assert_contract_version(
            step.env_contract_version,
            ENV_CONTRACT_VERSION,
            "env contract",
            strict_metadata=strict_metadata,
        )
        _assert_contract_version(
            step.native_schema_name,
            NATIVE_SCHEMA_NAME,
            "native schema name",
            strict_metadata=strict_metadata,
        )
        _assert_contract_version(
            step.native_schema_version,
            NATIVE_SCHEMA_VERSION,
            "native schema version",
            strict_metadata=strict_metadata,
        )
        _assert_contract_version(
            step.native_action_identity_version,
            NATIVE_ACTION_IDENTITY_VERSION,
            "native action identity version",
            strict_metadata=strict_metadata,
        )
        _assert_contract_version(
            step.native_legal_enumeration_version,
            NATIVE_LEGAL_ENUMERATION_VERSION,
            "native legal enumeration version",
            strict_metadata=strict_metadata,
        )
        _assert_contract_version(
            step.native_terminal_resolver_version,
            NATIVE_TERMINAL_RESOLVER_VERSION,
            "native terminal resolver version",
            strict_metadata=strict_metadata,
        )
        _assert_contract_version(
            step.rule_score_version,
            RULE_SCORE_VERSION,
            "rule score contract",
            strict_metadata=strict_metadata,
        )
        _assert_contract_version(
            step.rule_score_scale_version,
            RULE_SCORE_SCALE_VERSION,
            "rule score scale contract",
            strict_metadata=strict_metadata,
        )
        resolve_rule_score_scale_metadata(
            {
                "rule_score_scale": step.rule_score_scale,
                "rule_score_scale_version": step.rule_score_scale_version,
            },
            strict_metadata=strict_metadata,
        )
        _assert_contract_version(
            step.reward_spec_version,
            REWARD_SPEC_VERSION,
            "reward spec contract",
            strict_metadata=strict_metadata,
        )
        _assert_contract_version(
            step.style_context_version,
            STYLE_CONTEXT_VERSION,
            "style context contract",
            strict_metadata=strict_metadata,
        )


def _resolve_batch_rule_score_scale(steps: list[RolloutStep], *, strict_metadata: bool) -> float:
    scales = [
        resolve_rule_score_scale_metadata(
            {
                "rule_score_scale": step.rule_score_scale,
                "rule_score_scale_version": step.rule_score_scale_version,
            },
            strict_metadata=strict_metadata,
        )
        for step in steps
    ]
    first = scales[0]
    mismatched = [scale for scale in scales if abs(float(scale) - float(first)) > 1e-9]
    if mismatched:
        unique = sorted({float(scale) for scale in scales})
        raise ValueError(f"PPO batch mixes rule score scales: {unique}")
    return float(first)


def _assert_contract_version(
    actual: object | None,
    expected: object,
    label: str,
    *,
    strict_metadata: bool,
) -> None:
    if actual is None:
        if strict_metadata:
            raise ValueError(f"missing {label}")
        return
    if actual != expected:
        raise ValueError(f"unsupported {label}: {actual}")


def stack_obs_batches(obs_batches: list[ObsTensorBatch]) -> ObsTensorBatch:
    if not obs_batches:
        raise ValueError("obs_batches must not be empty")

    first_history = obs_batches[0].history_obs
    if any((obs.history_obs is None) != (first_history is None) for obs in obs_batches):
        raise ValueError("history_obs presence must match across all observations")

    extra_keys = set(obs_batches[0].extras)
    if any(set(obs.extras) != extra_keys for obs in obs_batches):
        raise ValueError("obs.extras keys must match across all observations")

    history = None
    if first_history is not None:
        history = torch.stack([obs.history_obs for obs in obs_batches], dim=0)

    extras = {
        key: torch.stack([obs.extras[key] for obs in obs_batches], dim=0)
        for key in extra_keys
    }
    return ObsTensorBatch(
        tile_obs=torch.stack([obs.tile_obs for obs in obs_batches], dim=0),
        scalar_obs=torch.stack([obs.scalar_obs for obs in obs_batches], dim=0),
        history_obs=history,
        extras=extras,
    )


__all__ = [
    "PPOBatch",
    "build_ppo_batch",
    "compute_returns_and_advantages",
    "pad_optional_legal_action_values",
    "pad_legal_action_tensors",
    "stack_obs_batches",
    "stack_optional_contexts",
]
