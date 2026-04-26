"""Interactive policy contracts and minimal policy scaffolds for keqingrl."""

from __future__ import annotations

import torch
import torch.nn as nn

from keqingrl.contracts import ActionSample, PolicyInput, PolicyOutput
from keqingrl.distribution import MaskedCategorical
from keqingrl.rule_score import (
    DEFAULT_RULE_SCORE_CONFIG,
    RuleScoreConfig,
    prior_logits_from_raw_scores,
)
from training.state_features import C_TILE, N_SCALAR


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
            nn.Mish(),
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(channels, momentum=0.01, eps=1e-3),
        )
        self.act = nn.Mish()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.net(x))


class _Head(nn.Module):
    def __init__(self, hidden_dim: int, out_dim: int) -> None:
        super().__init__()
        mid = max(32, hidden_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, mid),
            nn.Mish(),
            nn.Linear(mid, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class InteractivePolicy(nn.Module):
    def forward(self, policy_input: PolicyInput) -> PolicyOutput:  # pragma: no cover - interface only
        raise NotImplementedError

    def sample_action(self, policy_input: PolicyInput, *, greedy: bool = False) -> ActionSample:
        output = self.forward(policy_input)
        dist = MaskedCategorical(output.action_logits, policy_input.legal_action_mask)
        action_index = dist.greedy() if greedy else dist.sample()
        entropy = output.entropy if output.entropy is not None else dist.entropy()
        rank_probs = torch.softmax(output.rank_logits, dim=-1)
        return ActionSample(
            action_index=action_index,
            action_spec=_resolve_action_specs(policy_input, action_index),
            log_prob=dist.log_prob(action_index),
            entropy=entropy,
            value=output.value,
            rank_probs=rank_probs,
            aux=output.aux,
        )


class RandomInteractivePolicy(InteractivePolicy):
    """Zero-logit baseline policy for legality and rollout smoke tests."""

    def forward(self, policy_input: PolicyInput) -> PolicyOutput:
        device = policy_input.legal_action_mask.device
        dtype = policy_input.legal_action_features.dtype
        batch_size, max_actions = policy_input.legal_action_mask.shape
        logits = torch.zeros((batch_size, max_actions), device=device, dtype=dtype)
        value = torch.zeros((batch_size,), device=device, dtype=dtype)
        rank_logits = torch.zeros((batch_size, 4), device=device, dtype=dtype)
        entropy = MaskedCategorical(logits, policy_input.legal_action_mask).entropy()
        return PolicyOutput(
            action_logits=logits,
            value=value,
            rank_logits=rank_logits,
            entropy=entropy,
            aux={
                "rule_scores": torch.zeros_like(logits),
                "prior_logits": torch.zeros_like(logits),
                "neural_delta": torch.zeros_like(logits),
                "final_logits": logits,
            },
        )


class NeuralInteractivePolicy(InteractivePolicy):
    """A minimal state/action interaction model for variable legal-action policies."""

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        num_res_blocks: int = 2,
        c_tile: int = C_TILE,
        n_scalar: int = N_SCALAR,
        action_id_buckets: int = 4096,
        action_id_dim: int = 16,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.action_id_buckets = action_id_buckets

        self.input_proj = nn.Sequential(
            nn.Conv1d(c_tile, hidden_dim, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden_dim, momentum=0.01, eps=1e-3),
            nn.Mish(),
        )
        self.res_tower = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(num_res_blocks)])
        self.scalar_proj = nn.Sequential(
            nn.Linear(n_scalar, hidden_dim // 4),
            nn.Mish(),
            nn.Linear(hidden_dim // 4, hidden_dim // 4),
            nn.Mish(),
        )
        self.history_proj = nn.Sequential(
            nn.LazyLinear(hidden_dim // 4),
            nn.Mish(),
        )
        self.state_proj = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
        )
        self.rule_proj = nn.Sequential(
            nn.LazyLinear(hidden_dim // 4),
            nn.Mish(),
        )
        self.action_id_embed = nn.Embedding(action_id_buckets, action_id_dim)
        self.action_proj = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
        )
        self.policy_mlp = nn.Sequential(
            nn.LazyLinear(hidden_dim),
            nn.Mish(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.value_head = _Head(hidden_dim + hidden_dim // 4, 1)
        self.rank_head = _Head(hidden_dim, 4)

    def forward(self, policy_input: PolicyInput) -> PolicyOutput:
        h_state = self.encode_state(policy_input)
        h_rule = self.encode_rule(policy_input.rule_context)
        h_action = self.encode_actions(policy_input.legal_action_ids, policy_input.legal_action_features)

        h_state_expanded = h_state[:, None, :].expand(-1, h_action.size(1), -1)
        interaction = h_state_expanded * h_action
        logits = self.policy_mlp(torch.cat([h_state_expanded, h_action, interaction], dim=-1)).squeeze(-1)

        mask = policy_input.legal_action_mask.to(dtype=torch.bool)
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        value = self.value_head(torch.cat([h_state, h_rule], dim=-1)).squeeze(-1)
        rank_logits = self.rank_head(h_state)
        entropy = MaskedCategorical(logits, mask).entropy()

        return PolicyOutput(
            action_logits=logits,
            value=value,
            rank_logits=rank_logits,
            entropy=entropy,
            aux={"state_repr": h_state, "rule_repr": h_rule},
        )

    def encode_state(self, policy_input: PolicyInput) -> torch.Tensor:
        tile_obs = _cast_float(policy_input.obs.tile_obs)
        scalar_obs = _cast_float(policy_input.obs.scalar_obs)

        tile_latent = self.input_proj(tile_obs)
        tile_latent = self.res_tower(tile_latent)
        pooled = torch.cat([tile_latent.mean(dim=-1), tile_latent.amax(dim=-1)], dim=-1)
        parts = [pooled, self.scalar_proj(scalar_obs)]

        history_obs = policy_input.obs.history_obs
        if history_obs is not None:
            history_flat = _cast_float(history_obs).flatten(start_dim=1)
            parts.append(self.history_proj(history_flat))

        return self.state_proj(torch.cat(parts, dim=-1))

    def encode_rule(self, rule_context: torch.Tensor) -> torch.Tensor:
        return self.rule_proj(_cast_float(rule_context))

    def encode_actions(
        self,
        legal_action_ids: torch.Tensor,
        legal_action_features: torch.Tensor,
    ) -> torch.Tensor:
        action_ids = torch.remainder(legal_action_ids.long(), self.action_id_buckets)
        action_id_embed = self.action_id_embed(action_ids)
        action_features = _cast_float(legal_action_features)
        return self.action_proj(torch.cat([action_id_embed, action_features], dim=-1))


class RulePriorPolicy(InteractivePolicy):
    """Policy that samples directly from stored rule-prior logits."""

    def __init__(
        self,
        *,
        rule_score_scale: float = 1.0,
        rule_score_config: RuleScoreConfig = DEFAULT_RULE_SCORE_CONFIG,
    ) -> None:
        super().__init__()
        self.rule_score_scale = float(rule_score_scale)
        self.rule_score_config = rule_score_config

    def forward(self, policy_input: PolicyInput) -> PolicyOutput:
        prior_logits = _resolve_prior_logits(policy_input, self.rule_score_config)
        mask = policy_input.legal_action_mask.bool()
        logits = self.rule_score_scale * prior_logits
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        dtype = policy_input.legal_action_features.dtype
        device = policy_input.legal_action_features.device
        batch_size = int(mask.shape[0])
        value = torch.zeros((batch_size,), device=device, dtype=dtype)
        rank_logits = torch.zeros((batch_size, 4), device=device, dtype=dtype)
        entropy = MaskedCategorical(logits, mask).entropy()
        neural_delta = torch.zeros_like(logits)
        raw_scores = (
            torch.zeros_like(logits)
            if policy_input.raw_rule_scores is None
            else policy_input.raw_rule_scores.float()
        )
        return PolicyOutput(
            action_logits=logits,
            value=value,
            rank_logits=rank_logits,
            entropy=entropy,
            aux={
                "rule_scores": raw_scores,
                "prior_logits": prior_logits,
                "neural_delta": neural_delta,
                "final_logits": logits,
                "rule_score_scale": torch.tensor(self.rule_score_scale, device=device, dtype=dtype),
                "prior_temperature": torch.tensor(
                    self.rule_score_config.prior_temperature,
                    device=device,
                    dtype=dtype,
                ),
            },
        )


class RulePriorDeltaPolicy(NeuralInteractivePolicy):
    """Neural correction policy with zero-delta initialization."""

    def __init__(
        self,
        *,
        rule_score_scale: float = 1.0,
        rule_score_config: RuleScoreConfig = DEFAULT_RULE_SCORE_CONFIG,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.rule_score_scale = float(rule_score_scale)
        self.rule_score_config = rule_score_config
        self._zero_delta_output()

    def forward(self, policy_input: PolicyInput) -> PolicyOutput:
        h_state = self.encode_state(policy_input)
        h_rule = self.encode_rule(policy_input.rule_context)
        h_action = self.encode_actions(policy_input.legal_action_ids, policy_input.legal_action_features)

        h_state_expanded = h_state[:, None, :].expand(-1, h_action.size(1), -1)
        interaction = h_state_expanded * h_action
        neural_delta = self.policy_mlp(torch.cat([h_state_expanded, h_action, interaction], dim=-1)).squeeze(-1)

        prior_logits = _resolve_prior_logits(policy_input, self.rule_score_config)
        mask = policy_input.legal_action_mask.bool()
        logits = self.rule_score_scale * prior_logits + neural_delta
        logits = logits.masked_fill(~mask, torch.finfo(logits.dtype).min)
        value = self.value_head(torch.cat([h_state, h_rule], dim=-1)).squeeze(-1)
        rank_logits = self.rank_head(h_state)
        entropy = MaskedCategorical(logits, mask).entropy()
        raw_scores = (
            torch.zeros_like(logits)
            if policy_input.raw_rule_scores is None
            else policy_input.raw_rule_scores.float()
        )

        return PolicyOutput(
            action_logits=logits,
            value=value,
            rank_logits=rank_logits,
            entropy=entropy,
            aux={
                "state_repr": h_state,
                "rule_repr": h_rule,
                "rule_scores": raw_scores,
                "prior_logits": prior_logits,
                "neural_delta": neural_delta,
                "final_logits": logits,
                "rule_score_scale": torch.tensor(
                    self.rule_score_scale,
                    device=logits.device,
                    dtype=logits.dtype,
                ),
                "prior_temperature": torch.tensor(
                    self.rule_score_config.prior_temperature,
                    device=logits.device,
                    dtype=logits.dtype,
                ),
            },
        )

    def _zero_delta_output(self) -> None:
        for module in reversed(self.policy_mlp):
            if isinstance(module, nn.Linear):
                nn.init.zeros_(module.weight)
                nn.init.zeros_(module.bias)
                return


def _resolve_action_specs(policy_input: PolicyInput, action_index: torch.Tensor) -> list:
    if policy_input.legal_actions is None:
        raise ValueError("policy_input.legal_actions is required to resolve sampled ActionSpec objects")
    if len(policy_input.legal_actions) != int(action_index.shape[0]):
        raise ValueError("legal_actions batch size must match sampled action batch size")

    resolved = []
    for row_idx, col_idx in enumerate(action_index.tolist()):
        row_actions = policy_input.legal_actions[row_idx]
        if col_idx >= len(row_actions):
            raise IndexError(f"sampled action index {col_idx} exceeds legal action count {len(row_actions)}")
        resolved.append(row_actions[col_idx])
    return resolved


def _cast_float(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype == torch.float16 and tensor.device.type != "cuda":
        return tensor.float()
    if tensor.dtype in {torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8, torch.bool}:
        return tensor.float()
    return tensor


def _resolve_prior_logits(policy_input: PolicyInput, config: RuleScoreConfig) -> torch.Tensor:
    if policy_input.prior_logits is not None:
        prior_logits = policy_input.prior_logits.float()
    elif policy_input.raw_rule_scores is not None:
        prior_logits = prior_logits_from_raw_scores(
            policy_input.raw_rule_scores.float(),
            mask=policy_input.legal_action_mask,
            config=config,
        )
    else:
        raise ValueError("RulePrior policies require prior_logits or raw_rule_scores in PolicyInput")
    if prior_logits.shape != policy_input.legal_action_mask.shape:
        raise ValueError("prior_logits shape must match legal_action_mask")
    if not torch.isfinite(prior_logits[policy_input.legal_action_mask.bool()]).all():
        raise ValueError("prior_logits must be finite on legal actions")
    return prior_logits


__all__ = [
    "InteractivePolicy",
    "NeuralInteractivePolicy",
    "RandomInteractivePolicy",
    "RulePriorDeltaPolicy",
    "RulePriorPolicy",
]
