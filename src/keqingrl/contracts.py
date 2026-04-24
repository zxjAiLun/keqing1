"""Core policy contracts for the keqingrl family."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from keqingrl.actions import ActionSpec


@dataclass(frozen=True)
class ObsTensorBatch:
    tile_obs: torch.Tensor
    scalar_obs: torch.Tensor
    history_obs: torch.Tensor | None = None
    extras: dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyInput:
    obs: ObsTensorBatch
    legal_action_ids: torch.LongTensor
    legal_action_features: torch.Tensor
    legal_action_mask: torch.BoolTensor
    rule_context: torch.Tensor
    raw_rule_scores: torch.Tensor | None = None
    prior_logits: torch.Tensor | None = None
    style_context: torch.Tensor | None = None
    legal_actions: tuple[tuple["ActionSpec", ...], ...] | None = None
    recurrent_state: Any | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyOutput:
    action_logits: torch.Tensor
    value: torch.Tensor
    rank_logits: torch.Tensor
    entropy: torch.Tensor | None = None
    aux: dict[str, torch.Tensor] = field(default_factory=dict)
    next_recurrent_state: Any | None = None


@dataclass(frozen=True)
class ActionSample:
    action_index: torch.LongTensor
    action_spec: list["ActionSpec"]
    log_prob: torch.Tensor
    entropy: torch.Tensor
    value: torch.Tensor
    rank_probs: torch.Tensor


__all__ = ["ActionSample", "ObsTensorBatch", "PolicyInput", "PolicyOutput"]
