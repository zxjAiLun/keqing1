"""Masked action distributions for variable legal-action policies."""

from __future__ import annotations

import torch


class MaskedCategorical:
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor) -> None:
        if logits.shape != mask.shape:
            raise ValueError(f"logits shape {tuple(logits.shape)} must match mask shape {tuple(mask.shape)}")
        if logits.ndim < 2:
            raise ValueError("MaskedCategorical expects batched logits with shape [B, A]")
        if not torch.isfinite(logits).all():
            raise ValueError("MaskedCategorical logits must be finite")
        if not torch.all(mask.any(dim=-1)):
            raise ValueError("each batch row must contain at least one legal action")

        mask_bool = mask.to(dtype=torch.bool)
        mask_value = torch.finfo(logits.dtype).min
        self.logits = logits.masked_fill(~mask_bool, mask_value)
        self.mask = mask_bool
        self.dist = torch.distributions.Categorical(logits=self.logits)

    def sample(self) -> torch.Tensor:
        return self.dist.sample()

    def log_prob(self, action_index: torch.Tensor) -> torch.Tensor:
        action_index = action_index.long()
        if action_index.shape != self.logits.shape[:-1]:
            raise ValueError(
                f"action_index shape {tuple(action_index.shape)} must match batch shape "
                f"{tuple(self.logits.shape[:-1])}"
            )
        if (action_index < 0).any() or (action_index >= self.logits.shape[-1]).any():
            raise IndexError("action_index is outside the legal-action dimension")
        chosen_mask = self.mask.gather(-1, action_index.unsqueeze(-1)).squeeze(-1)
        if not torch.all(chosen_mask):
            raise ValueError("action_index must point to legal actions")
        return self.dist.log_prob(action_index)

    def entropy(self) -> torch.Tensor:
        return self.dist.entropy()

    def greedy(self) -> torch.Tensor:
        return self.logits.argmax(dim=-1)

    def probs(self) -> torch.Tensor:
        return self.dist.probs


__all__ = ["MaskedCategorical"]
