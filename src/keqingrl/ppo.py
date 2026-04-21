"""PPO losses and update helpers for keqingrl."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from keqingrl.buffer import PPOBatch
from keqingrl.distribution import MaskedCategorical
from keqingrl.policy import InteractivePolicy


@dataclass(frozen=True)
class PPOLossBreakdown:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_bonus: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    rank_loss: torch.Tensor | None = None


def compute_ppo_loss(
    policy: InteractivePolicy,
    batch: PPOBatch,
    *,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    rank_coef: float = 0.0,
    normalize_advantages: bool = True,
) -> PPOLossBreakdown:
    output = policy(batch.policy_input)
    dist = MaskedCategorical(output.action_logits, batch.policy_input.legal_action_mask)
    new_log_prob = dist.log_prob(batch.action_index)
    entropy = output.entropy if output.entropy is not None else dist.entropy()

    advantages = batch.advantages
    if normalize_advantages and advantages.numel() > 1:
        advantages = (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)

    ratio = torch.exp(new_log_prob - batch.old_log_prob)
    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
    policy_loss = -torch.min(unclipped, clipped).mean()

    value_loss = F.smooth_l1_loss(output.value, batch.returns)
    entropy_bonus = entropy.mean()
    approx_kl = 0.5 * (new_log_prob - batch.old_log_prob).pow(2).mean()
    clip_fraction = ((ratio - 1.0).abs() > clip_eps).float().mean()

    rank_loss = None
    total_loss = policy_loss + value_coef * value_loss - entropy_coef * entropy_bonus
    if rank_coef > 0.0 and batch.final_rank_target is not None:
        rank_loss = F.cross_entropy(output.rank_logits, batch.final_rank_target)
        total_loss = total_loss + rank_coef * rank_loss

    return PPOLossBreakdown(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_bonus=entropy_bonus,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        rank_loss=rank_loss,
    )


def ppo_update(
    policy: InteractivePolicy,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    *,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    rank_coef: float = 0.0,
    normalize_advantages: bool = True,
    max_grad_norm: float | None = None,
) -> PPOLossBreakdown:
    optimizer.zero_grad(set_to_none=True)
    losses = compute_ppo_loss(
        policy,
        batch,
        clip_eps=clip_eps,
        value_coef=value_coef,
        entropy_coef=entropy_coef,
        rank_coef=rank_coef,
        normalize_advantages=normalize_advantages,
    )
    losses.total_loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return losses


__all__ = ["PPOLossBreakdown", "compute_ppo_loss", "ppo_update"]
