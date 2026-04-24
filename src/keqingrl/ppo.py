"""PPO losses and update helpers for keqingrl."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from keqingrl.buffer import PPOBatch
from keqingrl.distribution import MaskedCategorical
from keqingrl.policy import InteractivePolicy
from keqingrl.rule_score import smoothed_prior_probs


@dataclass(frozen=True)
class PPOLossBreakdown:
    total_loss: torch.Tensor
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_bonus: torch.Tensor
    approx_kl: torch.Tensor
    clip_fraction: torch.Tensor
    rank_loss: torch.Tensor | None = None
    rule_kl: torch.Tensor | None = None
    rule_agreement: torch.Tensor | None = None
    avg_abs_neural_delta: torch.Tensor | None = None
    delta_norm: torch.Tensor | None = None


@dataclass(frozen=True)
class CriticPretrainLossBreakdown:
    total_loss: torch.Tensor
    value_loss: torch.Tensor
    rank_loss: torch.Tensor | None = None


def compute_ppo_loss(
    policy: InteractivePolicy,
    batch: PPOBatch,
    *,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    rank_coef: float = 0.0,
    rule_kl_coef: float = 0.0,
    prior_kl_eps: float = 1e-4,
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

    rule_kl = _rule_kl_from_output(output, dist, batch, prior_kl_eps=prior_kl_eps)
    if rule_kl is not None and rule_kl_coef > 0.0:
        total_loss = total_loss + float(rule_kl_coef) * rule_kl
    rule_agreement = _rule_agreement_from_output(output, batch)
    avg_abs_neural_delta, delta_norm = _delta_metrics_from_output(output, batch)

    return PPOLossBreakdown(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy_bonus=entropy_bonus,
        approx_kl=approx_kl,
        clip_fraction=clip_fraction,
        rank_loss=rank_loss,
        rule_kl=rule_kl,
        rule_agreement=rule_agreement,
        avg_abs_neural_delta=avg_abs_neural_delta,
        delta_norm=delta_norm,
    )


def compute_critic_pretrain_loss(
    policy: InteractivePolicy,
    batch: PPOBatch,
    *,
    value_coef: float = 1.0,
    rank_coef: float = 1.0,
) -> CriticPretrainLossBreakdown:
    output = policy(batch.policy_input)
    value_loss = F.smooth_l1_loss(output.value, batch.returns)
    total_loss = float(value_coef) * value_loss

    rank_loss = None
    if rank_coef > 0.0 and batch.final_rank_target is not None:
        rank_loss = F.cross_entropy(output.rank_logits, batch.final_rank_target)
        total_loss = total_loss + float(rank_coef) * rank_loss

    return CriticPretrainLossBreakdown(
        total_loss=total_loss,
        value_loss=value_loss,
        rank_loss=rank_loss,
    )


def critic_pretrain_update(
    policy: InteractivePolicy,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    *,
    value_coef: float = 1.0,
    rank_coef: float = 1.0,
    max_grad_norm: float | None = None,
) -> CriticPretrainLossBreakdown:
    optimizer.zero_grad(set_to_none=True)
    losses = compute_critic_pretrain_loss(
        policy,
        batch,
        value_coef=value_coef,
        rank_coef=rank_coef,
    )
    losses.total_loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return losses


def ppo_update(
    policy: InteractivePolicy,
    optimizer: torch.optim.Optimizer,
    batch: PPOBatch,
    *,
    clip_eps: float = 0.2,
    value_coef: float = 0.5,
    entropy_coef: float = 0.01,
    rank_coef: float = 0.0,
    rule_kl_coef: float = 0.0,
    prior_kl_eps: float = 1e-4,
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
        rule_kl_coef=rule_kl_coef,
        prior_kl_eps=prior_kl_eps,
        normalize_advantages=normalize_advantages,
    )
    losses.total_loss.backward()
    if max_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
    optimizer.step()
    return losses


def _rule_kl_from_output(
    output,
    dist: MaskedCategorical,
    batch: PPOBatch,
    *,
    prior_kl_eps: float,
) -> torch.Tensor | None:
    prior_logits = output.aux.get("prior_logits")
    if prior_logits is None:
        prior_logits = batch.policy_input.prior_logits
    if prior_logits is None:
        return None
    mask = batch.policy_input.legal_action_mask.bool()
    current_probs = dist.probs().masked_fill(~mask, 0.0)
    current_log_probs = torch.log(current_probs.clamp_min(1e-12))
    prior_probs = smoothed_prior_probs(prior_logits.float(), mask, eps=prior_kl_eps)
    prior_log_probs = torch.log(prior_probs.clamp_min(1e-12))
    kl = (current_probs * (current_log_probs - prior_log_probs)).masked_fill(~mask, 0.0)
    return kl.sum(dim=-1).mean()


def _rule_agreement_from_output(output, batch: PPOBatch) -> torch.Tensor | None:
    prior_logits = output.aux.get("prior_logits")
    if prior_logits is None:
        prior_logits = batch.policy_input.prior_logits
    if prior_logits is None:
        return None
    mask = batch.policy_input.legal_action_mask.bool()
    current = output.action_logits.masked_fill(~mask, torch.finfo(output.action_logits.dtype).min).argmax(dim=-1)
    prior = prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min).argmax(dim=-1)
    return (current == prior).float().mean()


def _delta_metrics_from_output(output, batch: PPOBatch) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    neural_delta = output.aux.get("neural_delta")
    if neural_delta is None:
        return None, None
    mask = batch.policy_input.legal_action_mask.bool()
    legal_delta = neural_delta.masked_select(mask)
    if legal_delta.numel() == 0:
        return None, None
    return legal_delta.abs().mean(), legal_delta.pow(2).mean().sqrt()


__all__ = [
    "CriticPretrainLossBreakdown",
    "PPOLossBreakdown",
    "compute_critic_pretrain_loss",
    "compute_ppo_loss",
    "critic_pretrain_update",
    "ppo_update",
]
