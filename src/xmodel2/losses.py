from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.to(values.device).float()
    while mask_f.ndim < values.ndim:
        mask_f = mask_f.unsqueeze(-1)
    return (values * mask_f).sum() / mask_f.sum().clamp_min(1.0)


def soft_label_cross_entropy(
    logits: torch.Tensor,
    target_distribution: torch.Tensor,
) -> torch.Tensor:
    target = target_distribution.float()
    target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-6)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    return -(target * log_probs).sum(dim=-1).mean()


def tied_top1_accuracy(
    logits: torch.Tensor,
    target_distribution: torch.Tensor,
) -> torch.Tensor:
    target = target_distribution.float()
    pred = logits.argmax(dim=-1)
    best_target = target.max(dim=-1, keepdim=True).values
    acceptable = target >= (best_target - 1e-6)
    return acceptable.gather(1, pred.unsqueeze(1)).float().mean()


__all__ = ["masked_mean", "soft_label_cross_entropy", "tied_top1_accuracy"]
