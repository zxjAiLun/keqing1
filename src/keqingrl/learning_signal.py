"""Learning-signal diagnostics for controlled discard-only KeqingRL research."""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import torch

from keqingrl.buffer import PPOBatch
from keqingrl.distribution import MaskedCategorical
from keqingrl.policy import InteractivePolicy
from keqingrl.ppo import compute_ppo_loss
from keqingrl.rollout import RolloutEpisode, RolloutStep


MODULE_PREFIXES = (
    "input_proj",
    "res_tower",
    "scalar_proj",
    "history_proj",
    "state_proj",
    "rule_proj",
    "action_id_embed",
    "action_proj",
    "policy_mlp",
    "value_head",
    "rank_head",
)


@dataclass(frozen=True)
class PpoDiagnosticConfig:
    clip_eps: float = 0.1
    value_coef: float = 0.5
    entropy_coef: float = 0.0
    rank_coef: float = 0.0
    rule_kl_coef: float = 0.0
    prior_kl_eps: float = 1e-4
    normalize_advantages: bool = True
    max_grad_norm: float | None = None


def seed_registry_hash(seeds: Sequence[int]) -> str:
    payload = json.dumps([int(seed) for seed in seeds], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def tensor_stats(values: torch.Tensor | Sequence[float], *, prefix: str) -> dict[str, float]:
    tensor = torch.as_tensor(values, dtype=torch.float32).flatten()
    if tensor.numel() == 0:
        return {
            f"{prefix}_mean": 0.0,
            f"{prefix}_std": 0.0,
            f"{prefix}_min": 0.0,
            f"{prefix}_max": 0.0,
        }
    return {
        f"{prefix}_mean": float(tensor.mean().detach().cpu()),
        f"{prefix}_std": float(tensor.std(unbiased=False).detach().cpu()),
        f"{prefix}_min": float(tensor.min().detach().cpu()),
        f"{prefix}_max": float(tensor.max().detach().cpu()),
    }


def top1_margin_diagnostics(policy: InteractivePolicy, batch: PPOBatch) -> dict[str, Any]:
    rows = top1_margin_rows(policy, batch)
    deltas = torch.tensor([float(row["delta_needed_to_flip_top1"]) for row in rows], dtype=torch.float32)
    return {
        "rows": rows,
        "summary": {
            "selected_prior_top1_rate": _mean(row["selected_is_prior_top1"] for row in rows),
            "selected_non_top1_positive_advantage_count": int(
                sum(
                    1
                    for row, advantage in zip(rows, batch.advantages.detach().cpu().tolist(), strict=True)
                    if not bool(row["selected_is_prior_top1"]) and float(advantage) > 0.0
                )
            ),
            "mean_delta_needed_to_flip_top1": _safe_mean_tensor(deltas),
            "p10_delta_needed_to_flip_top1": _safe_quantile(deltas, 0.10),
            "p50_delta_needed_to_flip_top1": _safe_quantile(deltas, 0.50),
            "p90_delta_needed_to_flip_top1": _safe_quantile(deltas, 0.90),
        },
    }


def top1_margin_rows(policy: InteractivePolicy, batch: PPOBatch) -> list[dict[str, Any]]:
    with torch.no_grad():
        output = policy(batch.policy_input)
    prior_logits = output.aux.get("prior_logits")
    if prior_logits is None:
        prior_logits = batch.policy_input.prior_logits
    if prior_logits is None:
        raise ValueError("top1 diagnostics require prior_logits")

    mask = batch.policy_input.legal_action_mask.bool()
    masked_prior = prior_logits.float().masked_fill(~mask, torch.finfo(prior_logits.float().dtype).min)
    prior_probs = torch.softmax(masked_prior, dim=-1).masked_fill(~mask, 0.0)
    rows: list[dict[str, Any]] = []
    for row_idx in range(masked_prior.shape[0]):
        legal_count = int(mask[row_idx].sum().item())
        action_idx = int(batch.action_index[row_idx].item())
        selected_logit = float(masked_prior[row_idx, action_idx].detach().cpu())
        selected_prob = float(prior_probs[row_idx, action_idx].detach().cpu())
        legal_logits = masked_prior[row_idx][mask[row_idx]]
        legal_probs = prior_probs[row_idx][mask[row_idx]]
        top_values, _top_positions = torch.topk(legal_logits, k=min(2, legal_count))
        prior_top1_logit = float(top_values[0].detach().cpu()) if legal_count else 0.0
        prior_second_logit = float(top_values[1].detach().cpu()) if legal_count > 1 else prior_top1_logit
        prior_top1_margin = max(0.0, prior_top1_logit - prior_second_logit)
        prior_top1_index = int(masked_prior[row_idx].argmax().item())
        selected_is_top1 = action_idx == prior_top1_index
        if selected_is_top1:
            delta_needed = prior_top1_margin
        else:
            delta_needed = max(0.0, prior_top1_logit - selected_logit)
        sorted_indices = torch.argsort(masked_prior[row_idx], descending=True)
        selected_prior_rank = 1 + next(
            rank for rank, candidate in enumerate(sorted_indices.tolist()) if int(candidate) == action_idx
        )
        rows.append(
            {
                "legal_count": legal_count,
                "selected_prior_rank": int(selected_prior_rank),
                "selected_prior_prob": selected_prob,
                "prior_top1_prob": float(legal_probs.max().detach().cpu()) if legal_count else 0.0,
                "selected_is_prior_top1": bool(selected_is_top1),
                "prior_top1_margin_to_second": float(prior_top1_margin),
                "delta_needed_to_flip_top1": float(delta_needed),
            }
        )
    return rows


def batch_diagnostic_rows(
    policy: InteractivePolicy,
    batch: PPOBatch,
    prepared_steps: Sequence[RolloutStep],
    episodes: Sequence[RolloutEpisode],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if len(prepared_steps) != int(batch.action_index.numel()):
        raise ValueError("prepared_steps length must match PPO batch size")
    episode_meta = _episode_metadata(episodes)
    margin_rows = top1_margin_rows(policy, batch)
    raw_advantages = batch.advantages.detach().cpu().float()
    normalized_advantages = _normalized_advantages(raw_advantages)
    returns = batch.returns.detach().cpu().float()
    rows: list[dict[str, Any]] = []
    for idx, (step, margin) in enumerate(zip(prepared_steps, margin_rows, strict=True)):
        meta = episode_meta.get(step.episode_id or step.game_id or "", {})
        final_ranks = meta.get("final_ranks", ())
        terminal_rewards = meta.get("terminal_rewards", ())
        rows.append(
            {
                "episode_id": step.episode_id or step.game_id or "",
                "step_index": step.step_id if step.step_id is not None else idx,
                "actor": int(step.actor),
                "action_type": step.action_spec.action_type.value,
                "action_index": int(batch.action_index[idx].item()),
                "action_canonical_key": step.action_spec.canonical_key,
                "old_log_prob": float(batch.old_log_prob[idx].detach().cpu()),
                "old_value": float(batch.old_value[idx].detach().cpu()),
                "reward": float(step.reward),
                "return": float(returns[idx]),
                "advantage_raw": float(raw_advantages[idx]),
                "advantage_normalized": float(normalized_advantages[idx]),
                "final_rank": _seat_tuple_value(final_ranks, step.actor),
                "terminal_reward": _seat_tuple_value(terminal_rewards, step.actor),
                "behavior_temperature": step.behavior_temperature,
                "rule_score_scale": step.rule_score_scale,
                "rule_score_scale_version": step.rule_score_scale_version,
                **margin,
            }
        )
    return rows, batch_diagnostic_summary(rows)


def batch_diagnostic_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rewards = torch.tensor([float(row["reward"]) for row in rows], dtype=torch.float32)
    returns = torch.tensor([float(row["return"]) for row in rows], dtype=torch.float32)
    advantages = torch.tensor([float(row["advantage_raw"]) for row in rows], dtype=torch.float32)
    deltas = torch.tensor([float(row["delta_needed_to_flip_top1"]) for row in rows], dtype=torch.float32)
    scale_values = sorted(
        {
            round(float(row["rule_score_scale"]), 12)
            for row in rows
            if row.get("rule_score_scale") is not None
        }
    )
    return {
        "batch_size": len(rows),
        "rule_score_scale": scale_values[0] if len(scale_values) == 1 else None,
        "rule_score_scale_values": tuple(float(value) for value in scale_values),
        "reward_nonzero_count": int((rewards != 0.0).sum().item()) if rewards.numel() else 0,
        **tensor_stats(rewards, prefix="reward"),
        **tensor_stats(returns, prefix="return"),
        **tensor_stats(advantages, prefix="advantage"),
        "advantage_positive_rate": _mean(float(row["advantage_raw"]) > 0.0 for row in rows),
        "advantage_negative_rate": _mean(float(row["advantage_raw"]) < 0.0 for row in rows),
        "advantage_zero_rate": _mean(float(row["advantage_raw"]) == 0.0 for row in rows),
        "selected_prior_top1_rate": _mean(bool(row["selected_is_prior_top1"]) for row in rows),
        "selected_non_top1_positive_advantage_count": int(
            sum(
                1
                for row in rows
                if not bool(row["selected_is_prior_top1"]) and float(row["advantage_raw"]) > 0.0
            )
        ),
        "mean_delta_needed_to_flip_top1": _safe_mean_tensor(deltas),
        "p10_delta_needed_to_flip_top1": _safe_quantile(deltas, 0.10),
        "p50_delta_needed_to_flip_top1": _safe_quantile(deltas, 0.50),
        "p90_delta_needed_to_flip_top1": _safe_quantile(deltas, 0.90),
    }


def loss_gradient_decomposition(
    policy: InteractivePolicy,
    batch: PPOBatch,
    *,
    config: PpoDiagnosticConfig | None = None,
) -> dict[str, Any]:
    cfg = config or PpoDiagnosticConfig()
    component_specs = (
        "policy_loss_only",
        "value_loss_only",
        "rank_loss_only",
        "entropy_only",
        "rule_kl_only",
        "total_loss",
    )
    rows = []
    for component in component_specs:
        policy.zero_grad(set_to_none=True)
        losses = compute_ppo_loss(
            policy,
            batch,
            clip_eps=cfg.clip_eps,
            value_coef=cfg.value_coef,
            entropy_coef=cfg.entropy_coef,
            rank_coef=cfg.rank_coef,
            rule_kl_coef=cfg.rule_kl_coef,
            prior_kl_eps=cfg.prior_kl_eps,
            normalize_advantages=cfg.normalize_advantages,
        )
        loss_tensor = _component_loss_tensor(component, losses)
        if loss_tensor is None:
            rows.append(_empty_gradient_row(component))
            continue
        loss_tensor.backward()
        rows.append({"component": component, "loss_value": float(loss_tensor.detach().cpu()), **gradient_norms(policy)})
    policy.zero_grad(set_to_none=True)
    return {"rows": rows, "summary": _gradient_summary(rows)}


def gradient_norms(policy: InteractivePolicy) -> dict[str, float]:
    sums = {prefix: 0.0 for prefix in MODULE_PREFIXES}
    final_weight = 0.0
    final_bias = 0.0
    hidden_sum = 0.0
    for name, parameter in policy.named_parameters():
        if parameter.grad is None:
            continue
        grad_sq = float(parameter.grad.detach().float().pow(2).sum().cpu())
        for prefix in MODULE_PREFIXES:
            if name.startswith(prefix + "."):
                sums[prefix] += grad_sq
                break
        if name == "policy_mlp.3.weight":
            final_weight += grad_sq
        elif name == "policy_mlp.3.bias":
            final_bias += grad_sq
        elif name.startswith("policy_mlp."):
            hidden_sum += grad_sq
    norms = {f"{prefix}_grad_norm": value ** 0.5 for prefix, value in sums.items()}
    actor_sq = sums["input_proj"] + sums["res_tower"] + sums["scalar_proj"] + sums["history_proj"] + sums["state_proj"] + sums["action_id_embed"] + sums["action_proj"] + sums["policy_mlp"]
    critic_sq = sums["value_head"]
    rank_sq = sums["rank_head"]
    actor_norm = actor_sq ** 0.5
    critic_norm = critic_sq ** 0.5
    norms.update(
        {
            "policy_mlp.final_linear.weight_grad_norm": final_weight ** 0.5,
            "policy_mlp.final_linear.bias_grad_norm": final_bias ** 0.5,
            "policy_mlp.hidden_layer_grad_norm": hidden_sum ** 0.5,
            "actor_grad_norm_total": actor_norm,
            "critic_grad_norm_total": critic_norm,
            "rank_grad_norm_total": rank_sq ** 0.5,
            "actor_to_critic_grad_ratio": actor_norm / max(critic_norm, 1e-12),
        }
    )
    return norms


def ppo_update_probe(
    policy: InteractivePolicy,
    batch: PPOBatch,
    *,
    lrs: Sequence[float] = (3e-5, 3e-4, 1e-3),
    update_epochs: Sequence[int] = (1, 4, 16),
    clip_eps: float = 0.1,
    normalize_advantages: bool = True,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lr in lrs:
        for epoch_count in update_epochs:
            probe_policy = copy.deepcopy(policy)
            optimizer = torch.optim.Adam(probe_policy.parameters(), lr=float(lr))
            previous_selected_logits, previous_prior_top1_logits = _selected_and_prior_top1_logits(probe_policy, batch)
            for epoch in range(1, int(epoch_count) + 1):
                optimizer.zero_grad(set_to_none=True)
                losses = compute_ppo_loss(
                    probe_policy,
                    batch,
                    clip_eps=clip_eps,
                    value_coef=0.0,
                    entropy_coef=0.0,
                    rank_coef=0.0,
                    rule_kl_coef=0.0,
                    normalize_advantages=normalize_advantages,
                )
                losses.total_loss.backward()
                optimizer.step()
                row = _post_update_probe_row(
                    probe_policy,
                    batch,
                    lr=float(lr),
                    target_update_epochs=int(epoch_count),
                    epoch=epoch,
                    previous_selected_logits=previous_selected_logits,
                    previous_prior_top1_logits=previous_prior_top1_logits,
                    clip_eps=clip_eps,
                )
                rows.append(row)
    return rows


def classify_learning_signal_blocker(
    batch_summary: dict[str, Any],
    gradient_summary: dict[str, Any],
    overfit_summary: dict[str, Any] | None = None,
) -> str:
    if float(batch_summary.get("advantage_std", 0.0)) <= 1e-8 or int(batch_summary.get("selected_non_top1_positive_advantage_count", 0)) == 0:
        return "REWARD_ADVANTAGE_TOO_SPARSE"
    actor_grad = float(gradient_summary.get("total_loss_actor_grad_norm_total", 0.0))
    final_grad = float(gradient_summary.get("total_loss_policy_mlp_final_grad_norm", 0.0))
    if actor_grad <= 0.0 or final_grad <= 0.0:
        return "ACTOR_GRADIENT_DEAD_OR_WRONG_SIGN"
    if overfit_summary is not None:
        if bool(overfit_summary.get("pass_actor_can_move")):
            return "BATCH_OVERFITS_BUT_ROLLOUT_FAILS"
        if bool(overfit_summary.get("prior_margin_too_large")):
            return "PRIOR_ANCHOR_TOO_STRONG"
    return "PPO_UPDATE_TOO_WEAK_OR_UNCLASSIFIED"


def _component_loss_tensor(component: str, losses: Any) -> torch.Tensor | None:
    if component == "policy_loss_only":
        return losses.policy_loss
    if component == "value_loss_only":
        return losses.value_loss
    if component == "rank_loss_only":
        return losses.rank_loss
    if component == "entropy_only":
        return -losses.entropy_bonus
    if component == "rule_kl_only":
        return losses.rule_kl
    if component == "total_loss":
        return losses.total_loss
    raise ValueError(f"unsupported loss component: {component}")


def _post_update_probe_row(
    policy: InteractivePolicy,
    batch: PPOBatch,
    *,
    lr: float,
    target_update_epochs: int,
    epoch: int,
    previous_selected_logits: torch.Tensor,
    previous_prior_top1_logits: torch.Tensor,
    clip_eps: float,
) -> dict[str, Any]:
    with torch.no_grad():
        output = policy(batch.policy_input)
        dist = MaskedCategorical(output.action_logits, batch.policy_input.legal_action_mask)
        new_log_prob = dist.log_prob(batch.action_index)
        ratio = torch.exp(new_log_prob - batch.old_log_prob)
        current_selected_logits, current_prior_top1_logits = _selected_and_prior_top1_logits(policy, batch)
        selected_delta = current_selected_logits - previous_selected_logits
        prior_top1_delta = current_prior_top1_logits - previous_prior_top1_logits
        positive_mask = batch.advantages.float() > 0.0
        negative_mask = batch.advantages.float() < 0.0
        delta_stats = _delta_stats(output, batch)
        rule_agreement = _rule_agreement(output, batch)
    return {
        "lr": lr,
        "target_update_epochs": target_update_epochs,
        "epoch": epoch,
        "post_approx_kl_vs_old": float(0.5 * (new_log_prob - batch.old_log_prob).pow(2).mean().detach().cpu()),
        "post_ratio_mean": float(ratio.mean().detach().cpu()),
        "post_ratio_std": float(ratio.std(unbiased=False).detach().cpu()),
        "post_ratio_min": float(ratio.min().detach().cpu()),
        "post_ratio_max": float(ratio.max().detach().cpu()),
        "post_clip_fraction": float(((ratio - 1.0).abs() > float(clip_eps)).float().mean().detach().cpu()),
        "neural_delta_abs_mean": delta_stats["neural_delta_abs_mean"],
        "neural_delta_abs_max": delta_stats["neural_delta_abs_max"],
        "top1_action_changed_rate": delta_stats["top1_action_changed_rate"],
        "rule_agreement": rule_agreement,
        "selected_action_logit_change_mean": _masked_mean(selected_delta, torch.ones_like(positive_mask, dtype=torch.bool)),
        "prior_top1_logit_change_mean": _masked_mean(prior_top1_delta, torch.ones_like(positive_mask, dtype=torch.bool)),
        "positive_adv_selected_logit_change_mean": _masked_mean(selected_delta, positive_mask),
        "negative_adv_selected_logit_change_mean": _masked_mean(selected_delta, negative_mask),
    }


def _selected_and_prior_top1_logits(policy: InteractivePolicy, batch: PPOBatch) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        output = policy(batch.policy_input)
        mask = batch.policy_input.legal_action_mask.bool()
        selected = output.action_logits.gather(1, batch.action_index.long().view(-1, 1)).squeeze(1)
        prior_logits = output.aux.get("prior_logits")
        if prior_logits is None:
            prior_logits = batch.policy_input.prior_logits
        if prior_logits is None:
            prior_top1 = selected.clone()
        else:
            prior_top1_index = prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min).argmax(dim=-1)
            prior_top1 = output.action_logits.gather(1, prior_top1_index.view(-1, 1)).squeeze(1)
    return selected.detach(), prior_top1.detach()


def _delta_stats(output: Any, batch: PPOBatch) -> dict[str, float]:
    mask = batch.policy_input.legal_action_mask.bool()
    neural_delta = output.aux.get("neural_delta")
    if neural_delta is None:
        return {"neural_delta_abs_mean": 0.0, "neural_delta_abs_max": 0.0, "top1_action_changed_rate": 0.0}
    legal_delta = neural_delta.masked_select(mask)
    prior_logits = output.aux.get("prior_logits")
    if prior_logits is None:
        prior_logits = batch.policy_input.prior_logits
    if prior_logits is None:
        top1_changed = 0.0
    else:
        min_current = torch.finfo(output.action_logits.dtype).min
        min_prior = torch.finfo(prior_logits.dtype).min
        current_top1 = output.action_logits.masked_fill(~mask, min_current).argmax(dim=-1)
        prior_top1 = prior_logits.masked_fill(~mask, min_prior).argmax(dim=-1)
        top1_changed = float((current_top1 != prior_top1).float().mean().detach().cpu())
    return {
        "neural_delta_abs_mean": 0.0 if legal_delta.numel() == 0 else float(legal_delta.abs().mean().detach().cpu()),
        "neural_delta_abs_max": 0.0 if legal_delta.numel() == 0 else float(legal_delta.abs().max().detach().cpu()),
        "top1_action_changed_rate": top1_changed,
    }


def _rule_agreement(output: Any, batch: PPOBatch) -> float | None:
    prior_logits = output.aux.get("prior_logits")
    if prior_logits is None:
        prior_logits = batch.policy_input.prior_logits
    if prior_logits is None:
        return None
    mask = batch.policy_input.legal_action_mask.bool()
    current = output.action_logits.masked_fill(~mask, torch.finfo(output.action_logits.dtype).min).argmax(dim=-1)
    prior = prior_logits.masked_fill(~mask, torch.finfo(prior_logits.dtype).min).argmax(dim=-1)
    return float((current == prior).float().mean().detach().cpu())


def _gradient_summary(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    summary: dict[str, float] = {}
    for row in rows:
        component = str(row["component"])
        summary[f"{component}_actor_grad_norm_total"] = float(row.get("actor_grad_norm_total", 0.0))
        summary[f"{component}_policy_mlp_final_grad_norm"] = float(
            row.get("policy_mlp.final_linear.weight_grad_norm", 0.0)
        )
    return summary


def _empty_gradient_row(component: str) -> dict[str, Any]:
    return {"component": component, "loss_value": None, **{f"{prefix}_grad_norm": 0.0 for prefix in MODULE_PREFIXES}}


def _episode_metadata(episodes: Sequence[RolloutEpisode]) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for index, episode in enumerate(episodes):
        keys = [episode.game_id, f"episode_{index}"]
        if episode.seed is not None:
            keys.append(str(episode.seed))
        payload = {
            "terminal_rewards": tuple(float(value) for value in episode.terminal_rewards),
            "final_ranks": tuple(int(value) for value in episode.final_ranks),
        }
        for key in keys:
            if key:
                metadata[str(key)] = payload
    return metadata


def _seat_tuple_value(values: Any, seat: int) -> float | int | None:
    try:
        return values[int(seat)]
    except Exception:
        return None


def _normalized_advantages(advantages: torch.Tensor) -> torch.Tensor:
    if advantages.numel() <= 1:
        return advantages.clone()
    return (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(1e-8)


def _safe_mean_tensor(tensor: torch.Tensor) -> float:
    return 0.0 if tensor.numel() == 0 else float(tensor.mean().detach().cpu())


def _safe_quantile(tensor: torch.Tensor, q: float) -> float:
    return 0.0 if tensor.numel() == 0 else float(torch.quantile(tensor, q).detach().cpu())


def _mean(values: Iterable[Any]) -> float:
    items = [float(value) for value in values]
    return 0.0 if not items else sum(items) / len(items)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> float:
    selected = values[mask]
    return 0.0 if selected.numel() == 0 else float(selected.mean().detach().cpu())


__all__ = [
    "PpoDiagnosticConfig",
    "batch_diagnostic_rows",
    "batch_diagnostic_summary",
    "classify_learning_signal_blocker",
    "gradient_norms",
    "loss_gradient_decomposition",
    "ppo_update_probe",
    "seed_registry_hash",
    "tensor_stats",
    "top1_margin_diagnostics",
    "top1_margin_rows",
]
