"""Rule-prior scoring utilities for KeqingRL-Lite."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

import keqing_core
from keqingrl.actions import ActionSpec, ActionType


@dataclass(frozen=True)
class RuleScoreConfig:
    clip_min: float = -10.0
    clip_max: float = 0.0
    prior_temperature: float = 1.0
    rule_score_scale: float = 1.0
    prior_kl_eps: float = 1e-4


@dataclass(frozen=True)
class RuleScoreEntry:
    action_index: int
    raw_score: float
    priority: int
    tie_break_rank: int
    components: dict[str, float | int | str | bool]


@dataclass(frozen=True)
class RuleScoreResult:
    raw_rule_scores: torch.Tensor
    prior_logits: torch.Tensor
    entries: tuple[RuleScoreEntry, ...]
    config: RuleScoreConfig


DEFAULT_RULE_SCORE_CONFIG = RuleScoreConfig()


def prior_logits_from_raw_scores(
    raw_rule_scores: torch.Tensor,
    *,
    mask: torch.Tensor | None = None,
    config: RuleScoreConfig = DEFAULT_RULE_SCORE_CONFIG,
) -> torch.Tensor:
    if config.prior_temperature <= 0.0:
        raise ValueError(f"prior_temperature must be positive, got {config.prior_temperature}")
    if config.clip_max != 0.0:
        raise ValueError("rule prior clip_max must stay at 0.0 to preserve best-action identity")
    if not torch.isfinite(raw_rule_scores).all():
        raise ValueError("raw_rule_scores must be finite")

    scores = raw_rule_scores.float()
    if mask is None:
        legal_scores = scores
    else:
        if mask.shape != scores.shape:
            raise ValueError("rule score mask shape must match raw_rule_scores")
        mask_bool = mask.bool()
        if not torch.all(mask_bool.any(dim=-1)):
            raise ValueError("each rule score row must contain at least one legal action")
        legal_scores = scores.masked_fill(~mask_bool, torch.finfo(scores.dtype).min)

    centered = scores - legal_scores.max(dim=-1, keepdim=True).values
    prior_logits = centered.clamp(min=float(config.clip_min), max=0.0)
    prior_logits = prior_logits / float(config.prior_temperature)
    if mask is not None:
        prior_logits = prior_logits.masked_fill(~mask.bool(), torch.finfo(prior_logits.dtype).min)
    if not torch.isfinite(prior_logits[mask.bool()] if mask is not None else prior_logits).all():
        raise ValueError("prior_logits must be finite on legal actions")
    return prior_logits


def smoothed_prior_probs(
    prior_logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = DEFAULT_RULE_SCORE_CONFIG.prior_kl_eps,
) -> torch.Tensor:
    if eps < 0.0:
        raise ValueError(f"prior smoothing eps must be non-negative, got {eps}")
    mask_bool = mask.bool()
    if prior_logits.shape != mask_bool.shape:
        raise ValueError("prior_logits shape must match mask shape")
    if not torch.all(mask_bool.any(dim=-1)):
        raise ValueError("each prior row must contain at least one legal action")

    masked = prior_logits.masked_fill(~mask_bool, torch.finfo(prior_logits.dtype).min)
    probs = torch.softmax(masked, dim=-1).masked_fill(~mask_bool, 0.0)
    legal_count = mask_bool.sum(dim=-1, keepdim=True).clamp_min(1).to(probs.dtype)
    uniform = mask_bool.to(probs.dtype) / legal_count
    smoothed = (1.0 - float(eps)) * probs + float(eps) * uniform
    return smoothed / smoothed.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def score_legal_actions(
    snapshot: Mapping[str, Any],
    actor: int,
    legal_actions: Sequence[ActionSpec | Mapping[str, Any]],
    *,
    config: RuleScoreConfig = DEFAULT_RULE_SCORE_CONFIG,
) -> RuleScoreResult:
    mjai_actions = [_to_rulebase_action(action, actor=actor) for action in legal_actions]
    try:
        raw_entries = keqing_core.score_rulebase_actions(dict(snapshot), int(actor), mjai_actions)
    except RuntimeError as exc:
        if "rulebase scoring capability" not in str(exc):
            raise
        raw_entries = _fallback_score_entries(dict(snapshot), int(actor), mjai_actions)
    if len(raw_entries) != len(legal_actions):
        raise RuntimeError("rulebase scorer returned a different action count")

    entries = tuple(
        RuleScoreEntry(
            action_index=int(entry["action_index"]),
            raw_score=float(entry["raw_score"]),
            priority=int(entry.get("priority", 0)),
            tie_break_rank=int(entry.get("tie_break_rank", 0)),
            components=dict(entry.get("components", {})),
        )
        for entry in raw_entries
    )
    raw_rule_scores = torch.tensor([entry.raw_score for entry in entries], dtype=torch.float32).unsqueeze(0)
    prior_logits = prior_logits_from_raw_scores(
        raw_rule_scores,
        mask=torch.ones_like(raw_rule_scores, dtype=torch.bool),
        config=config,
    )
    return RuleScoreResult(
        raw_rule_scores=raw_rule_scores.squeeze(0),
        prior_logits=prior_logits.squeeze(0),
        entries=entries,
        config=config,
    )


def _to_rulebase_action(action: ActionSpec | Mapping[str, Any], *, actor: int) -> dict[str, object]:
    if isinstance(action, ActionSpec):
        if action.action_type == ActionType.REACH_DISCARD:
            return {"type": "reach", "actor": int(actor)}
        return action.to_mjai_action(actor=int(actor))
    return dict(action)


def _fallback_score_entries(
    snapshot: dict[str, Any],
    actor: int,
    mjai_actions: list[dict[str, object]],
) -> list[dict[str, object]]:
    del snapshot, actor
    chosen_index = _fallback_chosen_index(mjai_actions)
    entries: list[dict[str, object]] = []
    for index, action in enumerate(mjai_actions):
        action_type = str(action.get("type", "unknown"))
        selected = index == chosen_index
        entries.append(
            {
                "action_index": index,
                "raw_score": 0.0 if selected else _fallback_penalty(action_type) - index * 0.001,
                "priority": 1000 if selected else 0,
                "tie_break_rank": -index,
                "components": {
                    "selected_by_rulebase": selected,
                    "action_type": action_type,
                    "legal_order": index,
                    "fallback": True,
                },
            }
        )
    return entries


def _fallback_chosen_index(mjai_actions: Sequence[Mapping[str, object]]) -> int:
    if not mjai_actions:
        return 0
    for preferred_type in ("hora", "reach", "dahai", "none"):
        for index, action in enumerate(mjai_actions):
            if str(action.get("type", "unknown")) == preferred_type:
                return index
    return 0


def _fallback_penalty(action_type: str) -> float:
    return {
        "hora": -1.0,
        "reach": -2.0,
        "dahai": -3.0,
        "ankan": -4.0,
        "kakan": -4.0,
        "daiminkan": -4.0,
        "pon": -5.0,
        "chi": -5.0,
        "none": -6.0,
    }.get(action_type, -8.0)


__all__ = [
    "DEFAULT_RULE_SCORE_CONFIG",
    "RuleScoreConfig",
    "RuleScoreEntry",
    "RuleScoreResult",
    "prior_logits_from_raw_scores",
    "score_legal_actions",
    "smoothed_prior_probs",
]
