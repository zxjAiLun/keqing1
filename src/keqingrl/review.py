"""Review/export helpers for keqingrl rollout traces."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from keqingrl.actions import ACTION_FLAG_REACH, ACTION_FLAG_TSUMOGIRI, ActionSpec, ActionType
from keqingrl.distribution import MaskedCategorical
from keqingrl.policy import InteractivePolicy
from keqingrl.rollout import RolloutEpisode, RolloutStep, rollout_step_policy_input
from mahjong_env.action_space import IDX_TO_TILE_NAME


@dataclass(frozen=True)
class ReviewCandidate:
    action_index: int
    action_label: str
    action_spec: ActionSpec
    prob: float
    logit: float
    feature_values: tuple[float, ...]
    action_canonical_key: str
    action_type: str
    raw_rule_score: float | None = None
    prior_logit: float | None = None
    neural_delta: float | None = None
    is_chosen: bool = False


@dataclass(frozen=True)
class StepReview:
    actor: int
    step_id: int | None
    game_id: str | None
    policy_version: int
    policy_name: str | None
    chosen_action: ReviewCandidate
    top_k: tuple[ReviewCandidate, ...]
    legal_action_count: int
    value: float
    recorded_value: float
    rank_probs: tuple[float, float, float, float]
    entropy: float | None
    recorded_log_prob: float | None
    recomputed_log_prob: float | None
    reward: float
    done: bool
    is_autopilot: bool = False
    is_learner_controlled: bool = True
    rulebase_chosen: str | None = None
    policy_chosen: str | None = None
    rule_kl: float | None = None
    rule_context: tuple[float, ...] = ()
    style_context: tuple[float, ...] = ()


@dataclass(frozen=True)
class EpisodeReview:
    game_id: str | None
    seed: int | None
    terminal_rewards: tuple[float, float, float, float]
    final_ranks: tuple[int, int, int, int]
    scores: tuple[int, int, int, int]
    steps: tuple[StepReview, ...]


@dataclass(frozen=True)
class ReviewPolicyFieldSummary:
    learner_step_count: int
    autopilot_step_count: int
    entropy_count: int
    log_prob_count: int
    neural_delta_count: int
    mean_entropy: float | None
    mean_recorded_log_prob: float | None
    mean_chosen_neural_delta: float | None


def format_action_spec(spec: ActionSpec) -> str:
    action_name = spec.action_type.name.lower()
    parts = [action_name]

    if spec.tile is not None:
        parts.append(str(IDX_TO_TILE_NAME[int(spec.tile)]))
    if spec.consumed:
        consumed = ",".join(str(IDX_TO_TILE_NAME[int(tile)]) for tile in spec.consumed)
        parts.append(f"[{consumed}]")
    if spec.from_who is not None:
        parts.append(f"from={spec.from_who}")
    flags = int(spec.flags)
    if flags & ACTION_FLAG_TSUMOGIRI:
        parts.append("tsumogiri")
        flags &= ~ACTION_FLAG_TSUMOGIRI
    if spec.action_type == ActionType.REACH_DISCARD:
        flags &= ~ACTION_FLAG_REACH
    if flags:
        parts.append(f"flags={flags}")
    return ":".join(parts)


def review_rollout_step(
    policy: InteractivePolicy | None,
    step: RolloutStep,
    *,
    top_k: int = 5,
    device: torch.device | str | None = None,
    policy_resolver: Callable[[RolloutStep], InteractivePolicy] | None = None,
) -> StepReview:
    if step.legal_actions is None:
        raise ValueError("rollout review requires step.legal_actions to be present")
    if top_k <= 0:
        raise ValueError(f"top_k must be positive, got {top_k}")

    active_policy = _resolve_review_policy(
        step,
        default_policy=policy,
        policy_resolver=policy_resolver,
    )
    target_device = _policy_device(active_policy) if device is None else torch.device(device)
    policy_input = rollout_step_policy_input(step, device=target_device)
    with torch.no_grad():
        output = active_policy(policy_input)
        dist = MaskedCategorical(output.action_logits, policy_input.legal_action_mask)
        probs = dist.probs()[0].detach().cpu()
        logits = output.action_logits[0].detach().cpu()
        raw_rule_scores = _optional_row(output.aux.get("rule_scores"), policy_input.raw_rule_scores)
        prior_logits = _optional_row(output.aux.get("prior_logits"), policy_input.prior_logits)
        neural_delta = _optional_row(output.aux.get("neural_delta"), None)
        rank_probs = torch.softmax(output.rank_logits[0], dim=-1).detach().cpu()
        entropy = (output.entropy if output.entropy is not None else dist.entropy())[0].detach().cpu()
        recomputed_log_prob = dist.log_prob(
            torch.tensor([step.action_index], dtype=torch.long, device=target_device)
        )[0].detach().cpu()
        value = output.value[0].detach().cpu()

    legal_count = len(step.legal_actions)
    ranked_indices = torch.argsort(probs[:legal_count], descending=True)
    top_indices = ranked_indices[: min(top_k, legal_count)].tolist()

    chosen_candidate = _candidate_from_step(
        step,
        probs,
        logits,
        step.action_index,
        raw_rule_scores=raw_rule_scores,
        prior_logits=prior_logits,
        neural_delta=neural_delta,
        is_chosen=True,
    )
    top_candidates = tuple(
        _candidate_from_step(
            step,
            probs,
            logits,
            action_index,
            raw_rule_scores=raw_rule_scores,
            prior_logits=prior_logits,
            neural_delta=neural_delta,
            is_chosen=(action_index == step.action_index),
        )
        for action_index in top_indices
    )

    return StepReview(
        actor=step.actor,
        step_id=step.step_id,
        game_id=step.game_id,
        policy_version=step.policy_version,
        policy_name=step.policy_name,
        chosen_action=chosen_candidate,
        top_k=top_candidates,
        legal_action_count=legal_count,
        value=float(value),
        recorded_value=float(step.value),
        rank_probs=tuple(float(value) for value in rank_probs),  # type: ignore[arg-type]
        entropy=None if step.is_autopilot else float(entropy),
        recorded_log_prob=None if step.is_autopilot else float(step.log_prob),
        recomputed_log_prob=None if step.is_autopilot else float(recomputed_log_prob),
        reward=float(step.reward),
        done=bool(step.done),
        is_autopilot=bool(step.is_autopilot),
        is_learner_controlled=bool(step.is_learner_controlled),
        rulebase_chosen=step.rulebase_chosen,
        policy_chosen=step.policy_chosen,
        rule_context=tuple(float(value) for value in step.rule_context.flatten().tolist()),
        style_context=()
        if step.style_context is None
        else tuple(float(value) for value in step.style_context.flatten().tolist()),
    )


def review_rollout_episode(
    policy: InteractivePolicy | None,
    episode: RolloutEpisode,
    *,
    top_k: int = 5,
    device: torch.device | str | None = None,
    policy_resolver: Callable[[RolloutStep], InteractivePolicy] | None = None,
) -> EpisodeReview:
    return EpisodeReview(
        game_id=episode.game_id,
        seed=episode.seed,
        terminal_rewards=episode.terminal_rewards,
        final_ranks=episode.final_ranks,
        scores=episode.scores,
        steps=tuple(
            review_rollout_step(
                policy,
                step,
                top_k=top_k,
                device=device,
                policy_resolver=policy_resolver,
            )
            for step in episode.steps
        ),
    )


def export_episode_review_jsonl(
    episode_review: EpisodeReview,
    output_path: str | Path,
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for step_review in episode_review.steps:
            handle.write(json.dumps(_step_review_dict(step_review), ensure_ascii=False) + "\n")
    return path


def summarize_review_policy_fields(episode_review: EpisodeReview) -> ReviewPolicyFieldSummary:
    learner_steps = [step for step in episode_review.steps if not step.is_autopilot]
    autopilot_steps = [step for step in episode_review.steps if step.is_autopilot]
    entropies = [step.entropy for step in learner_steps if step.entropy is not None]
    log_probs = [step.recorded_log_prob for step in learner_steps if step.recorded_log_prob is not None]
    deltas = [
        step.chosen_action.neural_delta
        for step in learner_steps
        if step.chosen_action.neural_delta is not None
    ]
    return ReviewPolicyFieldSummary(
        learner_step_count=len(learner_steps),
        autopilot_step_count=len(autopilot_steps),
        entropy_count=len(entropies),
        log_prob_count=len(log_probs),
        neural_delta_count=len(deltas),
        mean_entropy=_mean_optional(entropies),
        mean_recorded_log_prob=_mean_optional(log_probs),
        mean_chosen_neural_delta=_mean_optional(deltas),
    )


def _candidate_from_step(
    step: RolloutStep,
    probs: torch.Tensor,
    logits: torch.Tensor,
    action_index: int,
    *,
    raw_rule_scores: torch.Tensor | None,
    prior_logits: torch.Tensor | None,
    neural_delta: torch.Tensor | None,
    is_chosen: bool,
) -> ReviewCandidate:
    if step.legal_actions is None:
        raise ValueError("review candidate requires step.legal_actions")
    return ReviewCandidate(
        action_index=action_index,
        action_label=format_action_spec(step.legal_actions[action_index]),
        action_spec=step.legal_actions[action_index],
        prob=float(probs[action_index]),
        logit=float(logits[action_index]),
        feature_values=tuple(float(value) for value in step.legal_action_features[action_index].tolist()),
        action_canonical_key=step.legal_actions[action_index].canonical_key,
        action_type=step.legal_actions[action_index].action_type.name,
        raw_rule_score=None if raw_rule_scores is None else float(raw_rule_scores[action_index]),
        prior_logit=None if prior_logits is None else float(prior_logits[action_index]),
        neural_delta=None if step.is_autopilot or neural_delta is None else float(neural_delta[action_index]),
        is_chosen=is_chosen,
    )


def _step_review_dict(step_review: StepReview) -> dict[str, object]:
    def candidate_dict(candidate: ReviewCandidate) -> dict[str, object]:
        return {
            "action_index": candidate.action_index,
            "action": candidate.action_label,
            "prob": candidate.prob,
            "logit": candidate.logit,
            "final_logit": candidate.logit,
            "action_type": candidate.action_type,
            "action_canonical_key": candidate.action_canonical_key,
            "raw_rule_score": candidate.raw_rule_score,
            "prior_logit": candidate.prior_logit,
            "neural_delta": candidate.neural_delta,
            "feature_values": list(candidate.feature_values),
            "is_chosen": candidate.is_chosen,
        }

    return {
        "actor": step_review.actor,
        "step_id": step_review.step_id,
        "game_id": step_review.game_id,
        "policy_version": step_review.policy_version,
        "policy_name": step_review.policy_name,
        "chosen_action": candidate_dict(step_review.chosen_action),
        "top_k": [candidate_dict(candidate) for candidate in step_review.top_k],
        "legal_action_count": step_review.legal_action_count,
        "value": step_review.value,
        "recorded_value": step_review.recorded_value,
        "rank_probs": list(step_review.rank_probs),
        "entropy": step_review.entropy,
        "recorded_log_prob": step_review.recorded_log_prob,
        "recomputed_log_prob": step_review.recomputed_log_prob,
        "reward": step_review.reward,
        "done": step_review.done,
        "is_autopilot": step_review.is_autopilot,
        "is_learner_controlled": step_review.is_learner_controlled,
        "rulebase_chosen": step_review.rulebase_chosen,
        "policy_chosen": step_review.policy_chosen,
        "rule_kl": step_review.rule_kl,
        "rule_context": list(step_review.rule_context),
        "style_context": list(step_review.style_context),
    }


def _optional_row(
    primary: torch.Tensor | None,
    fallback: torch.Tensor | None,
) -> torch.Tensor | None:
    tensor = primary if primary is not None else fallback
    if tensor is None:
        return None
    return tensor[0].detach().cpu()


def _mean_optional(values: Sequence[float | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def _policy_device(policy: InteractivePolicy) -> torch.device:
    parameter = next(policy.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(policy.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


def build_policy_resolver(
    *,
    default_policy: InteractivePolicy | None = None,
    policies_by_name: Mapping[str, InteractivePolicy] | None = None,
    policies_by_version: Mapping[int, InteractivePolicy] | None = None,
) -> Callable[[RolloutStep], InteractivePolicy]:
    name_map = {} if policies_by_name is None else dict(policies_by_name)
    version_map = {} if policies_by_version is None else dict(policies_by_version)

    def _resolver(step: RolloutStep) -> InteractivePolicy:
        if step.policy_name is not None and step.policy_name in name_map:
            return name_map[step.policy_name]
        if step.policy_version in version_map:
            return version_map[step.policy_version]
        if default_policy is not None:
            return default_policy
        raise KeyError(
            "no review policy registered for rollout step "
            f"(policy_name={step.policy_name!r}, policy_version={step.policy_version})"
        )

    return _resolver


def _resolve_review_policy(
    step: RolloutStep,
    *,
    default_policy: InteractivePolicy | None,
    policy_resolver: Callable[[RolloutStep], InteractivePolicy] | None,
) -> InteractivePolicy:
    if policy_resolver is not None:
        return policy_resolver(step)
    if default_policy is not None:
        return default_policy
    raise ValueError("review requires either a policy or a policy_resolver")


__all__ = [
    "EpisodeReview",
    "ReviewCandidate",
    "ReviewPolicyFieldSummary",
    "StepReview",
    "build_policy_resolver",
    "export_episode_review_jsonl",
    "format_action_spec",
    "review_rollout_episode",
    "review_rollout_step",
    "summarize_review_policy_fields",
]
