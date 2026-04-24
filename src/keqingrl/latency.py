"""Latency smoke helpers for KeqingRL-Lite."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import torch

from keqingrl.env import DiscardOnlyMahjongEnv
from keqingrl.policy import InteractivePolicy
from keqingrl.contracts import ObsTensorBatch, PolicyInput


@dataclass(frozen=True)
class LatencyReport:
    decision_count: int
    game_count: int
    elapsed_seconds: float
    rust_rule_score_call_latency_ms: float
    python_policy_forward_latency_ms: float
    rust_env_step_latency_ms: float
    avg_decision_latency_ms: float
    p95_decision_latency_ms: float
    decisions_per_sec: float
    games_per_sec: float


def measure_latency_smoke(
    env: DiscardOnlyMahjongEnv,
    policy: InteractivePolicy,
    *,
    num_games: int = 1,
    seed: int | None = None,
    seed_stride: int = 1,
    max_steps: int = 512,
    device: torch.device | str | None = None,
) -> LatencyReport:
    if num_games <= 0:
        raise ValueError(f"num_games must be positive, got {num_games}")

    target_device = torch.device(device) if device is not None else _policy_device(policy)
    decision_count = 0
    observe_seconds = 0.0
    policy_seconds = 0.0
    step_seconds = 0.0
    decision_latencies_ms: list[float] = []
    total_start = perf_counter()

    for game_idx in range(num_games):
        game_seed = None if seed is None else seed + game_idx * seed_stride
        env.reset(seed=game_seed)

        for _step_idx in range(max_steps):
            actor = env.current_actor()
            if actor is None:
                if env.is_done():
                    break
                raise RuntimeError("environment has no current actor before episode termination")

            decision_start = perf_counter()
            start = perf_counter()
            policy_input_cpu = env.observe(actor)
            observe_seconds += perf_counter() - start

            policy_input = _policy_input_to_device(policy_input_cpu, target_device)
            start = perf_counter()
            with torch.no_grad():
                sample = policy.sample_action(policy_input)
            policy_seconds += perf_counter() - start

            action_index = int(sample.action_index[0].detach().cpu())
            action_spec = policy_input_cpu.legal_actions[0][action_index]
            start = perf_counter()
            result = env.step(actor, action_spec)
            step_seconds += perf_counter() - start

            decision_count += 1
            decision_latencies_ms.append((perf_counter() - decision_start) * 1000.0)
            if result.done:
                break
        else:
            raise RuntimeError(f"latency smoke exceeded max_steps={max_steps}")

    elapsed_seconds = perf_counter() - total_start
    divisor = max(1, decision_count)
    return LatencyReport(
        decision_count=decision_count,
        game_count=num_games,
        elapsed_seconds=elapsed_seconds,
        rust_rule_score_call_latency_ms=observe_seconds * 1000.0 / divisor,
        python_policy_forward_latency_ms=policy_seconds * 1000.0 / divisor,
        rust_env_step_latency_ms=step_seconds * 1000.0 / divisor,
        avg_decision_latency_ms=sum(decision_latencies_ms) / divisor,
        p95_decision_latency_ms=_percentile(decision_latencies_ms, 0.95),
        decisions_per_sec=decision_count / max(elapsed_seconds, 1e-12),
        games_per_sec=num_games / max(elapsed_seconds, 1e-12),
    )


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * float(q)))))
    return float(ordered[index])


def _policy_input_to_device(policy_input: PolicyInput, device: torch.device) -> PolicyInput:
    return PolicyInput(
        obs=ObsTensorBatch(
            tile_obs=policy_input.obs.tile_obs.to(device),
            scalar_obs=policy_input.obs.scalar_obs.to(device),
            history_obs=None
            if policy_input.obs.history_obs is None
            else policy_input.obs.history_obs.to(device),
            extras={key: value.to(device) for key, value in policy_input.obs.extras.items()},
        ),
        legal_action_ids=policy_input.legal_action_ids.to(device),
        legal_action_features=policy_input.legal_action_features.to(device),
        legal_action_mask=policy_input.legal_action_mask.to(device),
        rule_context=policy_input.rule_context.to(device),
        raw_rule_scores=None
        if policy_input.raw_rule_scores is None
        else policy_input.raw_rule_scores.to(device),
        prior_logits=None
        if policy_input.prior_logits is None
        else policy_input.prior_logits.to(device),
        style_context=None
        if policy_input.style_context is None
        else policy_input.style_context.to(device),
        legal_actions=policy_input.legal_actions,
        recurrent_state=policy_input.recurrent_state,
        metadata=policy_input.metadata,
    )


def _policy_device(policy: InteractivePolicy) -> torch.device:
    parameter = next(policy.parameters(), None)
    if parameter is not None:
        return parameter.device
    buffer = next(policy.buffers(), None)
    if buffer is not None:
        return buffer.device
    return torch.device("cpu")


__all__ = ["LatencyReport", "measure_latency_smoke"]
