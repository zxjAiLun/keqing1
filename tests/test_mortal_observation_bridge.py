from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from keqingrl.env import DiscardOnlyMahjongEnv
from keqingrl.mortal_observation import MortalObservationBridge, sanitize_event_for_mortal
from keqingrl.mortal_runtime import MortalTeacherRuntimeOutput
from keqingrl.mortal_teacher import (
    MORTAL_ACTION_MASK_EXTRA_KEY,
    MORTAL_ACTION_SPACE,
    MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY,
    MORTAL_ENCODED_OBS_EXTRA_KEY,
    MORTAL_Q_VALUES_EXTRA_KEY,
    assert_mortal_discard_mask_parity,
)
from keqingrl.opponent_pool import SeatPolicyAssignment
from keqingrl.policy import RulePriorPolicy
from keqingrl.selfplay import collect_policy_episode
from keqingrl.buffer import build_ppo_batch


def _requires_local_mortal() -> None:
    if not (Path("third_party/Mortal/mortal/libriichi.so").exists()):
        pytest.skip("local Mortal libriichi.so is not built")


class _EchoMortalRuntime:
    def evaluate(self, obs: torch.Tensor, masks: torch.Tensor) -> MortalTeacherRuntimeOutput:
        batch_size = int(obs.shape[0])
        q_values = torch.arange(MORTAL_ACTION_SPACE, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1)
        return MortalTeacherRuntimeOutput(
            q_values=q_values,
            action_mask=masks.bool(),
            actions=q_values.masked_fill(~masks.bool(), -torch.inf).argmax(dim=-1),
            is_greedy=torch.ones((batch_size,), dtype=torch.bool),
        )


class _CountingPlayerState:
    updates: list[tuple[int, str]] = []

    def __init__(self, actor: int) -> None:
        self.actor = int(actor)
        self.events: list[dict[str, object]] = []

    def update(self, event_json: str) -> None:
        event = json.loads(event_json)
        self.events.append(event)
        self.updates.append((self.actor, str(event.get("type"))))

    def encode_obs(self, version: int, at_kan_select: bool):
        _ = version, at_kan_select
        mask = torch.zeros((MORTAL_ACTION_SPACE,), dtype=torch.bool)
        mask[0] = True
        return torch.zeros((1012, 34), dtype=torch.float32), mask


def test_sanitize_event_for_mortal_skips_keqing_kakan_accepted_extension() -> None:
    assert sanitize_event_for_mortal({"type": "kakan_accepted", "actor": 0}) is None
    assert sanitize_event_for_mortal({"type": "none", "actor": 2}) is None
    assert sanitize_event_for_mortal({"type": "none"}) == {"type": "none"}
    assert sanitize_event_for_mortal({"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False}) == {
        "type": "dahai",
        "actor": 0,
        "pai": "1m",
        "tsumogiri": False,
    }


def test_mortal_observation_bridge_incrementally_replays_only_new_events() -> None:
    _CountingPlayerState.updates.clear()
    bridge = MortalObservationBridge()
    bridge._player_state_cls = _CountingPlayerState
    events = (
        {"type": "start_game", "names": ["a", "b", "c", "d"]},
        {"type": "start_kyoku", "bakaze": "E"},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
    )

    first = bridge.encode_from_events(events[:2], actor=0)
    second = bridge.encode_from_events(events, actor=0)

    assert first.replayed_event_count == 2
    assert second.replayed_event_count == 3
    assert _CountingPlayerState.updates == [
        (0, "start_game"),
        (0, "start_kyoku"),
        (0, "tsumo"),
    ]


def test_mortal_observation_bridge_resets_incremental_cache_on_event_rewind() -> None:
    _CountingPlayerState.updates.clear()
    bridge = MortalObservationBridge()
    bridge._player_state_cls = _CountingPlayerState
    events = (
        {"type": "start_game", "names": ["a", "b", "c", "d"]},
        {"type": "start_kyoku", "bakaze": "E"},
        {"type": "tsumo", "actor": 0, "pai": "1m"},
    )

    bridge.encode_from_events(events, actor=0)
    rewound = bridge.encode_from_events(events[:2], actor=0)

    assert rewound.replayed_event_count == 2
    assert _CountingPlayerState.updates == [
        (0, "start_game"),
        (0, "start_kyoku"),
        (0, "tsumo"),
        (0, "start_game"),
        (0, "start_kyoku"),
    ]


def test_keqingrl_reset_clears_mortal_observation_bridge_cache() -> None:
    class _Bridge:
        def __init__(self) -> None:
            self.reset_count = 0

        def reset_cache(self) -> None:
            self.reset_count += 1

    bridge = _Bridge()
    env = DiscardOnlyMahjongEnv(max_kyokus=1, mortal_observation_bridge=bridge)

    env.reset(seed=1)
    env.reset(seed=2)

    assert bridge.reset_count == 2


def test_mortal_observation_bridge_replays_keqingrl_current_turn_and_matches_discard_mask() -> None:
    _requires_local_mortal()
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    env.reset(seed=1)
    actor = env.current_actor()
    assert actor is not None

    bridge = MortalObservationBridge()
    encoded = bridge.encode_from_events(env._require_room().events, actor)

    assert encoded.obs.shape == (1012, 34)
    assert encoded.action_mask.shape == (MORTAL_ACTION_SPACE,)
    assert_mortal_discard_mask_parity(encoded.action_mask, env.legal_actions(actor))


def test_mortal_mapping_debug_context_includes_action_and_response_fields() -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    env.reset(seed=1)
    actor = env.current_actor()
    assert actor is not None

    context = env._mortal_mapping_debug_context(env._require_turn(actor))

    assert "legal_action_types" in context
    assert "legal_action_payloads" in context
    assert "mortal_events_tail" in context
    assert "control_action_types" in context
    assert "response_window" in context


def test_keqingrl_observe_can_attach_mortal_teacher_extras_from_events() -> None:
    _requires_local_mortal()
    env = DiscardOnlyMahjongEnv(
        max_kyokus=1,
        mortal_teacher_runtime=_EchoMortalRuntime(),
        mortal_observation_bridge=MortalObservationBridge(),
    )
    env.reset(seed=1)
    actor = env.current_actor()
    assert actor is not None

    policy_input = env.observe(actor)

    assert set(policy_input.obs.extras) == {MORTAL_Q_VALUES_EXTRA_KEY, MORTAL_ACTION_MASK_EXTRA_KEY}
    assert policy_input.obs.extras[MORTAL_Q_VALUES_EXTRA_KEY].shape == (1, MORTAL_ACTION_SPACE)
    assert policy_input.obs.extras[MORTAL_ACTION_MASK_EXTRA_KEY].shape == (1, MORTAL_ACTION_SPACE)
    assert_mortal_discard_mask_parity(policy_input.obs.extras[MORTAL_ACTION_MASK_EXTRA_KEY][0], env.legal_actions(actor))


def test_keqingrl_observe_can_defer_mortal_teacher_runtime_to_batch_eval() -> None:
    _requires_local_mortal()
    env = DiscardOnlyMahjongEnv(
        max_kyokus=1,
        mortal_teacher_runtime=None,
        mortal_observation_bridge=MortalObservationBridge(),
        mortal_teacher_defer_runtime=True,
    )
    env.reset(seed=1)
    actor = env.current_actor()
    assert actor is not None

    policy_input = env.observe(actor)

    assert set(policy_input.obs.extras) == {
        MORTAL_ENCODED_OBS_EXTRA_KEY,
        MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY,
    }
    assert policy_input.obs.extras[MORTAL_ENCODED_OBS_EXTRA_KEY].shape == (1, 1012, 34)
    assert policy_input.obs.extras[MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY].shape == (1, MORTAL_ACTION_SPACE)
    assert_mortal_discard_mask_parity(
        policy_input.obs.extras[MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY][0],
        env.legal_actions(actor),
    )


def test_keqingrl_observe_can_skip_mortal_teacher_extras_for_untrained_opponents() -> None:
    class _RaisingBridge:
        def encode_from_events(self, events, actor):
            _ = events, actor
            raise AssertionError("Mortal bridge should not run for skipped extras")

    env = DiscardOnlyMahjongEnv(
        max_kyokus=1,
        mortal_observation_bridge=_RaisingBridge(),
        mortal_teacher_defer_runtime=True,
    )
    env.reset(seed=1)
    actor = env.current_actor()
    assert actor is not None

    policy_input = env.observe(actor, include_mortal_teacher_extras=False)

    assert policy_input.obs.extras == {}


def test_selfplay_can_collect_mortal_teacher_events_without_observation_extras() -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    policy = RulePriorPolicy()
    seat_policies = tuple(
        SeatPolicyAssignment(policy=policy, greedy=True, name="learner", is_learner=True)
        for _ in range(4)
    )

    episode = collect_policy_episode(
        env,
        seat_policies,
        seed=1,
        max_steps=400,
        include_mortal_teacher_extras=False,
        collect_mortal_teacher_events=True,
    )
    learner_steps = [
        step
        for step in episode.steps
        if step.is_learner_controlled and not step.is_autopilot
    ]

    assert learner_steps
    assert all(step.obs.extras == {} for step in learner_steps)
    assert all(step.mortal_teacher_events for step in learner_steps)


def test_mortal_teacher_extras_survive_selfplay_and_ppo_batch_collation() -> None:
    _requires_local_mortal()
    env = DiscardOnlyMahjongEnv(
        max_kyokus=1,
        mortal_teacher_runtime=_EchoMortalRuntime(),
        mortal_observation_bridge=MortalObservationBridge(),
    )
    policy = RulePriorPolicy()
    seat_policies = tuple(
        SeatPolicyAssignment(policy=policy, greedy=True, name="learner", is_learner=True)
        for _ in range(4)
    )

    episode = collect_policy_episode(env, seat_policies, seed=1, max_steps=400)
    learner_steps = [
        step
        for step in episode.steps
        if step.is_learner_controlled and not step.is_autopilot
    ]
    assert learner_steps

    batch = build_ppo_batch(
        learner_steps,
        advantages=[0.0] * len(learner_steps),
        returns=[0.0] * len(learner_steps),
    )

    assert set(batch.policy_input.obs.extras) == {
        MORTAL_Q_VALUES_EXTRA_KEY,
        MORTAL_ACTION_MASK_EXTRA_KEY,
    }
    assert batch.policy_input.obs.extras[MORTAL_Q_VALUES_EXTRA_KEY].shape == (
        len(learner_steps),
        MORTAL_ACTION_SPACE,
    )
    assert batch.policy_input.obs.extras[MORTAL_ACTION_MASK_EXTRA_KEY].shape == (
        len(learner_steps),
        MORTAL_ACTION_SPACE,
    )
