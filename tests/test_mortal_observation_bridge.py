from __future__ import annotations

from pathlib import Path

import pytest
import torch

from keqingrl.env import DiscardOnlyMahjongEnv
from keqingrl.mortal_observation import MortalObservationBridge, sanitize_event_for_mortal
from keqingrl.mortal_runtime import MortalTeacherRuntimeOutput
from keqingrl.mortal_teacher import (
    MORTAL_ACTION_MASK_EXTRA_KEY,
    MORTAL_ACTION_SPACE,
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
