from __future__ import annotations

from dataclasses import replace
import json

import pytest
import torch

from keqingrl import (
    ACTION_FLAG_TSUMOGIRI,
    ActionSpec,
    ActionType,
    build_policy_resolver,
    DiscardOnlyMahjongEnv,
    InteractivePolicy,
    RolloutEpisode,
    RolloutStep,
    RandomInteractivePolicy,
    ACTION_FEATURE_CONTRACT_VERSION,
    ENV_CONTRACT_VERSION,
    NATIVE_ACTION_IDENTITY_VERSION,
    NATIVE_LEGAL_ENUMERATION_VERSION,
    NATIVE_SCHEMA_NAME,
    NATIVE_SCHEMA_VERSION,
    NATIVE_TERMINAL_RESOLVER_VERSION,
    OBSERVATION_CONTRACT_VERSION,
    REWARD_SPEC_VERSION,
    RULE_SCORE_VERSION,
    STYLE_CONTEXT_VERSION,
    bind_reach_discard,
    export_episode_review_jsonl,
    format_action_spec,
    review_rollout_episode,
    review_rollout_step,
    summarize_review_policy_fields,
)
from keqingrl.contracts import ObsTensorBatch, PolicyOutput


class _FixedReviewPolicy(InteractivePolicy):
    def __init__(self, logits: list[float], neural_delta: list[float] | None = None) -> None:
        super().__init__()
        self._logits = torch.tensor([logits], dtype=torch.float32)
        self._neural_delta = None if neural_delta is None else torch.tensor([neural_delta], dtype=torch.float32)

    def forward(self, policy_input) -> PolicyOutput:
        batch_size = int(policy_input.legal_action_mask.shape[0])
        logits = self._logits.expand(batch_size, -1).clone()
        logits = logits.masked_fill(~policy_input.legal_action_mask, torch.finfo(logits.dtype).min)
        aux = {}
        if self._neural_delta is not None:
            aux["neural_delta"] = self._neural_delta.expand(batch_size, -1).clone()
        return PolicyOutput(
            action_logits=logits,
            value=torch.full((batch_size,), 0.25, dtype=torch.float32),
            rank_logits=torch.tensor([[2.0, 1.0, 0.0, -1.0]], dtype=torch.float32).expand(batch_size, -1).clone(),
            aux=aux,
        )


def _synthetic_review_step() -> RolloutStep:
    legal_actions = (
        ActionSpec(ActionType.DISCARD, tile=0),
        bind_reach_discard(ActionSpec(ActionType.DISCARD, tile=1, flags=ACTION_FLAG_TSUMOGIRI)),
        ActionSpec(ActionType.TSUMO, tile=2),
        ActionSpec(ActionType.ANKAN, tile=27, consumed=(27, 27, 27, 27)),
        ActionSpec(ActionType.KAKAN, tile=13, consumed=(13, 13, 13, 13)),
    )
    return RolloutStep(
        obs=ObsTensorBatch(
            tile_obs=torch.zeros((4, 34), dtype=torch.float32),
            scalar_obs=torch.zeros((6,), dtype=torch.float32),
        ),
        legal_action_ids=torch.tensor([10, 20, 30, 40, 50], dtype=torch.long),
        legal_action_features=torch.tensor(
            [
                [0.0] * 8,
                [0.1] * 8,
                [0.2] * 8,
                [0.3] * 8,
                [0.4] * 8,
            ],
            dtype=torch.float32,
        ),
        legal_action_mask=torch.tensor([True, True, True, True, True], dtype=torch.bool),
        action_index=3,
        action_spec=legal_actions[3],
        log_prob=-0.5,
        value=0.1,
        entropy=0.7,
        reward=1.0,
        done=True,
        actor=0,
        policy_version=3,
        rule_context=torch.zeros((6,), dtype=torch.float32),
        legal_actions=legal_actions,
        game_id="review-synthetic",
        step_id=5,
        observation_contract_version=OBSERVATION_CONTRACT_VERSION,
        action_feature_contract_version=ACTION_FEATURE_CONTRACT_VERSION,
        env_contract_version=ENV_CONTRACT_VERSION,
        native_schema_name=NATIVE_SCHEMA_NAME,
        native_schema_version=NATIVE_SCHEMA_VERSION,
        native_action_identity_version=NATIVE_ACTION_IDENTITY_VERSION,
        native_legal_enumeration_version=NATIVE_LEGAL_ENUMERATION_VERSION,
        native_terminal_resolver_version=NATIVE_TERMINAL_RESOLVER_VERSION,
        rule_score_version=RULE_SCORE_VERSION,
        reward_spec_version=REWARD_SPEC_VERSION,
        style_context_version=STYLE_CONTEXT_VERSION,
    )


def _synthetic_response_review_step() -> RolloutStep:
    legal_actions = (
        ActionSpec(ActionType.RON, tile=3, from_who=0),
        ActionSpec(ActionType.PON, tile=3, consumed=(3, 3), from_who=0),
        ActionSpec(ActionType.DAIMINKAN, tile=3, consumed=(3, 3, 3), from_who=0),
        ActionSpec(ActionType.CHI, tile=3, consumed=(1, 2), from_who=0),
        ActionSpec(ActionType.PASS),
    )
    return RolloutStep(
        obs=ObsTensorBatch(
            tile_obs=torch.zeros((4, 34), dtype=torch.float32),
            scalar_obs=torch.zeros((6,), dtype=torch.float32),
        ),
        legal_action_ids=torch.tensor([60, 70, 80, 90, 100], dtype=torch.long),
        legal_action_features=torch.tensor(
            [
                [0.5] * 8,
                [0.6] * 8,
                [0.7] * 8,
                [0.8] * 8,
                [0.9] * 8,
            ],
            dtype=torch.float32,
        ),
        legal_action_mask=torch.tensor([True, True, True, True, True], dtype=torch.bool),
        action_index=3,
        action_spec=legal_actions[3],
        log_prob=-0.2,
        value=0.15,
        entropy=0.6,
        reward=0.0,
        done=False,
        actor=1,
        policy_version=4,
        rule_context=torch.zeros((6,), dtype=torch.float32),
        legal_actions=legal_actions,
        game_id="review-response",
        step_id=9,
        observation_contract_version=OBSERVATION_CONTRACT_VERSION,
        action_feature_contract_version=ACTION_FEATURE_CONTRACT_VERSION,
        env_contract_version=ENV_CONTRACT_VERSION,
        native_schema_name=NATIVE_SCHEMA_NAME,
        native_schema_version=NATIVE_SCHEMA_VERSION,
        native_action_identity_version=NATIVE_ACTION_IDENTITY_VERSION,
        native_legal_enumeration_version=NATIVE_LEGAL_ENUMERATION_VERSION,
        native_terminal_resolver_version=NATIVE_TERMINAL_RESOLVER_VERSION,
        rule_score_version=RULE_SCORE_VERSION,
        reward_spec_version=REWARD_SPEC_VERSION,
        style_context_version=STYLE_CONTEXT_VERSION,
    )


def _synthetic_ryukyoku_review_step() -> RolloutStep:
    legal_actions = (
        ActionSpec(ActionType.TSUMO, tile=2),
        ActionSpec(ActionType.RYUKYOKU),
        ActionSpec(ActionType.DISCARD, tile=0),
    )
    return RolloutStep(
        obs=ObsTensorBatch(
            tile_obs=torch.zeros((4, 34), dtype=torch.float32),
            scalar_obs=torch.zeros((6,), dtype=torch.float32),
        ),
        legal_action_ids=torch.tensor([110, 120, 130], dtype=torch.long),
        legal_action_features=torch.tensor(
            [
                [0.2] * 8,
                [0.3] * 8,
                [0.4] * 8,
            ],
            dtype=torch.float32,
        ),
        legal_action_mask=torch.tensor([True, True, True], dtype=torch.bool),
        action_index=1,
        action_spec=legal_actions[1],
        log_prob=-0.4,
        value=0.05,
        entropy=0.55,
        reward=0.0,
        done=True,
        actor=0,
        policy_version=6,
        rule_context=torch.zeros((6,), dtype=torch.float32),
        legal_actions=legal_actions,
        game_id="review-ryukyoku",
        step_id=12,
        observation_contract_version=OBSERVATION_CONTRACT_VERSION,
        action_feature_contract_version=ACTION_FEATURE_CONTRACT_VERSION,
        env_contract_version=ENV_CONTRACT_VERSION,
        native_schema_name=NATIVE_SCHEMA_NAME,
        native_schema_version=NATIVE_SCHEMA_VERSION,
        native_action_identity_version=NATIVE_ACTION_IDENTITY_VERSION,
        native_legal_enumeration_version=NATIVE_LEGAL_ENUMERATION_VERSION,
        native_terminal_resolver_version=NATIVE_TERMINAL_RESOLVER_VERSION,
        rule_score_version=RULE_SCORE_VERSION,
        reward_spec_version=REWARD_SPEC_VERSION,
        style_context_version=STYLE_CONTEXT_VERSION,
    )


def _synthetic_mixed_policy_episode() -> RolloutEpisode:
    step_a_actions = (
        ActionSpec(ActionType.DISCARD, tile=0),
        ActionSpec(ActionType.ANKAN, tile=27, consumed=(27, 27, 27, 27)),
    )
    step_b_actions = (
        ActionSpec(ActionType.PASS),
        ActionSpec(ActionType.RON, tile=3, from_who=0),
    )
    return RolloutEpisode(
        steps=(
            RolloutStep(
                obs=ObsTensorBatch(
                    tile_obs=torch.zeros((4, 34), dtype=torch.float32),
                    scalar_obs=torch.zeros((6,), dtype=torch.float32),
                ),
                legal_action_ids=torch.tensor([10, 20], dtype=torch.long),
                legal_action_features=torch.tensor([[0.1] * 8, [0.2] * 8], dtype=torch.float32),
                legal_action_mask=torch.tensor([True, True], dtype=torch.bool),
                action_index=1,
                action_spec=step_a_actions[1],
                log_prob=-0.3,
                value=0.1,
                entropy=0.4,
                reward=0.0,
                done=False,
                actor=0,
                policy_version=101,
                policy_name="learner",
                rule_context=torch.zeros((6,), dtype=torch.float32),
                legal_actions=step_a_actions,
                game_id="mixed-review",
                step_id=0,
                observation_contract_version=OBSERVATION_CONTRACT_VERSION,
                action_feature_contract_version=ACTION_FEATURE_CONTRACT_VERSION,
                env_contract_version=ENV_CONTRACT_VERSION,
                native_schema_name=NATIVE_SCHEMA_NAME,
                native_schema_version=NATIVE_SCHEMA_VERSION,
                native_action_identity_version=NATIVE_ACTION_IDENTITY_VERSION,
                native_legal_enumeration_version=NATIVE_LEGAL_ENUMERATION_VERSION,
                native_terminal_resolver_version=NATIVE_TERMINAL_RESOLVER_VERSION,
                rule_score_version=RULE_SCORE_VERSION,
                reward_spec_version=REWARD_SPEC_VERSION,
                style_context_version=STYLE_CONTEXT_VERSION,
            ),
            RolloutStep(
                obs=ObsTensorBatch(
                    tile_obs=torch.zeros((4, 34), dtype=torch.float32),
                    scalar_obs=torch.zeros((6,), dtype=torch.float32),
                ),
                legal_action_ids=torch.tensor([30, 40], dtype=torch.long),
                legal_action_features=torch.tensor([[0.3] * 8, [0.4] * 8], dtype=torch.float32),
                legal_action_mask=torch.tensor([True, True], dtype=torch.bool),
                action_index=1,
                action_spec=step_b_actions[1],
                log_prob=-0.6,
                value=0.2,
                entropy=0.5,
                reward=1.0,
                done=True,
                actor=1,
                policy_version=202,
                policy_name="opponent",
                rule_context=torch.zeros((6,), dtype=torch.float32),
                legal_actions=step_b_actions,
                game_id="mixed-review",
                step_id=1,
                observation_contract_version=OBSERVATION_CONTRACT_VERSION,
                action_feature_contract_version=ACTION_FEATURE_CONTRACT_VERSION,
                env_contract_version=ENV_CONTRACT_VERSION,
                native_schema_name=NATIVE_SCHEMA_NAME,
                native_schema_version=NATIVE_SCHEMA_VERSION,
                native_action_identity_version=NATIVE_ACTION_IDENTITY_VERSION,
                native_legal_enumeration_version=NATIVE_LEGAL_ENUMERATION_VERSION,
                native_terminal_resolver_version=NATIVE_TERMINAL_RESOLVER_VERSION,
                rule_score_version=RULE_SCORE_VERSION,
                reward_spec_version=REWARD_SPEC_VERSION,
                style_context_version=STYLE_CONTEXT_VERSION,
            ),
        ),
        terminal_rewards=(1.0, -1.0, 0.0, 0.0),
        final_ranks=(0, 3, 1, 2),
        scores=(27000, 23000, 25000, 25000),
        game_id="mixed-review",
    )


def test_review_rollout_episode_exposes_top_k_and_chosen_action(tmp_path) -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    policy = RandomInteractivePolicy()

    from keqingrl.selfplay import collect_discard_only_episode

    episode = collect_discard_only_episode(env, policy, seed=71, policy_version=9)
    review = review_rollout_episode(policy, episode, top_k=3)

    assert review.steps
    assert review.game_id == episode.game_id
    first = review.steps[0]
    assert first.legal_action_count == len(episode.steps[0].legal_actions)
    assert first.chosen_action.action_spec == episode.steps[0].action_spec
    assert first.chosen_action.action_label == format_action_spec(episode.steps[0].action_spec)
    assert 1 <= len(first.top_k) <= 3
    assert all(candidate.action_spec in episode.steps[0].legal_actions for candidate in first.top_k)
    assert len(first.rank_probs) == 4

    output_path = tmp_path / "keqingrl_review.jsonl"
    export_episode_review_jsonl(review, output_path)
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(review.steps)
    payload = json.loads(lines[0])
    assert payload["chosen_action"]["action"] == first.chosen_action.action_label
    assert payload["top_k"]


def test_review_rollout_step_and_export_support_atomic_self_turn_actions(tmp_path) -> None:
    step = _synthetic_review_step()
    policy = _FixedReviewPolicy([0.1, 1.3, 1.1, 0.9, 0.7])

    assert format_action_spec(step.legal_actions[1]) == "reach_discard:2m:tsumogiri"
    assert format_action_spec(step.legal_actions[2]) == "tsumo:3m"
    assert format_action_spec(step.legal_actions[3]) == "ankan:E:[E,E,E,E]"
    assert format_action_spec(step.legal_actions[4]) == "kakan:5p:[5p,5p,5p,5p]"

    step_review = review_rollout_step(policy, step, top_k=5)

    assert step_review.chosen_action.action_label == "ankan:E:[E,E,E,E]"
    assert [candidate.action_label for candidate in step_review.top_k] == [
        "reach_discard:2m:tsumogiri",
        "tsumo:3m",
        "ankan:E:[E,E,E,E]",
        "kakan:5p:[5p,5p,5p,5p]",
        "discard:1m",
    ]

    episode_review = review_rollout_episode(
        policy,
        RolloutEpisode(
            steps=(step,),
            terminal_rewards=(1.0, 0.0, -0.5, -0.5),
            final_ranks=(0, 1, 2, 3),
            scores=(26000, 24000, 25000, 25000),
            game_id="review-synthetic",
        ),
        top_k=5,
    )
    output_path = tmp_path / "atomic-self-turn-review.jsonl"
    export_episode_review_jsonl(episode_review, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8").strip())

    assert payload["chosen_action"]["action"] == "ankan:E:[E,E,E,E]"
    assert payload["chosen_action"]["feature_values"] == pytest.approx([0.3] * 8)
    assert [candidate["action"] for candidate in payload["top_k"]] == [
        "reach_discard:2m:tsumogiri",
        "tsumo:3m",
        "ankan:E:[E,E,E,E]",
        "kakan:5p:[5p,5p,5p,5p]",
        "discard:1m",
    ]
    assert payload["top_k"][0]["feature_values"] == pytest.approx([0.1] * 8)
    assert payload["top_k"][3]["feature_values"] == pytest.approx([0.4] * 8)


def test_review_rollout_step_and_export_support_response_actions(tmp_path) -> None:
    step = _synthetic_response_review_step()
    policy = _FixedReviewPolicy([1.4, 1.1, 0.9, 0.7, 0.2])

    assert format_action_spec(step.legal_actions[0]) == "ron:4m:from=0"
    assert format_action_spec(step.legal_actions[1]) == "pon:4m:[4m,4m]:from=0"
    assert format_action_spec(step.legal_actions[2]) == "daiminkan:4m:[4m,4m,4m]:from=0"
    assert format_action_spec(step.legal_actions[3]) == "chi:4m:[2m,3m]:from=0"
    assert format_action_spec(step.legal_actions[4]) == "pass"

    step_review = review_rollout_step(policy, step, top_k=5)

    assert step_review.chosen_action.action_label == "chi:4m:[2m,3m]:from=0"
    assert [candidate.action_label for candidate in step_review.top_k] == [
        "ron:4m:from=0",
        "pon:4m:[4m,4m]:from=0",
        "daiminkan:4m:[4m,4m,4m]:from=0",
        "chi:4m:[2m,3m]:from=0",
        "pass",
    ]

    episode_review = review_rollout_episode(
        policy,
        RolloutEpisode(
            steps=(step,),
            terminal_rewards=(0.0, 0.5, -0.5, 0.0),
            final_ranks=(1, 0, 3, 2),
            scores=(24000, 32000, 18000, 26000),
            game_id="review-response",
        ),
        top_k=5,
    )
    output_path = tmp_path / "response-review.jsonl"
    export_episode_review_jsonl(episode_review, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8").strip())

    assert payload["chosen_action"]["action"] == "chi:4m:[2m,3m]:from=0"
    assert payload["top_k"][0]["action"] == "ron:4m:from=0"
    assert payload["top_k"][4]["action"] == "pass"
    assert payload["top_k"][1]["feature_values"] == pytest.approx([0.6] * 8)


def test_review_rollout_step_and_export_support_ryukyoku_action(tmp_path) -> None:
    step = _synthetic_ryukyoku_review_step()
    policy = _FixedReviewPolicy([1.3, 1.1, 0.2])

    assert format_action_spec(step.legal_actions[0]) == "tsumo:3m"
    assert format_action_spec(step.legal_actions[1]) == "ryukyoku"
    assert format_action_spec(step.legal_actions[2]) == "discard:1m"

    step_review = review_rollout_step(policy, step, top_k=3)

    assert step_review.chosen_action.action_label == "ryukyoku"
    assert [candidate.action_label for candidate in step_review.top_k] == [
        "tsumo:3m",
        "ryukyoku",
        "discard:1m",
    ]

    episode_review = review_rollout_episode(
        policy,
        RolloutEpisode(
            steps=(step,),
            terminal_rewards=(0.0, 0.0, 0.0, 0.0),
            final_ranks=(0, 1, 2, 3),
            scores=(25000, 25000, 25000, 25000),
            game_id="review-ryukyoku",
        ),
        top_k=3,
    )
    output_path = tmp_path / "ryukyoku-review.jsonl"
    export_episode_review_jsonl(episode_review, output_path)
    payload = json.loads(output_path.read_text(encoding="utf-8").strip())

    assert payload["chosen_action"]["action"] == "ryukyoku"
    assert payload["chosen_action"]["feature_values"] == pytest.approx([0.3] * 8)
    assert [candidate["action"] for candidate in payload["top_k"]] == [
        "tsumo:3m",
        "ryukyoku",
        "discard:1m",
    ]


def test_review_autopilot_null_policy_fields_are_not_aggregated(tmp_path) -> None:
    learner_step = _synthetic_review_step()
    autopilot_step = replace(
        _synthetic_review_step(),
        is_autopilot=True,
        is_learner_controlled=False,
        step_id=6,
        policy_version=4,
        log_prob=-99.0,
        entropy=99.0,
        terminal_reason="tsumo",
        rulebase_chosen=learner_step.action_spec.canonical_key,
        policy_chosen=learner_step.action_spec.canonical_key,
    )
    episode = RolloutEpisode(
        steps=(learner_step, autopilot_step),
        terminal_rewards=(1.0, 0.0, -0.5, -0.5),
        final_ranks=(0, 1, 2, 3),
        scores=(26000, 24000, 25000, 25000),
        game_id="review-autopilot-null",
    )
    policy = _FixedReviewPolicy(
        [0.1, 1.3, 1.1, 0.9, 0.7],
        neural_delta=[0.01, 0.02, 0.03, 0.04, 0.05],
    )

    review = review_rollout_episode(policy, episode, top_k=3)

    learner_review, autopilot_review = review.steps
    assert learner_review.entropy is not None
    assert learner_review.recorded_log_prob == pytest.approx(learner_step.log_prob)
    assert learner_review.recomputed_log_prob is not None
    assert learner_review.chosen_action.neural_delta == pytest.approx(0.04)

    assert autopilot_review.is_autopilot is True
    assert autopilot_review.entropy is None
    assert autopilot_review.recorded_log_prob is None
    assert autopilot_review.recomputed_log_prob is None
    assert autopilot_review.chosen_action.neural_delta is None
    assert all(candidate.neural_delta is None for candidate in autopilot_review.top_k)

    summary = summarize_review_policy_fields(review)
    assert summary.learner_step_count == 1
    assert summary.autopilot_step_count == 1
    assert summary.autopilot_terminal_count == 1
    assert summary.terminal_reason_count == {"tsumo": 1}
    assert summary.entropy_count == 1
    assert summary.log_prob_count == 1
    assert summary.neural_delta_count == 1
    assert summary.mean_chosen_neural_delta == pytest.approx(0.04)

    output_path = tmp_path / "autopilot-null-review.jsonl"
    export_episode_review_jsonl(review, output_path)
    payloads = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert payloads[1]["is_autopilot"] is True
    assert payloads[1]["entropy"] is None
    assert payloads[1]["recorded_log_prob"] is None
    assert payloads[1]["recomputed_log_prob"] is None
    assert payloads[1]["terminal_reason"] == "tsumo"
    assert payloads[1]["chosen_action"]["neural_delta"] is None


def test_review_rollout_episode_can_route_mixed_policy_steps_via_resolver(tmp_path) -> None:
    learner_policy = _FixedReviewPolicy([0.2, 1.4])
    opponent_policy = _FixedReviewPolicy([1.2, 0.1])
    episode = _synthetic_mixed_policy_episode()
    resolver = build_policy_resolver(
        policies_by_name={
            "learner": learner_policy,
            "opponent": opponent_policy,
        }
    )

    review = review_rollout_episode(
        None,
        episode,
        top_k=2,
        policy_resolver=resolver,
    )

    assert [step.policy_name for step in review.steps] == ["learner", "opponent"]
    assert [step.policy_version for step in review.steps] == [101, 202]
    assert [step.top_k[0].action_label for step in review.steps] == [
        "ankan:E:[E,E,E,E]",
        "pass",
    ]

    output_path = tmp_path / "mixed-policy-review.jsonl"
    export_episode_review_jsonl(review, output_path)
    payloads = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]

    assert [payload["policy_name"] for payload in payloads] == ["learner", "opponent"]
    assert [payload["policy_version"] for payload in payloads] == [101, 202]
    assert [payload["top_k"][0]["action"] for payload in payloads] == [
        "ankan:E:[E,E,E,E]",
        "pass",
    ]
