from __future__ import annotations

import torch

from keqingrl import (
    ActionSample,
    ActionSpec,
    ActionType,
    DiscardOnlyMahjongEnv,
    DiscardOnlyIterationResult,
    EnvState,
    NeuralInteractivePolicy,
    InteractivePolicy,
    OpponentPool,
    OpponentPoolEntry,
    RandomInteractivePolicy,
    RolloutEpisode,
    RolloutStep,
    SeatPolicyAssignment,
    StepResult,
    build_episode_ppo_batch,
    build_episodes_ppo_batch,
    bind_reach_discard,
    collect_policy_episode,
    collect_selfplay_episode,
    collect_discard_only_episode,
    collect_discard_only_episodes,
    ppo_update,
    run_ppo_iteration,
    run_discard_only_ppo_iteration,
)
from keqingrl.contracts import ObsTensorBatch, PolicyInput


def _dummy_step(
    *,
    actor: int,
    step_id: int,
    action_count: int,
    reward: float = 0.0,
    done: bool = False,
    value: float = 0.0,
    action_spec: ActionSpec | None = None,
    legal_actions: tuple[ActionSpec, ...] | None = None,
    feature_dim: int = 3,
    action_index: int = 0,
) -> RolloutStep:
    chosen_action = ActionSpec(ActionType.DISCARD, tile=0) if action_spec is None else action_spec
    available_actions = (
        tuple(ActionSpec(ActionType.DISCARD, tile=index) for index in range(action_count))
        if legal_actions is None
        else legal_actions
    )
    return RolloutStep(
        obs=ObsTensorBatch(
            tile_obs=torch.zeros((4, 34), dtype=torch.float32),
            scalar_obs=torch.zeros((6,), dtype=torch.float32),
        ),
        legal_action_ids=torch.arange(action_count, dtype=torch.long),
        legal_action_features=torch.zeros((action_count, feature_dim), dtype=torch.float32),
        legal_action_mask=torch.ones((action_count,), dtype=torch.bool),
        action_index=action_index,
        action_spec=chosen_action,
        log_prob=0.0,
        value=value,
        entropy=0.0,
        reward=reward,
        done=done,
        actor=actor,
        policy_version=0,
        rule_context=torch.zeros((6,), dtype=torch.float32),
        legal_actions=available_actions,
        game_id="dummy",
        step_id=step_id,
    )


class _SingleStepActionEnv:
    def __init__(self, policy_input, chosen_actor: int = 0) -> None:
        self._policy_input = policy_input
        self._actor = chosen_actor
        self._done = False

    def reset(self, seed: int | None = None) -> EnvState:
        del seed
        self._done = False
        return EnvState(
            game_id="synthetic-episode",
            bakaze="E",
            kyoku=1,
            honba=0,
            kyotaku=0,
            scores=(25000, 25000, 25000, 25000),
            current_actor=self._actor,
            done=False,
            kyokus_completed=0,
        )

    def current_actor(self) -> int | None:
        return None if self._done else self._actor

    def is_done(self) -> bool:
        return self._done

    def observe(self, actor: int):
        assert actor == self._actor
        return self._policy_input

    def step(self, actor: int, action_spec: ActionSpec) -> StepResult:
        assert actor == self._actor
        self._done = True
        return StepResult(
            reward=1.0,
            done=True,
            next_actor=None,
            state=EnvState(
                game_id="synthetic-episode",
                bakaze="E",
                kyoku=1,
                honba=0,
                kyotaku=0,
                scores=(26000, 24000, 25000, 25000),
                current_actor=None,
                done=True,
                kyokus_completed=1,
            ),
            terminal_rewards=(1.0, 0.0, -0.5, -0.5),
            final_ranks=(0, 1, 2, 3),
            scores=(26000, 24000, 25000, 25000),
        )


class _TwoStepActorEnv:
    def __init__(self, policy_inputs: dict[int, PolicyInput]) -> None:
        self._policy_inputs = policy_inputs
        self._actor_sequence = (0, 1)
        self._cursor = 0
        self._done = False

    def reset(self, seed: int | None = None) -> EnvState:
        del seed
        self._cursor = 0
        self._done = False
        return EnvState(
            game_id="two-step-episode",
            bakaze="E",
            kyoku=1,
            honba=0,
            kyotaku=0,
            scores=(25000, 25000, 25000, 25000),
            current_actor=self._actor_sequence[0],
            done=False,
            kyokus_completed=0,
        )

    def current_actor(self) -> int | None:
        if self._done or self._cursor >= len(self._actor_sequence):
            return None
        return self._actor_sequence[self._cursor]

    def is_done(self) -> bool:
        return self._done

    def observe(self, actor: int) -> PolicyInput:
        assert actor == self.current_actor()
        return self._policy_inputs[actor]

    def step(self, actor: int, action_spec: ActionSpec) -> StepResult:
        current_input = self._policy_inputs[actor]
        assert action_spec in current_input.legal_actions[0]
        self._cursor += 1
        done = self._cursor >= len(self._actor_sequence)
        self._done = done
        next_actor = None if done else self._actor_sequence[self._cursor]
        return StepResult(
            reward=0.0 if not done else 1.0,
            done=done,
            next_actor=next_actor,
            state=EnvState(
                game_id="two-step-episode",
                bakaze="E",
                kyoku=1,
                honba=0,
                kyotaku=0,
                scores=(27000, 23000, 25000, 25000),
                current_actor=next_actor,
                done=done,
                kyokus_completed=1 if done else 0,
            ),
            terminal_rewards=(1.0, -1.0, 0.0, 0.0) if done else None,
            final_ranks=(0, 3, 1, 2) if done else None,
            scores=(27000, 23000, 25000, 25000) if done else None,
        )


class _ChosenIndexPolicy(InteractivePolicy):
    def __init__(self, chosen_index: int) -> None:
        super().__init__()
        self.chosen_index = chosen_index

    def forward(self, policy_input):  # pragma: no cover - unused in this test helper
        raise NotImplementedError

    def sample_action(self, policy_input, *, greedy: bool = False) -> ActionSample:
        del greedy
        return ActionSample(
            action_index=torch.tensor([self.chosen_index], dtype=torch.long),
            action_spec=[policy_input.legal_actions[0][self.chosen_index]],
            log_prob=torch.tensor([-0.25], dtype=torch.float32),
            entropy=torch.tensor([0.5], dtype=torch.float32),
            value=torch.tensor([0.2], dtype=torch.float32),
            rank_probs=torch.tensor([[0.4, 0.3, 0.2, 0.1]], dtype=torch.float32),
        )


def test_build_episode_ppo_batch_separates_actor_returns() -> None:
    episode = RolloutEpisode(
        steps=(
            _dummy_step(actor=0, step_id=0, action_count=2),
            _dummy_step(actor=1, step_id=1, action_count=3),
            _dummy_step(actor=0, step_id=2, action_count=4),
            _dummy_step(actor=1, step_id=3, action_count=2),
        ),
        terminal_rewards=(1.0, -1.0, 0.0, 0.0),
        final_ranks=(0, 3, 1, 2),
        scores=(32000, 18000, 28000, 22000),
        game_id="dummy",
    )

    advantages, returns, prepared_steps, batch = build_episode_ppo_batch(
        episode,
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert returns.tolist() == [1.0, -1.0, 1.0, -1.0]
    assert advantages.tolist() == [1.0, -1.0, 1.0, -1.0]
    assert [step.done for step in prepared_steps] == [False, False, True, True]
    assert batch.policy_input.legal_action_ids.shape == (4, 4)
    assert batch.policy_input.legal_action_features.shape == (4, 4, 3)
    assert batch.policy_input.legal_action_mask.tolist() == [
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
        [True, True, False, False],
    ]
    assert batch.final_rank_target.tolist() == [0, 3, 0, 3]


def test_build_ppo_batch_rejects_reordered_legal_actions() -> None:
    legal_actions = (
        ActionSpec(ActionType.DISCARD, tile=0),
        ActionSpec(ActionType.DISCARD, tile=1),
    )
    step = _dummy_step(
        actor=0,
        step_id=0,
        action_count=2,
        legal_actions=tuple(reversed(legal_actions)),
        action_spec=legal_actions[1],
        action_index=1,
    )

    try:
        build_episodes_ppo_batch(
            (
                RolloutEpisode(
                    steps=(step,),
                    terminal_rewards=(1.0, 0.0, 0.0, 0.0),
                    final_ranks=(0, 1, 2, 3),
                    scores=(30000, 25000, 25000, 20000),
                ),
            )
        )
    except ValueError as exc:
        assert "action-order contract" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected action-order contract violation")


def test_build_ppo_batch_rejects_contract_version_mismatch() -> None:
    step = _dummy_step(actor=0, step_id=0, action_count=2, reward=1.0, done=True)
    step = RolloutStep(
        obs=step.obs,
        legal_action_ids=step.legal_action_ids,
        legal_action_features=step.legal_action_features,
        legal_action_mask=step.legal_action_mask,
        action_index=step.action_index,
        action_spec=step.action_spec,
        log_prob=step.log_prob,
        value=step.value,
        entropy=step.entropy,
        reward=step.reward,
        done=step.done,
        actor=step.actor,
        policy_version=step.policy_version,
        rule_context=step.rule_context,
        legal_actions=step.legal_actions,
        game_id=step.game_id,
        step_id=step.step_id,
        observation_contract_version="old-observation-contract",
    )

    try:
        build_episodes_ppo_batch(
            (
                RolloutEpisode(
                    steps=(step,),
                    terminal_rewards=(1.0, 0.0, 0.0, 0.0),
                    final_ranks=(0, 1, 2, 3),
                    scores=(30000, 25000, 25000, 20000),
                ),
            )
        )
    except ValueError as exc:
        assert "unsupported observation contract" in str(exc)
    else:  # pragma: no cover - defensive
        raise AssertionError("expected rollout contract-version mismatch")


def test_collect_discard_only_episode_emits_terminal_metadata() -> None:
    torch.manual_seed(0)
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    policy = RandomInteractivePolicy()

    episode = collect_discard_only_episode(env, policy, seed=29, policy_version=7)

    assert episode.steps
    assert episode.game_id is not None
    assert len(episode.terminal_rewards) == 4
    assert tuple(sorted(episode.final_ranks)) == (0, 1, 2, 3)
    assert all(step.policy_version == 7 for step in episode.steps)
    assert all(step.game_id == episode.game_id for step in episode.steps)
    assert all(step.action_spec.action_type == ActionType.DISCARD for step in episode.steps)


def test_collect_discard_only_episodes_returns_requested_episode_count() -> None:
    torch.manual_seed(0)
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    policy = RandomInteractivePolicy()

    episodes = collect_discard_only_episodes(
        env,
        policy,
        num_episodes=2,
        seed=41,
        seed_stride=3,
        policy_version=5,
    )

    assert len(episodes) == 2
    assert [episode.seed for episode in episodes] == [41, 44]
    assert all(episode.steps for episode in episodes)
    assert all(step.policy_version == 5 for episode in episodes for step in episode.steps)


def test_build_episodes_ppo_batch_combines_multiple_episodes() -> None:
    episode_a = RolloutEpisode(
        steps=(
            _dummy_step(actor=0, step_id=0, action_count=2),
            _dummy_step(actor=0, step_id=1, action_count=4),
        ),
        terminal_rewards=(1.0, 0.0, 0.0, -1.0),
        final_ranks=(0, 1, 2, 3),
        scores=(32000, 27000, 23000, 18000),
        game_id="ep_a",
    )
    episode_b = RolloutEpisode(
        steps=(
            _dummy_step(actor=1, step_id=0, action_count=3),
            _dummy_step(actor=1, step_id=1, action_count=2),
        ),
        terminal_rewards=(0.0, 1.0, -1.0, 0.0),
        final_ranks=(2, 0, 3, 1),
        scores=(24000, 33000, 17000, 26000),
        game_id="ep_b",
    )

    advantages, returns, prepared_steps, batch = build_episodes_ppo_batch(
        (episode_a, episode_b),
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert advantages.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert returns.tolist() == [1.0, 1.0, 1.0, 1.0]
    assert len(prepared_steps) == 4
    assert batch.policy_input.legal_action_ids.shape == (4, 4)
    assert batch.final_rank_target.tolist() == [0, 0, 0, 0]


def test_collect_policy_episode_uses_seat_specific_policies_and_versions() -> None:
    actor0_actions = (
        ActionSpec(ActionType.DISCARD, tile=0),
        ActionSpec(ActionType.ANKAN, tile=27, consumed=(27, 27, 27, 27)),
    )
    actor1_actions = (
        ActionSpec(ActionType.PASS),
        ActionSpec(ActionType.RON, tile=1, from_who=0),
    )
    env = _TwoStepActorEnv(
        {
            0: PolicyInput(
                obs=ObsTensorBatch(
                    tile_obs=torch.zeros((1, 4, 34), dtype=torch.float32),
                    scalar_obs=torch.zeros((1, 6), dtype=torch.float32),
                ),
                legal_action_ids=torch.tensor([[10, 20]], dtype=torch.long),
                legal_action_features=torch.zeros((1, 2, 8), dtype=torch.float32),
                legal_action_mask=torch.tensor([[True, True]], dtype=torch.bool),
                rule_context=torch.zeros((1, 6), dtype=torch.float32),
                legal_actions=(actor0_actions,),
            ),
            1: PolicyInput(
                obs=ObsTensorBatch(
                    tile_obs=torch.zeros((1, 4, 34), dtype=torch.float32),
                    scalar_obs=torch.zeros((1, 6), dtype=torch.float32),
                ),
                legal_action_ids=torch.tensor([[30, 40]], dtype=torch.long),
                legal_action_features=torch.zeros((1, 2, 8), dtype=torch.float32),
                legal_action_mask=torch.tensor([[True, True]], dtype=torch.bool),
                rule_context=torch.zeros((1, 6), dtype=torch.float32),
                legal_actions=(actor1_actions,),
            ),
        }
    )
    seat_policies = (
        SeatPolicyAssignment(policy=_ChosenIndexPolicy(1), policy_version=11),
        SeatPolicyAssignment(policy=_ChosenIndexPolicy(0), policy_version=22),
        SeatPolicyAssignment(policy=_ChosenIndexPolicy(0), policy_version=33),
        SeatPolicyAssignment(policy=_ChosenIndexPolicy(0), policy_version=44),
    )

    episode = collect_policy_episode(env, seat_policies, seed=7)

    assert [step.actor for step in episode.steps] == [0, 1]
    assert [step.action_spec.action_type for step in episode.steps] == [
        ActionType.ANKAN,
        ActionType.PASS,
    ]
    assert [step.policy_version for step in episode.steps] == [11, 22]
    assert [step.policy_name for step in episode.steps] == [None, None]


def test_collect_selfplay_episode_uses_opponent_pool_for_nonlearner_seats() -> None:
    actor0_actions = (
        ActionSpec(ActionType.DISCARD, tile=0),
        ActionSpec(ActionType.KAKAN, tile=13, consumed=(13, 13, 13, 13)),
    )
    actor1_actions = (
        ActionSpec(ActionType.PASS),
        ActionSpec(ActionType.PON, tile=1, consumed=(1, 1), from_who=0),
    )
    env = _TwoStepActorEnv(
        {
            0: PolicyInput(
                obs=ObsTensorBatch(
                    tile_obs=torch.zeros((1, 4, 34), dtype=torch.float32),
                    scalar_obs=torch.zeros((1, 6), dtype=torch.float32),
                ),
                legal_action_ids=torch.tensor([[10, 20]], dtype=torch.long),
                legal_action_features=torch.zeros((1, 2, 8), dtype=torch.float32),
                legal_action_mask=torch.tensor([[True, True]], dtype=torch.bool),
                rule_context=torch.zeros((1, 6), dtype=torch.float32),
                legal_actions=(actor0_actions,),
            ),
            1: PolicyInput(
                obs=ObsTensorBatch(
                    tile_obs=torch.zeros((1, 4, 34), dtype=torch.float32),
                    scalar_obs=torch.zeros((1, 6), dtype=torch.float32),
                ),
                legal_action_ids=torch.tensor([[30, 40]], dtype=torch.long),
                legal_action_features=torch.zeros((1, 2, 8), dtype=torch.float32),
                legal_action_mask=torch.tensor([[True, True]], dtype=torch.bool),
                rule_context=torch.zeros((1, 6), dtype=torch.float32),
                legal_actions=(actor1_actions,),
            ),
        }
    )
    opponent_pool = OpponentPool(
        (
            OpponentPoolEntry(
                policy=_ChosenIndexPolicy(1),
                policy_version=99,
                greedy=True,
                name="fixed-opponent",
            ),
        )
    )

    episode = collect_selfplay_episode(
        env,
        _ChosenIndexPolicy(1),
        opponent_pool=opponent_pool,
        learner_seats=(0,),
        seed=13,
        policy_version=7,
    )

    assert [step.action_spec.action_type for step in episode.steps] == [
        ActionType.KAKAN,
        ActionType.PON,
    ]
    assert [step.policy_version for step in episode.steps] == [7, 99]
    assert [step.policy_name for step in episode.steps] == ["learner", "fixed-opponent"]


def test_collect_discard_only_episode_accepts_non_discard_self_turn_actions() -> None:
    legal_actions = (
        ActionSpec(ActionType.DISCARD, tile=0),
        bind_reach_discard(ActionSpec(ActionType.DISCARD, tile=1)),
        ActionSpec(ActionType.ANKAN, tile=27, consumed=(27, 27, 27, 27)),
        ActionSpec(ActionType.KAKAN, tile=13, consumed=(13, 13, 13, 13)),
    )
    policy_input = ObsTensorBatch(
        tile_obs=torch.zeros((1, 4, 34), dtype=torch.float32),
        scalar_obs=torch.zeros((1, 6), dtype=torch.float32),
    )
    env = _SingleStepActionEnv(
        policy_input=PolicyInput(
            obs=policy_input,
            legal_action_ids=torch.tensor([[10, 20, 30, 40]], dtype=torch.long),
            legal_action_features=torch.zeros((1, 4, 8), dtype=torch.float32),
            legal_action_mask=torch.tensor([[True, True, True, True]], dtype=torch.bool),
            rule_context=torch.zeros((1, 6), dtype=torch.float32),
            legal_actions=(legal_actions,),
            recurrent_state=None,
        )
    )
    policy = _ChosenIndexPolicy(chosen_index=1)

    episode = collect_discard_only_episode(env, policy, seed=17, policy_version=4)

    assert len(episode.steps) == 1
    assert episode.steps[0].action_spec.action_type == ActionType.REACH_DISCARD
    assert episode.steps[0].legal_actions == legal_actions
    assert episode.steps[0].legal_action_features.shape == (4, 8)


def test_collect_discard_only_episode_accepts_response_actions() -> None:
    legal_actions = (
        ActionSpec(ActionType.RON, tile=1, from_who=0),
        ActionSpec(ActionType.PON, tile=1, consumed=(1, 1), from_who=0),
        ActionSpec(ActionType.PASS),
    )
    policy_input = ObsTensorBatch(
        tile_obs=torch.zeros((1, 4, 34), dtype=torch.float32),
        scalar_obs=torch.zeros((1, 6), dtype=torch.float32),
    )
    env = _SingleStepActionEnv(
        policy_input=PolicyInput(
            obs=policy_input,
            legal_action_ids=torch.tensor([[10, 20, 30]], dtype=torch.long),
            legal_action_features=torch.zeros((1, 3, 8), dtype=torch.float32),
            legal_action_mask=torch.tensor([[True, True, True]], dtype=torch.bool),
            rule_context=torch.zeros((1, 6), dtype=torch.float32),
            legal_actions=(legal_actions,),
            recurrent_state=None,
        )
    )
    policy = _ChosenIndexPolicy(chosen_index=1)

    episode = collect_discard_only_episode(env, policy, seed=18, policy_version=5)

    assert len(episode.steps) == 1
    assert episode.steps[0].action_spec.action_type == ActionType.PON
    assert episode.steps[0].legal_actions == legal_actions
    assert episode.steps[0].legal_action_features.shape == (3, 8)


def test_collect_discard_only_episode_accepts_ryukyoku_action() -> None:
    legal_actions = (
        ActionSpec(ActionType.DISCARD, tile=0),
        ActionSpec(ActionType.RYUKYOKU),
    )
    policy_input = ObsTensorBatch(
        tile_obs=torch.zeros((1, 4, 34), dtype=torch.float32),
        scalar_obs=torch.zeros((1, 6), dtype=torch.float32),
    )
    env = _SingleStepActionEnv(
        policy_input=PolicyInput(
            obs=policy_input,
            legal_action_ids=torch.tensor([[10, 20]], dtype=torch.long),
            legal_action_features=torch.zeros((1, 2, 8), dtype=torch.float32),
            legal_action_mask=torch.tensor([[True, True]], dtype=torch.bool),
            rule_context=torch.zeros((1, 6), dtype=torch.float32),
            legal_actions=(legal_actions,),
            recurrent_state=None,
        )
    )
    policy = _ChosenIndexPolicy(chosen_index=1)

    episode = collect_discard_only_episode(env, policy, seed=19, policy_version=6)

    assert len(episode.steps) == 1
    assert episode.steps[0].action_spec.action_type == ActionType.RYUKYOKU
    assert episode.steps[0].legal_actions == legal_actions
    assert episode.steps[0].legal_action_features.shape == (2, 8)


def test_build_episode_ppo_batch_accepts_atomic_self_turn_actions() -> None:
    steps = (
        _dummy_step(
            actor=0,
            step_id=0,
            action_count=2,
            action_spec=bind_reach_discard(ActionSpec(ActionType.DISCARD, tile=1)),
            legal_actions=(
                ActionSpec(ActionType.DISCARD, tile=0),
                bind_reach_discard(ActionSpec(ActionType.DISCARD, tile=1)),
            ),
            feature_dim=8,
            action_index=1,
        ),
        _dummy_step(
            actor=0,
            step_id=1,
            action_count=3,
            action_spec=ActionSpec(ActionType.ANKAN, tile=27, consumed=(27, 27, 27, 27)),
            legal_actions=(
                ActionSpec(ActionType.DISCARD, tile=2),
                ActionSpec(ActionType.ANKAN, tile=27, consumed=(27, 27, 27, 27)),
                ActionSpec(ActionType.KAKAN, tile=13, consumed=(13, 13, 13, 13)),
            ),
            feature_dim=8,
            action_index=1,
        ),
        _dummy_step(
            actor=1,
            step_id=2,
            action_count=4,
            action_spec=ActionSpec(ActionType.KAKAN, tile=13, consumed=(13, 13, 13, 13)),
            legal_actions=(
                ActionSpec(ActionType.DISCARD, tile=3),
                ActionSpec(ActionType.TSUMO, tile=4),
                ActionSpec(ActionType.ANKAN, tile=28, consumed=(28, 28, 28, 28)),
                ActionSpec(ActionType.KAKAN, tile=13, consumed=(13, 13, 13, 13)),
            ),
            feature_dim=8,
            action_index=3,
        ),
    )
    episode = RolloutEpisode(
        steps=steps,
        terminal_rewards=(1.0, -1.0, 0.0, 0.0),
        final_ranks=(0, 3, 1, 2),
        scores=(32000, 18000, 28000, 22000),
        game_id="atomic-self-turn",
    )

    _advantages, _returns, prepared_steps, batch = build_episode_ppo_batch(episode)

    assert [step.action_spec.action_type for step in prepared_steps] == [
        ActionType.REACH_DISCARD,
        ActionType.ANKAN,
        ActionType.KAKAN,
    ]
    assert batch.policy_input.legal_action_features.shape == (3, 4, 8)
    assert batch.policy_input.legal_action_mask.tolist() == [
        [True, True, False, False],
        [True, True, True, False],
        [True, True, True, True],
    ]


def test_build_episode_ppo_batch_accepts_response_actions() -> None:
    steps = (
        _dummy_step(
            actor=1,
            step_id=0,
            action_count=3,
            action_spec=ActionSpec(ActionType.RON, tile=1, from_who=0),
            legal_actions=(
                ActionSpec(ActionType.RON, tile=1, from_who=0),
                ActionSpec(ActionType.PON, tile=1, consumed=(1, 1), from_who=0),
                ActionSpec(ActionType.PASS),
            ),
            feature_dim=8,
            action_index=0,
        ),
        _dummy_step(
            actor=2,
            step_id=1,
            action_count=4,
            action_spec=ActionSpec(ActionType.CHI, tile=3, consumed=(1, 2), from_who=1),
            legal_actions=(
                ActionSpec(ActionType.CHI, tile=3, consumed=(1, 2), from_who=1),
                ActionSpec(ActionType.PON, tile=3, consumed=(3, 3), from_who=1),
                ActionSpec(ActionType.DAIMINKAN, tile=3, consumed=(3, 3, 3), from_who=1),
                ActionSpec(ActionType.PASS),
            ),
            feature_dim=8,
            action_index=0,
        ),
        _dummy_step(
            actor=3,
            step_id=2,
            action_count=2,
            action_spec=ActionSpec(ActionType.PASS),
            legal_actions=(
                ActionSpec(ActionType.RON, tile=5, from_who=2),
                ActionSpec(ActionType.PASS),
            ),
            feature_dim=8,
            action_index=1,
        ),
    )
    episode = RolloutEpisode(
        steps=steps,
        terminal_rewards=(0.0, 1.0, -0.5, -0.5),
        final_ranks=(1, 0, 2, 3),
        scores=(25000, 32000, 22000, 21000),
        game_id="response-actions",
    )

    _advantages, _returns, prepared_steps, batch = build_episode_ppo_batch(episode)

    assert [step.action_spec.action_type for step in prepared_steps] == [
        ActionType.RON,
        ActionType.CHI,
        ActionType.PASS,
    ]
    assert batch.policy_input.legal_action_features.shape == (3, 4, 8)
    assert batch.policy_input.legal_action_mask.tolist() == [
        [True, True, True, False],
        [True, True, True, True],
        [True, True, False, False],
    ]


def test_build_episode_ppo_batch_accepts_ryukyoku_action() -> None:
    steps = (
        _dummy_step(
            actor=0,
            step_id=0,
            action_count=2,
            action_spec=ActionSpec(ActionType.RYUKYOKU),
            legal_actions=(
                ActionSpec(ActionType.DISCARD, tile=0),
                ActionSpec(ActionType.RYUKYOKU),
            ),
            feature_dim=8,
            action_index=1,
        ),
    )
    episode = RolloutEpisode(
        steps=steps,
        terminal_rewards=(0.0, 0.0, 0.0, 0.0),
        final_ranks=(0, 1, 2, 3),
        scores=(25000, 25000, 25000, 25000),
        game_id="ryukyoku-self-turn",
    )

    _advantages, _returns, prepared_steps, batch = build_episode_ppo_batch(episode)

    assert [step.action_spec.action_type for step in prepared_steps] == [ActionType.RYUKYOKU]
    assert batch.policy_input.legal_action_features.shape == (1, 2, 8)
    assert batch.policy_input.legal_action_mask.tolist() == [[True, True]]


def test_run_ppo_iteration_filters_batch_to_learner_seats_with_opponent_pool() -> None:
    learner_policy = NeuralInteractivePolicy(
        hidden_dim=32,
        num_res_blocks=1,
        c_tile=4,
        n_scalar=6,
        dropout=0.0,
    )
    opponent_policy = _ChosenIndexPolicy(0)
    optimizer = torch.optim.Adam(learner_policy.parameters(), lr=1e-3)
    env = _TwoStepActorEnv(
        {
            0: PolicyInput(
                obs=ObsTensorBatch(
                    tile_obs=torch.zeros((1, 4, 34), dtype=torch.float32),
                    scalar_obs=torch.zeros((1, 6), dtype=torch.float32),
                ),
                legal_action_ids=torch.tensor([[10, 20]], dtype=torch.long),
                legal_action_features=torch.zeros((1, 2, 8), dtype=torch.float32),
                legal_action_mask=torch.tensor([[True, True]], dtype=torch.bool),
                rule_context=torch.zeros((1, 6), dtype=torch.float32),
                legal_actions=(
                    (
                        ActionSpec(ActionType.DISCARD, tile=0),
                        ActionSpec(ActionType.ANKAN, tile=27, consumed=(27, 27, 27, 27)),
                    ),
                ),
            ),
            1: PolicyInput(
                obs=ObsTensorBatch(
                    tile_obs=torch.zeros((1, 4, 34), dtype=torch.float32),
                    scalar_obs=torch.zeros((1, 6), dtype=torch.float32),
                ),
                legal_action_ids=torch.tensor([[30, 40]], dtype=torch.long),
                legal_action_features=torch.zeros((1, 2, 8), dtype=torch.float32),
                legal_action_mask=torch.tensor([[True, True]], dtype=torch.bool),
                rule_context=torch.zeros((1, 6), dtype=torch.float32),
                legal_actions=(
                    (
                        ActionSpec(ActionType.PASS),
                        ActionSpec(ActionType.RON, tile=1, from_who=0),
                    ),
                ),
            ),
        }
    )
    opponent_pool = OpponentPool(
        (OpponentPoolEntry(policy=opponent_policy, policy_version=99, greedy=True),)
    )

    result = run_ppo_iteration(
        env,
        learner_policy,
        optimizer,
        num_episodes=1,
        opponent_pool=opponent_pool,
        learner_seats=(0,),
        update_epochs=1,
        seed=23,
        rank_coef=0.05,
        max_grad_norm=1.0,
    )

    assert isinstance(result, DiscardOnlyIterationResult)
    assert len(result.episodes) == 1
    assert result.metrics.total_steps == 2
    assert result.metrics.batch_size == 1
    assert [step.policy_version for step in result.episodes[0].steps] == [0, 99]
    assert [step.policy_name for step in result.episodes[0].steps] == ["learner", None]
    assert torch.isfinite(torch.tensor(result.metrics.mean_total_loss))


def test_ppo_update_runs_on_collected_mahjong_episode() -> None:
    torch.manual_seed(1)
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    policy = NeuralInteractivePolicy(
        hidden_dim=32,
        num_res_blocks=1,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    episode = collect_discard_only_episode(env, policy, seed=31, policy_version=3)
    _advantages, _returns, prepared_steps, batch = build_episode_ppo_batch(episode)
    losses = ppo_update(
        policy,
        optimizer,
        batch,
        clip_eps=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        rank_coef=0.05,
        max_grad_norm=1.0,
    )

    assert prepared_steps
    assert batch.policy_input.legal_action_ids.ndim == 2
    assert batch.policy_input.legal_action_features.ndim == 3
    assert torch.isfinite(losses.total_loss)
    assert torch.isfinite(losses.policy_loss)
    assert torch.isfinite(losses.value_loss)
    assert losses.rank_loss is not None


def test_run_discard_only_ppo_iteration_returns_finite_metrics() -> None:
    torch.manual_seed(2)
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    policy = NeuralInteractivePolicy(
        hidden_dim=32,
        num_res_blocks=1,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)

    result = run_discard_only_ppo_iteration(
        env,
        policy,
        optimizer,
        num_episodes=2,
        update_epochs=2,
        seed=53,
        rank_coef=0.05,
        max_grad_norm=1.0,
    )

    assert isinstance(result, DiscardOnlyIterationResult)
    assert len(result.episodes) == 2
    assert len(result.losses) == 2
    assert result.metrics.episode_count == 2
    assert result.metrics.total_steps == sum(len(episode.steps) for episode in result.episodes)
    assert result.metrics.batch_size == int(result.batch.action_index.numel())
    assert torch.isfinite(torch.tensor(result.metrics.mean_total_loss))
    assert torch.isfinite(torch.tensor(result.metrics.mean_policy_loss))
    assert torch.isfinite(torch.tensor(result.metrics.mean_value_loss))
    assert torch.isfinite(torch.tensor(result.metrics.mean_approx_kl))
    assert 0.0 <= result.metrics.first_place_rate <= 1.0
    assert 1.0 <= result.metrics.mean_rank <= 4.0
