from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from keqingrl.actions import ActionSpec, ActionType
from keqingrl.contracts import ObsTensorBatch, PolicyInput, PolicyOutput
from keqingrl.mortal_teacher import (
    MORTAL_ACTION_MASK_EXTRA_KEY,
    MORTAL_ACTION_SPACE,
    MORTAL_ACTION_TEACHER_CONTRACT_VERSION,
    MORTAL_DISCARD_TEACHER_CONTRACT_VERSION,
    MORTAL_Q_VALUES_EXTRA_KEY,
    MortalTeacherMappingError,
)
from mahjong_env.action_space import TILE_NAME_TO_IDX
from scripts.run_keqingrl_tempered_ratio_pilot import (
    _TEACHER_SOURCES,
    _delta_support_mask,
    _full_action_probe_stats,
    _load_mortal_teacher_runtime_for_args,
    _mortal_runtime_for_teacher,
    _reach_probe_stats,
    _requires_mortal_teacher_runtime,
    _rollout_env,
    _teacher_version,
    _terminal_coverage_gate_pass,
    _terminal_coverage_stats,
    _topk_teacher_context,
)


def _mortal_batch(*, extras: dict[str, torch.Tensor] | None = None) -> SimpleNamespace:
    legal_actions = (
        (
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["5m"]),
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["9m"]),
        ),
    )
    policy_input = PolicyInput(
        obs=ObsTensorBatch(
            tile_obs=torch.zeros((1, 1), dtype=torch.float32),
            scalar_obs=torch.zeros((1, 1), dtype=torch.float32),
            extras={} if extras is None else extras,
        ),
        legal_action_ids=torch.zeros((1, 3), dtype=torch.long),
        legal_action_features=torch.zeros((1, 3, 8), dtype=torch.float32),
        legal_action_mask=torch.ones((1, 3), dtype=torch.bool),
        rule_context=torch.zeros((1, 1), dtype=torch.float32),
        prior_logits=torch.tensor([[1.0, 3.0, 2.0]], dtype=torch.float32),
        legal_actions=legal_actions,
    )
    return SimpleNamespace(policy_input=policy_input, action_index=torch.tensor([0], dtype=torch.long))


def _mortal_action_batch(*, extras: dict[str, torch.Tensor] | None = None) -> SimpleNamespace:
    legal_actions = (
        (
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
            ActionSpec(ActionType.REACH_DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
            ActionSpec(ActionType.PASS),
        ),
    )
    policy_input = PolicyInput(
        obs=ObsTensorBatch(
            tile_obs=torch.zeros((1, 1), dtype=torch.float32),
            scalar_obs=torch.zeros((1, 1), dtype=torch.float32),
            extras={} if extras is None else extras,
        ),
        legal_action_ids=torch.zeros((1, 3), dtype=torch.long),
        legal_action_features=torch.zeros((1, 3, 8), dtype=torch.float32),
        legal_action_mask=torch.ones((1, 3), dtype=torch.bool),
        rule_context=torch.zeros((1, 1), dtype=torch.float32),
        prior_logits=torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32),
        legal_actions=legal_actions,
    )
    return SimpleNamespace(policy_input=policy_input, action_index=torch.tensor([1], dtype=torch.long))


def _mortal_action_batch_with_short_row(*, extras: dict[str, torch.Tensor] | None = None) -> SimpleNamespace:
    legal_actions = (
        (
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
            ActionSpec(ActionType.REACH_DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
            ActionSpec(ActionType.PASS),
        ),
        (ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"]),),
    )
    policy_input = PolicyInput(
        obs=ObsTensorBatch(
            tile_obs=torch.zeros((2, 1), dtype=torch.float32),
            scalar_obs=torch.zeros((2, 1), dtype=torch.float32),
            extras={} if extras is None else extras,
        ),
        legal_action_ids=torch.zeros((2, 3), dtype=torch.long),
        legal_action_features=torch.zeros((2, 3, 8), dtype=torch.float32),
        legal_action_mask=torch.tensor([[True, True, True], [True, False, False]], dtype=torch.bool),
        rule_context=torch.zeros((2, 1), dtype=torch.float32),
        prior_logits=torch.tensor([[3.0, 2.0, 1.0], [1.0, 0.0, 0.0]], dtype=torch.float32),
        legal_actions=legal_actions,
    )
    return SimpleNamespace(policy_input=policy_input, action_index=torch.tensor([1, 0], dtype=torch.long))


def _mortal_action_batch_with_duplicate_reach_sources(
    *,
    extras: dict[str, torch.Tensor] | None = None,
) -> SimpleNamespace:
    legal_actions = (
        (
            ActionSpec(ActionType.REACH_DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
            ActionSpec(ActionType.REACH_DISCARD, tile=TILE_NAME_TO_IDX["2m"]),
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
            ActionSpec(ActionType.PASS),
        ),
    )
    policy_input = PolicyInput(
        obs=ObsTensorBatch(
            tile_obs=torch.zeros((1, 1), dtype=torch.float32),
            scalar_obs=torch.zeros((1, 1), dtype=torch.float32),
            extras={} if extras is None else extras,
        ),
        legal_action_ids=torch.zeros((1, 4), dtype=torch.long),
        legal_action_features=torch.zeros((1, 4, 8), dtype=torch.float32),
        legal_action_mask=torch.ones((1, 4), dtype=torch.bool),
        rule_context=torch.zeros((1, 1), dtype=torch.float32),
        prior_logits=torch.tensor([[4.0, 3.9, 3.8, 1.0]], dtype=torch.float32),
        legal_actions=legal_actions,
    )
    return SimpleNamespace(policy_input=policy_input, action_index=torch.tensor([0], dtype=torch.long))


class _StaticPolicy:
    def __init__(self, action_logits: torch.Tensor, prior_logits: torch.Tensor) -> None:
        self.action_logits = action_logits
        self.prior_logits = prior_logits

    def __call__(self, policy_input: PolicyInput) -> PolicyOutput:
        batch_size = int(policy_input.legal_action_mask.shape[0])
        return PolicyOutput(
            action_logits=self.action_logits,
            value=torch.zeros((batch_size,), dtype=torch.float32),
            rank_logits=torch.zeros((batch_size, 4), dtype=torch.float32),
            aux={
                "prior_logits": self.prior_logits,
                "final_logits": self.action_logits,
            },
        )


def _q_values() -> torch.Tensor:
    return torch.arange(MORTAL_ACTION_SPACE, dtype=torch.float32).unsqueeze(0)


def _mortal_mask(*ids: int) -> torch.BoolTensor:
    mask = torch.zeros((1, MORTAL_ACTION_SPACE), dtype=torch.bool)
    mask[0, list(ids)] = True
    return mask


def test_tempered_ratio_pilot_mortal_discard_q_teacher_uses_mortal_extras() -> None:
    batch = _mortal_batch(
        extras={
            MORTAL_Q_VALUES_EXTRA_KEY: _q_values(),
            MORTAL_ACTION_MASK_EXTRA_KEY: _mortal_mask(0, 34, 8),
        }
    )

    teacher = _topk_teacher_context(
        batch,
        prior_logits=batch.policy_input.prior_logits,
        topk=2,
        teacher_source="mortal-discard-q",
        teacher_temperature=1.0,
    )

    assert teacher["topk_indices"].tolist() == [[1, 2]]
    assert teacher["teacher_topk_scores"].tolist() == [[34.0, 8.0]]
    assert teacher["teacher_argmax"].tolist() == [0]
    assert _teacher_version("mortal-discard-q") == MORTAL_DISCARD_TEACHER_CONTRACT_VERSION


def test_tempered_ratio_pilot_mortal_action_q_teacher_scores_non_discard_actions() -> None:
    batch = _mortal_action_batch(
        extras={
            MORTAL_Q_VALUES_EXTRA_KEY: _q_values(),
            MORTAL_ACTION_MASK_EXTRA_KEY: _mortal_mask(0, 37, 45),
        }
    )

    teacher = _topk_teacher_context(
        batch,
        prior_logits=batch.policy_input.prior_logits,
        topk=3,
        teacher_source="mortal-action-q",
        teacher_temperature=1.0,
    )

    assert teacher["topk_indices"].tolist() == [[0, 1, 2]]
    assert teacher["teacher_topk_scores"].tolist() == [[0.0, 37.0, 45.0]]
    assert teacher["teacher_argmax"].tolist() == [2]
    assert _teacher_version("mortal-action-q") == MORTAL_ACTION_TEACHER_CONTRACT_VERSION


def test_tempered_ratio_pilot_mortal_action_q_teacher_keeps_fixed_topk_with_short_rows() -> None:
    batch = _mortal_action_batch_with_short_row(
        extras={
            MORTAL_Q_VALUES_EXTRA_KEY: _q_values().repeat(2, 1),
            MORTAL_ACTION_MASK_EXTRA_KEY: _mortal_mask(0, 37, 45).repeat(2, 1),
        }
    )

    teacher = _topk_teacher_context(
        batch,
        prior_logits=batch.policy_input.prior_logits,
        topk=3,
        teacher_source="mortal-action-q",
        teacher_temperature=1.0,
    )

    assert teacher["topk_indices"].shape == (2, 3)
    assert teacher["teacher_topk_scores"][0].tolist() == [0.0, 37.0, 45.0]
    assert teacher["teacher_row_valid_mask"].tolist() == [True, False]


def test_tempered_ratio_pilot_mortal_action_q_topk_dedupes_duplicate_reach_sources() -> None:
    q_values = torch.zeros((1, MORTAL_ACTION_SPACE), dtype=torch.float32)
    q_values[0, 0] = 5.0
    q_values[0, 37] = 1.0
    q_values[0, 45] = 0.5
    batch = _mortal_action_batch_with_duplicate_reach_sources(
        extras={
            MORTAL_Q_VALUES_EXTRA_KEY: q_values,
            MORTAL_ACTION_MASK_EXTRA_KEY: _mortal_mask(0, 37, 45),
        }
    )

    teacher = _topk_teacher_context(
        batch,
        prior_logits=batch.policy_input.prior_logits,
        topk=3,
        teacher_source="mortal-action-q",
        teacher_temperature=1.0,
    )

    assert teacher["topk_indices"].tolist() == [[0, 2, 3]]
    assert teacher["teacher_topk_scores"].tolist() == [[1.0, 5.0, 0.5]]
    assert teacher["teacher_argmax"].tolist() == [1]
    assert teacher["teacher_row_valid_mask"].tolist() == [True]


def test_tempered_ratio_pilot_support_topk_dedupes_duplicate_reach_sources() -> None:
    batch = _mortal_action_batch_with_duplicate_reach_sources()

    support = _delta_support_mask(
        batch.policy_input,
        prior_logits=batch.policy_input.prior_logits,
        support_mode="topk",
        topk=3,
        margin_threshold=0.75,
    )

    assert support.tolist() == [[True, False, True, True]]


def test_tempered_ratio_pilot_reach_probe_counts_mortal_action_teacher_decisions() -> None:
    q_values = torch.zeros((1, MORTAL_ACTION_SPACE), dtype=torch.float32)
    q_values[0, 37] = 5.0
    batch = _mortal_action_batch(
        extras={
            MORTAL_Q_VALUES_EXTRA_KEY: q_values,
            MORTAL_ACTION_MASK_EXTRA_KEY: _mortal_mask(0, 37, 45),
        }
    )
    policy = _StaticPolicy(
        action_logits=torch.tensor([[0.0, 4.0, -1.0]], dtype=torch.float32),
        prior_logits=torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32),
    )

    stats = _reach_probe_stats(
        policy,
        batch,
        teacher_source="mortal-action-q",
        teacher_temperature=1.0,
    )

    assert stats["reach_opportunity_count"] == 1
    assert stats["prior_reach_rate"] == 0.0
    assert stats["policy_reach_rate"] == 1.0
    assert stats["selected_reach_rate"] == 1.0
    assert stats["reach_added_rate"] == 1.0
    assert stats["teacher_reach_available_count"] == 1
    assert stats["teacher_reach_rate"] == 1.0
    assert stats["teacher_policy_agree_rate"] == 1.0


def test_tempered_ratio_pilot_full_action_probe_counts_groups() -> None:
    q_values = torch.zeros((1, MORTAL_ACTION_SPACE), dtype=torch.float32)
    q_values[0, 45] = 5.0
    batch = _mortal_action_batch(
        extras={
            MORTAL_Q_VALUES_EXTRA_KEY: q_values,
            MORTAL_ACTION_MASK_EXTRA_KEY: _mortal_mask(0, 37, 45),
        }
    )
    policy = _StaticPolicy(
        action_logits=torch.tensor([[0.0, 1.0, 4.0]], dtype=torch.float32),
        prior_logits=torch.tensor([[3.0, 2.0, 1.0]], dtype=torch.float32),
    )

    stats = _full_action_probe_stats(
        policy,
        batch,
        teacher_source="mortal-action-q",
    )

    assert stats["non_discard_opportunity_count"] == 1
    assert stats["pass_opportunity_count"] == 1
    assert stats["response_opportunity_count"] == 1
    assert stats["pass_policy_rate"] == 1.0
    assert stats["pass_prior_rate"] == 0.0
    assert stats["pass_teacher_rate"] == 1.0
    assert stats["teacher_policy_agree_rate"] == 1.0


def test_tempered_ratio_pilot_terminal_coverage_marks_all_ryukyoku_unqualified() -> None:
    episode = SimpleNamespace(
        steps=(
            SimpleNamespace(
                action_spec=ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
                legal_actions=(
                    ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
                    ActionSpec(ActionType.PASS),
                ),
            ),
        ),
        initial_scores=(25000, 25000, 25000, 25000),
        scores=(25000, 25000, 25000, 25000),
    )

    stats = _terminal_coverage_stats((episode,))
    args = SimpleNamespace(
        terminal_coverage_gate=True,
        terminal_coverage_outcome_gate=False,
        terminal_coverage_min_score_changed_episode_rate=0.1,
        terminal_coverage_min_legal_terminal_rows=0,
        terminal_coverage_min_legal_agari_rows=1,
        terminal_coverage_min_prepared_legal_terminal_rows=0,
        terminal_coverage_min_prepared_legal_agari_rows=0,
        terminal_coverage_min_selected_agari_count=1,
        terminal_coverage_min_selected_agari_episode_rate=0.1,
    )

    assert stats["terminal_coverage_score_changed_episode_count"] == 0
    assert stats["terminal_coverage_legal_agari_row_count"] == 0
    assert stats["terminal_coverage_selected_agari_count"] == 0
    assert not _terminal_coverage_gate_pass(stats, args)


def test_tempered_ratio_pilot_terminal_coverage_does_not_gate_on_outcomes_by_default() -> None:
    episode = SimpleNamespace(
        steps=(
            SimpleNamespace(
                action_spec=ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
                legal_actions=(
                    ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"]),
                    ActionSpec(ActionType.TSUMO),
                ),
            ),
        ),
        initial_scores=(25000, 25000, 25000, 25000),
        scores=(25000, 25000, 25000, 25000),
    )

    stats = _terminal_coverage_stats((episode,))
    args = SimpleNamespace(
        terminal_coverage_gate=True,
        terminal_coverage_outcome_gate=False,
        terminal_coverage_min_score_changed_episode_rate=1.0,
        terminal_coverage_min_legal_terminal_rows=1,
        terminal_coverage_min_legal_agari_rows=1,
        terminal_coverage_min_prepared_legal_terminal_rows=0,
        terminal_coverage_min_prepared_legal_agari_rows=0,
        terminal_coverage_min_selected_agari_count=99,
        terminal_coverage_min_selected_agari_episode_rate=1.0,
    )

    assert stats["terminal_coverage_legal_agari_row_count"] == 1
    assert stats["terminal_coverage_selected_agari_count"] == 0
    assert _terminal_coverage_gate_pass(stats, args)


def test_tempered_ratio_pilot_terminal_coverage_counts_ron_rows() -> None:
    episode = SimpleNamespace(
        steps=(
            SimpleNamespace(
                action_spec=ActionSpec(ActionType.RON, from_who=1),
                legal_actions=(
                    ActionSpec(ActionType.PASS),
                    ActionSpec(ActionType.RON, from_who=1),
                ),
            ),
        ),
        initial_scores=(25000, 25000, 25000, 25000),
        scores=(32000, 18000, 25000, 25000),
    )

    stats = _terminal_coverage_stats((episode,))

    assert stats["terminal_coverage_score_changed_episode_count"] == 1
    assert stats["terminal_coverage_selected_ron_count"] == 1
    assert stats["terminal_coverage_selected_agari_episode_rate"] == 1.0
    assert stats["terminal_coverage_legal_agari_row_count"] == 1


def test_tempered_ratio_pilot_mortal_discard_q_teacher_fails_without_extras() -> None:
    batch = _mortal_batch()

    with pytest.raises(MortalTeacherMappingError, match="missing required keys"):
        _topk_teacher_context(
            batch,
            prior_logits=batch.policy_input.prior_logits,
            topk=2,
            teacher_source="mortal-discard-q",
            teacher_temperature=1.0,
        )


def test_tempered_ratio_pilot_does_not_register_xmodel_or_keqingv4_teacher_sources() -> None:
    sources = set(_TEACHER_SOURCES)

    assert "mortal-discard-q" in sources
    assert "mortal-action-q" in sources
    assert "xmodel" not in sources
    assert "xmodel1" not in sources
    assert "keqingv4" not in sources


def test_tempered_ratio_pilot_requires_checkpoint_for_mortal_teacher_source() -> None:
    configs = ({"mode": "teacher-ce", "teacher_source": "mortal-action-q"},)
    args = SimpleNamespace(
        mortal_teacher_checkpoint=None,
        mortal_root=Path("third_party/Mortal"),
        mortal_teacher_device=None,
    )

    assert _requires_mortal_teacher_runtime(configs)
    with pytest.raises(ValueError, match="--mortal-teacher-checkpoint"):
        _load_mortal_teacher_runtime_for_args(args, configs, device=torch.device("cpu"))


def test_tempered_ratio_pilot_loads_mortal_teacher_runtime_once(monkeypatch) -> None:
    runtime = object()
    calls: list[tuple[Path, Path, str]] = []

    def _fake_load(checkpoint_path: Path, *, mortal_root: Path, device: str):
        calls.append((checkpoint_path, mortal_root, device))
        return runtime

    monkeypatch.setattr(
        "scripts.run_keqingrl_tempered_ratio_pilot.load_mortal_teacher_runtime",
        _fake_load,
    )
    args = SimpleNamespace(
        mortal_teacher_checkpoint=Path("artifacts/mortal_training/mortal.pth"),
        mortal_root=Path("third_party/Mortal"),
        mortal_teacher_device="cuda:0",
    )

    loaded = _load_mortal_teacher_runtime_for_args(
        args,
        ({"mode": "teacher-ce", "teacher_source": "mortal-discard-q"},),
        device=torch.device("cpu"),
    )

    assert loaded is runtime
    assert calls == [
        (
            Path("artifacts/mortal_training/mortal.pth"),
            Path("third_party/Mortal"),
            "cuda:0",
        )
    ]
    assert _mortal_runtime_for_teacher(runtime, teacher_source="mortal-discard-q") is runtime
    assert _mortal_runtime_for_teacher(runtime, teacher_source="mortal-action-q") is runtime
    assert _mortal_runtime_for_teacher(runtime, teacher_source="rule-prior-topk") is None


def test_tempered_ratio_pilot_does_not_load_mortal_runtime_for_controls(monkeypatch) -> None:
    def _fail_load(*args, **kwargs):
        raise AssertionError("Mortal runtime should not load for non-Mortal teacher configs")

    monkeypatch.setattr(
        "scripts.run_keqingrl_tempered_ratio_pilot.load_mortal_teacher_runtime",
        _fail_load,
    )
    args = SimpleNamespace(
        mortal_teacher_checkpoint=None,
        mortal_root=Path("third_party/Mortal"),
        mortal_teacher_device=None,
    )

    loaded = _load_mortal_teacher_runtime_for_args(
        args,
        (
            {"mode": "none", "teacher_source": "none"},
            {"mode": "teacher-ce", "teacher_source": "rule-prior-topk"},
        ),
        device=torch.device("cpu"),
    )

    assert loaded is None


def test_tempered_ratio_pilot_rollout_env_uses_configured_mortal_root() -> None:
    runtime = object()
    args = SimpleNamespace(
        max_kyokus=1,
        mortal_root=Path("custom/Mortal"),
        self_turn_action_types=("DISCARD",),
        response_action_types=(),
        mortal_teacher_strict_extra_mask=True,
    )

    env = _rollout_env(args, mortal_teacher_runtime=runtime)

    assert env.mortal_teacher_runtime is runtime
    assert env.mortal_observation_bridge.mortal_root == Path("custom/Mortal")


def test_tempered_ratio_pilot_rollout_env_uses_configured_action_scope() -> None:
    args = SimpleNamespace(
        max_kyokus=1,
        mortal_root=Path("third_party/Mortal"),
        self_turn_action_types=("DISCARD", "REACH_DISCARD"),
        response_action_types=("PASS",),
        forced_autopilot_action_types=(),
        mortal_teacher_strict_extra_mask=False,
    )

    env = _rollout_env(args)

    assert env.self_turn_action_types == (ActionType.DISCARD, ActionType.REACH_DISCARD)
    assert env.response_action_types == (ActionType.PASS,)
    assert env.forced_autopilot_action_types == ()
    assert env.mortal_teacher_strict_extra_mask is False
