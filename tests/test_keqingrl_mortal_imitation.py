from __future__ import annotations

import json
import csv
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from keqingrl.actions import ActionSpec, ActionType
from keqingrl.contracts import ObsTensorBatch, PolicyInput, PolicyOutput
from keqingrl.mortal_teacher import (
    MORTAL_ACTION_MASK_EXTRA_KEY,
    MORTAL_ACTION_SPACE,
    MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY,
    MORTAL_ENCODED_OBS_EXTRA_KEY,
    MORTAL_KAN_ACTION_ID,
    MORTAL_PASS_ACTION_ID,
    MORTAL_Q_VALUES_EXTRA_KEY,
)
from mahjong_env.action_space import TILE_NAME_TO_IDX
from scripts.run_keqingrl_mortal_imitation import (
    ALLOWED_IMITATION_TEACHER_SOURCES,
    _ensure_mortal_encoded_observation_extras,
    _ensure_mortal_teacher_q_extras,
    _latest_checkpoint_rows,
    _load_imitation_candidates,
    _save_imitation_checkpoint,
    _validate_args,
    audit_mortal_action_mapping,
    mortal_imitation_loss,
    prepare_mortal_imitation_teacher_data,
)


def _tile(name: str) -> int:
    return int(TILE_NAME_TO_IDX[name])


def _q_values() -> torch.Tensor:
    return torch.arange(MORTAL_ACTION_SPACE, dtype=torch.float32).unsqueeze(0)


def _mask(*ids: int) -> torch.BoolTensor:
    mask = torch.zeros((1, MORTAL_ACTION_SPACE), dtype=torch.bool)
    mask[0, list(ids)] = True
    return mask


def _policy_input(
    legal_actions: tuple[ActionSpec, ...],
    *,
    mask_ids: tuple[int, ...],
    legal_mask: torch.BoolTensor | None = None,
    prior_logits: torch.Tensor | None = None,
) -> PolicyInput:
    action_count = max(len(legal_actions), 3)
    if legal_mask is None:
        legal_mask = torch.zeros((1, action_count), dtype=torch.bool)
        legal_mask[0, : len(legal_actions)] = True
    if prior_logits is None:
        prior_logits = torch.arange(action_count, 0, -1, dtype=torch.float32).unsqueeze(0)
    return PolicyInput(
        obs=ObsTensorBatch(
            tile_obs=torch.zeros((1, 1), dtype=torch.float32),
            scalar_obs=torch.zeros((1, 1), dtype=torch.float32),
            extras={
                MORTAL_Q_VALUES_EXTRA_KEY: _q_values(),
                MORTAL_ACTION_MASK_EXTRA_KEY: _mask(*mask_ids),
            },
        ),
        legal_action_ids=torch.zeros((1, action_count), dtype=torch.long),
        legal_action_features=torch.zeros((1, action_count, 8), dtype=torch.float32),
        legal_action_mask=legal_mask,
        rule_context=torch.zeros((1, 1), dtype=torch.float32),
        prior_logits=prior_logits,
        legal_actions=(legal_actions,),
    )


def _output(logits: list[float]) -> PolicyOutput:
    return PolicyOutput(
        action_logits=torch.tensor([logits], dtype=torch.float32),
        value=torch.zeros((1,), dtype=torch.float32),
        rank_logits=torch.zeros((1, 4), dtype=torch.float32),
        aux={"final_logits": torch.tensor([logits], dtype=torch.float32)},
    )


class _EchoDeferredMortalRuntime:
    def __init__(self) -> None:
        self.obs_shape: tuple[int, ...] | None = None
        self.mask_shape: tuple[int, ...] | None = None
        self.calls: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    def evaluate(self, obs: torch.Tensor, masks: torch.Tensor):
        self.obs_shape = tuple(obs.shape)
        self.mask_shape = tuple(masks.shape)
        self.calls.append((tuple(obs.shape), tuple(masks.shape)))
        batch_size = int(obs.shape[0])
        return SimpleNamespace(
            q_values=torch.arange(MORTAL_ACTION_SPACE, dtype=torch.float32).unsqueeze(0).repeat(batch_size, 1),
            action_mask=masks.bool(),
        )


class _FakeMortalObservationBridge:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[dict[str, object], ...], int]] = []
        self.reset_count = 0

    def reset_cache(self) -> None:
        self.reset_count += 1

    def encode_from_events(self, events: tuple[dict[str, object], ...], actor: int):
        self.calls.append((events, int(actor)))
        mask = torch.zeros((MORTAL_ACTION_SPACE,), dtype=torch.bool)
        mask[MORTAL_PASS_ACTION_ID] = True
        return SimpleNamespace(
            obs=torch.full((1012, 34), float(actor), dtype=torch.float32),
            action_mask=mask,
        )


def test_mortal_imitation_only_allows_mortal_action_q_teacher() -> None:
    assert ALLOWED_IMITATION_TEACHER_SOURCES == ("mortal-action-q",)
    args = SimpleNamespace(
        teacher_source="xmodel",
        mortal_teacher_checkpoint=Path("artifacts/mortal_training/mortal.pth"),
        teacher_temperature=1.0,
        teacher_topk=3,
        episodes=1,
        iterations=1,
        update_epochs=1,
    )

    with pytest.raises(ValueError, match="unsupported imitation teacher source"):
        _validate_args(args)


def test_mortal_imitation_requires_teacher_checkpoint() -> None:
    args = SimpleNamespace(
        teacher_source="mortal-action-q",
        mortal_teacher_checkpoint=None,
        teacher_temperature=1.0,
        teacher_topk=3,
        episodes=1,
        iterations=1,
        update_epochs=1,
    )

    with pytest.raises(ValueError, match="mortal-teacher-checkpoint"):
        _validate_args(args)


def test_mortal_imitation_audit_fails_closed_on_missing_controlled_legal_action() -> None:
    policy_input = _policy_input((ActionSpec(ActionType.PASS),), mask_ids=())

    result = audit_mortal_action_mapping(policy_input, strict_extra=False)

    assert result.summary["fail_closed_count"] == 1
    assert result.summary["missing_legal_count"] == 1
    assert result.audit_rows[0]["mismatch_kind"] == "missing_legal"


def test_mortal_imitation_extra_kan_is_audit_only_when_extra_mask_not_strict() -> None:
    policy_input = _policy_input(
        (ActionSpec(ActionType.PASS),),
        mask_ids=(MORTAL_PASS_ACTION_ID, MORTAL_KAN_ACTION_ID),
    )

    result = audit_mortal_action_mapping(policy_input, strict_extra=False)

    assert result.summary["fail_closed_count"] == 0
    assert result.summary["mapping_available_count"] == 1
    assert result.summary["extra_mortal_count"] == 1
    assert json.loads(str(result.audit_rows[0]["extra_mortal_action_ids_json"])) == [MORTAL_KAN_ACTION_ID]


def test_mortal_imitation_extra_kan_fails_closed_when_extra_mask_strict() -> None:
    policy_input = _policy_input(
        (ActionSpec(ActionType.PASS),),
        mask_ids=(MORTAL_PASS_ACTION_ID, MORTAL_KAN_ACTION_ID),
    )

    result = audit_mortal_action_mapping(policy_input, strict_extra=True)

    assert result.summary["fail_closed_count"] == 1
    assert result.summary["extra_mortal_count"] == 1


def test_mortal_imitation_loss_ignores_illegal_logits() -> None:
    legal_actions = (
        ActionSpec(ActionType.DISCARD, tile=_tile("1m")),
        ActionSpec(ActionType.PASS),
    )
    legal_mask = torch.tensor([[True, True, False]], dtype=torch.bool)
    policy_input = _policy_input(
        legal_actions,
        mask_ids=(0, MORTAL_PASS_ACTION_ID),
        legal_mask=legal_mask,
        prior_logits=torch.tensor([[2.0, 1.0, 100.0]], dtype=torch.float32),
    )

    low_illegal = mortal_imitation_loss(
        _output([0.0, 1.0, -100.0]),
        policy_input,
        teacher_support="topk",
        teacher_topk=2,
        teacher_temperature=1.0,
        strict_extra=False,
    )
    high_illegal = mortal_imitation_loss(
        _output([0.0, 1.0, 100.0]),
        policy_input,
        teacher_support="topk",
        teacher_topk=2,
        teacher_temperature=1.0,
        strict_extra=False,
    )

    assert torch.isclose(low_illegal.teacher_ce, high_illegal.teacher_ce)


def test_deferred_mortal_observation_extras_are_materialized_after_rollout() -> None:
    policy_input = PolicyInput(
        obs=ObsTensorBatch(
            tile_obs=torch.zeros((2, 1), dtype=torch.float32),
            scalar_obs=torch.zeros((2, 1), dtype=torch.float32),
            extras={},
        ),
        legal_action_ids=torch.zeros((2, 1), dtype=torch.long),
        legal_action_features=torch.zeros((2, 1, 8), dtype=torch.float32),
        legal_action_mask=torch.ones((2, 1), dtype=torch.bool),
        rule_context=torch.zeros((2, 1), dtype=torch.float32),
        legal_actions=((ActionSpec(ActionType.PASS),), (ActionSpec(ActionType.PASS),)),
    )
    prepared_steps = (
        SimpleNamespace(
            actor=0,
            episode_id="episode-a",
            mortal_teacher_events=({"type": "start_game"}, {"type": "tsumo", "actor": 0}),
        ),
        SimpleNamespace(
            actor=0,
            episode_id="episode-a",
            mortal_teacher_events=(
                {"type": "start_game"},
                {"type": "tsumo", "actor": 0},
                {"type": "dahai", "actor": 0},
            ),
        ),
    )
    bridge = _FakeMortalObservationBridge()

    materialized = _ensure_mortal_encoded_observation_extras(
        policy_input,
        prepared_steps=prepared_steps,
        bridge=bridge,
    )

    assert bridge.reset_count == 1
    assert len(bridge.calls) == 2
    assert set(materialized.obs.extras) == {
        MORTAL_ENCODED_OBS_EXTRA_KEY,
        MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY,
    }
    assert materialized.obs.extras[MORTAL_ENCODED_OBS_EXTRA_KEY].shape == (2, 1012, 34)
    assert materialized.obs.extras[MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY].shape == (2, MORTAL_ACTION_SPACE)


def test_deferred_mortal_teacher_q_extras_are_materialized_in_one_batch() -> None:
    encoded_mask = torch.zeros((2, MORTAL_ACTION_SPACE), dtype=torch.bool)
    encoded_mask[:, MORTAL_PASS_ACTION_ID] = True
    policy_input = PolicyInput(
        obs=ObsTensorBatch(
            tile_obs=torch.zeros((2, 1), dtype=torch.float32),
            scalar_obs=torch.zeros((2, 1), dtype=torch.float32),
            extras={
                MORTAL_ENCODED_OBS_EXTRA_KEY: torch.zeros((2, 1012, 34), dtype=torch.float32),
                MORTAL_ENCODED_ACTION_MASK_EXTRA_KEY: encoded_mask,
            },
        ),
        legal_action_ids=torch.zeros((2, 1), dtype=torch.long),
        legal_action_features=torch.zeros((2, 1, 8), dtype=torch.float32),
        legal_action_mask=torch.ones((2, 1), dtype=torch.bool),
        rule_context=torch.zeros((2, 1), dtype=torch.float32),
        legal_actions=((ActionSpec(ActionType.PASS),), (ActionSpec(ActionType.PASS),)),
    )
    runtime = _EchoDeferredMortalRuntime()

    materialized = _ensure_mortal_teacher_q_extras(policy_input, teacher_runtime=runtime, eval_batch_size=1)

    assert runtime.calls == [
        ((1, 1012, 34), (1, MORTAL_ACTION_SPACE)),
        ((1, 1012, 34), (1, MORTAL_ACTION_SPACE)),
    ]
    assert runtime.obs_shape == (1, 1012, 34)
    assert runtime.mask_shape == (1, MORTAL_ACTION_SPACE)
    assert set(materialized.obs.extras) == {MORTAL_Q_VALUES_EXTRA_KEY, MORTAL_ACTION_MASK_EXTRA_KEY}
    assert materialized.obs.extras[MORTAL_Q_VALUES_EXTRA_KEY].shape == (2, MORTAL_ACTION_SPACE)
    assert materialized.obs.extras[MORTAL_ACTION_MASK_EXTRA_KEY].shape == (2, MORTAL_ACTION_SPACE)
    assert materialized.legal_actions == policy_input.legal_actions


def test_cached_mortal_imitation_topk_loss_matches_uncached_loss() -> None:
    legal_actions = (
        ActionSpec(ActionType.DISCARD, tile=_tile("1m")),
        ActionSpec(ActionType.PASS),
    )
    policy_input = _policy_input(
        legal_actions,
        mask_ids=(0, MORTAL_PASS_ACTION_ID),
        prior_logits=torch.tensor([[2.0, 1.0, -100.0]], dtype=torch.float32),
    )
    output = _output([0.5, 1.5, 100.0])
    uncached = mortal_imitation_loss(
        output,
        policy_input,
        teacher_support="topk",
        teacher_topk=2,
        teacher_temperature=1.0,
        strict_extra=False,
    )
    teacher_data = prepare_mortal_imitation_teacher_data(
        policy_input,
        strict_extra=False,
        teacher_support="topk",
        teacher_topk=2,
    )
    cached = mortal_imitation_loss(
        output,
        policy_input,
        teacher_support="topk",
        teacher_topk=2,
        teacher_temperature=1.0,
        strict_extra=False,
        teacher_batch=teacher_data.teacher_batch,
    )

    assert torch.isclose(cached.teacher_ce, uncached.teacher_ce)
    assert torch.isclose(cached.teacher_kl, uncached.teacher_kl)
    assert teacher_data.summary["mapping_available_rate"] == 1.0


def test_mortal_imitation_full_legal_loss_uses_mapped_legal_actions() -> None:
    legal_actions = (
        ActionSpec(ActionType.DISCARD, tile=_tile("1m")),
        ActionSpec(ActionType.PASS),
    )
    policy_input = _policy_input(legal_actions, mask_ids=(0, MORTAL_PASS_ACTION_ID))

    loss = mortal_imitation_loss(
        _output([0.0, 2.0, 0.0]),
        policy_input,
        teacher_support="full-legal",
        teacher_topk=2,
        teacher_temperature=1.0,
        strict_extra=False,
    )

    assert loss.teacher_ce.item() >= 0.0
    assert loss.teacher_policy_agreement.item() == 1.0


def test_cached_mortal_imitation_full_legal_loss_ignores_illegal_logits() -> None:
    legal_actions = (
        ActionSpec(ActionType.DISCARD, tile=_tile("1m")),
        ActionSpec(ActionType.PASS),
    )
    legal_mask = torch.tensor([[True, True, False]], dtype=torch.bool)
    policy_input = _policy_input(
        legal_actions,
        mask_ids=(0, MORTAL_PASS_ACTION_ID),
        legal_mask=legal_mask,
        prior_logits=torch.tensor([[2.0, 1.0, 100.0]], dtype=torch.float32),
    )
    teacher_data = prepare_mortal_imitation_teacher_data(
        policy_input,
        strict_extra=False,
        teacher_support="full-legal",
        teacher_topk=2,
    )

    low_illegal = mortal_imitation_loss(
        _output([0.0, 1.0, -100.0]),
        policy_input,
        teacher_support="full-legal",
        teacher_topk=2,
        teacher_temperature=1.0,
        strict_extra=False,
        teacher_batch=teacher_data.teacher_batch,
    )
    high_illegal = mortal_imitation_loss(
        _output([0.0, 1.0, 100.0]),
        policy_input,
        teacher_support="full-legal",
        teacher_topk=2,
        teacher_temperature=1.0,
        strict_extra=False,
        teacher_batch=teacher_data.teacher_batch,
    )

    assert torch.isclose(low_illegal.teacher_ce, high_illegal.teacher_ce)


def test_mortal_imitation_checkpoint_summary_contains_teacher_metadata(tmp_path: Path) -> None:
    source_config = {
        "model": {"hidden_dim": 8, "num_res_blocks": 1, "dropout": 0.0},
        "config_key": {"opponent_mode": "rulebase"},
    }
    config_path = tmp_path / "source_config.json"
    config_path.write_text(json.dumps(source_config), encoding="utf-8")
    source_checkpoint = tmp_path / "source_policy.pt"
    torch.save({"policy_state_dict": {}, "rule_score_scale": 0.25}, source_checkpoint)
    args = SimpleNamespace(
        output_dir=tmp_path / "out",
        self_turn_action_types=("DISCARD", "REACH_DISCARD", "TSUMO", "RYUKYOKU"),
        response_action_types=("PASS", "RON", "PON", "CHI"),
        forced_autopilot_action_types=("TSUMO", "RON", "RYUKYOKU"),
        rule_score_scale=0.25,
        gamma=1.0,
        gae_lambda=0.95,
        episodes=1,
        iterations=1,
        lr=0.001,
        update_epochs=1,
        teacher_source="mortal-action-q",
        teacher_support="topk",
        teacher_topk=3,
        teacher_temperature=1.0,
        mortal_teacher_checkpoint=Path("artifacts/mortal_training/mortal.pth"),
        mortal_teacher_strict_extra_mask=False,
        support_policy_mode="support-only-topk",
        delta_support_mode="topk",
        delta_support_topk=3,
        delta_support_margin_threshold=0.75,
        outside_support_delta_mode="zero",
    )
    candidate = {
        "source_config_id": 93,
        "rerun_config_id": 0,
        "checkpoint_path": str(source_checkpoint),
        "checkpoint_sha256": "source-sha",
        "config_path": str(config_path),
    }
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    summary_row = {
        "top1_changed_vs_parent_rate": 0.1,
        "teacher_ce": 0.2,
        "teacher_kl": 0.3,
        "teacher_policy_agreement": 0.4,
        "mapping_row_count": 1,
        "mapping_available_count": 1,
        "fail_closed_count": 0,
    }

    row = _save_imitation_checkpoint(
        args,
        candidate,
        model,
        optimizer,
        config_id=0,
        iteration=0,
        summary_row=summary_row,
    )

    assert Path(row["checkpoint_path"]).exists()
    checkpoint = torch.load(row["checkpoint_path"], map_location="cpu")
    assert checkpoint["contract_metadata"]["teacher_source"] == "mortal-action-q"
    assert checkpoint["contract_metadata"]["teacher_target_type"] == "topk_distribution"
    assert row["checkpoint_sha256"]
    assert Path(row["checkpoint_path"]).name == "policy_iter_0001.pt"


def test_latest_checkpoint_rows_keeps_only_latest_per_config() -> None:
    rows = [
        {"config_id": 0, "checkpoint_path": "old0"},
        {"config_id": 1, "checkpoint_path": "old1"},
        {"config_id": 0, "checkpoint_path": "new0"},
    ]

    latest = _latest_checkpoint_rows(rows)

    assert latest == [
        {"config_id": 0, "checkpoint_path": "new0"},
        {"config_id": 1, "checkpoint_path": "old1"},
    ]


def test_imitation_checkpoints_do_not_overwrite_previous_iterations(tmp_path: Path) -> None:
    source_config = {
        "model": {"hidden_dim": 8, "num_res_blocks": 1, "dropout": 0.0},
        "config_key": {"opponent_mode": "rulebase"},
    }
    config_path = tmp_path / "source_config.json"
    config_path.write_text(json.dumps(source_config), encoding="utf-8")
    source_checkpoint = tmp_path / "source_policy.pt"
    torch.save({"policy_state_dict": {}, "rule_score_scale": 0.25}, source_checkpoint)
    args = SimpleNamespace(
        output_dir=tmp_path / "out",
        self_turn_action_types=("DISCARD", "REACH_DISCARD", "TSUMO", "RYUKYOKU"),
        response_action_types=("PASS", "RON", "PON", "CHI"),
        forced_autopilot_action_types=("TSUMO", "RON", "RYUKYOKU"),
        rule_score_scale=0.25,
        gamma=1.0,
        gae_lambda=0.95,
        episodes=1,
        iterations=2,
        lr=0.001,
        update_epochs=1,
        teacher_source="mortal-action-q",
        teacher_support="topk",
        teacher_topk=3,
        teacher_temperature=1.0,
        mortal_teacher_checkpoint=Path("artifacts/mortal_training/mortal.pth"),
        mortal_teacher_strict_extra_mask=False,
        support_policy_mode="support-only-topk",
        delta_support_mode="topk",
        delta_support_topk=3,
        delta_support_margin_threshold=0.75,
        outside_support_delta_mode="zero",
    )
    candidate = {
        "source_config_id": 93,
        "rerun_config_id": 0,
        "checkpoint_path": str(source_checkpoint),
        "checkpoint_sha256": "source-sha",
        "config_path": str(config_path),
    }
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    summary_row = {
        "top1_changed_vs_parent_rate": 0.1,
        "teacher_ce": 0.2,
        "teacher_kl": 0.3,
        "teacher_policy_agreement": 0.4,
        "mapping_row_count": 1,
        "mapping_available_count": 1,
        "fail_closed_count": 0,
    }

    first = _save_imitation_checkpoint(
        args,
        candidate,
        model,
        optimizer,
        config_id=0,
        iteration=0,
        summary_row=summary_row,
    )
    second = _save_imitation_checkpoint(
        args,
        candidate,
        model,
        optimizer,
        config_id=0,
        iteration=1,
        summary_row=summary_row,
    )

    assert first["checkpoint_path"] != second["checkpoint_path"]
    assert Path(first["checkpoint_path"]).exists()
    assert Path(second["checkpoint_path"]).exists()


def test_load_imitation_candidates_dedupes_stale_duplicate_checkpoint_rows(tmp_path: Path) -> None:
    summary_path = tmp_path / "checkpoint_summary.csv"
    fieldnames = [
        "source_config_id",
        "rerun_config_id",
        "checkpoint_path",
        "checkpoint_sha256",
        "config_path",
        "opponent_mode",
    ]
    rows = [
        {
            "source_config_id": "93",
            "rerun_config_id": "0",
            "checkpoint_path": "same.pt",
            "checkpoint_sha256": "old-sha",
            "config_path": "same.json",
            "opponent_mode": "rule_prior_greedy",
        },
        {
            "source_config_id": "93",
            "rerun_config_id": "0",
            "checkpoint_path": "same.pt",
            "checkpoint_sha256": "new-sha",
            "config_path": "same.json",
            "opponent_mode": "rule_prior_greedy",
        },
    ]
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    args = SimpleNamespace(
        candidate_summary=summary_path,
        source_config_ids=(93,),
        rerun_config_ids=(0,),
    )

    candidates = _load_imitation_candidates(args)

    assert len(candidates) == 1
    assert candidates[0]["checkpoint_sha256"] == "new-sha"
