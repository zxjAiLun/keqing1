from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from keqingrl.actions import ActionSpec, ActionType
from keqingrl.contracts import ObsTensorBatch, PolicyInput
from keqingrl.mortal_teacher import (
    MORTAL_ACTION_MASK_EXTRA_KEY,
    MORTAL_ACTION_SPACE,
    MORTAL_DISCARD_TEACHER_CONTRACT_VERSION,
    MORTAL_Q_VALUES_EXTRA_KEY,
    MortalTeacherMappingError,
)
from mahjong_env.action_space import TILE_NAME_TO_IDX
from scripts.run_keqingrl_tempered_ratio_pilot import (
    _TEACHER_SOURCES,
    _load_mortal_teacher_runtime_for_args,
    _mortal_runtime_for_teacher,
    _requires_mortal_teacher_runtime,
    _rollout_env,
    _teacher_version,
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
    return SimpleNamespace(policy_input=policy_input)


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
    assert "xmodel" not in sources
    assert "xmodel1" not in sources
    assert "keqingv4" not in sources


def test_tempered_ratio_pilot_requires_checkpoint_for_mortal_teacher_source() -> None:
    configs = ({"mode": "teacher-ce", "teacher_source": "mortal-discard-q"},)
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
    )

    env = _rollout_env(args, mortal_teacher_runtime=runtime)

    assert env.mortal_teacher_runtime is runtime
    assert env.mortal_observation_bridge.mortal_root == Path("custom/Mortal")
