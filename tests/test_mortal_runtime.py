from __future__ import annotations

import sys
from pathlib import Path

import torch

from keqingrl.mortal_runtime import MortalRuntimeError, load_mortal_teacher_runtime
from keqingrl.mortal_teacher import MORTAL_ACTION_MASK_EXTRA_KEY, MORTAL_ACTION_SPACE, MORTAL_Q_VALUES_EXTRA_KEY


def _write_fake_mortal_modules(mortal_root: Path) -> None:
    mortal_dir = mortal_root / "mortal"
    mortal_dir.mkdir(parents=True)
    (mortal_dir / "model.py").write_text(
        """
import torch

class Brain(torch.nn.Module):
    def __init__(self, *, version, conv_channels, num_blocks):
        super().__init__()
        self.version = version

class DQN(torch.nn.Module):
    def __init__(self, *, version):
        super().__init__()
        self.version = version
""",
        encoding="utf-8",
    )
    (mortal_dir / "engine.py").write_text(
        f"""
class MortalEngine:
    def __init__(self, brain, dqn, is_oracle, version, device, enable_amp, enable_rule_based_agari_guard, name):
        self.version = version

    def react_batch(self, obs, masks, invisible_obs):
        batch_size = len(obs)
        q_values = [[float(i) for i in range({MORTAL_ACTION_SPACE})] for _ in range(batch_size)]
        return [0 for _ in range(batch_size)], q_values, masks, [True for _ in range(batch_size)]
""",
        encoding="utf-8",
    )


def test_load_mortal_teacher_runtime_uses_checkpoint_and_returns_extras(tmp_path: Path, monkeypatch) -> None:
    mortal_root = tmp_path / "Mortal"
    _write_fake_mortal_modules(mortal_root)
    monkeypatch.delitem(sys.modules, "model", raising=False)
    monkeypatch.delitem(sys.modules, "engine", raising=False)
    checkpoint_path = tmp_path / "mortal.pth"
    torch.save(
        {
            "mortal": {},
            "current_dqn": {},
            "config": {
                "control": {"version": 4},
                "resnet": {"conv_channels": 1, "num_blocks": 0},
            },
        },
        checkpoint_path,
    )

    runtime = load_mortal_teacher_runtime(checkpoint_path, mortal_root=mortal_root, device="cpu")
    output = runtime.evaluate(
        torch.zeros((2, 1012, 34), dtype=torch.float32),
        torch.ones((2, MORTAL_ACTION_SPACE), dtype=torch.bool),
    )

    assert output.q_values.shape == (2, MORTAL_ACTION_SPACE)
    assert output.action_mask.shape == (2, MORTAL_ACTION_SPACE)
    assert output.actions.tolist() == [0, 0]
    assert output.is_greedy.tolist() == [True, True]
    assert set(output.extras()) == {MORTAL_Q_VALUES_EXTRA_KEY, MORTAL_ACTION_MASK_EXTRA_KEY}


def test_load_mortal_teacher_runtime_rejects_missing_checkpoint_keys(tmp_path: Path, monkeypatch) -> None:
    mortal_root = tmp_path / "Mortal"
    _write_fake_mortal_modules(mortal_root)
    monkeypatch.delitem(sys.modules, "model", raising=False)
    monkeypatch.delitem(sys.modules, "engine", raising=False)
    checkpoint_path = tmp_path / "broken.pth"
    torch.save({"mortal": {}, "config": {}}, checkpoint_path)

    try:
        load_mortal_teacher_runtime(checkpoint_path, mortal_root=mortal_root)
    except MortalRuntimeError as exc:
        assert "missing required keys" in str(exc)
    else:
        raise AssertionError("expected MortalRuntimeError")
