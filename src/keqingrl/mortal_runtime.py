"""Runtime loading for trained Mortal teacher checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Sequence

import numpy as np
import torch

from keqingrl.mortal_teacher import MORTAL_ACTION_MASK_EXTRA_KEY, MORTAL_ACTION_SPACE, MORTAL_Q_VALUES_EXTRA_KEY


class MortalRuntimeError(RuntimeError):
    """Raised when a trained Mortal teacher cannot be loaded or evaluated."""


@dataclass(frozen=True)
class MortalTeacherRuntimeOutput:
    q_values: torch.Tensor
    action_mask: torch.BoolTensor
    actions: torch.LongTensor
    is_greedy: torch.BoolTensor

    def extras(self) -> dict[str, torch.Tensor]:
        return {
            MORTAL_Q_VALUES_EXTRA_KEY: self.q_values,
            MORTAL_ACTION_MASK_EXTRA_KEY: self.action_mask,
        }


class MortalTeacherRuntime:
    """Thin wrapper around MortalEngine for already-Mortal-encoded tensors."""

    def __init__(self, engine, *, device: torch.device) -> None:
        self.engine = engine
        self.device = device

    def evaluate(
        self,
        obs: torch.Tensor | Sequence[object],
        masks: torch.Tensor | Sequence[object],
        invisible_obs: torch.Tensor | Sequence[object] | None = None,
    ) -> MortalTeacherRuntimeOutput:
        obs_list = _batch_to_numpy_list(obs, field_name="obs")
        mask_list = _batch_to_numpy_list(masks, field_name="masks")
        if len(obs_list) != len(mask_list):
            raise MortalRuntimeError(f"obs/masks batch size mismatch: {len(obs_list)} != {len(mask_list)}")
        invisible_list = None
        if invisible_obs is not None:
            invisible_list = _batch_to_numpy_list(invisible_obs, field_name="invisible_obs")
            if len(invisible_list) != len(obs_list):
                raise MortalRuntimeError(
                    f"obs/invisible_obs batch size mismatch: {len(obs_list)} != {len(invisible_list)}"
                )

        actions, q_values, returned_masks, is_greedy = self.engine.react_batch(obs_list, mask_list, invisible_list)
        q_tensor = torch.as_tensor(np.asarray(q_values), dtype=torch.float32)
        mask_tensor = torch.as_tensor(np.asarray(returned_masks), dtype=torch.bool)
        if q_tensor.ndim != 2 or int(q_tensor.shape[-1]) != MORTAL_ACTION_SPACE:
            raise MortalRuntimeError(f"Mortal q_values must have shape [B, {MORTAL_ACTION_SPACE}], got {tuple(q_tensor.shape)}")
        if mask_tensor.shape != q_tensor.shape:
            raise MortalRuntimeError(
                f"Mortal returned mask shape must match q_values: {tuple(mask_tensor.shape)} != {tuple(q_tensor.shape)}"
            )
        return MortalTeacherRuntimeOutput(
            q_values=q_tensor,
            action_mask=mask_tensor,
            actions=torch.as_tensor(actions, dtype=torch.long),
            is_greedy=torch.as_tensor(is_greedy, dtype=torch.bool),
        )


def load_mortal_teacher_runtime(
    checkpoint_path: Path,
    *,
    mortal_root: Path = Path("third_party/Mortal"),
    device: str | torch.device | None = None,
    enable_amp: bool = False,
    enable_rule_based_agari_guard: bool = True,
) -> MortalTeacherRuntime:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Mortal checkpoint does not exist: {checkpoint_path}")
    mortal_python_dir = (mortal_root / "mortal").resolve()
    _ensure_mortal_python_path(mortal_python_dir)

    from engine import MortalEngine  # noqa: PLC0415
    from model import Brain, DQN  # noqa: PLC0415

    resolved_device = torch.device("cpu" if device is None else device)
    state = torch.load(checkpoint_path, weights_only=True, map_location=torch.device("cpu"))
    version, conv_channels, num_blocks = _checkpoint_model_config(state)
    brain = Brain(version=version, conv_channels=conv_channels, num_blocks=num_blocks).eval()
    dqn = DQN(version=version).eval()
    brain.load_state_dict(state["mortal"])
    dqn.load_state_dict(state["current_dqn"])
    engine = MortalEngine(
        brain,
        dqn,
        is_oracle=False,
        version=version,
        device=resolved_device,
        enable_amp=bool(enable_amp),
        enable_rule_based_agari_guard=bool(enable_rule_based_agari_guard),
        name="mortal-teacher",
    )
    return MortalTeacherRuntime(engine, device=resolved_device)


def _checkpoint_model_config(state: object) -> tuple[int, int, int]:
    if not isinstance(state, dict):
        raise MortalRuntimeError("Mortal checkpoint must be a dict")
    missing = [key for key in ("mortal", "current_dqn", "config") if key not in state]
    if missing:
        raise MortalRuntimeError(f"Mortal checkpoint missing required keys: {missing}")
    config = state["config"]
    if not isinstance(config, dict):
        raise MortalRuntimeError("Mortal checkpoint config must be a dict")
    control = config.get("control", {})
    resnet = config.get("resnet", {})
    if not isinstance(control, dict) or not isinstance(resnet, dict):
        raise MortalRuntimeError("Mortal checkpoint config must contain control/resnet dicts")
    try:
        version = int(control.get("version", 4))
        conv_channels = int(resnet["conv_channels"])
        num_blocks = int(resnet["num_blocks"])
    except KeyError as exc:
        raise MortalRuntimeError(f"Mortal checkpoint config missing resnet key: {exc}") from exc
    return version, conv_channels, num_blocks


def _ensure_mortal_python_path(mortal_python_dir: Path) -> None:
    if not mortal_python_dir.exists():
        raise FileNotFoundError(f"Mortal python directory does not exist: {mortal_python_dir}")
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))


def _batch_to_numpy_list(value: torch.Tensor | Sequence[object], *, field_name: str) -> list[object]:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim < 2:
            raise MortalRuntimeError(f"{field_name} must be batched, got shape {tuple(tensor.shape)}")
        return [item.numpy() for item in tensor]
    result = list(value)
    if not result:
        raise MortalRuntimeError(f"{field_name} batch must not be empty")
    return result


__all__ = [
    "MortalRuntimeError",
    "MortalTeacherRuntime",
    "MortalTeacherRuntimeOutput",
    "load_mortal_teacher_runtime",
]
