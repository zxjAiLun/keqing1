"""Mortal observation encoding bridge for KeqingRL event logs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import torch

from keqingrl.mortal_teacher import MORTAL_ACTION_SPACE


MORTAL_OBSERVATION_BRIDGE_CONTRACT_VERSION = "mortal_observation_bridge_v1"

_MORTAL_UNSUPPORTED_ANNOUNCE_EVENTS = {
    # Keqing live battle emits this implementation detail before the dora event.
    # Mortal's PlayerState already consumed the preceding kakan event.
    "kakan_accepted",
}


class MortalObservationBridgeError(RuntimeError):
    """Raised when mjai events cannot be replayed into Mortal observation tensors."""


@dataclass(frozen=True)
class MortalEncodedObservation:
    obs: torch.Tensor
    action_mask: torch.BoolTensor
    replayed_event_count: int


class MortalObservationBridge:
    def __init__(
        self,
        *,
        mortal_root: Path = Path("third_party/Mortal"),
        version: int = 4,
        at_kan_select: bool = False,
    ) -> None:
        self.mortal_root = mortal_root
        self.version = int(version)
        self.at_kan_select = bool(at_kan_select)
        self._player_state_cls = None

    def encode_from_events(self, events: Sequence[dict[str, Any]], actor: int) -> MortalEncodedObservation:
        if not events:
            raise MortalObservationBridgeError("cannot encode Mortal observation from empty event log")
        player_state_cls = self._load_player_state_cls()
        player_state = player_state_cls(int(actor))
        replayed = 0
        for raw_event in events:
            event = sanitize_event_for_mortal(raw_event)
            if event is None:
                continue
            try:
                player_state.update(json.dumps(event, ensure_ascii=False))
            except Exception as exc:
                raise MortalObservationBridgeError(
                    f"Mortal PlayerState failed on event #{replayed}: {event}"
                ) from exc
            replayed += 1
        obs, action_mask = player_state.encode_obs(self.version, self.at_kan_select)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool)
        if obs_tensor.ndim != 2:
            raise MortalObservationBridgeError(f"Mortal obs must be rank-2, got {tuple(obs_tensor.shape)}")
        if mask_tensor.shape != (MORTAL_ACTION_SPACE,):
            raise MortalObservationBridgeError(
                f"Mortal action mask must have shape ({MORTAL_ACTION_SPACE},), got {tuple(mask_tensor.shape)}"
            )
        return MortalEncodedObservation(obs=obs_tensor, action_mask=mask_tensor, replayed_event_count=replayed)

    def _load_player_state_cls(self):
        if self._player_state_cls is not None:
            return self._player_state_cls
        mortal_python_dir = (self.mortal_root / "mortal").resolve()
        if not mortal_python_dir.exists():
            raise FileNotFoundError(f"Mortal python directory does not exist: {mortal_python_dir}")
        if str(mortal_python_dir) not in sys.path:
            sys.path.insert(0, str(mortal_python_dir))
        from libriichi.state import PlayerState  # noqa: PLC0415

        self._player_state_cls = PlayerState
        return self._player_state_cls


def sanitize_event_for_mortal(event: dict[str, Any]) -> dict[str, Any] | None:
    event_type = str(event.get("type", ""))
    if event_type in _MORTAL_UNSUPPORTED_ANNOUNCE_EVENTS:
        return None
    return dict(event)


__all__ = [
    "MORTAL_OBSERVATION_BRIDGE_CONTRACT_VERSION",
    "MortalEncodedObservation",
    "MortalObservationBridge",
    "MortalObservationBridgeError",
    "sanitize_event_for_mortal",
]
