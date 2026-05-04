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


@dataclass
class _CachedPlayerState:
    player_state: Any
    raw_event_count: int = 0
    replayed_event_count: int = 0
    first_event_fingerprint: str = ""
    last_event_fingerprint: str = ""


class MortalObservationBridge:
    def __init__(
        self,
        *,
        mortal_root: Path = Path("third_party/Mortal"),
        version: int = 4,
        at_kan_select: bool = False,
        enable_incremental_cache: bool = True,
    ) -> None:
        self.mortal_root = mortal_root
        self.version = int(version)
        self.at_kan_select = bool(at_kan_select)
        self.enable_incremental_cache = bool(enable_incremental_cache)
        self._player_state_cls = None
        self._state_cache: dict[int, _CachedPlayerState] = {}

    def reset_cache(self) -> None:
        self._state_cache.clear()

    def encode_from_events(self, events: Sequence[dict[str, Any]], actor: int) -> MortalEncodedObservation:
        if not events:
            raise MortalObservationBridgeError("cannot encode Mortal observation from empty event log")
        player_state_cls = self._load_player_state_cls()
        actor = int(actor)
        raw_events = tuple(events)
        cache = self._cached_player_state(player_state_cls, raw_events, actor)
        start_index = cache.raw_event_count
        for raw_event in raw_events[start_index:]:
            event = sanitize_event_for_mortal(raw_event)
            cache.raw_event_count += 1
            cache.last_event_fingerprint = _event_fingerprint(raw_event)
            if not cache.first_event_fingerprint:
                cache.first_event_fingerprint = cache.last_event_fingerprint
            if event is None:
                continue
            try:
                cache.player_state.update(json.dumps(event, ensure_ascii=False))
            except Exception as exc:
                raise MortalObservationBridgeError(
                    f"Mortal PlayerState failed on event #{cache.replayed_event_count}: {event}"
                ) from exc
            cache.replayed_event_count += 1
        obs, action_mask = cache.player_state.encode_obs(self.version, self.at_kan_select)
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        mask_tensor = torch.as_tensor(action_mask, dtype=torch.bool)
        if obs_tensor.ndim != 2:
            raise MortalObservationBridgeError(f"Mortal obs must be rank-2, got {tuple(obs_tensor.shape)}")
        if mask_tensor.shape != (MORTAL_ACTION_SPACE,):
            raise MortalObservationBridgeError(
                f"Mortal action mask must have shape ({MORTAL_ACTION_SPACE},), got {tuple(mask_tensor.shape)}"
            )
        return MortalEncodedObservation(
            obs=obs_tensor,
            action_mask=mask_tensor,
            replayed_event_count=cache.replayed_event_count,
        )

    def _cached_player_state(
        self,
        player_state_cls,
        events: Sequence[dict[str, Any]],
        actor: int,
    ) -> _CachedPlayerState:
        if not self.enable_incremental_cache:
            return _CachedPlayerState(player_state=player_state_cls(int(actor)))
        cache = self._state_cache.get(int(actor))
        if cache is None or not _cache_can_continue(cache, events):
            cache = _CachedPlayerState(player_state=player_state_cls(int(actor)))
            self._state_cache[int(actor)] = cache
        return cache

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
    if event_type == "none" and "actor" in event:
        return None
    return dict(event)


def _cache_can_continue(cache: _CachedPlayerState, events: Sequence[dict[str, Any]]) -> bool:
    if cache.raw_event_count == 0:
        return True
    if cache.raw_event_count > len(events):
        return False
    if not events:
        return False
    first = _event_fingerprint(events[0])
    if cache.first_event_fingerprint and first != cache.first_event_fingerprint:
        return False
    last_cached = _event_fingerprint(events[cache.raw_event_count - 1])
    return last_cached == cache.last_event_fingerprint


def _event_fingerprint(event: dict[str, Any]) -> str:
    return json.dumps(event, sort_keys=True, ensure_ascii=False, default=str)


__all__ = [
    "MORTAL_OBSERVATION_BRIDGE_CONTRACT_VERSION",
    "MortalEncodedObservation",
    "MortalObservationBridge",
    "MortalObservationBridgeError",
    "sanitize_event_for_mortal",
]
