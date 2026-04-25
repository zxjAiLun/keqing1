"""Structured action contracts for the keqingrl family."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Mapping, TYPE_CHECKING

import keqing_core

from mahjong_env.action_space import IDX_TO_TILE_NAME, TILE_NAME_TO_IDX
from mahjong_env.tiles import normalize_tile

if TYPE_CHECKING:
    from mahjong_env.types import ActionSpec as MahjongActionSpec

ACTION_FLAG_TSUMOGIRI = 1 << 0
ACTION_FLAG_REACH = 1 << 1
_TILE_SLOT_BASE = 35  # 0=None, 1..34=tile ids 0..33
_FROM_WHO_BASE = 5    # 0=None, 1..4=player 0..3
_FLAGS_BASE = 256
_MAX_CONSUMED = 3


class ActionType(IntEnum):
    DISCARD = 0
    REACH_DISCARD = 1
    TSUMO = 2
    RON = 3
    CHI = 4
    PON = 5
    DAIMINKAN = 6
    ANKAN = 7
    KAKAN = 8
    PASS = 9
    NUKI = 10
    RYUKYOKU = 11


@dataclass(frozen=True)
class ActionIdentity:
    version: int
    action_type: ActionType
    action_type_name: str
    actor: int | None
    tile: int | None
    consumed: tuple[int, ...]
    from_who: int | None
    flags: int
    canonical_key: str
    action_id: int
    supported: bool
    unsupported_reason: str | None = None


@dataclass(frozen=True)
class ActionSpec:
    """Structured policy action.

    Runtime ActionIdentity is owned by Rust. The Python canonical key remains a
    parity oracle and stable serialization convenience until all constructors
    hydrate identities directly from Rust.
    """

    action_type: ActionType
    tile: int | None = None
    consumed: tuple[int, ...] = ()
    from_who: int | None = None
    flags: int = 0
    canonical_key: str = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "canonical_key", make_action_canonical_key(self))

    def identity(self, *, actor: int | None = None) -> ActionIdentity:
        return make_action_identity(self, actor=actor)

    def to_mjai_action(self, *, actor: int) -> dict[str, object]:
        if self.action_type == ActionType.REACH_DISCARD:
            raise ValueError("REACH_DISCARD expands to multiple MJAI events; use to_mjai_events(actor=...)")

        if self.action_type == ActionType.PASS:
            return {"type": "none"}

        payload: dict[str, object] = {"type": _ACTION_TYPE_TO_MJAI[self.action_type], "actor": actor}
        tile_name = _tile_name(self.tile)
        if tile_name is not None and self.action_type != ActionType.RYUKYOKU:
            payload["pai"] = tile_name
        if self.consumed:
            payload["consumed"] = [_tile_name(tile_id) for tile_id in self.consumed]
        if self.from_who is not None and self.action_type in _USES_TARGET:
            payload["target"] = self.from_who
        if self.action_type == ActionType.DISCARD:
            payload["tsumogiri"] = bool(self.flags & ACTION_FLAG_TSUMOGIRI)
        return payload

    def to_mjai_events(self, *, actor: int) -> list[dict[str, object]]:
        return keqing_core.mjai_events_for_action(_identity_payload(self, actor=actor), actor=actor)


_ACTION_TYPE_TO_MJAI = {
    ActionType.DISCARD: "dahai",
    ActionType.REACH_DISCARD: "reach",
    ActionType.TSUMO: "hora",
    ActionType.RON: "hora",
    ActionType.CHI: "chi",
    ActionType.PON: "pon",
    ActionType.DAIMINKAN: "daiminkan",
    ActionType.ANKAN: "ankan",
    ActionType.KAKAN: "kakan",
    ActionType.PASS: "none",
    ActionType.NUKI: "nuki",
    ActionType.RYUKYOKU: "ryukyoku",
}
_USES_TARGET = {
    ActionType.RON,
    ActionType.CHI,
    ActionType.PON,
    ActionType.DAIMINKAN,
}


def encode_action_id(spec: ActionSpec) -> int:
    """Pack a structured action into a stable integer id for candidate encoding."""
    encoded = int(spec.action_type)
    encoded = encoded * _TILE_SLOT_BASE + _encode_tile_slot(spec.tile)
    for slot in range(_MAX_CONSUMED):
        tile = spec.consumed[slot] if slot < len(spec.consumed) else None
        encoded = encoded * _TILE_SLOT_BASE + _encode_tile_slot(tile)
    encoded = encoded * _FROM_WHO_BASE + _encode_from_who_slot(spec.from_who)
    encoded = encoded * _FLAGS_BASE + int(spec.flags)
    return encoded


def make_action_identity(spec: ActionSpec, *, actor: int | None = None) -> ActionIdentity:
    native = keqing_core.action_identity(_identity_payload(spec, actor=actor))
    return ActionIdentity(
        version=int(native["version"]),
        action_type=ActionType(int(native["action_type"])),
        action_type_name=str(native["action_type_name"]),
        actor=_optional_int(native.get("actor")),
        tile=_optional_int(native.get("tile")),
        consumed=tuple(int(value) for value in native.get("consumed", ())),
        from_who=_optional_int(native.get("from_who")),
        flags=int(native["flags"]),
        canonical_key=str(native["canonical_key"]),
        action_id=int(native["action_id"]),
        supported=bool(native["supported"]),
        unsupported_reason=None if native.get("unsupported_reason") is None else str(native["unsupported_reason"]),
    )


def _identity_payload(spec: ActionSpec, *, actor: int | None) -> dict[str, object]:
    return {
        "action_type": int(spec.action_type),
        "actor": actor,
        "tile": spec.tile,
        "consumed": list(spec.consumed),
        "from_who": spec.from_who,
        "flags": int(spec.flags),
    }


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def make_action_canonical_key(spec: ActionSpec) -> str:
    """Build the Python parity key; Rust ActionIdentity is the runtime source of truth."""
    consumed = ",".join(str(int(tile)) for tile in spec.consumed)
    tile = -1 if spec.tile is None else int(spec.tile)
    from_who = -1 if spec.from_who is None else int(spec.from_who)
    return (
        f"{int(spec.action_type)}"
        f"|tile={tile}"
        f"|consumed={consumed}"
        f"|from={from_who}"
        f"|flags={int(spec.flags)}"
    )


def decode_action_id(encoded: int) -> ActionSpec:
    if encoded < 0:
        raise ValueError("encoded action id must be non-negative")
    decoded = keqing_core.decode_action_id(int(encoded))
    return ActionSpec(
        action_type=ActionType(int(decoded["action_type"])),
        tile=_optional_int(decoded.get("tile")),
        consumed=tuple(int(value) for value in decoded.get("consumed", ())),
        from_who=_optional_int(decoded.get("from_who")),
        flags=int(decoded["flags"]),
    )


def action_from_mjai(action: Mapping[str, object]) -> ActionSpec:
    action_type = str(action.get("type", "none"))
    tile = _tile_id(action.get("pai"))
    consumed = tuple(_tile_id(value) for value in action.get("consumed", ()) or ())
    target = action.get("target")
    actor = _maybe_int(action.get("actor"))

    if action_type == "dahai":
        flags = ACTION_FLAG_TSUMOGIRI if bool(action.get("tsumogiri", False)) else 0
        return ActionSpec(action_type=ActionType.DISCARD, tile=tile, flags=flags)
    if action_type == "reach":
        raise ValueError("bare mjai reach is not a complete keqingrl action; bind it to a discard first")
    if action_type == "hora":
        target_actor = _maybe_int(target)
        if target is None or (actor is not None and target_actor == actor):
            return ActionSpec(action_type=ActionType.TSUMO, tile=tile)
        return ActionSpec(action_type=ActionType.RON, tile=tile, from_who=int(target_actor))
    if action_type == "chi":
        return ActionSpec(action_type=ActionType.CHI, tile=tile, consumed=consumed, from_who=_maybe_int(target))
    if action_type == "pon":
        return ActionSpec(action_type=ActionType.PON, tile=tile, consumed=consumed, from_who=_maybe_int(target))
    if action_type == "daiminkan":
        return ActionSpec(action_type=ActionType.DAIMINKAN, tile=tile, consumed=consumed, from_who=_maybe_int(target))
    if action_type == "ankan":
        return ActionSpec(action_type=ActionType.ANKAN, tile=tile, consumed=consumed)
    if action_type == "kakan":
        return ActionSpec(action_type=ActionType.KAKAN, tile=tile, consumed=consumed)
    if action_type == "ryukyoku":
        return ActionSpec(action_type=ActionType.RYUKYOKU)
    if action_type == "nuki":
        return ActionSpec(action_type=ActionType.NUKI, tile=tile)
    return ActionSpec(action_type=ActionType.PASS)


def action_from_mahjong_spec(spec: "MahjongActionSpec") -> ActionSpec:
    if spec.type == "reach":
        raise ValueError("mahjong reach spec is not an atomic keqingrl action; bind it to a discard first")
    return action_from_mjai(spec.to_mjai())


def bind_reach_discard(discard_spec: ActionSpec) -> ActionSpec:
    if discard_spec.action_type != ActionType.DISCARD:
        raise ValueError(f"reach_discard requires a discard action, got {discard_spec.action_type.name}")
    if discard_spec.tile is None:
        raise ValueError("reach_discard requires a discard tile")
    return ActionSpec(
        action_type=ActionType.REACH_DISCARD,
        tile=discard_spec.tile,
        flags=int(discard_spec.flags) | ACTION_FLAG_REACH,
    )


def _tile_id(value: object) -> int | None:
    if value is None:
        return None
    normalized = normalize_tile(str(value))
    return int(TILE_NAME_TO_IDX[normalized])


def _tile_name(tile: int | None) -> str | None:
    if tile is None:
        return None
    return str(IDX_TO_TILE_NAME[int(tile)])


def _encode_tile_slot(tile: int | None) -> int:
    if tile is None:
        return 0
    return int(tile) + 1


def _decode_tile_slot(slot: int) -> int | None:
    if slot == 0:
        return None
    return int(slot) - 1


def _encode_from_who_slot(from_who: int | None) -> int:
    if from_who is None:
        return 0
    if not 0 <= int(from_who) <= 3:
        raise ValueError(f"from_who must be in [0, 3], got {from_who}")
    return int(from_who) + 1


def _decode_from_who_slot(slot: int) -> int | None:
    if slot == 0:
        return None
    return int(slot) - 1


def _maybe_int(value: object) -> int | None:
    return None if value is None else int(value)


__all__ = [
    "ACTION_FLAG_REACH",
    "ACTION_FLAG_TSUMOGIRI",
    "ActionIdentity",
    "ActionSpec",
    "ActionType",
    "bind_reach_discard",
    "action_from_mahjong_spec",
    "action_from_mjai",
    "decode_action_id",
    "encode_action_id",
    "make_action_identity",
    "make_action_canonical_key",
]
