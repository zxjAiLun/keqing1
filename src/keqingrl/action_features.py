"""KeqingRL action-feature contract builders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import keqing_core

from keqingrl.actions import ActionSpec, ActionType
from mahjong_env.action_space import IDX_TO_TILE_NAME
from mahjong_env.feature_tracker import SnapshotFeatureTracker
from mahjong_env.tiles import normalize_tile, tile_is_aka, tile_to_34

ACTION_FEATURE_CONTRACT_VERSION_V1 = "keqingrl_action_feature_v1"
ACTION_FEATURE_CONTRACT_VERSION_V2 = "keqingrl_action_feature_v2"
ACTION_FEATURE_DIM_V1 = 8
ACTION_FEATURE_DIM_V2 = 32

_DORA_NEXT = {
    "1m": "2m",
    "2m": "3m",
    "3m": "4m",
    "4m": "5m",
    "5m": "6m",
    "6m": "7m",
    "7m": "8m",
    "8m": "9m",
    "9m": "1m",
    "1p": "2p",
    "2p": "3p",
    "3p": "4p",
    "4p": "5p",
    "5p": "6p",
    "6p": "7p",
    "7p": "8p",
    "8p": "9p",
    "9p": "1p",
    "1s": "2s",
    "2s": "3s",
    "3s": "4s",
    "4s": "5s",
    "5s": "6s",
    "6s": "7s",
    "7s": "8s",
    "8s": "9s",
    "9s": "1s",
    "E": "S",
    "S": "W",
    "W": "N",
    "N": "E",
    "P": "F",
    "F": "C",
    "C": "P",
}


def build_keqingrl_action_features(
    snapshot: Mapping[str, object],
    legal_actions: Sequence[ActionSpec],
    *,
    remaining_wall: int | float,
    contract_version: str = ACTION_FEATURE_CONTRACT_VERSION_V2,
) -> list[list[float]]:
    """Build action features for the active KeqingRL contract.

    v2 preserves the old 8 columns as a strict prefix and appends response/call
    context that Mortal uses heavily when comparing PASS/CHI/PON/RON.
    """

    tracker = SnapshotFeatureTracker.from_state(snapshot, actor=int(snapshot["actor"]))
    base_rows = keqing_core.build_keqingrl_action_features_typed(
        tracker.hand_counts34,
        tracker.visible_counts34,
        [int(spec.action_type) for spec in legal_actions],
        [-1 if spec.tile is None else int(spec.tile) for spec in legal_actions],
        [int(spec.flags) for spec in legal_actions],
        float(remaining_wall),
    )
    if contract_version == ACTION_FEATURE_CONTRACT_VERSION_V1:
        return base_rows
    if contract_version != ACTION_FEATURE_CONTRACT_VERSION_V2:
        raise ValueError(f"unsupported action feature contract: {contract_version!r}")

    row_context = _row_context(snapshot, legal_actions, tracker=tracker)
    return [
        [*base, *_v2_extra_features(snapshot, spec, legal_actions, tracker=tracker, row_context=row_context)]
        for base, spec in zip(base_rows, legal_actions, strict=True)
    ]


def action_feature_dim(contract_version: str = ACTION_FEATURE_CONTRACT_VERSION_V2) -> int:
    if contract_version == ACTION_FEATURE_CONTRACT_VERSION_V1:
        return ACTION_FEATURE_DIM_V1
    if contract_version == ACTION_FEATURE_CONTRACT_VERSION_V2:
        return ACTION_FEATURE_DIM_V2
    raise ValueError(f"unsupported action feature contract: {contract_version!r}")


def _row_context(
    snapshot: Mapping[str, object],
    legal_actions: Sequence[ActionSpec],
    *,
    tracker: SnapshotFeatureTracker,
) -> dict[str, float]:
    action_types = {spec.action_type for spec in legal_actions}
    last_discard_tile = _last_discard_tile(snapshot)
    target_tile = last_discard_tile
    if target_tile is None:
        for spec in legal_actions:
            if spec.action_type in {ActionType.CHI, ActionType.PON, ActionType.DAIMINKAN, ActionType.RON}:
                target_tile = _tile_name(spec.tile)
                break
    target_id = _tile_id(target_tile)
    return {
        "actor": float(int(snapshot.get("actor", -1))),
        "is_response": float(bool(action_types & {ActionType.PASS, ActionType.RON, ActionType.CHI, ActionType.PON, ActionType.DAIMINKAN})),
        "can_pass": float(ActionType.PASS in action_types),
        "can_ron": float(ActionType.RON in action_types),
        "can_chi": float(ActionType.CHI in action_types),
        "can_pon": float(ActionType.PON in action_types),
        "can_daiminkan": float(ActionType.DAIMINKAN in action_types),
        "target_tile_norm": _tile_norm(target_id),
        "target_is_aka": float(bool(target_tile and tile_is_aka(str(target_tile)))),
        "target_is_dora": float(_is_dora_tile(target_tile, snapshot)),
        "target_visible_count": _count_feature(tracker.visible_counts34, target_id),
        "target_hand_count": _count_feature(tracker.hand_counts34, target_id),
    }


def _v2_extra_features(
    snapshot: Mapping[str, object],
    spec: ActionSpec,
    legal_actions: Sequence[ActionSpec],
    *,
    tracker: SnapshotFeatureTracker,
    row_context: Mapping[str, float],
) -> list[float]:
    del snapshot, legal_actions, tracker
    consumed = tuple(int(tile) for tile in spec.consumed)
    consumed_ids = [tile for tile in consumed if 0 <= tile < 34]
    min_consumed = min(consumed_ids) if consumed_ids else None
    max_consumed = max(consumed_ids) if consumed_ids else None
    actor = int(row_context.get("actor", -1)) if "actor" in row_context else None
    return [
        row_context["is_response"],
        row_context["can_pass"],
        row_context["can_ron"],
        row_context["can_chi"],
        row_context["can_pon"],
        row_context["can_daiminkan"],
        float(spec.action_type == ActionType.PASS),
        float(spec.action_type in {ActionType.RON, ActionType.TSUMO}),
        float(spec.action_type in {ActionType.CHI, ActionType.PON, ActionType.DAIMINKAN}),
        float(spec.action_type == ActionType.CHI),
        float(spec.action_type == ActionType.PON),
        float(spec.action_type == ActionType.DAIMINKAN),
        float(spec.action_type == ActionType.REACH_DISCARD),
        float(spec.action_type == ActionType.DISCARD),
        row_context["target_tile_norm"],
        row_context["target_is_aka"],
        row_context["target_is_dora"],
        row_context["target_visible_count"],
        row_context["target_hand_count"],
        min(len(consumed_ids) / 4.0, 1.0),
        _tile_norm(min_consumed),
        _tile_norm(max_consumed),
        _chi_shape_value(spec),
        _from_who_relative(spec.from_who, actor),
    ]


def _last_discard_tile(snapshot: Mapping[str, object]) -> str | None:
    last_discard = snapshot.get("last_discard")
    if isinstance(last_discard, Mapping):
        value = last_discard.get("pai_raw") or last_discard.get("pai")
        return str(value) if value else None
    return None


def _tile_name(tile_id: int | None) -> str | None:
    if tile_id is None:
        return None
    return IDX_TO_TILE_NAME.get(int(tile_id))


def _tile_id(tile_name: str | None) -> int | None:
    if not tile_name:
        return None
    tile_id = tile_to_34(normalize_tile(str(tile_name)))
    return tile_id if tile_id >= 0 else None


def _tile_norm(tile_id: int | None) -> float:
    return 0.0 if tile_id is None or tile_id < 0 else float(tile_id) / 33.0


def _count_feature(counts34: Sequence[int], tile_id: int | None) -> float:
    if tile_id is None or tile_id < 0 or tile_id >= len(counts34):
        return 0.0
    return float(counts34[int(tile_id)]) / 4.0


def _is_dora_tile(tile_name: str | None, snapshot: Mapping[str, object]) -> bool:
    if not tile_name:
        return False
    normalized = normalize_tile(str(tile_name))
    for marker in snapshot.get("dora_markers", ()) or ():
        if _DORA_NEXT.get(normalize_tile(str(marker))) == normalized:
            return True
    return False


def _chi_shape_value(spec: ActionSpec) -> float:
    if spec.action_type != ActionType.CHI or spec.tile is None or len(spec.consumed) < 2:
        return 0.0
    called = _numbered_tile(int(spec.tile))
    consumed = sorted(_numbered_tile(int(tile)) for tile in spec.consumed)
    if called is None or any(item is None for item in consumed):
        return 0.0
    suit, number = called
    consumed_numbers = [item[1] for item in consumed if item is not None and item[0] == suit]
    if len(consumed_numbers) != 2:
        return 0.0
    if consumed_numbers == [number + 1, number + 2]:
        return 1.0 / 3.0
    if consumed_numbers == [number - 1, number + 1]:
        return 2.0 / 3.0
    if consumed_numbers == [number - 2, number - 1]:
        return 1.0
    return 0.0


def _numbered_tile(tile_id: int) -> tuple[str, int] | None:
    name = IDX_TO_TILE_NAME.get(int(tile_id))
    if not name or len(name) != 2 or not name[0].isdigit() or name[1] not in {"m", "p", "s"}:
        return None
    return name[1], int(name[0])


def _from_who_relative(from_who: int | None, actor: int | None) -> float:
    if from_who is None or actor is None or actor < 0:
        return 0.0
    return float((int(from_who) - int(actor)) % 4) / 3.0
