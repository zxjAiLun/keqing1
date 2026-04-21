from __future__ import annotations

import pytest

from keqingrl.actions import (
    ACTION_FLAG_REACH,
    ACTION_FLAG_TSUMOGIRI,
    ActionSpec,
    ActionType,
    action_from_mahjong_spec,
    action_from_mjai,
    bind_reach_discard,
    decode_action_id,
    encode_action_id,
)
from mahjong_env.action_space import TILE_NAME_TO_IDX
from mahjong_env.types import ActionSpec as MahjongActionSpec


def test_action_spec_atomic_round_trip_through_mjai() -> None:
    actor = 2
    specs = [
        ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["1m"], flags=ACTION_FLAG_TSUMOGIRI),
        ActionSpec(ActionType.TSUMO, tile=TILE_NAME_TO_IDX["C"]),
        ActionSpec(ActionType.RON, tile=TILE_NAME_TO_IDX["9p"], from_who=1),
        ActionSpec(ActionType.CHI, tile=TILE_NAME_TO_IDX["2m"], consumed=(TILE_NAME_TO_IDX["3m"], TILE_NAME_TO_IDX["4m"]), from_who=3),
        ActionSpec(ActionType.PON, tile=TILE_NAME_TO_IDX["5p"], consumed=(TILE_NAME_TO_IDX["5p"], TILE_NAME_TO_IDX["5p"]), from_who=1),
        ActionSpec(ActionType.DAIMINKAN, tile=TILE_NAME_TO_IDX["7s"], consumed=(TILE_NAME_TO_IDX["7s"], TILE_NAME_TO_IDX["7s"], TILE_NAME_TO_IDX["7s"]), from_who=0),
        ActionSpec(ActionType.ANKAN, tile=TILE_NAME_TO_IDX["E"], consumed=(TILE_NAME_TO_IDX["E"], TILE_NAME_TO_IDX["E"], TILE_NAME_TO_IDX["E"])),
        ActionSpec(ActionType.KAKAN, tile=TILE_NAME_TO_IDX["P"], consumed=(TILE_NAME_TO_IDX["P"], TILE_NAME_TO_IDX["P"], TILE_NAME_TO_IDX["P"])),
        ActionSpec(ActionType.PASS),
        ActionSpec(ActionType.RYUKYOKU),
    ]

    for spec in specs:
        payload = spec.to_mjai_action(actor=actor)
        assert action_from_mjai(payload) == spec


def test_reach_discard_expands_to_two_mjai_events() -> None:
    spec = ActionSpec(ActionType.REACH_DISCARD, tile=TILE_NAME_TO_IDX["4m"], flags=ACTION_FLAG_TSUMOGIRI)
    events = spec.to_mjai_events(actor=1)
    assert events == [
        {"type": "reach", "actor": 1},
        {"type": "dahai", "actor": 1, "pai": "4m", "tsumogiri": True},
    ]


def test_bind_reach_discard_preserves_discard_tile_and_tsumogiri() -> None:
    discard = ActionSpec(
        ActionType.DISCARD,
        tile=TILE_NAME_TO_IDX["4m"],
        flags=ACTION_FLAG_TSUMOGIRI,
    )

    assert bind_reach_discard(discard) == ActionSpec(
        ActionType.REACH_DISCARD,
        tile=TILE_NAME_TO_IDX["4m"],
        flags=ACTION_FLAG_REACH | ACTION_FLAG_TSUMOGIRI,
    )


def test_encode_decode_action_id_round_trip() -> None:
    specs = [
        ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["9m"]),
        ActionSpec(ActionType.REACH_DISCARD, tile=TILE_NAME_TO_IDX["3s"], flags=ACTION_FLAG_TSUMOGIRI),
        ActionSpec(ActionType.RON, tile=TILE_NAME_TO_IDX["6p"], from_who=2),
        ActionSpec(ActionType.CHI, tile=TILE_NAME_TO_IDX["3m"], consumed=(TILE_NAME_TO_IDX["1m"], TILE_NAME_TO_IDX["2m"]), from_who=1),
        ActionSpec(ActionType.PASS),
    ]

    for spec in specs:
        assert decode_action_id(encode_action_id(spec)) == spec


def test_action_from_mahjong_spec_converts_atomic_self_turn_actions() -> None:
    cases = [
        (
            MahjongActionSpec(type="dahai", actor=0, pai="4m", tsumogiri=False),
            ActionSpec(ActionType.DISCARD, tile=TILE_NAME_TO_IDX["4m"]),
        ),
        (
            MahjongActionSpec(type="hora", actor=0, target=0, pai="7m"),
            ActionSpec(ActionType.TSUMO, tile=TILE_NAME_TO_IDX["7m"]),
        ),
        (
            MahjongActionSpec(type="ryukyoku", actor=0),
            ActionSpec(ActionType.RYUKYOKU),
        ),
        (
            MahjongActionSpec(
                type="ankan",
                actor=0,
                pai="E",
                consumed=("E", "E", "E", "E"),
            ),
            ActionSpec(
                ActionType.ANKAN,
                tile=TILE_NAME_TO_IDX["E"],
                consumed=(
                    TILE_NAME_TO_IDX["E"],
                    TILE_NAME_TO_IDX["E"],
                    TILE_NAME_TO_IDX["E"],
                    TILE_NAME_TO_IDX["E"],
                ),
            ),
        ),
        (
            MahjongActionSpec(
                type="kakan",
                actor=0,
                pai="5p",
                consumed=("5p", "5p", "5p", "5p"),
            ),
            ActionSpec(
                ActionType.KAKAN,
                tile=TILE_NAME_TO_IDX["5p"],
                consumed=(
                    TILE_NAME_TO_IDX["5p"],
                    TILE_NAME_TO_IDX["5p"],
                    TILE_NAME_TO_IDX["5p"],
                    TILE_NAME_TO_IDX["5p"],
                ),
            ),
        ),
    ]

    for raw_spec, expected in cases:
        assert action_from_mahjong_spec(raw_spec) == expected


def test_action_from_mahjong_spec_rejects_bare_reach() -> None:
    with pytest.raises(ValueError, match="reach"):
        action_from_mahjong_spec(MahjongActionSpec(type="reach", actor=0))


def test_action_from_mjai_rejects_bare_reach() -> None:
    with pytest.raises(ValueError, match="reach"):
        action_from_mjai({"type": "reach", "actor": 0})
