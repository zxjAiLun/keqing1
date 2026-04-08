import numpy as np

from mahjong_env.state import GameState, apply_event
from keqingv3.features import C_TILE, N_SCALAR, encode


def _parse_hand(s: str):
    out = []
    digits = ""
    honors = {"1": "E", "2": "S", "3": "W", "4": "N", "5": "P", "6": "F", "7": "C"}
    for ch in s:
        if ch.isdigit():
            digits += ch
        else:
            if ch in "mps":
                for d in digits:
                    out.append((d if d != "0" else "5") + ch)
            elif ch == "z":
                out.extend(honors[d] for d in digits)
            digits = ""
    return out


def _base_state(hand):
    return {
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "actor": 0,
        "hand": hand,
        "discards": [[], [], [], []],
        "melds": [[], [], [], []],
        "reached": [False, False, False, False],
        "furiten": [False, False, False, False],
        "dora_markers": [],
    }


def test_encode_v3_shape_constants():
    state = _base_state(_parse_hand("123456m345p67s55z"))
    tile_feat, scalar = encode(state, 0)

    assert tile_feat.shape == (C_TILE, 34)
    assert scalar.shape == (N_SCALAR,)


def test_encode_v3_progress_and_seen_tiles_reference_case():
    hand14 = _parse_hand("4m2067p4s5z0s")
    hand13 = list(hand14)
    hand13.remove("4m")
    state = _base_state(hand13)

    tile_feat, scalar = encode(state, 0)

    # 1-shanten, 14 live ukeire, 2p/3s/6s/5z
    assert np.isclose(scalar[8], 1.0 / 8.0)
    assert np.isclose(scalar[38], 14.0 / 34.0)
    assert tile_feat[54, 10] == 1.0  # 2p
    assert tile_feat[54, 20] == 1.0  # 3s
    assert tile_feat[54, 23] == 1.0  # 6s
    assert tile_feat[54, 31] == 1.0  # P(5z)

    # seen_tile_ratio should reflect own hand visibility
    assert np.isclose(tile_feat[56, 10], 0.25)  # 2p seen once in hand
    assert np.isclose(tile_feat[56, 13], 0.25)  # 5p(red normalized to 5p) seen once
    # good-shape / improvement are intentionally disabled in v3 now
    assert np.isclose(scalar[53], 0.0)
    assert np.isclose(scalar[54], 0.0)
    assert 0.0 <= scalar[55] <= 1.0
    assert np.count_nonzero(tile_feat[55]) == 0


def test_encode_v3_uses_snapshot_feature_tracker_from_gamestate():
    state = GameState()
    apply_event(state, {"type": "start_game"})
    apply_event(
        state,
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                _parse_hand("123456m345p67s55z"),
                _parse_hand("123456789m12344p"),
                _parse_hand("123456789p12344s"),
                _parse_hand("123456789s12344m"),
            ],
        },
    )
    snap = state.snapshot(0)

    assert "feature_tracker" in snap
    expected_counts = [0] * 34
    for tile in snap["hand"]:
        if tile.endswith("m"):
            expected_counts[int(tile[0]) - 1] += 1
        elif tile.endswith("p"):
            expected_counts[9 + int(tile[0]) - 1] += 1
        elif tile.endswith("s"):
            expected_counts[18 + int(tile[0]) - 1] += 1
        else:
            honors = {"E": 27, "S": 28, "W": 29, "N": 30, "P": 31, "F": 32, "C": 33}
            expected_counts[honors[tile]] += 1
    assert tuple(snap["feature_tracker"]["hand_counts34"]) == tuple(expected_counts)

    tile_feat, scalar = encode(snap, 0)
    assert tile_feat.shape == (C_TILE, 34)
    assert scalar.shape == (N_SCALAR,)
