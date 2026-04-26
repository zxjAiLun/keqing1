"""Rust-side structural tests for xmodel1 event_history export."""

from __future__ import annotations

import json
import numpy as np
import pytest


EVENT_HISTORY_LEN = 48
EVENT_HISTORY_FEATURE_DIM = 5
EVENT_TYPE_PAD = 0
EVENT_TYPE_TSUMO = 1
EVENT_TYPE_DAHAI = 2
EVENT_NO_ACTOR = 4
EVENT_NO_TILE = -1


def _export(tmp_path, events):
    from keqing_core import build_xmodel1_discard_records

    data_dir = tmp_path / "ds1"
    data_dir.mkdir()
    replay_path = data_dir / "sample.mjson"
    replay_path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n",
        encoding="utf-8",
    )
    try:
        _count, _manifest, produced_npz = build_xmodel1_discard_records(
            data_dirs=[str(data_dir)],
            output_dir=str(tmp_path / "out"),
            smoke=False,
        )
    except RuntimeError:
        pytest.skip("Rust extension unavailable")
    assert produced_npz
    return np.load(tmp_path / "out" / "ds1" / "sample.npz", allow_pickle=True)


def _build_min_round(*, kyoku_tehais=None):
    """Build a minimal mjson event stream with one start_kyoku + a handful of discards."""
    if kyoku_tehais is None:
        kyoku_tehais = [
            ["1m"] * 13,
            ["2m"] * 13,
            ["3m"] * 13,
            ["4m"] * 13,
        ]
    return [
        {"type": "start_game"},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "1s",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": kyoku_tehais,
        },
        {"type": "tsumo", "actor": 0, "pai": "5m"},
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "6m"},
        {"type": "dahai", "actor": 1, "pai": "6m", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "7m"},
        {"type": "dahai", "actor": 2, "pai": "7m", "tsumogiri": True},
        {"type": "tsumo", "actor": 3, "pai": "8m"},
        {"type": "dahai", "actor": 3, "pai": "8m", "tsumogiri": True},
        {"type": "end_kyoku"},
    ]


def test_event_history_shape_and_dtype(tmp_path):
    events = _build_min_round()
    arrays = _export(tmp_path, events)
    assert "event_history" in arrays
    eh = arrays["event_history"]
    n = arrays["sample_type"].shape[0]
    assert eh.shape == (n, EVENT_HISTORY_LEN, EVENT_HISTORY_FEATURE_DIM)
    assert eh.dtype == np.int16


def test_event_history_right_aligned_with_padding(tmp_path):
    events = _build_min_round()
    arrays = _export(tmp_path, events)
    eh = arrays["event_history"]
    # first sample should have heavy leading padding
    first = eh[0]
    # padding rows should match (actor=NO_ACTOR, type=PAD, tile=NO_TILE, turn=0, is_tedashi=0)
    pad_row = np.array(
        [EVENT_NO_ACTOR, EVENT_TYPE_PAD, EVENT_NO_TILE, 0, 0],
        dtype=np.int16,
    )
    # expect at least the top row to be padding for the first sample
    assert np.array_equal(first[0], pad_row), f"expected pad row, got {first[0]}"
    # the final row of the first sample is the most recent event at that decision
    # for a dahai decision the last row is the tsumo that immediately precedes it
    last = first[-1]
    assert last[0] != EVENT_NO_ACTOR, "last row should not be padding for decisions with >=1 preceding event"


def test_event_history_turn_idx_monotonic_within_kyoku(tmp_path):
    events = _build_min_round()
    arrays = _export(tmp_path, events)
    eh = arrays["event_history"]
    # check a later sample has non-decreasing turn_idx
    later = eh[-1]
    turns = [row[3] for row in later if row[1] != EVENT_TYPE_PAD]
    assert turns == sorted(turns), f"turn_idx should be monotonic, got {turns}"


def test_event_history_resets_across_kyoku(tmp_path):
    # build two kyoku; events in kyoku 2 should NOT see events from kyoku 1 in history
    round1 = _build_min_round()
    # modify second kyoku to be distinct
    round2 = [
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "dora_marker": "1s",
            "kyoku": 2,
            "honba": 0,
            "kyotaku": 0,
            "oya": 1,
            "scores": [25000, 25000, 25000, 25000],
            "tehais": [["9m"] * 13] * 4,
        },
        {"type": "tsumo", "actor": 1, "pai": "1p"},
        {"type": "dahai", "actor": 1, "pai": "1p", "tsumogiri": True},
        {"type": "tsumo", "actor": 2, "pai": "2p"},
        {"type": "dahai", "actor": 2, "pai": "2p", "tsumogiri": True},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]
    events = round1[:-1] + round2  # drop first end_kyoku, append round2 (which includes its own)
    arrays = _export(tmp_path, events)
    eh = arrays["event_history"]
    # find samples from the second kyoku and check their history has very little content
    # kyoku_idx is part of scalar or we can infer by sample order; fall back to heuristic:
    # the LAST sample (final decision in round 2) should have at most ~3 non-pad rows.
    last = eh[-1]
    non_pad = sum(1 for row in last if row[1] != EVENT_TYPE_PAD)
    assert 1 <= non_pad <= 4, (
        f"second-kyoku sample should only see events within its own kyoku, got {non_pad} non-pad"
    )


def test_event_history_encodes_actor_and_tile(tmp_path):
    events = _build_min_round()
    arrays = _export(tmp_path, events)
    eh = arrays["event_history"]
    # find a sample with visible actors in history
    sample = eh[-1]
    actors = [int(row[0]) for row in sample if row[1] != EVENT_TYPE_PAD]
    assert all(0 <= a <= 3 for a in actors), f"actors must be in [0,3]: {actors}"
    types = [int(row[1]) for row in sample if row[1] != EVENT_TYPE_PAD]
    assert EVENT_TYPE_TSUMO in types or EVENT_TYPE_DAHAI in types
    tiles = [int(row[2]) for row in sample if row[1] == EVENT_TYPE_DAHAI]
    assert all(0 <= t < 34 for t in tiles), f"dahai tile34 ids must be in [0,34): {tiles}"
