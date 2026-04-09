from __future__ import annotations

from xmodel1.preprocess import events_to_xmodel1_arrays


def test_xmodel1_events_to_arrays_smoke():
    events = [
        {"type": "start_game", "names": ["A", "B", "C", "D"]},
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
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
                ["1s"] * 13,
                ["2s"] * 13,
                ["3s"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "4p"},
        {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
    ]
    arrays = events_to_xmodel1_arrays(events, replay_id="fixture.mjson")
    assert arrays is not None
    assert arrays["state_tile_feat"].shape[0] >= 1
    assert arrays["state_scalar"].shape[1] == 56
    assert arrays["candidate_feat"].shape[1:] == (14, 21)
    assert arrays["candidate_flags"].shape[1:] == (14, 10)
