from __future__ import annotations

import numpy as np

from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_HISTORY_SUMMARY_DIM,
    XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM,
)
from xmodel1.candidate_quality import build_candidate_features
from xmodel1.candidate_quality import build_special_candidate_arrays
from xmodel1.preprocess import events_to_xmodel1_arrays
from xmodel1.schema import (
    XMODEL1_SAMPLE_TYPE_CALL,
    XMODEL1_SAMPLE_TYPE_HORA,
    XMODEL1_SPECIAL_TYPE_CHI_HIGH,
    XMODEL1_SPECIAL_TYPE_CHI_LOW,
    XMODEL1_SPECIAL_TYPE_CHI_MID,
    XMODEL1_SPECIAL_TYPE_HORA,
    XMODEL1_SPECIAL_TYPE_NONE,
    XMODEL1_SPECIAL_TYPE_PON,
    XMODEL1_SPECIAL_TYPE_REACH,
)


def _tsumo_hora_events() -> list[dict]:
    return [
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
                ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "4m", "5m", "9s", "9s"],
                ["1s"] * 13,
                ["2s"] * 13,
                ["3s"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "6m"},
        {"type": "hora", "actor": 0, "target": 0, "pai": "6m", "deltas": [1000, -500, -500, 0], "ura_markers": []},
    ]


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
    assert arrays["candidate_feat"].shape[1:] == (14, XMODEL1_CANDIDATE_FEATURE_DIM)
    assert arrays["candidate_flags"].shape[1:] == (14, XMODEL1_CANDIDATE_FLAG_DIM)
    assert arrays["special_candidate_feat"].shape[1:] == (12, XMODEL1_SPECIAL_CANDIDATE_FEATURE_DIM)
    assert arrays["special_candidate_mask"].shape[1] == 12
    assert arrays["history_summary"].shape[1:] == (XMODEL1_HISTORY_SUMMARY_DIM,)
    assert "event_history" not in arrays


def test_xmodel1_events_to_arrays_emits_history_summary():
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
    history_summary = arrays["history_summary"][0]
    assert history_summary.shape == (XMODEL1_HISTORY_SUMMARY_DIM,)
    assert np.isfinite(history_summary).all()
    assert float(history_summary[4]) >= 0.0


def test_xmodel1_events_to_arrays_exports_reach_dama_special_candidates():
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
    mask = arrays["special_candidate_mask"][0]
    type_ids = arrays["special_candidate_type_id"][0]
    active = type_ids[mask > 0]
    assert XMODEL1_SPECIAL_TYPE_REACH in active
    assert arrays["chosen_special_candidate_idx"][0] >= 0


def test_build_special_candidate_arrays_exports_call_none_candidates():
    state = {
        "hand": ["2m", "3m", "4m", "5m", "6m", "7m", "5p", "5p", "6p", "7p", "E", "E", "P"],
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "oya": 0,
        "reached": [False, False, False, False],
    }
    legal_actions = [
        {"type": "pon", "actor": 0, "pai": "5p", "consumed": ["5p", "5p"], "target": 1},
        {"type": "none"},
    ]
    feat, type_id, mask, quality, rank, hard_bad, chosen_idx = build_special_candidate_arrays(
        state,
        0,
        legal_actions,
        {"type": "none"},
    )
    active = type_id[mask > 0]
    assert XMODEL1_SPECIAL_TYPE_PON in active
    assert XMODEL1_SPECIAL_TYPE_NONE in active
    assert quality.shape == (12,)
    assert rank.shape == (12,)
    assert hard_bad.shape == (12,)
    assert chosen_idx >= 0


def test_build_special_candidate_arrays_exports_three_concrete_chi_candidates():
    state = {
        "hand": ["2m", "4m", "5m", "6m", "7m", "8m", "3p", "4p", "5p", "6s", "7s", "8s", "P"],
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "oya": 0,
        "reached": [False, False, False, False],
    }
    legal_actions = [
        {"type": "chi", "actor": 0, "pai": "2m", "consumed": ["3m", "4m"], "target": 1},
        {"type": "chi", "actor": 0, "pai": "5m", "consumed": ["4m", "6m"], "target": 1},
        {"type": "chi", "actor": 0, "pai": "7m", "consumed": ["5m", "6m"], "target": 1},
        {"type": "none"},
    ]
    feat, type_id, mask, _quality, _rank, _hard_bad, _chosen_idx = build_special_candidate_arrays(
        state,
        0,
        legal_actions,
        {"type": "none"},
    )
    active = set(type_id[mask > 0])
    assert XMODEL1_SPECIAL_TYPE_CHI_LOW in active
    assert XMODEL1_SPECIAL_TYPE_CHI_MID in active
    assert XMODEL1_SPECIAL_TYPE_CHI_HIGH in active


def test_build_special_candidate_arrays_emits_richer_special_semantics():
    state = {
        "hand": ["2m", "3m", "4m", "5m", "6m", "7m", "5p", "5p", "6p", "7p", "E", "E", "P"],
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "oya": 0,
        "reached": [False, True, False, False],
        "dora_marker": "4p",
    }
    legal_actions = [
        {"type": "pon", "actor": 0, "pai": "5p", "consumed": ["5p", "5p"], "target": 1},
        {"type": "none"},
    ]
    feat, type_id, mask, quality, rank, hard_bad, chosen_idx = build_special_candidate_arrays(
        state,
        0,
        legal_actions,
        {"type": "pon", "pai": "5p", "consumed": ["5p", "5p"], "target": 1},
    )
    active_slots = np.where(mask > 0)[0]
    assert len(active_slots) >= 2
    pon_slot = int(np.where(type_id == XMODEL1_SPECIAL_TYPE_PON)[0][0])
    none_slot = int(np.where(type_id == XMODEL1_SPECIAL_TYPE_NONE)[0][0])
    assert float(feat[pon_slot, 12]) >= 0.0  # action_dora_bonus
    assert float(feat[pon_slot, 14]) >= 0.0  # risk_proxy_shimocha
    assert float(feat[none_slot, 8]) == 0.0  # none after_value_norm
    assert quality[pon_slot] != 0.0
    assert rank[pon_slot] in {0, 1, 2, 3}
    assert hard_bad[pon_slot] in {0, 1}
    assert chosen_idx == pon_slot


def test_build_candidate_features_emits_after_state_path_metrics():
    state = {
        "hand": ["1m", "2m", "2m", "3m", "4m", "5m", "6m", "3p", "4p", "5p", "4s", "5s", "6s", "7s"],
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "oya": 0,
        "reached": [False, True, False, False],
        "dora_marker": "1m",
    }
    feat, flags, quality, rank, hard_bad = build_candidate_features(state, 0, {"type": "dahai", "pai": "1m"})

    assert feat.shape == (XMODEL1_CANDIDATE_FEATURE_DIM,)
    assert flags.shape == (XMODEL1_CANDIDATE_FLAG_DIM,)
    assert float(feat[21]) > 0.0  # confirmed_han_floor_norm
    assert float(feat[16]) == 1.0  # tanyao_path
    assert float(feat[6]) > 0.0  # after_dora_count
    assert np.isfinite(feat).all()
    assert quality != 0.0
    assert rank in {0, 1, 2, 3}
    assert hard_bad in {0, 1}


def test_xmodel1_events_to_arrays_exports_none_special_only_sample():
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
                ["5p", "5p", "7s", "7s", "8s", "8s", "9s", "9s", "E", "E", "S", "S", "W"],
                ["2s"] * 13,
                ["3s"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "5p"},
        {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "W"},
    ]
    arrays = events_to_xmodel1_arrays(events, replay_id="none_fixture.mjson")
    assert arrays is not None
    special_rows = np.where(arrays["sample_type"] == XMODEL1_SAMPLE_TYPE_CALL)[0]
    assert len(special_rows) >= 1
    row = special_rows[0]
    assert int(arrays["chosen_candidate_idx"][row]) == -1
    assert int(arrays["action_idx_target"][row]) == 44
    active = arrays["special_candidate_type_id"][row][arrays["special_candidate_mask"][row] > 0]
    assert XMODEL1_SPECIAL_TYPE_NONE in active


def test_xmodel1_events_to_arrays_keeps_hora_as_dedicated_special_sample():
    arrays = events_to_xmodel1_arrays(_tsumo_hora_events(), replay_id="hora_fixture.mjson")
    assert arrays is not None
    hora_rows = np.where(arrays["sample_type"] == XMODEL1_SAMPLE_TYPE_HORA)[0]
    assert len(hora_rows) == 1
    row = int(hora_rows[0])
    assert int(arrays["chosen_candidate_idx"][row]) == -1
    assert int(arrays["action_idx_target"][row]) == 42
    chosen_special_idx = int(arrays["chosen_special_candidate_idx"][row])
    assert chosen_special_idx >= 0
    assert int(arrays["special_candidate_type_id"][row, chosen_special_idx]) == XMODEL1_SPECIAL_TYPE_HORA
