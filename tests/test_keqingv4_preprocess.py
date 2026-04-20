from __future__ import annotations

import json
from pathlib import Path
import subprocess

import keqingv4.preprocess_features as v4_pf
import numpy as np
import pytest
from mahjong_env.action_space import ANKAN_IDX, CHI_LOW_IDX, action_to_idx
from keqing_core import (
    build_keqingv4_call_summary,
    build_keqingv4_discard_summary,
    build_keqingv4_special_summary,
    build_keqingv4_typed_summaries,
    enumerate_keqingv4_live_draw_weights,
    enumerate_keqingv4_post_meld_discards,
    enumerate_keqingv4_reach_discards,
    project_keqingv4_call_snapshot,
    project_keqingv4_discard_snapshot,
    project_keqingv4_reach_snapshot,
    project_keqingv4_rinshan_draw_snapshot,
)
from keqingv4.cached_dataset import CachedMjaiDatasetV4
from keqingv4.preprocess_features import (
    _best_discard_summary_vector,
    _call_action_slot,
    _projected_call_summary_vector,
    _project_call_state,
    _visible_counts34_from_state,
    build_typed_action_summaries,
)
from mahjong_env.replay import _calc_normal_progress as _replay_calc_normal_progress
from training.preprocess import KeqingV4PreprocessAdapter, events_to_cached_arrays
from training.cache_schema import KEQINGV4_OPPORTUNITY_DIM, KEQINGV4_SUMMARY_DIM


def test_build_typed_action_summaries_smoke():
    discard_state = {
        "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
        "tsumo_pai": "4p",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    discard_legal = [
        {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
        {"type": "dahai", "actor": 0, "pai": "1p", "tsumogiri": False},
        {"type": "reach", "actor": 0},
    ]
    discard_summary, call_summary, special_summary, opportunity = build_typed_action_summaries(
        discard_state,
        0,
        discard_legal,
    )
    assert discard_summary.shape == (34, KEQINGV4_SUMMARY_DIM)
    assert call_summary.shape == (8, KEQINGV4_SUMMARY_DIM)
    assert special_summary.shape == (3, KEQINGV4_SUMMARY_DIM)
    assert opportunity.shape == (KEQINGV4_OPPORTUNITY_DIM,)
    assert opportunity.tolist() == [1, 0, 0]
    assert float(discard_summary[12].sum()) > 0.0
    assert float(special_summary[0].sum()) > 0.0

    call_state = {
        "hand": ["1m", "2m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"],
        "melds": [[], [], [], []],
        "discards": [["3m"], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    call_legal = [
        {"type": "chi", "actor": 1, "pai": "3m", "consumed": ["1m", "2m"], "target": 0},
        {"type": "chi", "actor": 1, "pai": "3m", "consumed": ["2m", "4m"], "target": 0},
        {"type": "none"},
    ]
    _, call_summary, _, opportunity = build_typed_action_summaries(call_state, 1, call_legal)
    assert float(call_summary[:3].sum()) > 0.0
    assert float(call_summary[7].sum()) > 0.0
    assert opportunity.tolist() == [0, 0, 1]


def test_build_typed_action_summaries_falls_back_only_for_missing_rust_capability(monkeypatch):
    state = {
        "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
        "tsumo_pai": "4p",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    legal = [
        {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
        {"type": "dahai", "actor": 0, "pai": "1p", "tsumogiri": False},
        {"type": "reach", "actor": 0},
    ]
    monkeypatch.setattr(
        v4_pf,
        "_rust_build_keqingv4_typed_summaries",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Rust keqingv4 typed summaries capability is not available")
        ),
    )

    discard_summary, call_summary, special_summary, opportunity = build_typed_action_summaries(state, 0, legal)

    assert discard_summary.shape == (34, KEQINGV4_SUMMARY_DIM)
    assert call_summary.shape == (8, KEQINGV4_SUMMARY_DIM)
    assert special_summary.shape == (3, KEQINGV4_SUMMARY_DIM)
    assert float(discard_summary[12].sum()) > 0.0
    assert float(special_summary[0].sum()) > 0.0
    assert opportunity.tolist() == [1, 0, 0]


def test_build_typed_action_summaries_propagates_unexpected_rust_bridge_errors(monkeypatch):
    state = {
        "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
        "tsumo_pai": "4p",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    legal = [
        {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
        {"type": "reach", "actor": 0},
    ]
    monkeypatch.setattr(
        v4_pf,
        "_rust_build_keqingv4_typed_summaries",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("typed summary bridge drift")),
    )

    with pytest.raises(RuntimeError, match="typed summary bridge drift"):
        build_typed_action_summaries(state, 0, legal)


def test_build_typed_action_summaries_emits_explicit_opportunity_contract():
    state = {
        "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
        "tsumo_pai": "4p",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    legal = [
        {"type": "reach", "actor": 0},
        {"type": "hora", "actor": 0, "pai": "4p"},
        {"type": "none"},
    ]
    _discard, _call, _special, opportunity = build_typed_action_summaries(state, 0, legal)
    assert opportunity.shape == (KEQINGV4_OPPORTUNITY_DIM,)
    assert opportunity.tolist() == [1, 1, 1]


def test_call_none_summary_matches_current_vector():
    state = {
        "hand": ["1m", "2m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"],
        "melds": [[], [], [], []],
        "discards": [["3m"], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    current_hand, visible_counts34, current_progress, current_open_hand, current_vector = v4_pf._current_summary_context(state, 1)
    assert current_hand
    legal = [{"type": "none"}]
    rust_summary = build_keqingv4_call_summary(state, 1, legal)
    np.testing.assert_allclose(rust_summary[7], current_vector, rtol=1e-6, atol=1e-6)


def test_call_action_slot_matches_expected_family_positions():
    assert _call_action_slot({"type": "none"}) == 7
    assert _call_action_slot({"type": "pon", "pai": "5p", "consumed": ["5p", "5p"]}) == 3
    assert _call_action_slot({"type": "daiminkan", "pai": "5p", "consumed": ["5p", "5p", "5p"]}) == 4
    assert _call_action_slot({"type": "ankan", "pai": "5p", "consumed": ["5p", "5p", "5p", "5p"]}) == 5
    assert _call_action_slot({"type": "kakan", "pai": "5p", "consumed": ["5p", "5p", "5p"]}) == 6
    assert _call_action_slot({"type": "chi", "pai": "3m", "consumed": ["1m", "2m"]}) == 2
    assert _call_action_slot({"type": "chi", "pai": "3m", "consumed": ["2m", "4m"]}) == 1
    assert _call_action_slot({"type": "chi", "pai": "3m", "consumed": ["4m", "5m"]}) == 0


def test_rust_typed_summaries_bridge_matches_individual_bridges():
    state = {
        "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
        "tsumo_pai": "5s",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    legal = [
        {"type": "dahai", "actor": 0, "pai": "5s", "tsumogiri": True},
        {"type": "dahai", "actor": 0, "pai": "1p", "tsumogiri": False},
        {"type": "reach", "actor": 0},
    ]
    try:
        d_typed, c_typed, s_typed = build_keqingv4_typed_summaries(state, 0, legal)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    d = build_keqingv4_discard_summary(state, 0, legal)
    c = build_keqingv4_call_summary(state, 0, legal)
    s = build_keqingv4_special_summary(state, 0, legal)
    np.testing.assert_allclose(d_typed, d, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(c_typed, c, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(s_typed, s, rtol=1e-6, atol=1e-6)


def test_rust_discard_summary_bridge_matches_python_core_fields():
    state = {
        "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
        "tsumo_pai": "5s",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    legal = [
        {"type": "dahai", "actor": 0, "pai": "5s", "tsumogiri": True},
        {"type": "dahai", "actor": 0, "pai": "1p", "tsumogiri": False},
    ]
    try:
        rust_summary = build_keqingv4_discard_summary(state, 0, legal)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    visible = _visible_counts34_from_state(state)
    current_hand = [*state["hand"], state["tsumo_pai"]]
    current_progress = _replay_calc_normal_progress(current_hand, [], list(visible))
    current_guards = v4_pf._current_yaku_guards(state, 0, current_hand, [])
    after_1p = v4_pf._remove_tiles_once(current_hand, ["1p"])
    expected = v4_pf._summary_vector(
        hand_tiles=after_1p,
        melds=[],
        visible_counts34=visible,
        state=state,
        actor=0,
        current_shanten=current_progress.shanten,
        open_hand_flag=0.0,
        current_tanyao_keep=current_guards[0],
        current_yakuhai_pair=current_guards[1],
        current_chiitoi_path=current_guards[2],
        current_iipeiko_path=current_guards[3],
        current_pinfu_like_path=current_guards[4],
    )
    np.testing.assert_allclose(rust_summary[9], expected, rtol=1e-6, atol=1e-6)


def test_discard_summary_yaku_break_detects_chiitoi_path_break():
    state = {
        "hand": ["1m", "1m", "2m", "2m", "3p", "3p", "4p", "4p", "5s", "5s", "E", "E", "7m"],
        "tsumo_pai": "8m",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    legal = [
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
        {"type": "dahai", "actor": 0, "pai": "7m", "tsumogiri": False},
        {"type": "dahai", "actor": 0, "pai": "8m", "tsumogiri": True},
    ]
    rust_summary = build_keqingv4_discard_summary(state, 0, legal)
    visible = _visible_counts34_from_state(state)
    current_progress = _replay_calc_normal_progress([*state["hand"], state["tsumo_pai"]], [], list(visible))
    current_hand = [*state["hand"], state["tsumo_pai"]]
    current_guards = v4_pf._current_yaku_guards(state, 0, current_hand, [])
    break_vec = v4_pf._summary_vector(
        hand_tiles=v4_pf._remove_tiles_once(current_hand, ["1m"]),
        melds=[],
        visible_counts34=visible,
        state=state,
        actor=0,
        current_shanten=current_progress.shanten,
        open_hand_flag=0.0,
        current_tanyao_keep=current_guards[0],
        current_yakuhai_pair=current_guards[1],
        current_chiitoi_path=current_guards[2],
        current_iipeiko_path=current_guards[3],
        current_pinfu_like_path=current_guards[4],
    )
    keep_vec = v4_pf._summary_vector(
        hand_tiles=v4_pf._remove_tiles_once(current_hand, ["7m"]),
        melds=[],
        visible_counts34=visible,
        state=state,
        actor=0,
        current_shanten=current_progress.shanten,
        open_hand_flag=0.0,
        current_tanyao_keep=current_guards[0],
        current_yakuhai_pair=current_guards[1],
        current_chiitoi_path=current_guards[2],
        current_iipeiko_path=current_guards[3],
        current_pinfu_like_path=current_guards[4],
    )
    assert float(break_vec[22]) == 1.0
    assert float(keep_vec[22]) == 0.0
    assert float(keep_vec[20]) > float(break_vec[20])
    assert float(keep_vec[23]) > float(break_vec[23])
    np.testing.assert_allclose(rust_summary[0], break_vec, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rust_summary[6], keep_vec, rtol=1e-6, atol=1e-6)


def test_discard_summary_yaku_break_detects_iipeiko_path_break():
    state = {
        "hand": ["1m", "1m", "2m", "2m", "3m", "3m", "4p", "5p", "6p", "7s", "8s", "9s", "E"],
        "tsumo_pai": "N",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    legal = [
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
        {"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": False},
        {"type": "dahai", "actor": 0, "pai": "N", "tsumogiri": True},
    ]
    rust_summary = build_keqingv4_discard_summary(state, 0, legal)
    visible = _visible_counts34_from_state(state)
    current_progress = _replay_calc_normal_progress([*state["hand"], state["tsumo_pai"]], [], list(visible))
    current_hand = [*state["hand"], state["tsumo_pai"]]
    current_guards = v4_pf._current_yaku_guards(state, 0, current_hand, [])
    break_vec = v4_pf._summary_vector(
        hand_tiles=v4_pf._remove_tiles_once(current_hand, ["1m"]),
        melds=[],
        visible_counts34=visible,
        state=state,
        actor=0,
        current_shanten=current_progress.shanten,
        open_hand_flag=0.0,
        current_tanyao_keep=current_guards[0],
        current_yakuhai_pair=current_guards[1],
        current_chiitoi_path=current_guards[2],
        current_iipeiko_path=current_guards[3],
        current_pinfu_like_path=current_guards[4],
    )
    keep_vec = v4_pf._summary_vector(
        hand_tiles=v4_pf._remove_tiles_once(current_hand, ["E"]),
        melds=[],
        visible_counts34=visible,
        state=state,
        actor=0,
        current_shanten=current_progress.shanten,
        open_hand_flag=0.0,
        current_tanyao_keep=current_guards[0],
        current_yakuhai_pair=current_guards[1],
        current_chiitoi_path=current_guards[2],
        current_iipeiko_path=current_guards[3],
        current_pinfu_like_path=current_guards[4],
    )
    assert float(break_vec[22]) == 1.0
    assert float(keep_vec[22]) == 0.0
    assert float(keep_vec[20]) > float(break_vec[20])
    assert float(keep_vec[23]) > float(break_vec[23])
    np.testing.assert_allclose(rust_summary[0], break_vec, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(rust_summary[27], keep_vec, rtol=1e-6, atol=1e-6)


def test_summary_vector_one_shanten_shape_slots_can_be_nonzero():
    state = {
        "hand": ["1m", "2m", "3p", "3p", "3s", "4m", "4p", "4s", "5p", "5p", "5s", "7s", "7s"],
        "tsumo_pai": None,
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    visible = _visible_counts34_from_state(state)
    current_hand = list(state["hand"])
    current_progress = _replay_calc_normal_progress(current_hand, [], list(visible))
    assert current_progress.shanten == 1
    current_guards = v4_pf._current_yaku_guards(state, 0, current_hand, [])
    vec = v4_pf._summary_vector(
        hand_tiles=current_hand,
        melds=[],
        visible_counts34=visible,
        state=state,
        actor=0,
        current_shanten=current_progress.shanten,
        open_hand_flag=0.0,
        current_tanyao_keep=current_guards[0],
        current_yakuhai_pair=current_guards[1],
        current_chiitoi_path=current_guards[2],
        current_iipeiko_path=current_guards[3],
        current_pinfu_like_path=current_guards[4],
    )
    assert float(vec[5]) > 0.0 or float(vec[6]) > 0.0


def test_discard_summary_one_shanten_closed_value_proxies_can_be_positive():
    state = {
        "hand": ["1m", "1m", "2m", "2m", "3p", "3p", "4p", "4p", "5s", "5s", "E", "E", "7m"],
        "tsumo_pai": "8m",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    legal = [
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
    ]
    rust_summary = build_keqingv4_discard_summary(state, 0, legal)
    visible = _visible_counts34_from_state(state)
    current_hand = [*state["hand"], state["tsumo_pai"]]
    current_progress = _replay_calc_normal_progress(current_hand, [], list(visible))
    current_guards = v4_pf._current_yaku_guards(state, 0, current_hand, [])
    vec = v4_pf._summary_vector(
        hand_tiles=v4_pf._remove_tiles_once(current_hand, ["1m"]),
        melds=[],
        visible_counts34=visible,
        state=state,
        actor=0,
        current_shanten=current_progress.shanten,
        open_hand_flag=0.0,
        current_tanyao_keep=current_guards[0],
        current_yakuhai_pair=current_guards[1],
        current_chiitoi_path=current_guards[2],
        current_iipeiko_path=current_guards[3],
        current_pinfu_like_path=current_guards[4],
    )
    assert float(vec[21]) > 0.0
    assert float(vec[23]) > 0.0
    np.testing.assert_allclose(rust_summary[0], vec, rtol=1e-6, atol=1e-6)


def test_discard_summary_one_shanten_future_truth_beats_old_shape_only_proxy():
    state = {
        "hand": ["1m", "1m", "2m", "2m", "3p", "3p", "4p", "4p", "5s", "5s", "E", "E", "7m"],
        "tsumo_pai": "8m",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "last_tsumo": ["8m", None, None, None],
        "last_tsumo_raw": ["8m", None, None, None],
    }
    legal = [
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
    ]
    rust_summary = build_keqingv4_discard_summary(state, 0, legal)
    visible = _visible_counts34_from_state(state)
    current_hand = [*state["hand"], state["tsumo_pai"]]
    current_progress = _replay_calc_normal_progress(current_hand, [], list(visible))
    current_guards = v4_pf._current_yaku_guards(state, 0, current_hand, [])
    after_hand = v4_pf._remove_tiles_once(current_hand, ["1m"])
    vec = v4_pf._summary_vector(
        hand_tiles=after_hand,
        melds=[],
        visible_counts34=visible,
        state=state,
        actor=0,
        current_shanten=current_progress.shanten,
        open_hand_flag=0.0,
        current_tanyao_keep=current_guards[0],
        current_yakuhai_pair=current_guards[1],
        current_chiitoi_path=current_guards[2],
        current_iipeiko_path=current_guards[3],
        current_pinfu_like_path=current_guards[4],
    )
    progress = _replay_calc_normal_progress(after_hand, [], list(visible))
    assert progress.shanten == 1
    good_shape_live, improvement_live, tenpai_live, best_tenpai_waits_live = v4_pf._one_shanten_draw_analysis(
        after_hand,
        [],
        visible,
    )
    old_shape_bonus = min(
        1.0,
        0.6 * (good_shape_live / 34.0)
        + 0.4 * (improvement_live / 34.0)
        + 0.35 * (min(best_tenpai_waits_live, 12.0) / 12.0),
    )
    old_shape_only_proxy = min(1.0, tenpai_live / 34.0) * min(1.0, 0.35 * old_shape_bonus)
    assert float(vec[20]) > old_shape_only_proxy
    assert float(vec[21]) > 0.0
    assert float(vec[23]) > old_shape_only_proxy
    np.testing.assert_allclose(rust_summary[0], vec, rtol=1e-6, atol=1e-6)


def test_discard_summary_open_one_shanten_future_truth_can_lift_value_and_yakuhai_proxy():
    state = {
        "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "2p", "2p", "3p", "5s"],
        "tsumo_pai": "8s",
        "melds": [[{"type": "pon", "pai": "E", "consumed": ["E", "E"], "target": 1}], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "last_tsumo": ["8s", None, None, None],
        "last_tsumo_raw": ["8s", None, None, None],
    }
    legal = [
        {"type": "dahai", "actor": 0, "pai": "8s", "tsumogiri": True},
    ]
    rust_summary = build_keqingv4_discard_summary(state, 0, legal)
    visible = _visible_counts34_from_state(state)
    current_hand = [*state["hand"], state["tsumo_pai"]]
    current_progress = _replay_calc_normal_progress(current_hand, state["melds"][0], list(visible))
    current_guards = v4_pf._current_yaku_guards(state, 0, current_hand, state["melds"][0])
    after_hand = list(state["hand"])
    after_progress = _replay_calc_normal_progress(after_hand, state["melds"][0], list(visible))
    assert after_progress.shanten == 1
    exact_metrics = v4_pf._exact_one_shanten_truth_metrics(
        state=state,
        actor=0,
        after_hand=after_hand,
        progress=after_progress,
        visible_counts34=visible,
        open_hand_flag=1.0,
    )
    vec = v4_pf._summary_vector(
        hand_tiles=after_hand,
        melds=state["melds"][0],
        visible_counts34=visible,
        state=state,
        actor=0,
        current_shanten=current_progress.shanten,
        open_hand_flag=1.0,
        current_tanyao_keep=current_guards[0],
        current_yakuhai_pair=current_guards[1],
        current_chiitoi_path=current_guards[2],
        current_iipeiko_path=current_guards[3],
        current_pinfu_like_path=current_guards[4],
    )
    assert exact_metrics[0] > 0.0
    assert exact_metrics[1] == 0.0
    assert exact_metrics[3] > 0.0
    assert float(vec[20]) > 0.0
    assert float(vec[21]) >= float(exact_metrics[1])
    assert float(vec[23]) == 0.0
    assert float(vec[27]) > 0.0
    np.testing.assert_allclose(rust_summary[25], vec, rtol=1e-6, atol=1e-6)


def test_discard_summary_closed_two_shanten_future_truth_can_lift_value_proxies():
    state = {
        "hand": ["2m", "2m", "3m", "4m", "5m", "6m", "2p", "2p", "3p", "5s", "7s", "8s", "9s"],
        "tsumo_pai": "9p",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "last_tsumo": ["9p", None, None, None],
        "last_tsumo_raw": ["9p", None, None, None],
    }
    legal = [
        {"type": "dahai", "actor": 0, "pai": "9p", "tsumogiri": True},
    ]
    rust_summary = build_keqingv4_discard_summary(state, 0, legal)
    visible = _visible_counts34_from_state(state)
    current_hand = [*state["hand"], state["tsumo_pai"]]
    current_progress = _replay_calc_normal_progress(current_hand, [], list(visible))
    current_guards = v4_pf._current_yaku_guards(state, 0, current_hand, [])
    after_hand = list(state["hand"])
    after_progress = _replay_calc_normal_progress(after_hand, [], list(visible))
    assert after_progress.shanten == 2
    future_metrics = v4_pf._future_truth_metrics(
        state=state,
        actor=0,
        after_hand=after_hand,
        progress=after_progress,
        visible_counts34=visible,
        open_hand_flag=0.0,
    )
    vec = v4_pf._summary_vector(
        hand_tiles=after_hand,
        melds=[],
        visible_counts34=visible,
        state=state,
        actor=0,
        current_shanten=current_progress.shanten,
        open_hand_flag=0.0,
        current_tanyao_keep=current_guards[0],
        current_yakuhai_pair=current_guards[1],
        current_chiitoi_path=current_guards[2],
        current_iipeiko_path=current_guards[3],
        current_pinfu_like_path=current_guards[4],
    )
    assert future_metrics[0] > 0.0
    assert future_metrics[1] > 0.0
    assert float(vec[20]) > 0.0
    assert float(vec[21]) > 0.0
    assert float(vec[23]) > 0.0
    np.testing.assert_allclose(rust_summary[17], vec, rtol=1e-6, atol=1e-6)


def test_discard_summary_open_two_shanten_future_truth_can_lift_yakuhai_route():
    state = {
        "hand": ["2m", "2m", "3m", "4m", "5m", "6m", "2p", "2p", "3p", "5s"],
        "tsumo_pai": "8s",
        "melds": [[{"type": "pon", "pai": "E", "consumed": ["E", "E"], "target": 1}], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "last_tsumo": ["8s", None, None, None],
        "last_tsumo_raw": ["8s", None, None, None],
    }
    legal = [
        {"type": "dahai", "actor": 0, "pai": "8s", "tsumogiri": True},
    ]
    rust_summary = build_keqingv4_discard_summary(state, 0, legal)
    visible = _visible_counts34_from_state(state)
    current_hand = [*state["hand"], state["tsumo_pai"]]
    current_progress = _replay_calc_normal_progress(current_hand, state["melds"][0], list(visible))
    current_guards = v4_pf._current_yaku_guards(state, 0, current_hand, state["melds"][0])
    after_hand = list(state["hand"])
    after_progress = _replay_calc_normal_progress(after_hand, state["melds"][0], list(visible))
    assert after_progress.shanten == 2
    future_metrics = v4_pf._future_truth_metrics(
        state=state,
        actor=0,
        after_hand=after_hand,
        progress=after_progress,
        visible_counts34=visible,
        open_hand_flag=1.0,
    )
    vec = v4_pf._summary_vector(
        hand_tiles=after_hand,
        melds=state["melds"][0],
        visible_counts34=visible,
        state=state,
        actor=0,
        current_shanten=current_progress.shanten,
        open_hand_flag=1.0,
        current_tanyao_keep=current_guards[0],
        current_yakuhai_pair=current_guards[1],
        current_chiitoi_path=current_guards[2],
        current_iipeiko_path=current_guards[3],
        current_pinfu_like_path=current_guards[4],
    )
    assert future_metrics[0] > 0.0
    assert future_metrics[1] == 0.0
    assert future_metrics[3] > 0.0
    assert float(vec[20]) > 0.0
    assert float(vec[23]) == 0.0
    assert float(vec[27]) > 0.0
    np.testing.assert_allclose(rust_summary[25], vec, rtol=1e-6, atol=1e-6)


def test_future_truth_stays_non_recursive_above_train_ready_depth():
    state = {
        "hand": ["7s", "9s", "3m", "9m", "8p", "7p", "4p", "8s", "1p", "S", "3p", "1s", "7m"],
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    visible = _visible_counts34_from_state(state)
    progress = _replay_calc_normal_progress(state["hand"], [], list(visible))
    assert progress.shanten == 3

    metrics_closed = v4_pf._future_truth_metrics(
        state=state,
        actor=0,
        after_hand=state["hand"],
        progress=progress,
        visible_counts34=visible,
        open_hand_flag=0.0,
    )
    metrics_open = v4_pf._future_truth_metrics(
        state=state,
        actor=0,
        after_hand=state["hand"],
        progress=progress,
        visible_counts34=visible,
        open_hand_flag=1.0,
    )

    assert metrics_closed == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert metrics_open == (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def test_discard_summary_closed_tenpai_without_floor_still_gets_truth_backed_value_proxy():
    state = {
        "hand": ["1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "7s", "8s", "5p", "5p"],
        "tsumo_pai": "9s",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["2p"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "last_tsumo": ["9s", None, None, None],
        "last_tsumo_raw": ["9s", None, None, None],
    }
    legal = [
        {"type": "dahai", "actor": 0, "pai": "9s", "tsumogiri": True},
    ]
    rust_summary = build_keqingv4_discard_summary(state, 0, legal)
    visible = _visible_counts34_from_state(state)
    current_hand = [*state["hand"], state["tsumo_pai"]]
    current_progress = _replay_calc_normal_progress(current_hand, [], list(visible))
    current_guards = v4_pf._current_yaku_guards(state, 0, current_hand, [])
    vec = v4_pf._summary_vector(
        hand_tiles=v4_pf._remove_tiles_once(current_hand, ["9s"]),
        melds=[],
        visible_counts34=visible,
        state=state,
        actor=0,
        current_shanten=current_progress.shanten,
        open_hand_flag=0.0,
        current_tanyao_keep=current_guards[0],
        current_yakuhai_pair=current_guards[1],
        current_chiitoi_path=current_guards[2],
        current_iipeiko_path=current_guards[3],
        current_pinfu_like_path=current_guards[4],
    )
    assert float(vec[1]) == 1.0
    assert float(vec[20]) > 0.0
    assert float(vec[21]) >= float(vec[20])
    assert float(vec[23]) > 0.0
    np.testing.assert_allclose(rust_summary[26], vec, rtol=1e-6, atol=1e-6)


def test_keqingv4_events_to_arrays_and_cached_dataset_smoke(tmp_path: Path):
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
        {"type": "tsumo", "actor": 0, "pai": "5s"},
        {"type": "dahai", "actor": 0, "pai": "5s", "tsumogiri": True},
    ]

    arrays = events_to_cached_arrays(
        events,
        adapter=KeqingV4PreprocessAdapter(),
        value_strategy="mc_return",
        encode_module="training.state_features",
    )
    assert arrays is not None
    assert "pts_given_win_target" in arrays
    assert "pts_given_dealin_target" in arrays
    assert "opp_tenpai_target" in arrays
    assert "event_history" in arrays
    assert "v4_opportunity" in arrays
    assert arrays["v4_discard_summary"].shape[1:] == (34, KEQINGV4_SUMMARY_DIM)
    assert arrays["v4_call_summary"].shape[1:] == (8, KEQINGV4_SUMMARY_DIM)
    assert arrays["v4_special_summary"].shape[1:] == (3, KEQINGV4_SUMMARY_DIM)
    assert arrays["opp_tenpai_target"].shape[1:] == (3,)
    assert arrays["event_history"].shape[1:] == (48, 5)
    assert arrays["v4_opportunity"].shape[1:] == (KEQINGV4_OPPORTUNITY_DIM,)
    assert np.any(arrays["event_history"][0, :, 1] != 0)

    cache_path = tmp_path / "sample.npz"
    np.savez(cache_path, **{k: v for k, v in arrays.items() if not k.startswith("_")})
    dataset = CachedMjaiDatasetV4([cache_path], shuffle=False, buffer_size=2, seed=7, aug_perms=0)
    sample = next(iter(dataset))
    assert sample[8].shape == ()
    assert sample[9].shape == ()
    assert sample[10].shape == (3,)
    assert sample[14].shape == (48, 5)
    assert sample[16].shape == (KEQINGV4_OPPORTUNITY_DIM,)
    assert sample[17].shape == (34, KEQINGV4_SUMMARY_DIM)
    assert sample[18].shape == (8, KEQINGV4_SUMMARY_DIM)
    assert sample[19].shape == (3, KEQINGV4_SUMMARY_DIM)


def test_keqingv4_cached_dataset_fails_fast_on_contract_mismatch(tmp_path: Path):
    broken = tmp_path / "broken.npz"
    np.savez(
        broken,
        tile_feat=np.zeros((1, 57, 34), dtype=np.float16),
        scalar=np.zeros((1, 56), dtype=np.float16),
        mask=np.zeros((1, 45), dtype=np.uint8),
        action_idx=np.zeros((1,), dtype=np.int16),
        value=np.zeros((1,), dtype=np.float32),
        score_delta_target=np.zeros((1,), dtype=np.float32),
        win_target=np.zeros((1,), dtype=np.float32),
        dealin_target=np.zeros((1,), dtype=np.float32),
        pts_given_win_target=np.zeros((1,), dtype=np.float32),
        pts_given_dealin_target=np.zeros((1,), dtype=np.float32),
        opp_tenpai_target=np.zeros((1, 3), dtype=np.float32),
        event_history=np.zeros((1, 48, 5), dtype=np.int16),
        v4_opportunity=np.zeros((1, KEQINGV4_OPPORTUNITY_DIM), dtype=np.uint8),
        v4_discard_summary=np.zeros((1, 34, KEQINGV4_SUMMARY_DIM), dtype=np.float16),
        v4_call_summary=np.zeros((1, 7, KEQINGV4_SUMMARY_DIM), dtype=np.float16),
        v4_special_summary=np.zeros((1, 3, KEQINGV4_SUMMARY_DIM), dtype=np.float16),
    )
    dataset = CachedMjaiDatasetV4([broken], shuffle=False, buffer_size=1, seed=7, aug_perms=0)
    with pytest.raises(ValueError, match="v4_call_summary shape mismatch"):
        next(iter(dataset))


def test_call_summary_uses_best_post_meld_discard_projection():
    state = {
        "hand": ["1m", "2m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"],
        "melds": [[], [], [], []],
        "discards": [["3m"], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    chi_action = {"type": "chi", "actor": 1, "pai": "3m", "consumed": ["1m", "2m"], "target": 0}
    _, call_summary, _, _opportunity = build_typed_action_summaries(state, 1, [chi_action, {"type": "none"}])

    projected_hand, projected_melds, projected_open_hand, needs_rinshan = _project_call_state(state, 1, chi_action) or (None, None, None, None)
    assert projected_hand is not None
    assert needs_rinshan is False

    current_progress = _replay_calc_normal_progress(state["hand"], [], list(_visible_counts34_from_state(state)))
    expected = _best_discard_summary_vector(
        hand_tiles=projected_hand,
        melds=projected_melds,
        visible_counts34=_visible_counts34_from_state(state),
        state=state,
        actor=1,
        current_shanten=current_progress.shanten,
        open_hand_flag=projected_open_hand,
    )

    chi_slot = action_to_idx(chi_action) - CHI_LOW_IDX
    np.testing.assert_allclose(call_summary[chi_slot], expected, rtol=1e-6, atol=1e-6)
    assert float(call_summary[chi_slot][11]) == 1.0


def test_rust_call_summary_bridge_matches_python_core_fields():
    state = {
        "hand": ["1m", "2m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"],
        "melds": [[], [], [], []],
        "discards": [["3m"], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    legal = [
        {"type": "chi", "actor": 1, "pai": "3m", "consumed": ["1m", "2m"], "target": 0},
        {"type": "none"},
    ]
    try:
        rust_summary = build_keqingv4_call_summary(state, 1, legal)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    projected_hand, projected_melds, projected_open_hand, needs_rinshan = _project_call_state(state, 1, legal[0]) or (None, None, None, None)
    assert projected_hand is not None
    assert needs_rinshan is False
    current_progress = _replay_calc_normal_progress(state["hand"], [], list(_visible_counts34_from_state(state)))
    expected = _best_discard_summary_vector(
        hand_tiles=projected_hand,
        melds=projected_melds,
        visible_counts34=_visible_counts34_from_state(state),
        state=state,
        actor=1,
        current_shanten=current_progress.shanten,
        open_hand_flag=projected_open_hand,
    )
    chi_slot = action_to_idx(legal[0]) - CHI_LOW_IDX
    np.testing.assert_allclose(rust_summary[chi_slot], expected, rtol=1e-6, atol=1e-6)


def _best_rust_projected_discard_summary(snapshot: dict, actor: int) -> np.ndarray:
    legal = enumerate_keqingv4_post_meld_discards(snapshot, actor)
    discard_summary = build_keqingv4_discard_summary(snapshot, actor, legal)
    active_rows = [row for row in discard_summary if float(row[13]) > 0.0]
    assert active_rows
    return max(active_rows, key=v4_pf._summary_score)


def test_rust_call_summary_matches_projected_snapshot_discard_path():
    state = {
        "hand": ["1m", "2m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"],
        "melds": [[], [], [], []],
        "discards": [["3m"], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "last_discard": {"pai": "3m"},
        "last_kakan": None,
        "actor_to_move": 1,
        "tsumo_pai": None,
    }
    chi_action = {"type": "chi", "actor": 1, "pai": "3m", "consumed": ["1m", "2m"], "target": 0}
    legal = [chi_action]
    rust_summary = build_keqingv4_call_summary(state, 1, legal)
    projected = project_keqingv4_call_snapshot(state, 1, chi_action)
    expected = _best_rust_projected_discard_summary(projected, 1)
    chi_slot = _call_action_slot(chi_action)
    assert chi_slot is not None
    np.testing.assert_allclose(rust_summary[chi_slot], expected, rtol=1e-6, atol=1e-6)


def test_rust_call_snapshot_projection_matches_python_projection():
    state = {
        "hand": ["1m", "2m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"],
        "melds": [[], [], [], []],
        "discards": [["3m"], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "last_discard": {"pai": "3m"},
        "last_kakan": None,
        "actor_to_move": 1,
        "tsumo_pai": None,
    }
    chi_action = {"type": "chi", "actor": 1, "pai": "3m", "consumed": ["1m", "2m"], "target": 0}
    try:
        projected = project_keqingv4_call_snapshot(state, 1, chi_action)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    expected_hand, expected_melds, _expected_open, _needs_rinshan = _project_call_state(state, 1, chi_action)
    assert projected is not None
    assert projected["hand"] == expected_hand
    assert projected["melds"][1] == expected_melds
    assert projected["last_discard"] is None
    assert projected["last_kakan"] is None
    assert projected["actor_to_move"] == 1
    assert projected["tsumo_pai"] is None


def test_rust_discard_snapshot_projection_matches_python_projection():
    state = {
        "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "actor_to_move": 0,
        "tsumo_pai": "5s",
    }
    try:
        projected = project_keqingv4_discard_snapshot(state, 0, "1p")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    expected_hand = v4_pf._remove_tiles_once(state["hand"], ["1p"])
    assert projected["hand"] == expected_hand
    assert projected["discards"][0] == ["1p"]
    assert projected["tsumo_pai"] is None


def test_rust_rinshan_draw_snapshot_projection_matches_python_projection():
    state = {
        "hand": ["2m", "3m", "4m", "5m", "6m", "7m", "2p", "3p", "4p", "5p", "6p", "7p", "P"],
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "last_tsumo": [None, None, None, None],
        "last_tsumo_raw": [None, None, None, None],
        "actor_to_move": 0,
        "tsumo_pai": None,
    }
    try:
        projected = project_keqingv4_rinshan_draw_snapshot(state, 0, "5mr")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert projected["hand"][-1] == "5mr"
    assert projected["tsumo_pai"] == "5mr"
    assert projected["last_tsumo"][0] == "5m"
    assert projected["last_tsumo_raw"][0] == "5mr"
    assert projected["actor_to_move"] == 0
    assert projected["last_discard"] is None
    assert projected["last_kakan"] is None


def test_rust_post_meld_discard_enumeration_matches_python_helper():
    state = {
        "hand": ["1m", "1m", "2m", "2m", "3m", "5pr"],
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    try:
        rust_actions = enumerate_keqingv4_post_meld_discards(state, 0)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    expected_actions = [
        {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
        {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": False},
        {"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": False},
        {"type": "dahai", "actor": 0, "pai": "5pr", "tsumogiri": False},
    ]
    assert rust_actions == expected_actions


def test_rust_live_draw_weight_enumeration_matches_python_visible_count_logic():
    state = {
        "hand": ["1m", "2m", "3m"],
        "melds": [[{"type": "chi", "pai": "3p", "consumed": ["1p", "2p"], "target": 1}], [], [], []],
        "discards": [["5s"], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "tsumo_pai": "E",
    }
    try:
        rust_weights = enumerate_keqingv4_live_draw_weights(state)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    visible = _visible_counts34_from_state(state)
    expected = []
    for tile34, seen in enumerate(visible):
        live = max(0, 4 - int(seen))
        if live > 0:
            expected.append((v4_pf._TILE34_STR[tile34], live))
    assert rust_weights == expected


def test_rust_reach_discard_enumeration_matches_python_helper(monkeypatch):
    state = {
        "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
        "tsumo_pai": "5s",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "last_tsumo": ["5s", None, None, None],
        "last_tsumo_raw": ["5s", None, None, None],
    }
    try:
        rust_discards = enumerate_keqingv4_reach_discards(state, 0)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    from collections import Counter
    expected = v4_pf._reach_discard_candidates(Counter([*state["hand"], state["tsumo_pai"]]), "5s", "5s")
    assert rust_discards == expected


def test_rust_reach_snapshot_projection_sets_reach_flags():
    state = {
        "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "pending_reach": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "tsumo_pai": "5s",
    }
    try:
        projected = project_keqingv4_reach_snapshot(state, 0, "1p")
    except RuntimeError as exc:
        pytest.skip(str(exc))
    assert projected["reached"][0] is True
    assert projected["pending_reach"][0] is False
    assert projected["discards"][0] == ["1p"]
    assert projected["tsumo_pai"] is None


def test_ankan_summary_uses_rinshan_projection_and_stays_closed():
    state = {
        "hand": ["1m", "1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p"],
        "tsumo_pai": "2p",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["9m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    ankan_action = {"type": "ankan", "actor": 0, "pai": "1m", "consumed": ["1m", "1m", "1m", "1m"]}
    _, call_summary, _, _opportunity = build_typed_action_summaries(state, 0, [ankan_action])

    projected_hand, projected_melds, projected_open_hand, needs_rinshan = _project_call_state(state, 0, ankan_action) or (None, None, None, None)
    assert projected_hand is not None
    assert needs_rinshan is True
    assert projected_open_hand == 0.0

    expected = _projected_call_summary_vector(
        state=state,
        actor=0,
        action=ankan_action,
    )
    assert expected is not None

    ankan_slot = ANKAN_IDX - CHI_LOW_IDX
    np.testing.assert_allclose(call_summary[ankan_slot], expected, rtol=1e-6, atol=1e-6)
    assert float(call_summary[ankan_slot][11]) == 0.0


def test_rust_ankan_summary_matches_projected_rinshan_snapshot_path():
    state = {
        "hand": ["1m", "1m", "1m", "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p"],
        "tsumo_pai": "2p",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["9m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "last_tsumo": ["2p", None, None, None],
        "last_tsumo_raw": ["2p", None, None, None],
        "actor_to_move": 0,
    }
    ankan_action = {"type": "ankan", "actor": 0, "pai": "1m", "consumed": ["1m", "1m", "1m", "1m"]}
    rust_summary = build_keqingv4_call_summary(state, 0, [ankan_action])
    projected = project_keqingv4_call_snapshot(state, 0, ankan_action)
    weighted = np.zeros((KEQINGV4_SUMMARY_DIM,), dtype=np.float32)
    total = 0.0
    for draw_tile, live in enumerate_keqingv4_live_draw_weights(projected):
        draw_snapshot = project_keqingv4_rinshan_draw_snapshot(projected, 0, draw_tile)
        weighted += float(live) * _best_rust_projected_discard_summary(draw_snapshot, 0)
        total += float(live)
    assert total > 0.0
    expected = weighted / total
    ankan_slot = _call_action_slot(ankan_action)
    assert ankan_slot is not None
    np.testing.assert_allclose(rust_summary[ankan_slot], expected, rtol=1e-6, atol=1e-6)


def test_reach_special_summary_uses_best_declaration_discard(monkeypatch):
    state = {
        "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
        "tsumo_pai": "5s",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    monkeypatch.setattr(
        v4_pf,
        "_rust_build_keqingv4_typed_summaries",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Rust keqingv4 typed summary capability is not available")
        ),
    )
    monkeypatch.setattr(v4_pf, "_reach_discard_candidates", lambda hand, last_tsumo, last_tsumo_raw: [("1p", False), ("5s", True)])

    _discard_summary, _call_summary, special_summary, _opportunity = build_typed_action_summaries(
        state,
        0,
        [{"type": "reach", "actor": 0}],
    )

    visible = _visible_counts34_from_state(state)
    current_hand = [*state["hand"], state["tsumo_pai"]]
    current_progress = _replay_calc_normal_progress(current_hand, [], list(visible))
    current_guards = v4_pf._current_yaku_guards(state, 0, current_hand, [])
    after_1p = v4_pf._remove_tiles_once(current_hand, ["1p"])
    after_5s = v4_pf._remove_tiles_once(current_hand, ["5s"])
    vec_1p = v4_pf._summary_vector(
        hand_tiles=after_1p,
        melds=[],
        visible_counts34=visible,
        state=state,
        actor=0,
        current_shanten=current_progress.shanten,
        open_hand_flag=0.0,
        current_tanyao_keep=current_guards[0],
        current_yakuhai_pair=current_guards[1],
        current_chiitoi_path=current_guards[2],
        current_iipeiko_path=current_guards[3],
        current_pinfu_like_path=current_guards[4],
    )
    vec_5s = v4_pf._summary_vector(
        hand_tiles=after_5s,
        melds=[],
        visible_counts34=visible,
        state=state,
        actor=0,
        current_shanten=current_progress.shanten,
        open_hand_flag=0.0,
        current_tanyao_keep=current_guards[0],
        current_yakuhai_pair=current_guards[1],
        current_chiitoi_path=current_guards[2],
        current_iipeiko_path=current_guards[3],
        current_pinfu_like_path=current_guards[4],
    )
    expected = vec_1p if v4_pf._summary_score(vec_1p) >= v4_pf._summary_score(vec_5s) else vec_5s

    np.testing.assert_allclose(special_summary[0], expected, rtol=1e-6, atol=1e-6)
    assert float(special_summary[0][11]) == 0.0


def test_rust_special_summary_bridge_matches_python_reach_core_fields(monkeypatch):
    state = {
        "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
        "tsumo_pai": "5s",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    legal = [{"type": "reach", "actor": 0}]
    try:
        rust_summary = build_keqingv4_special_summary(state, 0, legal)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    try:
        rust_discards = enumerate_keqingv4_reach_discards(state, 0)
    except RuntimeError as exc:
        pytest.skip(str(exc))
    visible = _visible_counts34_from_state(state)
    current_hand = [*state["hand"], state["tsumo_pai"]]
    current_progress = _replay_calc_normal_progress(current_hand, [], list(visible))
    current_guards = v4_pf._current_yaku_guards(state, 0, current_hand, [])
    best_score = -1e18
    expected = None
    for pai_out, _tsumogiri in rust_discards:
        after_hand = v4_pf._remove_tiles_once(current_hand, [pai_out])
        vec = v4_pf._summary_vector(
            hand_tiles=after_hand,
            melds=[],
            visible_counts34=visible,
            state=state,
            actor=0,
            current_shanten=current_progress.shanten,
            open_hand_flag=0.0,
            current_tanyao_keep=current_guards[0],
            current_yakuhai_pair=current_guards[1],
            current_chiitoi_path=current_guards[2],
            current_iipeiko_path=current_guards[3],
            current_pinfu_like_path=current_guards[4],
        )
        score = v4_pf._summary_score(vec)
        if score > best_score:
            best_score = score
            expected = vec
    assert expected is not None
    np.testing.assert_allclose(rust_summary[0], expected, rtol=1e-6, atol=1e-6)


def test_hora_special_summary_encodes_terminal_win_semantics():
    state = {
        "hand": ["2m", "3m", "4m", "5m", "6m", "7m", "2p", "3p", "4p", "6p", "7p", "8p", "P"],
        "tsumo_pai": "P",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "_hora_is_rinshan": True,
    }
    _discard_summary, _call_summary, special_summary, _opportunity = build_typed_action_summaries(
        state,
        0,
        [{"type": "hora", "actor": 0, "target": 0, "pai": "P", "is_rinshan": True}],
    )

    hora_vec = special_summary[1]
    assert float(hora_vec[0]) == -0.125
    assert float(hora_vec[1]) == 1.0
    assert float(hora_vec[2]) == 1.0  # tsumo
    assert float(hora_vec[3]) == 0.0  # not ron
    assert float(hora_vec[4]) == 1.0  # rinshan
    assert float(hora_vec[12]) == 1.0
    assert float(hora_vec[13]) == 1.0


def test_ryukyoku_special_summary_encodes_kyuushu_kyuuhai_shape():
    state = {
        "hand": ["1m", "9m", "1p", "9p", "1s", "9s", "E", "S", "W", "N", "P", "F", "C"],
        "tsumo_pai": "5m",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "scores": [25000, 25000, 25000, 25000],
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
    }
    _discard_summary, _call_summary, special_summary, _opportunity = build_typed_action_summaries(
        state,
        0,
        [{"type": "ryukyoku"}],
    )

    ryukyoku_vec = special_summary[2]
    assert np.isclose(float(ryukyoku_vec[2]), 13.0 / 13.0)
    assert float(ryukyoku_vec[3]) == 1.0
    assert np.isclose(float(ryukyoku_vec[4]), 13.0 / 9.0)
    assert float(ryukyoku_vec[12]) == 1.0
    assert float(ryukyoku_vec[13]) == 1.0


def test_preprocess_keqingv4_script_runs(tmp_path: Path):
    input_dir = tmp_path / "converted" / "ds1"
    input_dir.mkdir(parents=True)
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
        {"type": "tsumo", "actor": 0, "pai": "5s"},
        {"type": "dahai", "actor": 0, "pai": "5s", "tsumogiri": True},
    ]
    with (input_dir / "sample.mjson").open("w", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    output_dir = tmp_path / "processed_v4"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/preprocess_keqingv4.py",
            "--config",
            "configs/keqingv4_preprocess.yaml",
            "--data_dirs",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--jobs",
            "1",
            "--progress-every",
            "1",
        ],
        cwd=str(Path(__file__).resolve().parents[1]),
        check=True,
        capture_output=True,
        text=True,
    )
    exported = output_dir / "ds1" / "sample.npz"
    assert exported.exists()
    assert "Rust keqingv4 export completed:" in result.stdout
    with np.load(exported, allow_pickle=False) as data:
        assert "pts_given_win_target" in data.files
        assert "pts_given_dealin_target" in data.files
        assert "opp_tenpai_target" in data.files
        assert "event_history" in data.files
        assert data["v4_discard_summary"].shape[1:] == (34, KEQINGV4_SUMMARY_DIM)
        assert data["v4_call_summary"].shape[1:] == (8, KEQINGV4_SUMMARY_DIM)
        assert data["v4_special_summary"].shape[1:] == (3, KEQINGV4_SUMMARY_DIM)
        assert data["opp_tenpai_target"].shape[1:] == (3,)
        assert data["event_history"].shape[1:] == (48, 5)
