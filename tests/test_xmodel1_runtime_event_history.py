from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
import types

import keqing_core
from inference.default_context import DefaultDecisionContextBuilder
from inference.keqing_adapter import KeqingModelAdapter
from mahjong_env.event_history import EVENT_TYPE_DAHAI, EVENT_TYPE_TSUMO, compute_event_history as shared_compute_event_history
from mahjong_env.replay import build_replay_samples_mc_return
from mahjong_env.state import GameState
from xmodel1.features import compute_event_history as xmodel1_compute_event_history
from xmodel1.model import Xmodel1Model


@pytest.fixture(autouse=True)
def _enable_rust_for_runtime_context_tests():
    previous = keqing_core.is_enabled()
    keqing_core.enable_rust(True)
    yield
    keqing_core.enable_rust(previous)


def _inject_shanten_waits_stub(snap, *, hand_list, melds_list, model_version):
    snap["shanten"] = 1
    snap["waits_count"] = 0
    snap["waits_tiles"] = [0] * 34


def _minimal_snapshot(actor: int) -> dict:
    return {
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "dora_markers": [],
        "ura_dora_markers": [],
        "actor": actor,
        "hand": [],
        "discards": [[], [], [], []],
        "melds": [[], [], [], []],
        "reached": [False, False, False, False],
        "pending_reach": [False, False, False, False],
        "actor_to_move": None,
        "last_discard": None,
        "last_kakan": None,
        "last_tsumo": [None, None, None, None],
        "last_tsumo_raw": [None, None, None, None],
        "remaining_wall": None,
        "pending_rinshan_actor": None,
        "ryukyoku_tenpai_players": [],
        "furiten": [False, False, False, False],
        "sutehai_furiten": [False, False, False, False],
        "riichi_furiten": [False, False, False, False],
        "doujun_furiten": [False, False, False, False],
        "ippatsu_eligible": [False, False, False, False],
        "rinshan_tsumo": [False, False, False, False],
    }


def test_default_context_builder_injects_training_aligned_xmodel1_event_history():
    builder = DefaultDecisionContextBuilder(
        model_version="xmodel1",
        riichi_state=None,
        inject_shanten_waits=_inject_shanten_waits_stub,
    )
    state = GameState()
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
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
                ["1s"] * 13,
                ["2s"] * 13,
                ["3s"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 1, "pai": "6m"},
        {"type": "dahai", "actor": 1, "pai": "6m", "tsumogiri": True},
    ]
    assert builder.build(state, 0, events[0]) is None
    assert builder.build(state, 0, events[1]) is None
    assert builder.build(state, 0, events[2]) is None
    ctx = builder.build(state, 0, events[3])
    assert ctx is not None
    event_history = ctx.model_snap["event_history"]
    non_pad = event_history[event_history[:, 1] != 0]
    assert non_pad.shape[0] == 1
    assert int(non_pad[-1, 1]) == EVENT_TYPE_TSUMO
    assert not np.any(
        (event_history[:, 1] == EVENT_TYPE_DAHAI) & (event_history[:, 2] == 5)
    ), "current discard event should be excluded from xmodel1 runtime history"


def test_default_context_builder_prefers_rust_replay_snapshot(monkeypatch):
    builder = DefaultDecisionContextBuilder(
        model_version="xmodel1",
        riichi_state=None,
        inject_shanten_waits=_inject_shanten_waits_stub,
        enumerate_legal_actions_fn=lambda snap, seat: [],
    )
    state = GameState()
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
            "tehais": [["1m"] * 13, ["2m"] * 13, ["3m"] * 13, ["4m"] * 13],
        },
        {"type": "tsumo", "actor": 1, "pai": "6m"},
        {"type": "dahai", "actor": 1, "pai": "6m", "tsumogiri": True},
    ]
    assert builder.build(state, 0, events[0]) is None
    assert builder.build(state, 0, events[1]) is None
    assert builder.build(state, 0, events[2]) is None

    calls: list[int] = []

    def fake_replay_state_snapshot(history, actor):
        calls.append(len(history))
        snap = _minimal_snapshot(actor)
        snap["snapshot_source"] = "rust"
        snap["history_len"] = len(history)
        return snap

    monkeypatch.setattr("inference.default_context.keqing_core.is_enabled", lambda: True)
    monkeypatch.setattr("inference.default_context.keqing_core.replay_state_snapshot", fake_replay_state_snapshot)
    state.snapshot = lambda actor: (_ for _ in ()).throw(AssertionError("python snapshot should stay unused"))

    ctx = builder.build(state, 0, events[3])

    assert ctx is not None
    assert calls == [3, 4]
    assert ctx.runtime_snap["snapshot_source"] == "rust"
    assert ctx.model_snap["snapshot_source"] == "rust"


def test_default_context_builder_falls_back_only_for_missing_rust_snapshot_capability(monkeypatch):
    builder = DefaultDecisionContextBuilder(
        model_version="xmodel1",
        riichi_state=None,
        inject_shanten_waits=_inject_shanten_waits_stub,
        enumerate_legal_actions_fn=lambda snap, seat: [],
    )
    state = GameState()
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
            "tehais": [["1m"] * 13, ["2m"] * 13, ["3m"] * 13, ["4m"] * 13],
        },
        {"type": "tsumo", "actor": 1, "pai": "6m"},
        {"type": "dahai", "actor": 1, "pai": "6m", "tsumogiri": True},
    ]
    assert builder.build(state, 0, events[0]) is None
    assert builder.build(state, 0, events[1]) is None
    assert builder.build(state, 0, events[2]) is None

    calls = {"snapshot": 0}
    original_snapshot = state.snapshot

    def counting_snapshot(self, actor):
        calls["snapshot"] += 1
        snap = original_snapshot(actor)
        snap["snapshot_source"] = "python"
        return snap

    monkeypatch.setattr("inference.default_context.keqing_core.is_enabled", lambda: True)
    monkeypatch.setattr(
        "inference.default_context.keqing_core.replay_state_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Rust replay state snapshot capability is not available")
        ),
    )
    state.snapshot = types.MethodType(counting_snapshot, state)

    ctx = builder.build(state, 0, events[3])

    assert ctx is not None
    assert calls["snapshot"] == 2
    assert ctx.runtime_snap["snapshot_source"] == "python"
    assert ctx.model_snap["snapshot_source"] == "python"


def test_default_context_builder_propagates_unexpected_rust_snapshot_errors(monkeypatch):
    builder = DefaultDecisionContextBuilder(
        model_version="xmodel1",
        riichi_state=None,
        inject_shanten_waits=_inject_shanten_waits_stub,
        enumerate_legal_actions_fn=lambda snap, seat: [],
    )
    state = GameState()
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
            "tehais": [["1m"] * 13, ["2m"] * 13, ["3m"] * 13, ["4m"] * 13],
        },
        {"type": "tsumo", "actor": 1, "pai": "6m"},
        {"type": "dahai", "actor": 1, "pai": "6m", "tsumogiri": True},
    ]
    assert builder.build(state, 0, events[0]) is None
    assert builder.build(state, 0, events[1]) is None
    assert builder.build(state, 0, events[2]) is None

    monkeypatch.setattr("inference.default_context.keqing_core.is_enabled", lambda: True)
    monkeypatch.setattr(
        "inference.default_context.keqing_core.replay_state_snapshot",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("rust snapshot drift")),
    )
    state.snapshot = lambda actor: (_ for _ in ()).throw(AssertionError("python snapshot should stay unused"))

    with pytest.raises(RuntimeError, match="rust snapshot drift"):
        builder.build(state, 0, events[3])


def test_xmodel1_features_event_history_reuses_shared_owner():
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
            "tehais": [["1m"] * 13, ["2m"] * 13, ["3m"] * 13, ["4m"] * 13],
        },
        {"type": "foo", "actor": 1, "pai": "1m"},
    ]
    assert np.array_equal(
        xmodel1_compute_event_history(events, len(events)),
        shared_compute_event_history(events, len(events)),
    )


def _sample_discard_state():
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
    samples = build_replay_samples_mc_return(events, strict_legal_labels=True)
    sample = next(s for s in samples if s.label_action.get("type") == "dahai")
    snap = dict(sample.state)
    snap["legal_actions"] = sample.legal_actions
    return snap, sample.actor


def test_keqing_adapter_xmodel1_forward_passes_event_history(tmp_path: Path, monkeypatch):
    ckpt = tmp_path / "xmodel1_test.pth"
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=56,
        candidate_feature_dim=35,
        candidate_flag_dim=10,
        hidden_dim=32,
        num_res_blocks=1,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": {
                "model_name": "xmodel1",
                "state_tile_channels": 57,
                "state_scalar_dim": 56,
                "candidate_feature_dim": 35,
                "candidate_flag_dim": 10,
                "hidden_dim": 32,
                "num_res_blocks": 1,
            },
            "model_version": "xmodel1",
        },
        ckpt,
    )
    snap, actor = _sample_discard_state()
    event_history = np.zeros((48, 5), dtype=np.int16)
    event_history[-1] = np.array([0, EVENT_TYPE_TSUMO, 12, 0, 0], dtype=np.int16)
    snap["event_history"] = event_history
    adapter = KeqingModelAdapter.from_checkpoint(ckpt, device=torch.device("cpu"))
    captured: dict[str, torch.Tensor | None] = {}
    original_forward = adapter.model.forward

    def _wrapped_forward(*args, **kwargs):
        captured["event_history"] = kwargs.get("event_history")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(adapter.model, "forward", _wrapped_forward)
    result = adapter.forward(snap, actor)

    assert result.policy_logits.shape == (45,)
    assert captured["event_history"] is not None
    assert captured["event_history"].shape == (1, 48, 5)
    assert captured["event_history"][0, -1].tolist() == [0, EVENT_TYPE_TSUMO, 12, 0, 0]
