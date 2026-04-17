from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from inference.default_context import DefaultDecisionContextBuilder
from inference.keqing_adapter import KeqingModelAdapter
from mahjong_env.event_history import EVENT_TYPE_DAHAI, EVENT_TYPE_TSUMO
from mahjong_env.replay import build_supervised_samples
from mahjong_env.state import GameState
from xmodel1.model import Xmodel1Model


def _inject_shanten_waits_stub(snap, *, hand_list, melds_list, model_version):
    snap["shanten"] = 1
    snap["waits_count"] = 0
    snap["waits_tiles"] = [0] * 34


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
    samples = build_supervised_samples(events, value_strategy="mc_return", strict_legal_labels=True)
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
