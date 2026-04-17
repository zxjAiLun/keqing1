from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from inference.keqing_adapter import KeqingModelAdapter
from mahjong_env.replay import build_supervised_samples
from keqingv4.model import KeqingV4Model
from training.cache_schema import KEQINGV4_SUMMARY_DIM


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
        {"type": "tsumo", "actor": 0, "pai": "5s"},
        {"type": "dahai", "actor": 0, "pai": "5s", "tsumogiri": True},
    ]
    samples = build_supervised_samples(events, value_strategy="mc_return", strict_legal_labels=True)
    sample = next(s for s in samples if s.label_action.get("type") == "dahai")
    snap = dict(sample.state)
    snap["legal_actions"] = sample.legal_actions
    return snap, sample.actor


def test_keqingv4_keqing_adapter_forward_with_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "keqingv4_test.pth"
    model = KeqingV4Model(
        hidden_dim=64,
        num_res_blocks=2,
        action_embed_dim=16,
        context_dim=12,
        dropout=0.0,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": {
                "model_name": "keqingv4",
                "hidden_dim": 64,
                "num_res_blocks": 2,
                "action_embed_dim": 16,
                "context_dim": 12,
                "dropout": 0.0,
            },
            "model_version": "keqingv4",
        },
        ckpt,
    )
    snap, actor = _sample_discard_state()
    adapter = KeqingModelAdapter.from_checkpoint(ckpt, device=torch.device("cpu"))
    result = adapter.forward(snap, actor)
    assert adapter.model_version == "keqingv4"
    assert result.policy_logits.shape == (45,)


def test_keqingv4_keqing_adapter_reuses_runtime_summary_cache(tmp_path: Path):
    ckpt = tmp_path / "keqingv4_cache_test.pth"
    model = KeqingV4Model(
        hidden_dim=64,
        num_res_blocks=2,
        action_embed_dim=16,
        context_dim=12,
        dropout=0.0,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": {
                "model_name": "keqingv4",
                "hidden_dim": 64,
                "num_res_blocks": 2,
                "action_embed_dim": 16,
                "context_dim": 12,
                "dropout": 0.0,
            },
            "model_version": "keqingv4",
        },
        ckpt,
    )
    snap, actor = _sample_discard_state()
    adapter = KeqingModelAdapter.from_checkpoint(ckpt, device=torch.device("cpu"))

    original_builder = adapter._runtime_v4_summary_builder
    calls = {"count": 0}

    def counting_builder(snap_arg, actor_arg, legal_actions_arg):
        calls["count"] += 1
        return original_builder(snap_arg, actor_arg, legal_actions_arg)

    adapter._runtime_v4_summary_builder = counting_builder

    result1 = adapter.forward(snap, actor)
    result2 = adapter.forward(snap, actor)

    assert calls["count"] == 1
    assert result1.policy_logits.shape == (45,)
    assert result2.policy_logits.shape == (45,)


def test_keqingv4_keqing_adapter_prefers_core_typed_summary_bridge(tmp_path: Path, monkeypatch):
    ckpt = tmp_path / "keqingv4_bridge_test.pth"
    model = KeqingV4Model(
        hidden_dim=64,
        num_res_blocks=2,
        action_embed_dim=16,
        context_dim=12,
        dropout=0.0,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": {
                "model_name": "keqingv4",
                "hidden_dim": 64,
                "num_res_blocks": 2,
                "action_embed_dim": 16,
                "context_dim": 12,
                "dropout": 0.0,
            },
            "model_version": "keqingv4",
        },
        ckpt,
    )
    snap, actor = _sample_discard_state()

    import keqing_core
    import keqingv4.preprocess_features as preprocess_features

    calls = {"core": 0}

    def fake_core_builder(snapshot, actor_arg, legal_actions):
        del snapshot, actor_arg, legal_actions
        calls["core"] += 1
        return (
            np.zeros((34, KEQINGV4_SUMMARY_DIM), dtype=np.float32),
            np.zeros((8, KEQINGV4_SUMMARY_DIM), dtype=np.float32),
            np.zeros((3, KEQINGV4_SUMMARY_DIM), dtype=np.float32),
        )

    def fail_python_builder(*args, **kwargs):
        raise AssertionError("python fallback should not be used when core typed bridge works")

    monkeypatch.setattr(keqing_core, "build_keqingv4_typed_summaries", fake_core_builder)
    monkeypatch.setattr(preprocess_features, "build_typed_action_summaries", fail_python_builder)

    adapter = KeqingModelAdapter.from_checkpoint(ckpt, device=torch.device("cpu"))
    result = adapter.forward(snap, actor)

    assert calls["core"] == 1
    assert result.policy_logits.shape == (45,)
