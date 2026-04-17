from __future__ import annotations

import numpy as np

import xmodel1.adapter as xadapter
import xmodel1.features as xfeatures
from xmodel1.adapter import Xmodel1Adapter
from xmodel1.model import Xmodel1Model


def test_xmodel1_adapter_build_runtime_batch_accepts_tuple_candidate_arrays(monkeypatch):
    monkeypatch.setattr(
        xadapter,
        "build_runtime_candidate_arrays",
        lambda *args, **kwargs: (
            np.ones((14, 35), dtype=np.float32),
            np.array([3] + [-1] * 13, dtype=np.int16),
            np.array([1] + [0] * 13, dtype=np.uint8),
            np.zeros((14, 10), dtype=np.uint8),
        ),
    )
    monkeypatch.setattr(
        xfeatures,
        "encode",
        lambda snap, actor, state_scalar_dim=56: (
            np.zeros((57, 34), dtype=np.float32),
            np.zeros((state_scalar_dim,), dtype=np.float32),
        ),
    )

    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=56,
        candidate_feature_dim=35,
        candidate_flag_dim=10,
        hidden_dim=32,
        num_res_blocks=1,
    )
    adapter = Xmodel1Adapter(model, device="cpu")

    batch = adapter.build_runtime_batch({"hand": ["1m"]}, actor=0)

    assert batch["state_tile_feat"].shape == (1, 57, 34)
    assert batch["state_scalar"].shape == (1, 56)
    assert batch["candidate_feat"].shape == (1, 14, 35)
    assert batch["candidate_tile_id"][0, 0].item() == 3
    assert batch["chosen_candidate_idx"].tolist() == [0]
    assert batch["event_history"].shape == (1, 48, 5)


def test_xmodel1_adapter_build_runtime_batch_preserves_snapshot_event_history(monkeypatch):
    monkeypatch.setattr(
        xadapter,
        "build_runtime_candidate_arrays",
        lambda *args, **kwargs: (
            np.ones((14, 35), dtype=np.float32),
            np.array([3] + [-1] * 13, dtype=np.int16),
            np.array([1] + [0] * 13, dtype=np.uint8),
            np.zeros((14, 10), dtype=np.uint8),
        ),
    )
    monkeypatch.setattr(
        xfeatures,
        "encode",
        lambda snap, actor, state_scalar_dim=56: (
            np.zeros((57, 34), dtype=np.float32),
            np.zeros((state_scalar_dim,), dtype=np.float32),
        ),
    )

    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=56,
        candidate_feature_dim=35,
        candidate_flag_dim=10,
        hidden_dim=32,
        num_res_blocks=1,
    )
    adapter = Xmodel1Adapter(model, device="cpu")
    event_history = np.zeros((48, 5), dtype=np.int16)
    event_history[-1] = np.array([1, 2, 5, 1, 1], dtype=np.int16)

    batch = adapter.build_runtime_batch({"hand": ["1m"], "event_history": event_history}, actor=0)

    assert batch["event_history"][0, -1].tolist() == [1, 2, 5, 1, 1]
