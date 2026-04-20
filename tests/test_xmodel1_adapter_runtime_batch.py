from __future__ import annotations

import numpy as np

from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_HISTORY_SUMMARY_DIM,
)
import xmodel1.adapter as xadapter
import xmodel1.features as xfeatures
from xmodel1.adapter import Xmodel1Adapter
from xmodel1.model import Xmodel1Model


def test_xmodel1_adapter_build_runtime_batch_accepts_tuple_candidate_arrays(monkeypatch):
    monkeypatch.setattr(
        xadapter,
        "build_runtime_candidate_arrays",
        lambda *args, **kwargs: (
            np.ones((14, XMODEL1_CANDIDATE_FEATURE_DIM), dtype=np.float32),
            np.array([3] + [-1] * 13, dtype=np.int16),
            np.array([1] + [0] * 13, dtype=np.uint8),
            np.zeros((14, XMODEL1_CANDIDATE_FLAG_DIM), dtype=np.uint8),
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
        candidate_feature_dim=XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim=XMODEL1_CANDIDATE_FLAG_DIM,
        hidden_dim=32,
        num_res_blocks=1,
    )
    adapter = Xmodel1Adapter(model, device="cpu")

    batch = adapter.build_runtime_batch({"hand": ["1m"]}, actor=0)

    assert batch["state_tile_feat"].shape == (1, 57, 34)
    assert batch["state_scalar"].shape == (1, 56)
    assert batch["candidate_feat"].shape == (1, 14, XMODEL1_CANDIDATE_FEATURE_DIM)
    assert batch["candidate_tile_id"][0, 0].item() == 3
    assert batch["chosen_candidate_idx"].tolist() == [0]
    assert batch["history_summary"].shape == (1, XMODEL1_HISTORY_SUMMARY_DIM)


def test_xmodel1_adapter_build_runtime_batch_preserves_snapshot_event_history(monkeypatch):
    monkeypatch.setattr(
        xadapter,
        "build_runtime_candidate_arrays",
        lambda *args, **kwargs: (
            np.ones((14, XMODEL1_CANDIDATE_FEATURE_DIM), dtype=np.float32),
            np.array([3] + [-1] * 13, dtype=np.int16),
            np.array([1] + [0] * 13, dtype=np.uint8),
            np.zeros((14, XMODEL1_CANDIDATE_FLAG_DIM), dtype=np.uint8),
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
        candidate_feature_dim=XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim=XMODEL1_CANDIDATE_FLAG_DIM,
        hidden_dim=32,
        num_res_blocks=1,
    )
    adapter = Xmodel1Adapter(model, device="cpu")
    history_summary = np.zeros((XMODEL1_HISTORY_SUMMARY_DIM,), dtype=np.float16)
    history_summary[-1] = 0.5

    batch = adapter.build_runtime_batch({"hand": ["1m"], "history_summary": history_summary}, actor=0)

    assert batch["history_summary"][0, -1].item() == 0.5
