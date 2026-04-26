from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from inference.keqing_adapter import KeqingModelAdapter
from mahjong_env.legal_actions import enumerate_legal_actions
from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_MAX_RESPONSE_CANDIDATES,
    XMODEL1_RULE_CONTEXT_DIM,
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
)
from xmodel1.checkpoint import build_xmodel1_checkpoint_metadata
from xmodel1.model import Xmodel1Model


def _sample_discard_state():
    snap = {
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"],
        "tsumo_pai": "4p",
        "melds": [[], [], [], []],
        "discards": [[], [], [], []],
        "dora_markers": ["1m"],
        "reached": [False, False, False, False],
        "pending_reach": [False, False, False, False],
        "last_tsumo": [None, None, None, None],
        "last_tsumo_raw": [None, None, None, None],
        "last_discard": None,
        "last_kakan": None,
        "actor_to_move": 0,
    }
    actor = 0
    snap["legal_actions"] = [a.to_mjai() if hasattr(a, "to_mjai") else dict(a) for a in enumerate_legal_actions(snap, actor)]
    return snap, actor


def test_xmodel1_keqing_adapter_forward_with_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "xmodel1_test.pth"
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=56,
        candidate_feature_dim=XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim=XMODEL1_CANDIDATE_FLAG_DIM,
        hidden_dim=32,
        num_res_blocks=1,
    )
    cfg = {
        "model_name": "xmodel1",
        "state_tile_channels": 57,
        "state_scalar_dim": 56,
        "candidate_feature_dim": XMODEL1_CANDIDATE_FEATURE_DIM,
        "candidate_flag_dim": XMODEL1_CANDIDATE_FLAG_DIM,
        "hidden_dim": 32,
        "num_res_blocks": 1,
        "schema_name": XMODEL1_SCHEMA_NAME,
        "schema_version": XMODEL1_SCHEMA_VERSION,
    }
    payload = {"model": model.state_dict()}
    payload.update(build_xmodel1_checkpoint_metadata(cfg=cfg, model=model))
    torch.save(payload, ckpt)
    snap, actor = _sample_discard_state()
    adapter = KeqingModelAdapter.from_checkpoint(ckpt, device=torch.device("cpu"))
    result = adapter.forward(snap, actor)
    assert result.policy_logits.shape == (45,)
    assert result.xmodel1 is not None
    assert result.xmodel1.discard_logits.shape[0] == 14
    assert result.xmodel1.response_logits.shape[0] == XMODEL1_MAX_RESPONSE_CANDIDATES
    assert len(result.aux.rank_probs) == 4
    assert isinstance(result.aux.final_score_delta, float)
    assert isinstance(result.aux.rank_pt_value, float)


def test_xmodel1_keqing_adapter_rejects_runtime_rule_context_shape_drift(tmp_path: Path):
    ckpt = tmp_path / "xmodel1_bad_rule_context.pth"
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=56,
        candidate_feature_dim=XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim=XMODEL1_CANDIDATE_FLAG_DIM,
        hidden_dim=32,
        num_res_blocks=1,
    )
    cfg = {
        "model_name": "xmodel1",
        "state_tile_channels": 57,
        "state_scalar_dim": 56,
        "candidate_feature_dim": XMODEL1_CANDIDATE_FEATURE_DIM,
        "candidate_flag_dim": XMODEL1_CANDIDATE_FLAG_DIM,
        "hidden_dim": 32,
        "num_res_blocks": 1,
        "schema_name": XMODEL1_SCHEMA_NAME,
        "schema_version": XMODEL1_SCHEMA_VERSION,
    }
    payload = {"model": model.state_dict()}
    payload.update(build_xmodel1_checkpoint_metadata(cfg=cfg, model=model))
    torch.save(payload, ckpt)
    snap, actor = _sample_discard_state()
    snap["rule_context"] = np.zeros((XMODEL1_RULE_CONTEXT_DIM - 1,), dtype=np.float32)
    adapter = KeqingModelAdapter.from_checkpoint(ckpt, device=torch.device("cpu"))
    with pytest.raises(RuntimeError, match="rule_context contract drift"):
        adapter.forward(snap, actor)
