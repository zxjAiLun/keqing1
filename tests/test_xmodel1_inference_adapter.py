from __future__ import annotations

from pathlib import Path

import torch

from inference.keqing_adapter import KeqingModelAdapter
from mahjong_env.replay import build_replay_samples_mc_return
from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_MAX_RESPONSE_CANDIDATES,
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
)
from xmodel1.model import Xmodel1Model


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
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": {
                "model_name": "xmodel1",
                "state_tile_channels": 57,
                "state_scalar_dim": 56,
                "candidate_feature_dim": XMODEL1_CANDIDATE_FEATURE_DIM,
                "candidate_flag_dim": XMODEL1_CANDIDATE_FLAG_DIM,
                "hidden_dim": 32,
                "num_res_blocks": 1,
                "schema_name": XMODEL1_SCHEMA_NAME,
                "schema_version": XMODEL1_SCHEMA_VERSION,
            },
            "model_version": "xmodel1",
            "schema_name": XMODEL1_SCHEMA_NAME,
            "schema_version": XMODEL1_SCHEMA_VERSION,
        },
        ckpt,
    )
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
