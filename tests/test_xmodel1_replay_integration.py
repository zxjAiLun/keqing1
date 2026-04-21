from __future__ import annotations

from pathlib import Path

import torch

from replay.api import run_replay_single_raw
from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
)
from xmodel1.model import Xmodel1Model


def _save_xmodel1_ckpt(path: Path) -> None:
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=64,
        candidate_feature_dim=XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim=XMODEL1_CANDIDATE_FLAG_DIM,
        hidden_dim=64,
        num_res_blocks=2,
        dropout=0.0,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": {
                "model_name": "xmodel1",
                "state_tile_channels": 57,
                "state_scalar_dim": 64,
                "candidate_feature_dim": XMODEL1_CANDIDATE_FEATURE_DIM,
                "candidate_flag_dim": XMODEL1_CANDIDATE_FLAG_DIM,
                "hidden_dim": 64,
                "num_res_blocks": 2,
                "dropout": 0.0,
                "schema_name": XMODEL1_SCHEMA_NAME,
                "schema_version": XMODEL1_SCHEMA_VERSION,
            },
            "model_version": "xmodel1",
            "schema_name": XMODEL1_SCHEMA_NAME,
            "schema_version": XMODEL1_SCHEMA_VERSION,
        },
        path,
    )


def test_replay_api_accepts_xmodel1_bot_type(tmp_path: Path):
    ckpt = tmp_path / "xmodel1.pth"
    _save_xmodel1_ckpt(ckpt)
    events = [
        {"type": "start_game", "names": ["P0", "P1", "P2", "P3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "names": ["P0", "P1", "P2", "P3"],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "1m", "5p", "5p", "7s", "7s", "E", "E", "S", "S", "W", "N", "F"],
                ["1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p", "S", "S", "W", "F"],
                ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "E", "E", "N", "W"],
                ["?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?", "?"],
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "5m"},
    ]
    bot = run_replay_single_raw(events, player_id=0, checkpoint=ckpt, input_type="url", bot_type="xmodel1")
    assert bot is not None
    assert getattr(bot, "_model_version", None) == "xmodel1"
    assert len(bot.decision_log) >= 1
