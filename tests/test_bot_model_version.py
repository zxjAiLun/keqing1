from pathlib import Path

import torch

from inference.runtime_bot import RuntimeBot
from keqingv4.model import KeqingV4Model
from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
)
from xmodel1.model import Xmodel1Model


def test_runtime_bot_auto_detects_xmodel1_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "xmodel1.pth"
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=64,
        candidate_feature_dim=XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim=XMODEL1_CANDIDATE_FLAG_DIM,
        hidden_dim=64,
        num_res_blocks=2,
        dropout=0.0,
    )
    torch.save({"model": model.state_dict()}, ckpt)

    bot = RuntimeBot(player_id=0, model_path=ckpt, device="cpu")

    assert bot._model_version == "xmodel1"


def test_runtime_bot_explicit_v4_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "v4.pth"
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

    bot = RuntimeBot(player_id=0, model_path=ckpt, device="cpu", model_version="keqingv4")

    assert bot._model_version == "keqingv4"
