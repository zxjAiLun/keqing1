from __future__ import annotations

from pathlib import Path

import torch

from inference.bot_registry import create_runtime_bot
from inference.runtime_bot import RuntimeBot
from xmodel1.model import Xmodel1Model


def test_create_runtime_bot_supports_xmodel1(tmp_path: Path):
    ckpt = tmp_path / "xmodel1.pth"
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=64,
        candidate_feature_dim=21,
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
                "state_scalar_dim": 64,
                "candidate_feature_dim": 21,
                "candidate_flag_dim": 10,
                "hidden_dim": 32,
                "num_res_blocks": 1,
            },
            "model_version": "xmodel1",
        },
        ckpt,
    )
    bot = create_runtime_bot(
        bot_name="xmodel1",
        player_id=0,
        project_root=".",
        model_path=ckpt,
        device="cpu",
        verbose=False,
        beam_k=0,
        beam_lambda=0.0,
    )
    assert isinstance(bot, RuntimeBot)
    assert bot._model_version == "xmodel1"
