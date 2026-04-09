from __future__ import annotations

from pathlib import Path

import torch

from inference.runtime_bot import RuntimeBot
from mahjong_env.state import GameState, apply_event
from xmodel1.model import Xmodel1Model


def _save_xmodel1_ckpt(path: Path) -> None:
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=64,
        candidate_feature_dim=21,
        candidate_flag_dim=10,
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
                "candidate_feature_dim": 21,
                "candidate_flag_dim": 10,
                "hidden_dim": 64,
                "num_res_blocks": 2,
                "dropout": 0.0,
            },
        },
        path,
    )


def _setup_actor0_state() -> GameState:
    state = GameState()
    apply_event(state, {"type": "start_game", "names": ["P0", "P1", "P2", "P3"]})
    apply_event(
        state,
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
    )
    return state


def test_runtime_bot_auto_detects_xmodel1_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "xmodel1.pth"
    _save_xmodel1_ckpt(ckpt)
    bot = RuntimeBot(player_id=0, model_path=ckpt, device="cpu")
    assert bot._model_version == "xmodel1"


def test_runtime_bot_xmodel1_tsumo_react_smoke(tmp_path: Path):
    ckpt = tmp_path / "xmodel1.pth"
    _save_xmodel1_ckpt(ckpt)
    bot = RuntimeBot(player_id=0, model_path=ckpt, device="cpu", beam_k=0, model_version="xmodel1")
    bot.game_state = _setup_actor0_state()
    chosen = bot.react({"type": "tsumo", "actor": 0, "pai": "5m"})
    assert chosen is not None
    assert chosen.get("type") in {"dahai", "reach", "ankan", "kakan", "hora"}
