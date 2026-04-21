from __future__ import annotations

from pathlib import Path

import keqing_core
import pytest
import torch

from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
)
from inference.keqing_adapter import KeqingModelAdapter
from inference.runtime_bot import RuntimeBot
from mahjong_env.state import GameState, apply_event
from mahjong_env.tiles import tile_to_34
from xmodel1.features import build_runtime_candidate_arrays
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
                "schema_name": XMODEL1_SCHEMA_NAME,
                "schema_version": XMODEL1_SCHEMA_VERSION,
                "hidden_dim": 64,
                "num_res_blocks": 2,
                "dropout": 0.0,
            },
            "schema_name": XMODEL1_SCHEMA_NAME,
            "schema_version": XMODEL1_SCHEMA_VERSION,
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


def test_xmodel1_rust_runtime_candidates_include_tsumo_pai():
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")

    previous = keqing_core.is_enabled()
    keqing_core.enable_rust(True)
    try:
        snap = _setup_actor0_state().snapshot(0)
        snap["tsumo_pai"] = "9m"
        legal_actions = [
            {"type": "dahai", "actor": 0, "pai": pai, "tsumogiri": False}
            for pai in snap["hand"]
        ]
        legal_actions.append({"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": True})

        _candidate_feat, candidate_tile_id, candidate_mask, _candidate_flags = build_runtime_candidate_arrays(
            snap,
            0,
            legal_actions,
        )

        active_tile_ids = [
            int(tile_id)
            for tile_id, mask in zip(candidate_tile_id.tolist(), candidate_mask.tolist())
            if int(mask) > 0
        ]
        assert tile_to_34("9m") in active_tile_ids
        assert len(active_tile_ids) == len(
            {action["pai"] for action in legal_actions if action["type"] == "dahai"}
        )
    finally:
        keqing_core.enable_rust(previous)


def test_keqing_model_adapter_rejects_partial_xmodel1_checkpoint(tmp_path: Path):
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=56,
        candidate_feature_dim=XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim=XMODEL1_CANDIDATE_FLAG_DIM,
        hidden_dim=32,
        num_res_blocks=1,
        dropout=0.0,
    )
    state_dict = model.state_dict()
    state_dict.pop("dealin_head.2.bias")
    ckpt = tmp_path / "partial_xmodel1.pth"
    torch.save(
        {
            "model": state_dict,
            "cfg": {
                "model_name": "xmodel1",
                "state_tile_channels": 57,
                "candidate_feature_dim": XMODEL1_CANDIDATE_FEATURE_DIM,
                "candidate_flag_dim": XMODEL1_CANDIDATE_FLAG_DIM,
                "schema_name": XMODEL1_SCHEMA_NAME,
                "schema_version": XMODEL1_SCHEMA_VERSION,
                "hidden_dim": 32,
                "num_res_blocks": 1,
                "dropout": 0.0,
            },
            "schema_name": XMODEL1_SCHEMA_NAME,
            "schema_version": XMODEL1_SCHEMA_VERSION,
        },
        ckpt,
    )

    with pytest.raises(RuntimeError, match="Refusing partial load"):
        KeqingModelAdapter.from_checkpoint(ckpt, device=torch.device("cpu"), model_version="xmodel1")
