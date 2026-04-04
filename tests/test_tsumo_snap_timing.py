from pathlib import Path

import numpy as np
import torch

import inference.runtime_bot as bot_mod
from keqingv1.features import C_TILE, N_SCALAR
from keqingv1.model import MahjongModel as MahjongModelV1
from inference.runtime_bot import RuntimeBot
from mahjong_env.state import GameState, apply_event
from mahjong_env.types import Action


def _save_ckpt(path: Path) -> None:
    torch.save({"model": MahjongModelV1().state_dict()}, path)


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


def test_keqing_bot_tsumo_uses_pre_apply_decision_snapshot_for_model_and_log(
    tmp_path: Path, monkeypatch
):
    ckpt = tmp_path / "v1.pth"
    _save_ckpt(ckpt)
    bot = RuntimeBot(player_id=0, model_path=ckpt, device="cpu", beam_k=0)
    bot.game_state = _setup_actor0_state()

    captured: dict[str, dict] = {}

    def fake_encode(snap: dict, actor: int):
        captured["model_snap"] = {
            "actor": actor,
            "hand": list(snap.get("hand", [])),
            "tsumo_pai": snap.get("tsumo_pai"),
            "waits_count": snap.get("waits_count"),
            "shanten": snap.get("shanten"),
        }
        return (
            np.zeros((C_TILE, 34), dtype=np.float32),
            np.zeros((N_SCALAR,), dtype=np.float32),
        )

    def fake_enumerate_legal_actions(snap: dict, actor: int):
        captured["legal_snap"] = {
            "actor": actor,
            "hand": list(snap.get("hand", [])),
            "tsumo_pai": snap.get("tsumo_pai"),
            "waits_count": snap.get("waits_count"),
            "shanten": snap.get("shanten"),
        }
        return [Action(type="dahai", actor=actor, pai="5m", tsumogiri=True)]

    bot._encode = fake_encode
    monkeypatch.setattr(bot_mod, "enumerate_legal_actions", fake_enumerate_legal_actions)

    chosen = bot.react({"type": "tsumo", "actor": 0, "pai": "5m"})

    assert chosen == {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True}

    assert len(captured["legal_snap"]["hand"]) == 14
    assert captured["legal_snap"]["tsumo_pai"] is None

    assert len(captured["model_snap"]["hand"]) == 13
    assert captured["model_snap"]["tsumo_pai"] == "5m"

    entry = bot.decision_log[-1]
    assert len(entry["hand"]) == 13
    assert entry["tsumo_pai"] == "5m"


def test_keqing_bot_tsumo_decision_snapshot_matches_training_contract(
    tmp_path: Path, monkeypatch
):
    ckpt = tmp_path / "v1.pth"
    _save_ckpt(ckpt)
    bot = RuntimeBot(player_id=0, model_path=ckpt, device="cpu", beam_k=0)
    bot.game_state = _setup_actor0_state()

    model_snaps: list[dict] = []

    def fake_encode(snap: dict, actor: int):
        model_snaps.append(
            {
                "actor": actor,
                "hand": list(snap.get("hand", [])),
                "tsumo_pai": snap.get("tsumo_pai"),
                "waits_count": snap.get("waits_count"),
                "shanten": snap.get("shanten"),
            }
        )
        return (
            np.zeros((C_TILE, 34), dtype=np.float32),
            np.zeros((N_SCALAR,), dtype=np.float32),
        )

    monkeypatch.setattr(
        bot_mod,
        "enumerate_legal_actions",
        lambda snap, actor: [Action(type="dahai", actor=actor, pai="5m", tsumogiri=True)],
    )
    bot._encode = fake_encode

    pre_snap = bot.game_state.snapshot(0)
    pre_hand = list(pre_snap["hand"])

    bot.react({"type": "tsumo", "actor": 0, "pai": "5m"})

    assert len(model_snaps) == 1
    model_snap = model_snaps[0]
    assert model_snap["hand"] == pre_hand
    assert model_snap["tsumo_pai"] == "5m"
