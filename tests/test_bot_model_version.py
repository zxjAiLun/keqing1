from pathlib import Path

import torch

from inference.runtime_bot import RuntimeBot
from keqingv1.model import MahjongModel as MahjongModelV1
from keqingv3.model import MahjongModel as MahjongModelV3
from keqingv31.model import KeqingV31Model


def _save_ckpt(path: Path, model: torch.nn.Module) -> None:
    torch.save({"model": model.state_dict()}, path)


def test_keqing_bot_auto_detects_v1_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "v1.pth"
    _save_ckpt(ckpt, MahjongModelV1())
    bot = RuntimeBot(player_id=0, model_path=ckpt, device="cpu")
    assert bot._model_version == "keqingv1"


def test_keqing_bot_auto_detects_v3_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "v3.pth"
    _save_ckpt(ckpt, MahjongModelV3())
    bot = RuntimeBot(player_id=0, model_path=ckpt, device="cpu")
    assert bot._model_version == "keqingv3"


def test_keqing_bot_explicit_v31_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "v31.pth"
    model = KeqingV31Model(hidden_dim=256, num_res_blocks=5, action_embed_dim=48)
    torch.save({"model": model.state_dict(), "cfg": {"model_name": "keqingv31", "hidden_dim": 256, "num_res_blocks": 5, "action_embed_dim": 48}, "model_version": "keqingv31"}, ckpt)
    bot = RuntimeBot(player_id=0, model_path=ckpt, device="cpu", model_version="keqingv31")
    assert bot._model_version == "keqingv31"
