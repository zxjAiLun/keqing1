from pathlib import Path

import torch

from keqingv1.bot import KeqingBot
from keqingv1.model import MahjongModel as MahjongModelV1
from keqingv3.model import MahjongModel as MahjongModelV3


def _save_ckpt(path: Path, model: torch.nn.Module) -> None:
    torch.save({"model": model.state_dict()}, path)


def test_keqing_bot_auto_detects_v1_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "v1.pth"
    _save_ckpt(ckpt, MahjongModelV1())
    bot = KeqingBot(player_id=0, model_path=ckpt, device="cpu")
    assert bot._model_version == "keqingv1"


def test_keqing_bot_auto_detects_v3_checkpoint(tmp_path: Path):
    ckpt = tmp_path / "v3.pth"
    _save_ckpt(ckpt, MahjongModelV3())
    bot = KeqingBot(player_id=0, model_path=ckpt, device="cpu")
    assert bot._model_version == "keqingv3"
