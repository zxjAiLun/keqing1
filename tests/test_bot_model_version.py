from pathlib import Path

import torch

import inference.bot_registry as bot_registry
import inference.runtime_bot as runtime_bot_module
from inference.runtime_bot import RuntimeBot
from keqingv4.checkpoint import build_keqingv4_checkpoint_metadata
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
            **build_keqingv4_checkpoint_metadata(
                cfg={
                    "model_name": "keqingv4",
                    "hidden_dim": 64,
                    "num_res_blocks": 2,
                    "action_embed_dim": 16,
                    "context_dim": 12,
                    "dropout": 0.0,
                },
                model=model,
            ),
        },
        ckpt,
    )

    bot = RuntimeBot(player_id=0, model_path=ckpt, device="cpu", model_version="keqingv4")

    assert bot._model_version == "keqingv4"


def test_runtime_bot_rebuilds_scorer_with_rank_pt_lambda(monkeypatch):
    captured = {}

    class FakeAdapter:
        def __init__(self):
            self.model_version = "keqingv4"
            self._encode = lambda snap, actor: (snap, actor)
            self.model = object()

    class FakeScorer:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(
        runtime_bot_module.KeqingModelAdapter,
        "from_checkpoint",
        classmethod(lambda cls, *args, **kwargs: FakeAdapter()),
    )
    monkeypatch.setattr(runtime_bot_module, "DefaultActionScorer", FakeScorer)

    bot = RuntimeBot(
        player_id=0,
        model_path=Path("fake.pth"),
        device="cpu",
        model_version="keqingv4",
        rank_pt_lambda=0.15,
    )

    assert bot.rank_pt_lambda == 0.15
    assert captured["rank_pt_lambda"] == 0.15
    assert captured["adapter"].model_version == "keqingv4"


def test_create_runtime_bot_passes_rank_pt_lambda(monkeypatch):
    captured = {}

    class FakeRuntimeBot:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr(bot_registry, "RuntimeBot", FakeRuntimeBot)

    bot = bot_registry.create_runtime_bot(
        bot_name="keqingv4",
        player_id=1,
        project_root=Path("/tmp/project"),
        device="cpu",
        rank_pt_lambda=0.2,
    )

    assert isinstance(bot, FakeRuntimeBot)
    assert captured["rank_pt_lambda"] == 0.2
    assert captured["model_version"] == "keqingv4"
