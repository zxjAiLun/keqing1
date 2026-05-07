from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts import generate_mortal_riichienv_replays as generator


def test_new_bots_share_one_mortal_engine(monkeypatch) -> None:
    calls = []

    class FakeMortalReviewBot:
        def __init__(
            self,
            *,
            player_id,
            model_path,
            mortal_root,
            device,
            enable_review_log,
            shared_mortal_engine=None,
            shared_model=None,
        ):
            calls.append(
                {
                    "player_id": player_id,
                    "shared_mortal_engine": shared_mortal_engine,
                    "shared_model": shared_model,
                }
            )
            self.player_id = player_id
            self._mortal_engine = shared_mortal_engine if shared_mortal_engine is not None else object()
            self.model = shared_model if shared_model is not None else object()

    monkeypatch.setattr(generator, "MortalReviewBot", FakeMortalReviewBot)
    args = SimpleNamespace(model=Path("mortal.pth"), mortal_root=Path("third_party/Mortal"), device="cpu")

    bots = generator._new_bots(args)

    assert sorted(bots) == [0, 1, 2, 3]
    assert calls[0]["shared_mortal_engine"] is None
    assert calls[0]["shared_model"] is None
    shared_engine = bots[0]._mortal_engine
    shared_model = bots[0].model
    for call in calls[1:]:
        assert call["shared_mortal_engine"] is shared_engine
        assert call["shared_model"] is shared_model


def test_make_env_reports_constructor_seed(monkeypatch) -> None:
    calls = []

    class FakeRiichiEnv:
        def __init__(self, *, game_mode, seed=None):
            calls.append({"game_mode": game_mode, "seed": seed})

    monkeypatch.setattr(generator, "RiichiEnv", FakeRiichiEnv)

    _env, seed_info = generator._make_env(game_mode="4p-red-half", seed=123)

    assert calls == [{"game_mode": "4p-red-half", "seed": 123}]
    assert seed_info.applied is True
    assert seed_info.mode == "constructor"
    assert seed_info.requested_seed == 123


def test_make_env_reports_fallback_unseeded_when_seed_api_missing(monkeypatch) -> None:
    calls = []

    class FakeRiichiEnv:
        def __init__(self, *, game_mode, seed=None):
            if seed is not None:
                raise TypeError("seed is unsupported")
            calls.append({"game_mode": game_mode, "seed": seed})

        def reset(self):
            return {}

    monkeypatch.setattr(generator, "RiichiEnv", FakeRiichiEnv)

    _env, seed_info = generator._make_env(game_mode="4p-red-half", seed=123)

    assert calls == [{"game_mode": "4p-red-half", "seed": None}]
    assert seed_info.applied is False
    assert seed_info.mode == "fallback_unseeded"
    assert seed_info.requested_seed == 123


def test_make_env_reports_reset_seed_when_available(monkeypatch) -> None:
    calls = []

    class FakeRiichiEnv:
        def __init__(self, *, game_mode, seed=None):
            if seed is not None:
                raise TypeError("seed is unsupported")
            calls.append({"game_mode": game_mode, "seed": seed})

        def reset(self, *, seed=None):
            return {"seed": seed}

    monkeypatch.setattr(generator, "RiichiEnv", FakeRiichiEnv)

    _env, seed_info = generator._make_env(game_mode="4p-red-half", seed=123)

    assert calls == [{"game_mode": "4p-red-half", "seed": None}]
    assert seed_info.applied is True
    assert seed_info.mode == "reset"
    assert seed_info.requested_seed == 123
