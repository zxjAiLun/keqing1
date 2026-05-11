from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from scripts import generate_mortal_riichienv_replays as generator


class FakeAction:
    def __init__(self, payload):
        self.payload = payload

    def to_mjai(self):
        return dict(self.payload)


def _compact_meta(scores_by_action_id):
    mask_bits = 0
    q_values = []
    for action_id in sorted(scores_by_action_id):
        mask_bits |= 1 << action_id
        q_values.append(scores_by_action_id[action_id])
    return {"mask_bits": mask_bits, "q_values": q_values}


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


def test_style_profile_schema_requires_four_weights() -> None:
    profile = generator.StyleProfile("bad", (1.0, 0.0, 0.0))

    try:
        generator.validate_style_profile(profile)
    except ValueError as exc:
        assert "length 4" in str(exc)
    else:
        raise AssertionError("expected invalid style profile to fail")


def test_base_style_policy_keeps_base_action() -> None:
    policy = generator.MortalStylePolicy(
        profile=generator.DEFAULT_STYLE_PROFILES["base"],
        style_alpha=0.25,
    )
    base_action = {"type": "none"}
    legal_actions = [FakeAction({"type": "hora", "actor": 0, "target": 0}), FakeAction(base_action)]

    decision = policy.select_action(
        base_action=base_action,
        mortal_meta=_compact_meta({43: 0.0, 45: 1.0}),
        legal_actions=legal_actions,
    )

    assert decision.action == base_action
    assert decision.meta["applied"] is False


def test_style_policy_bias_selects_only_from_legal_actions() -> None:
    policy = generator.MortalStylePolicy(
        profile=generator.DEFAULT_STYLE_PROFILES["atk_fuuro"],
        style_alpha=1.0,
    )
    legal_actions = [
        FakeAction({"type": "hora", "actor": 0, "target": 0}),
        FakeAction({"type": "none"}),
    ]

    decision = policy.select_action(
        base_action={"type": "none"},
        mortal_meta=_compact_meta({37: 100.0, 43: 0.0, 45: 1.0}),
        legal_actions=legal_actions,
    )

    assert decision.action["type"] == "hora"
    assert decision.meta["changed"] is True
    assert decision.meta["candidate_count"] == 2


def test_assign_style_profiles_rotates_deterministically() -> None:
    profiles = generator.parse_style_profiles("base,atk_fuuro,def_menzen")

    first = generator.assign_style_profiles(game_id=0, profiles=profiles, seed_key=1, mode="rotate")
    second = generator.assign_style_profiles(game_id=1, profiles=profiles, seed_key=1, mode="rotate")

    assert [first[seat].style_id for seat in range(4)] == ["base", "atk_fuuro", "def_menzen", "base"]
    assert [second[seat].style_id for seat in range(4)] == ["atk_fuuro", "def_menzen", "base", "atk_fuuro"]


def test_build_style_metrics_groups_by_profile() -> None:
    players = [
        {
            "seat": 0,
            **generator.DEFAULT_STYLE_PROFILES["base"].to_json(style_alpha=0.25),
        },
        {
            "seat": 1,
            **generator.DEFAULT_STYLE_PROFILES["atk_fuuro"].to_json(style_alpha=0.25),
        },
        {
            "seat": 2,
            **generator.DEFAULT_STYLE_PROFILES["def_menzen"].to_json(style_alpha=0.25),
        },
        {
            "seat": 3,
            **generator.DEFAULT_STYLE_PROFILES["base"].to_json(style_alpha=0.25),
        },
    ]
    game = {
        "events": [
            {"type": "start_kyoku"},
            {"type": "reach", "actor": 1},
            {"type": "pon", "actor": 1},
            {"type": "hora", "actor": 1, "target": 2},
            {"type": "ryukyoku"},
        ],
        "sidecar": {
            "by_actor": {
                "1": [
                    {
                        "mortal_meta": {
                            "style_policy": {
                                "changed": True,
                            }
                        }
                    }
                ]
            }
        },
        "summary": {
            "players": players,
            "ranks": [2, 1, 4, 3],
            "scores": [27000, 35000, 12000, 26000],
        },
    }

    metrics = generator.build_style_metrics([game])

    assert metrics["by_style"]["base"]["games"] == 2
    assert metrics["by_style"]["base"]["rank_counts"] == [0, 1, 1, 0]
    assert metrics["by_style"]["atk_fuuro"]["win_count"] == 1
    assert metrics["by_style"]["atk_fuuro"]["call_count"] == 1
    assert metrics["by_style"]["atk_fuuro"]["riichi_count"] == 1
    assert metrics["by_style"]["atk_fuuro"]["style_changed_count"] == 1
    assert metrics["by_style"]["def_menzen"]["deal_in_count"] == 1
