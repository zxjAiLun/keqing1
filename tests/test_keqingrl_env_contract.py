from __future__ import annotations

from types import SimpleNamespace

import pytest

from keqingrl import ActionSpec, ActionType, DiscardOnlyMahjongEnv, action_from_mahjong_spec, bind_reach_discard
from keqingrl.env import _TurnContext
from mahjong_env.types import ActionSpec as MahjongActionSpec


def _self_turn_snapshot() -> dict[str, object]:
    return {
        "actor": 0,
        "hand": ["4m", "7m"],
        "last_tsumo": [None, None, None, None],
        "last_tsumo_raw": [None, None, None, None],
    }


def _response_snapshot() -> dict[str, object]:
    return {
        "actor": 1,
        "hand": ["2m", "3m", "4m", "4m", "4m"],
        "last_discard": {"actor": 0, "pai": "4m", "pai_raw": "4m"},
    }


def _self_turn_raw_legal_actions() -> tuple[MahjongActionSpec, ...]:
    return (
        MahjongActionSpec(type="dahai", actor=0, pai="4m", tsumogiri=False),
        MahjongActionSpec(type="dahai", actor=0, pai="7m", tsumogiri=True),
        MahjongActionSpec(type="reach", actor=0),
        MahjongActionSpec(type="hora", actor=0, target=0, pai="7m"),
        MahjongActionSpec(type="ankan", actor=0, pai="E", consumed=("E", "E", "E", "E")),
        MahjongActionSpec(type="kakan", actor=0, pai="5p", consumed=("5p", "5p", "5p", "5p")),
        MahjongActionSpec(type="ryukyoku", actor=0),
    )


def _response_raw_legal_actions() -> tuple[MahjongActionSpec, ...]:
    return (
        MahjongActionSpec(type="hora", actor=1, target=0, pai="4m"),
        MahjongActionSpec(type="pon", actor=1, target=0, pai="4m", consumed=("4m", "4m")),
        MahjongActionSpec(type="daiminkan", actor=1, target=0, pai="4m", consumed=("4m", "4m", "4m")),
        MahjongActionSpec(type="chi", actor=1, target=0, pai="4m", consumed=("2m", "3m")),
        MahjongActionSpec(type="none"),
    )


def test_discard_only_env_keeps_ordered_legal_actions_stable_within_turn() -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    state = env.reset(seed=7)

    assert state.current_actor is not None
    actor = state.current_actor

    first = env.legal_actions(actor)
    second = env.legal_actions(actor)
    observed = env.observe(actor)

    assert first == second
    assert [id(spec) for spec in first] == [id(spec) for spec in second]
    assert observed.legal_actions == (first,)
    assert observed.legal_action_ids.shape == (1, len(first))
    assert observed.legal_action_features.shape == (1, len(first), 8)
    assert observed.legal_action_mask.shape == (1, len(first))
    assert all(spec.action_type == ActionType.DISCARD for spec in first)


def test_discard_only_env_rejects_action_not_in_current_legal_order() -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    state = env.reset(seed=11)

    assert state.current_actor is not None
    actor = state.current_actor

    with pytest.raises(ValueError):
        env.step(actor, ActionSpec(ActionType.PASS))


@pytest.mark.parametrize(
    ("raw_actions", "expected_type"),
    [
        (
            (
                MahjongActionSpec(type="hora", actor=0, target=0, pai="7m"),
                MahjongActionSpec(type="dahai", actor=0, pai="4m", tsumogiri=False),
                MahjongActionSpec(type="dahai", actor=0, pai="7m", tsumogiri=True),
            ),
            "hora",
        ),
        (
            (
                MahjongActionSpec(type="hora", actor=1, target=0, pai="4m"),
                MahjongActionSpec(type="none"),
            ),
            "hora",
        ),
        (
            (
                MahjongActionSpec(type="ryukyoku", actor=0),
                MahjongActionSpec(type="dahai", actor=0, pai="4m", tsumogiri=False),
            ),
            "ryukyoku",
        ),
    ],
)
def test_forced_terminal_actions_preempt_learner_gate(
    monkeypatch: pytest.MonkeyPatch,
    raw_actions: tuple[MahjongActionSpec, ...],
    expected_type: str,
) -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    response_window = raw_actions[0].type == "hora" and raw_actions[0].target != raw_actions[0].actor
    room = SimpleNamespace(
        phase="active",
        events=[],
        state=SimpleNamespace(
            actor_to_move=0 if not response_window else 1,
            last_discard=None if not response_window else {"actor": 0, "pai": "4m"},
            last_kakan=None,
        ),
    )
    dispatched: list[dict[str, object]] = []

    env.room = room
    monkeypatch.setattr(env.manager, "prepare_turn", lambda _room, _actor: None)
    monkeypatch.setattr("keqingrl.env.enumerate_legal_action_specs", lambda _snapshot, _actor: raw_actions)
    monkeypatch.setattr(env, "_snapshot_with_rl_fields", lambda actor: {"actor": actor})
    monkeypatch.setattr(
        env,
        "_choose_rulebase_raw_action",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("rulebase gate should not run")),
    )

    def _record_dispatch(_room, _actor: int, payload: dict[str, object]) -> None:
        dispatched.append(payload)
        env._done = True

    monkeypatch.setattr(env, "_dispatch_manager_action", _record_dispatch)

    env._advance_until_decision()

    assert dispatched == [raw_actions[0].to_mjai()]
    assert dispatched[0]["type"] == expected_type
    assert env._turn is None
    assert env.is_done()


def test_discard_only_env_step_advances_or_finishes_episode() -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    state = env.reset(seed=19)

    assert state.current_actor is not None
    actor = state.current_actor
    chosen = env.legal_actions(actor)[0]

    result = env.step(actor, chosen)

    assert result.state.game_id == state.game_id
    assert result.done == env.is_done()
    if not result.done:
        assert result.next_actor is not None
        next_actions = env.legal_actions(result.next_actor)
        assert next_actions
        assert all(spec.action_type == ActionType.DISCARD for spec in next_actions)


def test_discard_only_env_can_finish_bounded_episode_and_emit_terminal_rewards() -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    state = env.reset(seed=23)

    assert state.current_actor is not None
    last_result = None
    for _ in range(200):
        actor = env.current_actor()
        assert actor is not None
        chosen = env.legal_actions(actor)[0]
        last_result = env.step(actor, chosen)
        if last_result.done:
            break

    assert last_result is not None
    assert last_result.done is True
    assert last_result.terminal_rewards is not None
    assert len(last_result.terminal_rewards) == 4
    assert max(abs(value) for value in last_result.terminal_rewards) <= 1.5
    assert last_result.final_ranks is not None
    assert tuple(sorted(last_result.final_ranks)) == (0, 1, 2, 3)


def test_discard_only_env_validates_supported_self_turn_action_scope() -> None:
    DiscardOnlyMahjongEnv(
        self_turn_action_types=(
            ActionType.DISCARD,
            ActionType.REACH_DISCARD,
        )
    )
    DiscardOnlyMahjongEnv(
        self_turn_action_types=(
            ActionType.DISCARD,
            ActionType.RYUKYOKU,
        )
    )

    with pytest.raises(ValueError, match="include ActionType.DISCARD"):
        DiscardOnlyMahjongEnv(self_turn_action_types=(ActionType.TSUMO,))

    with pytest.raises(ValueError, match="unsupported self-turn action types"):
        DiscardOnlyMahjongEnv(
            self_turn_action_types=(
                ActionType.DISCARD,
                ActionType.PASS,
            )
        )

    DiscardOnlyMahjongEnv(
        response_action_types=(
            ActionType.RON,
            ActionType.CHI,
            ActionType.PON,
            ActionType.DAIMINKAN,
            ActionType.PASS,
        )
    )

    with pytest.raises(ValueError, match="unsupported response action types"):
        DiscardOnlyMahjongEnv(
            response_action_types=(
                ActionType.RON,
                ActionType.TSUMO,
            )
        )


def test_collect_controlled_self_turn_actions_expanded_scope_preserves_order() -> None:
    env = DiscardOnlyMahjongEnv(
        self_turn_action_types=(
            ActionType.DISCARD,
            ActionType.REACH_DISCARD,
            ActionType.TSUMO,
            ActionType.ANKAN,
            ActionType.KAKAN,
        )
    )
    env._enumerate_reach_discard_candidates = lambda snapshot, actor: [("7m", True), ("4m", False)]  # type: ignore[method-assign]

    controlled_pairs = env._collect_controlled_self_turn_actions(
        _self_turn_snapshot(),
        _self_turn_raw_legal_actions(),
    )

    assert [spec.action_type for spec, _raw in controlled_pairs] == [
        ActionType.DISCARD,
        ActionType.DISCARD,
        ActionType.REACH_DISCARD,
        ActionType.REACH_DISCARD,
        ActionType.TSUMO,
        ActionType.ANKAN,
        ActionType.KAKAN,
    ]
    assert [spec.tile for spec, _dispatch in controlled_pairs[:4]] == [
        action_from_mahjong_spec(MahjongActionSpec(type="dahai", actor=0, pai="4m")).tile,
        action_from_mahjong_spec(MahjongActionSpec(type="dahai", actor=0, pai="7m", tsumogiri=True)).tile,
        action_from_mahjong_spec(MahjongActionSpec(type="dahai", actor=0, pai="4m")).tile,
        action_from_mahjong_spec(MahjongActionSpec(type="dahai", actor=0, pai="7m", tsumogiri=True)).tile,
    ]
    assert [dispatch[0]["type"] for _spec, dispatch in controlled_pairs] == [
        "dahai",
        "dahai",
        "reach",
        "reach",
        "hora",
        "ankan",
        "kakan",
    ]
    assert [dispatch[-1].get("pai") for _spec, dispatch in controlled_pairs[:4]] == ["4m", "7m", "4m", "7m"]


def test_collect_controlled_self_turn_actions_default_scope_keeps_discard_only() -> None:
    env = DiscardOnlyMahjongEnv()

    controlled_pairs = env._collect_controlled_self_turn_actions(
        _self_turn_snapshot(),
        _self_turn_raw_legal_actions(),
    )

    assert [spec.action_type for spec, _dispatch in controlled_pairs] == [
        ActionType.DISCARD,
        ActionType.DISCARD,
    ]
    assert [dispatch[0]["type"] for _spec, dispatch in controlled_pairs] == ["dahai", "dahai"]


def test_collect_controlled_self_turn_actions_includes_ryukyoku_when_enabled() -> None:
    env = DiscardOnlyMahjongEnv(
        self_turn_action_types=(
            ActionType.DISCARD,
            ActionType.TSUMO,
            ActionType.ANKAN,
            ActionType.KAKAN,
            ActionType.RYUKYOKU,
        )
    )

    controlled_pairs = env._collect_controlled_self_turn_actions(
        _self_turn_snapshot(),
        _self_turn_raw_legal_actions(),
    )

    assert [spec.action_type for spec, _dispatch in controlled_pairs] == [
        ActionType.DISCARD,
        ActionType.DISCARD,
        ActionType.TSUMO,
        ActionType.ANKAN,
        ActionType.KAKAN,
        ActionType.RYUKYOKU,
    ]
    assert [dispatch[0]["type"] for _spec, dispatch in controlled_pairs] == [
        "dahai",
        "dahai",
        "hora",
        "ankan",
        "kakan",
        "ryukyoku",
    ]


def test_collect_controlled_response_actions_preserves_raw_order() -> None:
    env = DiscardOnlyMahjongEnv(
        response_action_types=(
            ActionType.RON,
            ActionType.CHI,
            ActionType.PON,
            ActionType.DAIMINKAN,
            ActionType.PASS,
        )
    )

    controlled_pairs = env._collect_controlled_response_actions(_response_raw_legal_actions())

    assert [spec.action_type for spec, _dispatch in controlled_pairs] == [
        ActionType.RON,
        ActionType.PON,
        ActionType.DAIMINKAN,
        ActionType.CHI,
        ActionType.PASS,
    ]
    assert [dispatch[0]["type"] for _spec, dispatch in controlled_pairs] == [
        "hora",
        "pon",
        "daiminkan",
        "chi",
        "none",
    ]


def test_collect_controlled_response_actions_default_scope_keeps_auto_response() -> None:
    env = DiscardOnlyMahjongEnv()

    controlled_pairs = env._collect_controlled_response_actions(_response_raw_legal_actions())

    assert controlled_pairs == []


@pytest.mark.parametrize(
    ("chosen_raw", "expected_action_type", "expected_raw_type"),
    [
        (MahjongActionSpec(type="hora", actor=0, target=0, pai="7m"), ActionType.TSUMO, "hora"),
        (
            MahjongActionSpec(type="dahai", actor=0, pai="4m", tsumogiri=False),
            ActionType.REACH_DISCARD,
            "reach",
        ),
        (
            MahjongActionSpec(type="ankan", actor=0, pai="E", consumed=("E", "E", "E", "E")),
            ActionType.ANKAN,
            "ankan",
        ),
        (
            MahjongActionSpec(type="kakan", actor=0, pai="5p", consumed=("5p", "5p", "5p", "5p")),
            ActionType.KAKAN,
            "kakan",
        ),
        (
            MahjongActionSpec(type="ryukyoku", actor=0),
            ActionType.RYUKYOKU,
            "ryukyoku",
        ),
    ],
)
def test_discard_only_env_step_dispatches_atomic_self_turn_actions(
    monkeypatch: pytest.MonkeyPatch,
    chosen_raw: MahjongActionSpec,
    expected_action_type: ActionType,
    expected_raw_type: str,
) -> None:
    env = DiscardOnlyMahjongEnv(
        self_turn_action_types=(
            ActionType.DISCARD,
            ActionType.REACH_DISCARD,
            ActionType.TSUMO,
            ActionType.ANKAN,
            ActionType.KAKAN,
            ActionType.RYUKYOKU,
        )
    )
    raw_actions = _self_turn_raw_legal_actions()
    discard_action = action_from_mahjong_spec(raw_actions[0])
    tsumogiri_discard_action = action_from_mahjong_spec(raw_actions[1])
    reach_action = bind_reach_discard(discard_action)
    legal_actions = (
        discard_action,
        tsumogiri_discard_action,
        reach_action,
        action_from_mahjong_spec(raw_actions[3]),
        action_from_mahjong_spec(raw_actions[4]),
        action_from_mahjong_spec(raw_actions[5]),
        action_from_mahjong_spec(raw_actions[6]),
    )
    chosen_action = next(spec for spec in legal_actions if spec.action_type == expected_action_type)

    room = SimpleNamespace(
        game_id="dispatch-test",
        state=SimpleNamespace(
            bakaze="E",
            kyoku=1,
            honba=0,
            kyotaku=0,
            scores=(25000, 25000, 25000, 25000),
        ),
    )
    applied: list[dict[str, object]] = []

    monkeypatch.setattr(env, "_require_room", lambda: room)
    monkeypatch.setattr(
        env,
        "_require_turn",
        lambda actor: _TurnContext(
            actor=0,
            snapshot={"actor": 0},
            legal_actions=legal_actions,
                dispatch_actions=(
                    (raw_actions[0].to_mjai(),),
                    (raw_actions[1].to_mjai(),),
                    ({"type": "reach", "actor": 0}, raw_actions[0].to_mjai()),
                    (raw_actions[3].to_mjai(),),
                    (raw_actions[4].to_mjai(),),
                    (raw_actions[5].to_mjai(),),
                    (raw_actions[6].to_mjai(),),
                ),
            ),
        )
    monkeypatch.setattr(env, "_advance_until_decision", lambda: None)
    monkeypatch.setattr(
        env,
        "state",
        lambda: SimpleNamespace(
            game_id="dispatch-test",
            bakaze="E",
            kyoku=1,
            honba=0,
            kyotaku=0,
            scores=(25000, 25000, 25000, 25000),
            current_actor=None,
            done=False,
            kyokus_completed=0,
        ),
    )

    def _record_apply_action(_room, actor: int, payload: dict[str, object]) -> None:
        assert actor == 0
        applied.append(payload)

    def _record_ryukyoku(_room, tenpai=None) -> None:
        del tenpai
        applied.append({"type": "ryukyoku"})

    monkeypatch.setattr(env.manager, "apply_action", _record_apply_action)
    monkeypatch.setattr(env.manager, "ryukyoku", _record_ryukyoku)

    result = env.step(0, chosen_action)

    assert result.done is False
    if expected_action_type == ActionType.REACH_DISCARD:
        assert applied == [
            {"type": "reach", "actor": 0},
            chosen_raw.to_mjai(),
        ]
    elif expected_action_type == ActionType.RYUKYOKU:
        assert applied == [{"type": "ryukyoku"}]
    else:
        assert applied == [chosen_raw.to_mjai()]
    assert applied[0]["type"] == expected_raw_type


@pytest.mark.parametrize(
    ("chosen_raw", "expected_action_type", "expected_raw_type"),
    [
        (MahjongActionSpec(type="hora", actor=1, target=0, pai="4m"), ActionType.RON, "hora"),
        (
            MahjongActionSpec(type="pon", actor=1, target=0, pai="4m", consumed=("4m", "4m")),
            ActionType.PON,
            "pon",
        ),
        (
            MahjongActionSpec(type="daiminkan", actor=1, target=0, pai="4m", consumed=("4m", "4m", "4m")),
            ActionType.DAIMINKAN,
            "daiminkan",
        ),
        (
            MahjongActionSpec(type="chi", actor=1, target=0, pai="4m", consumed=("2m", "3m")),
            ActionType.CHI,
            "chi",
        ),
        (MahjongActionSpec(type="none"), ActionType.PASS, "none"),
    ],
)
def test_discard_only_env_step_dispatches_response_actions(
    monkeypatch: pytest.MonkeyPatch,
    chosen_raw: MahjongActionSpec,
    expected_action_type: ActionType,
    expected_raw_type: str,
) -> None:
    env = DiscardOnlyMahjongEnv(
        response_action_types=(
            ActionType.RON,
            ActionType.CHI,
            ActionType.PON,
            ActionType.DAIMINKAN,
            ActionType.PASS,
        )
    )
    raw_actions = _response_raw_legal_actions()
    legal_actions = tuple(action_from_mahjong_spec(spec) for spec in raw_actions)
    chosen_action = next(spec for spec in legal_actions if spec.action_type == expected_action_type)

    room = SimpleNamespace(
        game_id="response-dispatch-test",
        state=SimpleNamespace(
            bakaze="E",
            kyoku=1,
            honba=0,
            kyotaku=0,
            scores=(25000, 25000, 25000, 25000),
        ),
    )
    applied: list[dict[str, object]] = []

    monkeypatch.setattr(env, "_require_room", lambda: room)
    monkeypatch.setattr(
        env,
        "_require_turn",
        lambda actor: _TurnContext(
            actor=1,
            snapshot=_response_snapshot(),
            legal_actions=legal_actions,
            dispatch_actions=tuple((raw.to_mjai(),) for raw in raw_actions),
        ),
    )
    monkeypatch.setattr(env, "_advance_until_decision", lambda: None)
    monkeypatch.setattr(
        env,
        "state",
        lambda: SimpleNamespace(
            game_id="response-dispatch-test",
            bakaze="E",
            kyoku=1,
            honba=0,
            kyotaku=0,
            scores=(25000, 25000, 25000, 25000),
            current_actor=None,
            done=False,
            kyokus_completed=0,
        ),
    )

    def _record_apply_action(_room, actor: int, payload: dict[str, object]) -> None:
        assert actor == 1
        applied.append(payload)

    monkeypatch.setattr(env.manager, "apply_action", _record_apply_action)

    result = env.step(1, chosen_action)

    assert result.done is False
    assert applied == [chosen_raw.to_mjai()]
    assert applied[0]["type"] == expected_raw_type
