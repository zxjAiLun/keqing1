from __future__ import annotations

from types import SimpleNamespace

import pytest

import keqing_core

from keqingrl import ActionSpec, ActionType, DiscardOnlyMahjongEnv, action_from_mahjong_spec, bind_reach_discard
from keqingrl.actions import ACTION_FLAG_TSUMOGIRI
from keqingrl.env import _TurnContext
from mahjong_env.feature_tracker import SnapshotFeatureTracker
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


def test_raw_reach_uses_debug_key_not_canonical_identity() -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    raw_reach = MahjongActionSpec(type="reach", actor=0)

    assert env._raw_action_canonical_key(raw_reach) is None
    assert env._raw_action_debug_key(raw_reach) == "raw:reach|actor=0|tile=None"


def test_non_discard_self_turn_actions_are_not_autopilot_dispatchable_in_discard_only_scope() -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    raw_reach = MahjongActionSpec(type="reach", actor=0)
    raw_kakan = MahjongActionSpec(type="kakan", actor=0, pai="E", consumed=("E", "E", "E"))

    assert env._is_raw_action_controlled(raw_reach, response=False) is False
    assert env._is_raw_action_autopilot_dispatchable(raw_reach, response=False) is False
    assert env._is_raw_action_controlled(raw_kakan, response=False) is False
    assert env._is_raw_action_autopilot_dispatchable(raw_kakan, response=False) is False


def test_runtime_rulebase_missing_capability_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)

    def _missing_rulebase(*_args, **_kwargs):
        raise RuntimeError("Rust rulebase capability is not available")

    monkeypatch.setattr(keqing_core, "choose_rulebase_action", _missing_rulebase)

    with pytest.raises(RuntimeError, match="KeqingRL runtime requires Rust rulebase scoring"):
        env._choose_rulebase_raw_action(
            _self_turn_snapshot(),
            0,
            (MahjongActionSpec(type="dahai", actor=0, pai="4m", tsumogiri=False),),
        )

    assert env._choose_rulebase_raw_action_optional(
        _self_turn_snapshot(),
        0,
        (MahjongActionSpec(type="dahai", actor=0, pai="4m", tsumogiri=False),),
    ) is None


def test_runtime_legal_actions_are_keqing_core_boundary(monkeypatch: pytest.MonkeyPatch) -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)

    monkeypatch.setattr(
        keqing_core,
        "enumerate_public_legal_action_specs",
        lambda *_args: [
            {"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": False},
            {"type": "dahai", "actor": 0, "pai": "7m", "tsumogiri": True},
        ],
    )

    legal = env._enumerate_runtime_legal_actions({"actor": 0}, 0)

    assert [action.pai for action in legal] == ["4m", "7m"]


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
        remaining_wall=lambda: 42,
        state=SimpleNamespace(
            actor_to_move=0 if not response_window else 1,
            last_discard=None if not response_window else {"actor": 0, "pai": "4m"},
            last_kakan=None,
        ),
    )
    dispatched: list[dict[str, object]] = []

    env.room = room
    monkeypatch.setattr(env.manager, "prepare_turn", lambda _room, _actor: None)
    monkeypatch.setattr(env, "_enumerate_runtime_legal_actions", lambda _snapshot, _actor: raw_actions)
    monkeypatch.setattr(env, "_snapshot_with_rl_fields", lambda actor: {"actor": actor, "hand": [], "melds": [[], [], [], []], "discards": [[], [], [], []], "dora_markers": []})
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
    events = env.drain_autopilot_events()
    expected_action_type = (
        ActionType.RYUKYOKU
        if expected_type == "ryukyoku"
        else ActionType.RON
        if response_window
        else ActionType.TSUMO
    )
    assert len(events) == 1
    assert events[0].action_spec.action_type == expected_action_type
    assert events[0].terminal_reason == expected_action_type.name.lower()
    assert events[0].policy_input.metadata["is_autopilot"] is True
    assert events[0].policy_input.metadata["is_learner_controlled"] is False
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


def test_collect_controlled_self_turn_actions_filters_late_reach() -> None:
    env = DiscardOnlyMahjongEnv(
        self_turn_action_types=(
            ActionType.DISCARD,
            ActionType.REACH_DISCARD,
        )
    )
    env._enumerate_reach_discard_candidates = lambda _snapshot, _actor: [("7m", True), ("4m", False)]  # type: ignore[method-assign]
    snapshot = {
        **_self_turn_snapshot(),
        "remaining_wall": 1,
        "scores": [25000, 25000, 25000, 25000],
        "reached": [False, False, False, False],
        "pending_reach": [False, False, False, False],
        "melds": [[], [], [], []],
    }

    controlled_pairs = env._collect_controlled_self_turn_actions(
        snapshot,
        _self_turn_raw_legal_actions(),
    )

    assert [spec.action_type for spec, _dispatch in controlled_pairs] == [
        ActionType.DISCARD,
        ActionType.DISCARD,
    ]


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


def test_collect_controlled_response_actions_autopilots_bare_pass() -> None:
    env = DiscardOnlyMahjongEnv(response_action_types=(ActionType.PASS,))

    controlled_pairs = env._collect_controlled_response_actions((MahjongActionSpec(type="none"),))

    assert controlled_pairs == []


def test_mortal_events_for_response_turn_drops_trailing_reach_accepted() -> None:
    env = DiscardOnlyMahjongEnv(response_action_types=(ActionType.PASS, ActionType.CHI))
    env.room = SimpleNamespace(
        events=(
            {"type": "reach", "actor": 3},
            {"type": "dahai", "actor": 3, "pai": "3s", "tsumogiri": False},
            {"type": "reach_accepted", "actor": 3},
        )
    )
    turn = _TurnContext(
        actor=0,
        snapshot={"actor": 0},
        legal_actions=(ActionSpec(ActionType.PASS),),
        dispatch_actions=(({"type": "none"},),),
        rulebase_chosen=None,
        control_action_types=("PASS",),
    )

    events = env._mortal_events_for_turn(turn)

    assert tuple(events) == env.room.events[:-1]


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


def test_runtime_legal_enumeration_fails_closed_without_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    monkeypatch.setattr(
        "keqingrl.env.keqing_core.enumerate_public_legal_action_specs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Rust public legal action capability is not available")),
    )
    monkeypatch.setattr(
        "keqingrl.env._mahjong_spec_from_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("python fallback should stay unused")),
    )

    with pytest.raises(RuntimeError, match="requires Rust legal action enumeration"):
        env._enumerate_runtime_legal_actions({"actor": 0}, 0)


def test_runtime_terminal_resolver_fails_closed_without_rust(monkeypatch: pytest.MonkeyPatch) -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    raw_actions = (MahjongActionSpec(type="hora", actor=0, target=0, pai="7m"),)
    monkeypatch.setattr(
        "keqingrl.env.keqing_core.resolve_terminal_action",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Rust terminal resolver capability is not available")),
    )

    with pytest.raises(RuntimeError, match="requires Rust terminal resolver"):
        env._resolve_forced_terminal_action({"actor": 0}, 0, raw_actions)


def test_autopilot_trace_uses_rust_terminal_resolver_reason(monkeypatch: pytest.MonkeyPatch) -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    env.room = SimpleNamespace(remaining_wall=lambda: 42)
    raw_actions = (MahjongActionSpec(type="hora", actor=1, target=0, pai="4m"),)
    monkeypatch.setattr(
        "keqingrl.env.keqing_core.resolve_terminal_action",
        lambda *_args, **_kwargs: {
            "action_index": 0,
            "action": raw_actions[0].to_mjai(),
            "terminal_reason": "rust_resolved_ron",
        },
    )

    resolved = env._resolve_forced_terminal_action({"actor": 1}, 1, raw_actions)
    assert resolved is not None
    env._record_autopilot_event(
        {"actor": 1, "hand": [], "melds": [[], [], [], []], "discards": [[], [], [], []], "dora_markers": []},
        1,
        resolved.raw_spec,
        rulebase_action=resolved.raw_spec,
        terminal_reason=resolved.terminal_reason,
    )

    events = env.drain_autopilot_events()
    assert len(events) == 1
    assert events[0].terminal_reason == "rust_resolved_ron"
    assert events[0].policy_input.metadata["terminal_reason"] == "rust_resolved_ron"


def test_dispatch_syncs_runtime_state_from_rust_apply() -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    env.reset(seed=7)
    room = env._require_room()
    actor = env.current_actor()
    assert actor is not None
    dispatch_action = env._require_turn(actor).dispatch_actions[0][0]

    env._dispatch_manager_action(room, actor, dispatch_action)

    assert env._rust_synced_event_count == len(room.events)
    rust_snapshot = keqing_core.replay_state_snapshot(room.events, actor)
    assert room.state.actor_to_move == rust_snapshot["actor_to_move"]
    assert room.state.last_tsumo == rust_snapshot["last_tsumo"]
    assert room.state.players[actor].discards == rust_snapshot["discards"][actor]
    assert sorted(room.state.players[actor].hand.elements()) == rust_snapshot["hand"]


def test_dispatch_fails_closed_without_rust_state_replay(monkeypatch: pytest.MonkeyPatch) -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    env.reset(seed=7)
    room = env._require_room()
    actor = env.current_actor()
    assert actor is not None
    dispatch_action = env._require_turn(actor).dispatch_actions[0][0]
    monkeypatch.setattr(
        "keqingrl.env.keqing_core.replay_state_snapshot",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Rust replay state snapshot capability is not available")),
    )

    with pytest.raises(RuntimeError, match="Rust replay state snapshot capability"):
        env._dispatch_manager_action(room, actor, dispatch_action)


def _python_action_features_reference(
    snapshot: dict[str, object],
    spec: ActionSpec,
    *,
    remaining_wall: int,
) -> list[float]:
    tracker = SnapshotFeatureTracker.from_state(snapshot, actor=int(snapshot["actor"]))
    tile = -1 if spec.tile is None else int(spec.tile)
    tile_norm = 0.0 if tile < 0 else float(tile) / 33.0
    hand_count = 0.0 if tile < 0 else float(tracker.hand_counts34[tile]) / 4.0
    visible = 0.0 if tile < 0 else float(tracker.visible_counts34[tile]) / 4.0
    is_honor = 1.0 if tile >= 27 else 0.0
    is_terminal = 1.0 if 0 <= tile < 27 and tile % 9 in (0, 8) else 0.0
    return [
        float(spec.action_type) / float(max(1, len(ActionType) - 1)),
        tile_norm,
        1.0 if spec.flags & ACTION_FLAG_TSUMOGIRI else 0.0,
        hand_count,
        visible,
        is_honor,
        is_terminal,
        float(remaining_wall) / 70.0,
    ]


def test_rust_action_feature_parity_covers_non_discard_unlock_scope() -> None:
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    room = SimpleNamespace(remaining_wall=lambda: 42)
    env.room = room
    snapshot = {
        "actor": 1,
        "hand": ["1m", "2m", "3m", "4m", "5m", "5mr", "7p", "8p", "9p", "E", "E", "P", "C"],
        "melds": [
            [],
            [{"type": "pon", "pai": "4s", "consumed": ["4s", "4s"], "target": 0}],
            [{"type": "chi", "pai": "3m", "consumed": ["1m", "2m"], "target": 1}],
            [],
        ],
        "discards": [
            [{"pai": "9m", "tsumogiri": False}],
            [{"pai": "1p", "tsumogiri": True}],
            [],
            ["N"],
        ],
        "dora_markers": ["5pr"],
        "last_tsumo": [None, "7p", None, None],
        "tsumo_pai": "7p",
    }
    specs = (
        ActionSpec(ActionType.REACH_DISCARD, tile=0, flags=ACTION_FLAG_TSUMOGIRI),
        ActionSpec(ActionType.PASS),
        ActionSpec(ActionType.PON, tile=3, consumed=(3, 3), from_who=0),
        ActionSpec(ActionType.CHI, tile=2, consumed=(0, 1), from_who=0),
        ActionSpec(ActionType.DAIMINKAN, tile=27, consumed=(27, 27, 27), from_who=0),
        ActionSpec(ActionType.ANKAN, tile=31, consumed=(31, 31, 31, 31)),
        ActionSpec(ActionType.KAKAN, tile=13, consumed=(13,)),
    )

    actual = env._action_features_batch(snapshot, specs)
    expected = [
        _python_action_features_reference(snapshot, spec, remaining_wall=42)
        for spec in specs
    ]

    assert len(actual) == len(expected)
    for actual_row, expected_row in zip(actual, expected):
        assert actual_row == pytest.approx(expected_row)
    assert keqing_core.keqingrl_action_feature_dim() == 8


def test_rust_action_feature_generation_fails_closed_without_capability(monkeypatch: pytest.MonkeyPatch) -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    env.room = SimpleNamespace(remaining_wall=lambda: 42)
    monkeypatch.setattr(
        "keqingrl.env.keqing_core.build_keqingrl_action_features_typed",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("Rust KeqingRL typed action feature capability is not available")),
    )

    with pytest.raises(RuntimeError, match="requires Rust typed action feature generation"):
        env._action_features({"actor": 0, "hand": ["1m"]}, ActionSpec(ActionType.PASS))


def test_env_snapshot_uses_rust_replay_not_manager_shanten(monkeypatch: pytest.MonkeyPatch) -> None:
    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    env.reset(seed=13)
    actor = env.current_actor()
    assert actor is not None
    monkeypatch.setattr(
        env.manager,
        "get_snap_with_shanten",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Python shanten snapshot should not be used")),
    )

    snapshot = env._snapshot_with_rl_fields(actor)

    assert snapshot["actor"] == actor
    assert "shanten" in snapshot
    assert "waits_count" in snapshot
    assert "waits_tiles" in snapshot


def test_battle_manager_shanten_waits_are_rust_owned(monkeypatch: pytest.MonkeyPatch) -> None:
    from gateway.battle import BattleManager
    from mahjong_env.replay import _calc_shanten_waits

    env = DiscardOnlyMahjongEnv(max_kyokus=1)
    env.reset(seed=17)
    room = env._require_room()
    actor = env.current_actor()
    assert actor is not None
    monkeypatch.setattr(
        "mahjong_env.replay._calc_shanten_waits",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("Python waits fallback should not be used")),
    )
    del _calc_shanten_waits

    snapshot = BattleManager().get_snap_with_shanten(room, actor)

    assert snapshot["actor"] == actor
    assert isinstance(snapshot["shanten"], int)
    assert isinstance(snapshot["waits_count"], int)
    assert len(snapshot["waits_tiles"]) == 34
