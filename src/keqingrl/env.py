"""Discard-only Mahjong env scaffolding for keqingrl."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import keqing_core
import torch

from gateway.battle import BattleConfig, BattleManager, BattleRoom
from keqingrl.actions import (
    ActionSpec,
    ActionType,
    action_from_mahjong_spec,
    bind_reach_discard,
    encode_action_id,
)
from keqingrl.contracts import ObsTensorBatch, PolicyInput
from keqingrl.metadata import (
    ACTION_FEATURE_CONTRACT_VERSION,
    ENV_CONTRACT_VERSION,
    NATIVE_ACTION_IDENTITY_VERSION,
    NATIVE_LEGAL_ENUMERATION_VERSION,
    NATIVE_SCHEMA_NAME,
    NATIVE_SCHEMA_VERSION,
    NATIVE_TERMINAL_RESOLVER_VERSION,
    OBSERVATION_CONTRACT_VERSION,
    REWARD_SPEC_VERSION,
    RULE_SCORE_VERSION,
    STYLE_CONTEXT_VERSION,
)
from keqingrl.rewards import (
    DEFAULT_PT_MAP,
    build_rule_context,
    terminal_rank_rewards,
)
from keqingrl.rule_score import DEFAULT_RULE_SCORE_CONFIG, RuleScoreConfig, score_legal_actions
from keqingrl.style import DEFAULT_STYLE_CONTEXT, StyleContext
from mahjong_env.feature_tracker import SnapshotFeatureTracker
from mahjong_env.final_rank import final_ranks
from mahjong_env.types import ActionSpec as MahjongActionSpec
from training.state_features import encode as encode_state_features

_KEQINGRL_NATIVE_SCHEMA_KWARGS = {
    "schema_name": "keqingrl_native_boundary",
    "schema_version": 1,
    "action_identity_version": 1,
    "legal_enumeration_version": 1,
    "terminal_resolver_version": 1,
}

_SUPPORTED_SELF_TURN_ACTION_TYPES = frozenset(
    {
        ActionType.DISCARD,
        ActionType.REACH_DISCARD,
        ActionType.TSUMO,
        ActionType.ANKAN,
        ActionType.KAKAN,
        ActionType.RYUKYOKU,
    }
)
_SUPPORTED_RESPONSE_ACTION_TYPES = frozenset(
    {
        ActionType.RON,
        ActionType.CHI,
        ActionType.PON,
        ActionType.DAIMINKAN,
        ActionType.PASS,
    }
)


@dataclass(frozen=True)
class EnvState:
    game_id: str
    bakaze: str
    kyoku: int
    honba: int
    kyotaku: int
    scores: tuple[int, int, int, int]
    current_actor: int | None
    done: bool
    kyokus_completed: int


@dataclass(frozen=True)
class StepResult:
    reward: float
    done: bool
    next_actor: int | None
    state: EnvState
    terminal_rewards: tuple[float, float, float, float] | None = None
    final_ranks: tuple[int, int, int, int] | None = None
    scores: tuple[int, int, int, int] | None = None


@dataclass(frozen=True)
class AutopilotTraceEvent:
    actor: int
    policy_input: PolicyInput
    action_index: int
    action_spec: ActionSpec
    terminal_reason: str | None
    rulebase_chosen: str | None
    policy_chosen: str


@dataclass(frozen=True)
class _TurnContext:
    actor: int
    snapshot: dict[str, object]
    legal_actions: tuple[ActionSpec, ...]
    dispatch_actions: tuple[tuple[dict[str, object], ...], ...]
    rulebase_chosen: str | None = None
    control_action_types: tuple[str, ...] = ()


@dataclass(frozen=True)
class _ResolvedTerminalAction:
    raw_spec: MahjongActionSpec
    terminal_reason: str


class DiscardOnlyMahjongEnv:
    """A conservative interactive env where the policy controls a bounded action subset."""

    def __init__(
        self,
        *,
        pt_map: Sequence[int | float] = DEFAULT_PT_MAP,
        rank_score_scale: float = 0.0,
        max_kyokus: int | None = None,
        manager: BattleManager | None = None,
        self_turn_action_types: tuple[ActionType, ...] = (ActionType.DISCARD,),
        response_action_types: tuple[ActionType, ...] = (),
        forced_autopilot_action_types: tuple[ActionType, ...] = (
            ActionType.TSUMO,
            ActionType.RON,
            ActionType.RYUKYOKU,
        ),
        style_context: StyleContext = DEFAULT_STYLE_CONTEXT,
        rule_score_config: RuleScoreConfig = DEFAULT_RULE_SCORE_CONFIG,
    ) -> None:
        keqing_core.enable_rust(True)
        self.native_schema_info = keqing_core.require_native_schema(**_KEQINGRL_NATIVE_SCHEMA_KWARGS)
        self.manager = manager or BattleManager()
        self.pt_map = tuple(float(value) for value in pt_map)
        self.rank_score_scale = float(rank_score_scale)
        self.max_kyokus = max_kyokus
        configured_action_types = tuple(ActionType(value) for value in self_turn_action_types)
        if ActionType.DISCARD not in configured_action_types:
            raise ValueError("self_turn_action_types must include ActionType.DISCARD")
        unsupported_action_types = [
            action_type
            for action_type in configured_action_types
            if action_type not in _SUPPORTED_SELF_TURN_ACTION_TYPES
        ]
        if unsupported_action_types:
            unsupported = ", ".join(action_type.name for action_type in unsupported_action_types)
            raise ValueError(f"unsupported self-turn action types for this slice: {unsupported}")
        configured_response_action_types = tuple(ActionType(value) for value in response_action_types)
        unsupported_response_action_types = [
            action_type
            for action_type in configured_response_action_types
            if action_type not in _SUPPORTED_RESPONSE_ACTION_TYPES
        ]
        if unsupported_response_action_types:
            unsupported = ", ".join(action_type.name for action_type in unsupported_response_action_types)
            raise ValueError(f"unsupported response action types for this slice: {unsupported}")
        self.self_turn_action_types = configured_action_types
        self._self_turn_action_type_set = set(configured_action_types)
        self.response_action_types = configured_response_action_types
        self._response_action_type_set = set(configured_response_action_types)
        self.forced_autopilot_action_types = tuple(ActionType(value) for value in forced_autopilot_action_types)
        self._forced_autopilot_action_type_set = set(self.forced_autopilot_action_types)
        self.style_context = style_context
        self.style_context_tensor = style_context.to_tensor()
        self.rule_score_config = rule_score_config
        self.rule_context = build_rule_context(
            self.pt_map,
            rank_score_scale=self.rank_score_scale,
            is_hanchan=max_kyokus is None or max_kyokus > 1,
        )
        self.room: BattleRoom | None = None
        self._turn: _TurnContext | None = None
        self._base_seed: int | None = None
        self._seed_cursor = 0
        self._done = False
        self._completed_kyokus = 0
        self._game_start_oya = 0
        self._autopilot_events: list[AutopilotTraceEvent] = []
        self._rust_synced_event_count = 0

    def reset(self, seed: int | None = None) -> EnvState:
        if self.room is not None:
            self.manager.close_room(self.room.game_id)

        self._base_seed = seed
        self._seed_cursor = 0
        self._done = False
        self._completed_kyokus = 0
        self._turn = None
        self._autopilot_events.clear()
        self.room = self.manager.create_room(_default_battle_config())
        self._rust_synced_event_count = 0
        self.manager.start_kyoku(self.room, seed=self._next_seed())
        self._sync_rust_runtime_state(self.room, actor_hint=int(self.room.state.oya))
        self._game_start_oya = int(self.room.state.oya)
        self._advance_until_decision()
        return self.state()

    def state(self) -> EnvState:
        room = self._require_room()
        return EnvState(
            game_id=room.game_id,
            bakaze=room.state.bakaze,
            kyoku=room.state.kyoku,
            honba=room.state.honba,
            kyotaku=room.state.kyotaku,
            scores=tuple(int(score) for score in room.state.scores),  # type: ignore[arg-type]
            current_actor=None if self._done or self._turn is None else self._turn.actor,
            done=self._done,
            kyokus_completed=self._completed_kyokus,
        )

    def current_actor(self) -> int | None:
        return self.state().current_actor

    def is_done(self) -> bool:
        return self._done

    def legal_actions(self, actor: int) -> tuple[ActionSpec, ...]:
        turn = self._require_turn(actor)
        return turn.legal_actions

    def drain_autopilot_events(self) -> tuple[AutopilotTraceEvent, ...]:
        events = tuple(self._autopilot_events)
        self._autopilot_events.clear()
        return events

    def observe(self, actor: int) -> PolicyInput:
        turn = self._require_turn(actor)
        tile_obs_np, scalar_obs_np = encode_state_features(turn.snapshot, actor)

        tile_obs = torch.from_numpy(tile_obs_np).float().unsqueeze(0)
        scalar_obs = torch.from_numpy(scalar_obs_np).float().unsqueeze(0)
        legal_action_ids = torch.tensor(
            [[encode_action_id(spec) for spec in turn.legal_actions]],
            dtype=torch.long,
        )
        legal_action_features = torch.tensor(
            [self._action_features_batch(turn.snapshot, turn.legal_actions)],
            dtype=torch.float32,
        )
        legal_action_mask = torch.ones((1, len(turn.legal_actions)), dtype=torch.bool)
        rule_scores = score_legal_actions(
            turn.snapshot,
            actor,
            turn.legal_actions,
            config=self.rule_score_config,
        )
        return PolicyInput(
            obs=ObsTensorBatch(tile_obs=tile_obs, scalar_obs=scalar_obs),
            legal_action_ids=legal_action_ids,
            legal_action_features=legal_action_features,
            legal_action_mask=legal_action_mask,
            rule_context=self.rule_context.unsqueeze(0).clone(),
            raw_rule_scores=rule_scores.raw_rule_scores.unsqueeze(0),
            prior_logits=rule_scores.prior_logits.unsqueeze(0),
            style_context=self.style_context_tensor.unsqueeze(0).clone(),
            legal_actions=(turn.legal_actions,),
            metadata={
                "rulebase_chosen": turn.rulebase_chosen,
                "control_action_types": turn.control_action_types,
                "is_learner_controlled": True,
                "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
                "action_feature_contract_version": ACTION_FEATURE_CONTRACT_VERSION,
                "env_contract_version": ENV_CONTRACT_VERSION,
                "native_schema_name": self.native_schema_info.get("schema_name", NATIVE_SCHEMA_NAME),
                "native_schema_version": self.native_schema_info.get("schema_version", NATIVE_SCHEMA_VERSION),
                "native_action_identity_version": self.native_schema_info.get("action_identity_version", NATIVE_ACTION_IDENTITY_VERSION),
                "native_legal_enumeration_version": self.native_schema_info.get("legal_enumeration_version", NATIVE_LEGAL_ENUMERATION_VERSION),
                "native_terminal_resolver_version": self.native_schema_info.get("terminal_resolver_version", NATIVE_TERMINAL_RESOLVER_VERSION),
                "rule_score_version": RULE_SCORE_VERSION,
                "reward_spec_version": REWARD_SPEC_VERSION,
                "style_context_version": STYLE_CONTEXT_VERSION,
            },
        )

    def step(self, actor: int, action_spec: ActionSpec) -> StepResult:
        room = self._require_room()
        turn = self._require_turn(actor)
        action_index = self._resolve_action_index(turn, action_spec)
        for dispatch_action in turn.dispatch_actions[action_index]:
            self._dispatch_manager_action(room, actor, dispatch_action)
        self._turn = None
        self._advance_until_decision()

        terminal_rewards = None
        reward = 0.0
        final_rank_targets = None
        scores = None
        if self._done:
            terminal_rewards = self.terminal_rewards()
            reward = float(terminal_rewards[actor])
            final_rank_targets = self.final_ranks()
            scores = tuple(int(score) for score in room.state.scores)  # type: ignore[arg-type]

        next_actor = self.current_actor()
        return StepResult(
            reward=reward,
            done=self._done,
            next_actor=next_actor,
            state=self.state(),
            terminal_rewards=terminal_rewards,
            final_ranks=final_rank_targets,
            scores=scores,
        )

    def terminal_rewards(self) -> tuple[float, float, float, float]:
        room = self._require_room()
        if not self._done:
            raise RuntimeError("terminal_rewards() requires a finished episode")
        return terminal_rank_rewards(
            room.state.scores,
            self.pt_map,
            initial_oya=self._game_start_oya,
            normalize=True,
        )

    def final_ranks(self) -> tuple[int, int, int, int]:
        room = self._require_room()
        if not self._done:
            raise RuntimeError("final_ranks() requires a finished episode")
        return final_ranks(room.state.scores, initial_oya=self._game_start_oya)

    def _advance_until_decision(self) -> None:
        room = self._require_room()
        self._turn = None

        while not self._done:
            if room.phase == "ended":
                self._completed_kyokus += 1
                if self.max_kyokus is not None and self._completed_kyokus >= self.max_kyokus:
                    self._mark_done()
                    return
                if self.manager.is_game_ended(room) or not self.manager.next_kyoku(room):
                    self._mark_done()
                    return
                self.manager.start_kyoku(room, seed=self._next_seed())
                self._sync_rust_runtime_state(room, actor_hint=int(room.state.oya))
                continue

            actor = room.state.actor_to_move
            if actor is None:
                self._mark_done()
                return

            before_event_count = len(room.events)
            self.manager.prepare_turn(room, actor)
            if len(room.events) != before_event_count:
                self._sync_rust_runtime_state(room, actor_hint=actor)
            if room.phase == "ended":
                continue

            actor = room.state.actor_to_move
            if actor is None:
                self._mark_done()
                return

            snapshot = self._snapshot_with_rl_fields(actor)
            raw_legal_actions = self._enumerate_runtime_legal_actions(snapshot, actor)
            if self._is_response_window(actor):
                forced_response = self._resolve_forced_terminal_action(snapshot, actor, raw_legal_actions)
                if forced_response is not None:
                    self._record_autopilot_event(
                        snapshot,
                        actor,
                        forced_response.raw_spec,
                        rulebase_action=forced_response.raw_spec,
                        terminal_reason=forced_response.terminal_reason,
                    )
                    self._dispatch_manager_action(room, actor, forced_response.raw_spec.to_mjai())
                    continue
                rulebase_response = self._choose_rulebase_raw_action(snapshot, actor, raw_legal_actions)
                if rulebase_response is not None and not self._is_raw_action_controlled(
                    rulebase_response,
                    response=True,
                ):
                    self._record_autopilot_event(snapshot, actor, rulebase_response, rulebase_action=rulebase_response)
                    self._dispatch_manager_action(room, actor, rulebase_response.to_mjai())
                    continue
                controlled_response_pairs = self._collect_controlled_response_actions(raw_legal_actions)
                auto_response = self._auto_response_action(
                    raw_legal_actions,
                    has_controlled_actions=bool(controlled_response_pairs),
                )
                if auto_response is not None:
                    self._record_autopilot_event(snapshot, actor, auto_response, rulebase_action=rulebase_response)
                    self._dispatch_manager_action(room, actor, auto_response.to_mjai())
                    continue
                if not controlled_response_pairs:
                    raise RuntimeError(f"keqingrl env found no controllable response action for actor {actor}")
                self._turn = _TurnContext(
                    actor=actor,
                    snapshot=snapshot,
                    legal_actions=tuple(action for action, _ in controlled_response_pairs),
                    dispatch_actions=tuple(dispatch for _, dispatch in controlled_response_pairs),
                    rulebase_chosen=self._raw_action_canonical_key(rulebase_response),
                    control_action_types=tuple(action_type.name for action_type in self.response_action_types),
                )
                return

            forced_self_action = self._resolve_forced_terminal_action(snapshot, actor, raw_legal_actions)
            if forced_self_action is not None:
                self._record_autopilot_event(
                    snapshot,
                    actor,
                    forced_self_action.raw_spec,
                    rulebase_action=forced_self_action.raw_spec,
                    terminal_reason=forced_self_action.terminal_reason,
                )
                self._dispatch_manager_action(room, actor, forced_self_action.raw_spec.to_mjai())
                continue

            rulebase_self_action = self._choose_rulebase_raw_action(snapshot, actor, raw_legal_actions)
            if rulebase_self_action is not None and not self._is_raw_action_controlled(
                rulebase_self_action,
                response=False,
            ):
                self._record_autopilot_event(snapshot, actor, rulebase_self_action, rulebase_action=rulebase_self_action)
                self._dispatch_manager_action(room, actor, rulebase_self_action.to_mjai())
                continue

            controlled_pairs = self._collect_controlled_self_turn_actions(snapshot, raw_legal_actions)
            auto_action = self._auto_self_action(
                raw_legal_actions,
                has_controlled_actions=bool(controlled_pairs),
            )
            if auto_action is not None:
                self._record_autopilot_event(snapshot, actor, auto_action, rulebase_action=rulebase_self_action)
                self._dispatch_manager_action(room, actor, auto_action.to_mjai())
                continue

            if not controlled_pairs:
                raise RuntimeError(f"keqingrl env found no controllable self-turn action for actor {actor}")

            self._turn = _TurnContext(
                actor=actor,
                snapshot=snapshot,
                legal_actions=tuple(action for action, _ in controlled_pairs),
                dispatch_actions=tuple(dispatch for _, dispatch in controlled_pairs),
                rulebase_chosen=self._raw_action_canonical_key(rulebase_self_action),
                control_action_types=tuple(action_type.name for action_type in self.self_turn_action_types),
            )
            return

    def _record_autopilot_event(
        self,
        snapshot: dict[str, object],
        actor: int,
        raw_action: MahjongActionSpec,
        *,
        rulebase_action: MahjongActionSpec | None,
        terminal_reason: str | None = None,
    ) -> None:
        try:
            action_spec = action_from_mahjong_spec(raw_action)
        except ValueError:
            return
        policy_input = self._policy_input_for_trace_action(
            snapshot,
            actor,
            action_spec,
            rulebase_chosen=self._raw_action_canonical_key(rulebase_action),
            terminal_reason=terminal_reason if terminal_reason is not None else self._terminal_reason_for_action(action_spec),
        )
        self._autopilot_events.append(
            AutopilotTraceEvent(
                actor=int(actor),
                policy_input=policy_input,
                action_index=0,
                action_spec=action_spec,
                terminal_reason=terminal_reason if terminal_reason is not None else self._terminal_reason_for_action(action_spec),
                rulebase_chosen=self._raw_action_canonical_key(rulebase_action),
                policy_chosen=action_spec.canonical_key,
            )
        )

    def _policy_input_for_trace_action(
        self,
        snapshot: dict[str, object],
        actor: int,
        action_spec: ActionSpec,
        *,
        rulebase_chosen: str | None,
        terminal_reason: str | None,
    ) -> PolicyInput:
        try:
            tile_obs_np, scalar_obs_np = encode_state_features(snapshot, actor)
            tile_obs = torch.from_numpy(tile_obs_np).float().unsqueeze(0)
            scalar_obs = torch.from_numpy(scalar_obs_np).float().unsqueeze(0)
        except Exception:
            tile_obs = torch.zeros((1, 4, 34), dtype=torch.float32)
            scalar_obs = torch.zeros((1, 6), dtype=torch.float32)
        legal_actions = (action_spec,)
        legal_action_ids = torch.tensor([[encode_action_id(action_spec)]], dtype=torch.long)
        legal_action_features = torch.tensor(
            [self._action_features_batch(snapshot, legal_actions)],
            dtype=torch.float32,
        )
        legal_action_mask = torch.ones((1, 1), dtype=torch.bool)
        try:
            rule_scores = score_legal_actions(
                snapshot,
                actor,
                legal_actions,
                config=self.rule_score_config,
            )
            raw_rule_scores = rule_scores.raw_rule_scores.unsqueeze(0)
            prior_logits = rule_scores.prior_logits.unsqueeze(0)
        except Exception:
            raw_rule_scores = torch.zeros((1, 1), dtype=torch.float32)
            prior_logits = torch.zeros((1, 1), dtype=torch.float32)
        return PolicyInput(
            obs=ObsTensorBatch(tile_obs=tile_obs, scalar_obs=scalar_obs),
            legal_action_ids=legal_action_ids,
            legal_action_features=legal_action_features,
            legal_action_mask=legal_action_mask,
            rule_context=self.rule_context.unsqueeze(0).clone(),
            raw_rule_scores=raw_rule_scores,
            prior_logits=prior_logits,
            style_context=self.style_context_tensor.unsqueeze(0).clone(),
            legal_actions=(legal_actions,),
            metadata={
                "rulebase_chosen": rulebase_chosen,
                "control_action_types": (),
                "is_autopilot": True,
                "is_learner_controlled": False,
                "terminal_reason": terminal_reason,
                "observation_contract_version": OBSERVATION_CONTRACT_VERSION,
                "action_feature_contract_version": ACTION_FEATURE_CONTRACT_VERSION,
                "env_contract_version": ENV_CONTRACT_VERSION,
                "native_schema_name": self.native_schema_info.get("schema_name", NATIVE_SCHEMA_NAME),
                "native_schema_version": self.native_schema_info.get("schema_version", NATIVE_SCHEMA_VERSION),
                "native_action_identity_version": self.native_schema_info.get("action_identity_version", NATIVE_ACTION_IDENTITY_VERSION),
                "native_legal_enumeration_version": self.native_schema_info.get("legal_enumeration_version", NATIVE_LEGAL_ENUMERATION_VERSION),
                "native_terminal_resolver_version": self.native_schema_info.get("terminal_resolver_version", NATIVE_TERMINAL_RESOLVER_VERSION),
                "rule_score_version": RULE_SCORE_VERSION,
                "reward_spec_version": REWARD_SPEC_VERSION,
                "style_context_version": STYLE_CONTEXT_VERSION,
            },
        )

    def _terminal_reason_for_action(self, action_spec: ActionSpec) -> str | None:
        if action_spec.action_type == ActionType.TSUMO:
            return "tsumo"
        if action_spec.action_type == ActionType.RON:
            return "ron"
        if action_spec.action_type == ActionType.RYUKYOKU:
            return "ryukyoku"
        return None

    def _enumerate_runtime_legal_actions(self, snapshot: dict[str, object], actor: int) -> tuple[MahjongActionSpec, ...]:
        try:
            raw_items = keqing_core.enumerate_public_legal_action_specs(snapshot, actor)
        except RuntimeError as exc:
            if keqing_core.is_missing_rust_capability_error(exc):
                raise RuntimeError("KeqingRL runtime requires Rust legal action enumeration") from exc
            raise
        return tuple(_mahjong_spec_from_payload(item) for item in raw_items)

    def _resolve_forced_terminal_action(
        self,
        snapshot: dict[str, object],
        actor: int,
        raw_legal_actions: Sequence[MahjongActionSpec],
    ) -> _ResolvedTerminalAction | None:
        raw_payloads = [raw.to_mjai() for raw in raw_legal_actions]
        forced_action_types = [action_type.name for action_type in self.forced_autopilot_action_types]
        try:
            resolved = keqing_core.resolve_terminal_action(snapshot, actor, raw_payloads, forced_action_types)
        except RuntimeError as exc:
            if keqing_core.is_missing_rust_capability_error(exc):
                raise RuntimeError("KeqingRL runtime requires Rust terminal resolver") from exc
            raise
        if resolved is None:
            return None
        return _ResolvedTerminalAction(
            raw_spec=raw_legal_actions[int(resolved["action_index"])],
            terminal_reason=str(resolved["terminal_reason"]),
        )

    def _forced_autopilot_action(
        self,
        raw_legal_actions: Sequence[MahjongActionSpec],
    ) -> MahjongActionSpec | None:
        for raw_spec in raw_legal_actions:
            action_type = self._raw_action_type(raw_spec)
            if action_type in self._forced_autopilot_action_type_set:
                return raw_spec
        return None

    def _choose_rulebase_raw_action(
        self,
        snapshot: dict[str, object],
        actor: int,
        raw_legal_actions: Sequence[MahjongActionSpec],
    ) -> MahjongActionSpec | None:
        if not raw_legal_actions:
            return None
        raw_payloads = [raw.to_mjai() for raw in raw_legal_actions]
        try:
            chosen_payload = keqing_core.choose_rulebase_action(snapshot, actor, raw_payloads)
        except RuntimeError as exc:
            if not (
                keqing_core.is_missing_rust_capability_error(exc)
                or "rulebase capability" in str(exc)
            ):
                raise
            return None
        if chosen_payload is None:
            return None
        for raw_spec, payload in zip(raw_legal_actions, raw_payloads):
            if payload == chosen_payload:
                return raw_spec
        return None

    def _is_raw_action_controlled(
        self,
        raw_spec: MahjongActionSpec,
        *,
        response: bool,
    ) -> bool:
        action_type = self._raw_action_type(raw_spec)
        if response:
            return action_type in self._response_action_type_set
        return action_type in self._self_turn_action_type_set

    def _raw_action_type(self, raw_spec: MahjongActionSpec) -> ActionType | None:
        if raw_spec.type == "reach":
            return ActionType.REACH_DISCARD
        try:
            return action_from_mahjong_spec(raw_spec).action_type
        except ValueError:
            return None

    def _raw_action_canonical_key(self, raw_spec: MahjongActionSpec | None) -> str | None:
        if raw_spec is None:
            return None
        if raw_spec.type == "reach":
            return f"{int(ActionType.REACH_DISCARD)}|tile=-1|consumed=|from=-1|flags=0"
        try:
            return action_from_mahjong_spec(raw_spec).canonical_key
        except ValueError:
            return None

    def _is_response_window(self, actor: int) -> bool:
        room = self._require_room()
        is_discard_response = bool(
            room.state.last_discard and room.state.last_discard.get("actor") != actor
        )
        is_kakan_response = bool(
            room.state.last_kakan and room.state.last_kakan.get("actor") != actor
        )
        return is_discard_response or is_kakan_response

    def _auto_response_action(
        self,
        raw_legal_actions: Sequence[MahjongActionSpec],
        *,
        has_controlled_actions: bool,
    ) -> MahjongActionSpec | None:
        if has_controlled_actions:
            return None
        return next(
            (
                spec
                for spec in raw_legal_actions
                if spec.type in {"hora", "none"}
            ),
            None,
        )

    def _auto_self_action(
        self,
        raw_legal_actions: Sequence[MahjongActionSpec],
        *,
        has_controlled_actions: bool,
    ) -> MahjongActionSpec | None:
        if ActionType.TSUMO not in self._self_turn_action_type_set:
            tsumo_action = next(
                (spec for spec in raw_legal_actions if self._is_self_turn_hora(spec)),
                None,
            )
            if tsumo_action is not None:
                return tsumo_action

        if has_controlled_actions:
            return None

        for action_type in ("hora", "ryukyoku", "none"):
            if action_type == "hora":
                chosen = next(
                    (spec for spec in raw_legal_actions if self._is_self_turn_hora(spec)),
                    None,
                )
            else:
                chosen = next(
                    (spec for spec in raw_legal_actions if spec.type == action_type),
                    None,
                )
            if chosen is not None:
                return chosen
        return None

    def _convert_supported_self_turn_raw_action(
        self,
        raw_spec: MahjongActionSpec,
    ) -> ActionSpec | None:
        if raw_spec.type == "dahai":
            action_spec = action_from_mahjong_spec(raw_spec)
        elif raw_spec.type == "hora":
            if not self._is_self_turn_hora(raw_spec):
                return None
            action_spec = action_from_mahjong_spec(raw_spec)
        elif raw_spec.type in {"ankan", "kakan"}:
            action_spec = action_from_mahjong_spec(raw_spec)
        elif raw_spec.type == "ryukyoku":
            action_spec = action_from_mahjong_spec(raw_spec)
        else:
            return None

        if action_spec.action_type not in self._self_turn_action_type_set:
            return None
        return action_spec

    def _collect_controlled_self_turn_actions(
        self,
        snapshot: dict[str, object],
        raw_legal_actions: Sequence[MahjongActionSpec],
    ) -> list[tuple[ActionSpec, tuple[dict[str, object], ...]]]:
        controlled_pairs: list[tuple[ActionSpec, tuple[dict[str, object], ...]]] = []
        for raw_spec in raw_legal_actions:
            if raw_spec.type == "reach":
                if ActionType.REACH_DISCARD in self._self_turn_action_type_set:
                    controlled_pairs.extend(
                        self._collect_reach_discard_actions(snapshot, raw_legal_actions)
                    )
                continue
            action_spec = self._convert_supported_self_turn_raw_action(raw_spec)
            if action_spec is not None:
                controlled_pairs.append((action_spec, (raw_spec.to_mjai(),)))
        return controlled_pairs

    def _convert_supported_response_raw_action(
        self,
        raw_spec: MahjongActionSpec,
    ) -> ActionSpec | None:
        if raw_spec.type == "hora":
            if self._is_self_turn_hora(raw_spec):
                return None
            action_spec = action_from_mahjong_spec(raw_spec)
        elif raw_spec.type in {"chi", "pon", "daiminkan", "none"}:
            action_spec = action_from_mahjong_spec(raw_spec)
        else:
            return None

        if action_spec.action_type not in self._response_action_type_set:
            return None
        return action_spec

    def _collect_controlled_response_actions(
        self,
        raw_legal_actions: Sequence[MahjongActionSpec],
    ) -> list[tuple[ActionSpec, tuple[dict[str, object], ...]]]:
        controlled_pairs: list[tuple[ActionSpec, tuple[dict[str, object], ...]]] = []
        for raw_spec in raw_legal_actions:
            action_spec = self._convert_supported_response_raw_action(raw_spec)
            if action_spec is not None:
                controlled_pairs.append((action_spec, (raw_spec.to_mjai(),)))
        return controlled_pairs

    def _collect_reach_discard_actions(
        self,
        snapshot: dict[str, object],
        raw_legal_actions: Sequence[MahjongActionSpec],
    ) -> list[tuple[ActionSpec, tuple[dict[str, object], ...]]]:
        actor = int(snapshot["actor"])
        candidate_keys = set(self._enumerate_reach_discard_candidates(snapshot, actor))
        if not candidate_keys:
            return []

        bindings: list[tuple[ActionSpec, tuple[dict[str, object], ...]]] = []
        seen_keys: set[tuple[str, bool]] = set()
        for raw_spec in raw_legal_actions:
            if raw_spec.type != "dahai":
                continue
            key = (str(raw_spec.pai), bool(raw_spec.tsumogiri))
            if key in seen_keys or key not in candidate_keys:
                continue
            discard_action = action_from_mahjong_spec(raw_spec)
            bindings.append(
                (
                    bind_reach_discard(discard_action),
                    (
                        {"type": "reach", "actor": actor},
                        raw_spec.to_mjai(),
                    ),
                )
            )
            seen_keys.add(key)
        return bindings

    def _enumerate_reach_discard_candidates(
        self,
        snapshot: dict[str, object],
        actor: int,
    ) -> list[tuple[str, bool]]:
        try:
            return list(keqing_core.enumerate_keqingv4_reach_discards(snapshot, actor))
        except RuntimeError as exc:
            if keqing_core.is_missing_rust_capability_error(exc):
                raise RuntimeError("KeqingRL runtime requires Rust reach-discard enumeration") from exc
            raise

    def _is_self_turn_hora(self, raw_spec: MahjongActionSpec) -> bool:
        return raw_spec.type == "hora" and (
            raw_spec.target is None or raw_spec.target == raw_spec.actor
        )

    def _resolve_action_index(self, turn: _TurnContext, action_spec: ActionSpec) -> int:
        identity_matches = [
            index for index, candidate in enumerate(turn.legal_actions) if candidate is action_spec
        ]
        if len(identity_matches) == 1:
            return identity_matches[0]
        if len(identity_matches) > 1:
            raise RuntimeError("identity-resolved action matched multiple legal entries")

        value_matches = [
            index for index, candidate in enumerate(turn.legal_actions) if candidate == action_spec
        ]
        if len(value_matches) == 1:
            return value_matches[0]
        if len(value_matches) > 1:
            raise ValueError(
                "ambiguous action_spec match; pass back the exact ActionSpec instance from legal_actions"
            )
        raise ValueError("action_spec is not part of the current ordered legal-action list")

    def _dispatch_manager_action(
        self,
        room: BattleRoom,
        actor: int,
        action: dict[str, object],
    ) -> None:
        before_event_count = len(getattr(room, "events", ()))
        if action.get("type") == "ryukyoku":
            self.manager.ryukyoku(room)
        else:
            self.manager.apply_action(room, actor, action)
        events = getattr(room, "events", None)
        if events is not None and len(events) != before_event_count:
            self._sync_rust_runtime_state(room, actor_hint=actor)

    def _sync_rust_runtime_state(self, room: BattleRoom, *, actor_hint: int | None = None) -> None:
        event_count = len(room.events)
        if event_count <= self._rust_synced_event_count:
            return
        self.manager.sync_state_from_rust_events(room, actor_hint=actor_hint)
        self._rust_synced_event_count = event_count

    def _action_features_batch(
        self,
        snapshot: dict[str, object],
        specs: Sequence[ActionSpec],
    ) -> list[list[float]]:
        tracker = SnapshotFeatureTracker.from_state(snapshot, actor=int(snapshot["actor"]))
        try:
            return keqing_core.build_keqingrl_action_features_typed(
                tracker.hand_counts34,
                tracker.visible_counts34,
                [int(spec.action_type) for spec in specs],
                [-1 if spec.tile is None else int(spec.tile) for spec in specs],
                [int(spec.flags) for spec in specs],
                self._require_room().remaining_wall(),
            )
        except RuntimeError as exc:
            if keqing_core.is_missing_rust_capability_error(exc):
                raise RuntimeError("KeqingRL runtime requires Rust typed action feature generation") from exc
            raise

    def _action_features(self, snapshot: dict[str, object], spec: ActionSpec) -> list[float]:
        return self._action_features_batch(snapshot, (spec,))[0]

    def _action_feature_payload(self, spec: ActionSpec, *, actor: int) -> dict[str, object]:
        return {
            "action_type": int(spec.action_type),
            "actor": actor,
            "tile": spec.tile,
            "consumed": list(spec.consumed),
            "from_who": spec.from_who,
            "flags": int(spec.flags),
        }

    def _snapshot_with_rl_fields(self, actor: int) -> dict[str, object]:
        room = self._require_room()
        snapshot = keqing_core.replay_state_snapshot(room.events, actor)
        snapshot["actor"] = actor
        snapshot["tsumo_pai"] = room.state.last_tsumo_raw[actor] or room.state.last_tsumo[actor]
        self._inject_rust_shanten_waits(snapshot, actor)
        snapshot["_hora_is_haitei"] = room.remaining_wall() == 0
        snapshot["_hora_is_houtei"] = room.remaining_wall() == 0
        snapshot["_hora_is_rinshan"] = room.state.players[actor].rinshan_tsumo
        snapshot["_hora_is_chankan"] = bool(
            room.state.last_kakan and room.state.last_kakan.get("actor") != actor
        )
        return snapshot

    def _inject_rust_shanten_waits(self, snapshot: dict[str, object], actor: int) -> None:
        tracker = SnapshotFeatureTracker.from_state(snapshot, actor=actor)
        summary = keqing_core.summarize_3n1(tracker.hand_counts34, tracker.visible_counts34)
        snapshot["shanten"] = int(summary[0])
        snapshot["waits_count"] = int(summary[1])
        snapshot["waits_tiles"] = [bool(value) for value in summary[2]]

    def _next_seed(self) -> int | None:
        if self._base_seed is None:
            return None
        seed = self._base_seed + self._seed_cursor
        self._seed_cursor += 1
        return seed

    def _mark_done(self) -> None:
        room = self._require_room()
        if not room.events or room.events[-1].get("type") != "end_game":
            room.events.append({"type": "end_game"})
            self._sync_rust_runtime_state(room, actor_hint=0)
        self._done = True
        self._turn = None

    def _require_room(self) -> BattleRoom:
        if self.room is None:
            raise RuntimeError("env.reset(...) must be called before using the environment")
        return self.room

    def _require_turn(self, actor: int) -> _TurnContext:
        if self._done:
            raise RuntimeError("the episode is finished")
        if self._turn is None:
            raise RuntimeError("environment is not positioned at a controllable decision")
        if self._turn.actor != actor:
            raise ValueError(f"expected actor {self._turn.actor}, got {actor}")
        return self._turn


def _default_battle_config() -> BattleConfig:
    return BattleConfig(
        player_count=4,
        players=[
            {"id": seat, "name": f"RL{seat}", "type": "bot"}
            for seat in range(4)
        ],
    )


def _mahjong_spec_from_payload(item: dict[str, object]) -> MahjongActionSpec:
    return MahjongActionSpec(
        type=str(item["type"]),
        actor=item.get("actor"),
        pai=item.get("pai"),
        consumed=tuple(item.get("consumed") or ()),
        target=item.get("target"),
        tsumogiri=item.get("tsumogiri"),
    )


__all__ = ["AutopilotTraceEvent", "DiscardOnlyMahjongEnv", "EnvState", "StepResult"]
