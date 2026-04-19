from __future__ import annotations

import copy
from collections import Counter
from typing import Any, Optional

from inference import DefaultDecisionContextBuilder, DefaultRuntimeReviewExporter
from inference.contracts import DecisionResult, ModelAuxOutputs, ScoredCandidate
from mahjong_env.replay import _calc_shanten_waits
from mahjong_env.state import GameState, apply_event
from mahjong_env.tiles import normalize_tile, tile_to_34

from inference.runtime_bot import enumerate_legal_actions, inject_shanten_waits

_WIND_ORDER = ["E", "S", "W", "N"]
_DRAGON_TILES = {"P", "F", "C"}


def _remove_one_tile(hand: list[str], target: str) -> list[str]:
    norm_target = normalize_tile(target)
    removed = False
    result: list[str] = []
    for tile in hand:
        if not removed and normalize_tile(tile) == norm_target:
            removed = True
            continue
        result.append(tile)
    return result


class RulebaseBot:
    """项目内生的规则型 bot。

    不引入第二套 rules truth，只基于当前 GameState / legal_actions / shanten helpers
    做最小可用 heuristics。
    """

    def __init__(self, player_id: int, verbose: bool = False):
        self.player_id = player_id
        self.verbose = verbose
        self.decision_log: list[dict] = []
        self.game_state = GameState()
        self._context_builder = DefaultDecisionContextBuilder(
            model_version="rulebase",
            riichi_state=None,
            inject_shanten_waits=inject_shanten_waits,
            enumerate_legal_actions_fn=lambda snap, seat: enumerate_legal_actions(
                snap, seat
            ),
        )
        self._review_exporter = DefaultRuntimeReviewExporter()
        self.model = None

    def reset(self) -> None:
        self.decision_log.clear()
        self.game_state = GameState()

    def react(self, event: dict, gt_action: Optional[dict] = None) -> Optional[dict]:
        actor = self.player_id
        ctx = self._context_builder.build(self.game_state, actor, event)
        if ctx is None:
            return None

        legal_actions = ctx.legal_actions
        if not legal_actions:
            return None

        chosen, candidates = self._choose_action(ctx)
        decision = DecisionResult(
            chosen=chosen,
            candidates=candidates,
            model_value=0.0,
            model_aux=ModelAuxOutputs(),
        )
        entry = self._review_exporter.build_decision_entry(
            step=len(self.decision_log),
            ctx=ctx,
            decision=decision,
            gt_action=gt_action,
            actor=actor,
        )
        self.decision_log.append(entry)
        return chosen

    def _choose_action(
        self,
        ctx,
    ) -> tuple[dict[str, Any], list[ScoredCandidate]]:
        actor = ctx.actor
        runtime_snap = ctx.runtime_snap
        legal_actions = ctx.legal_actions

        chosen = legal_actions[0]
        scored_candidates: list[ScoredCandidate] = []

        hora_actions = [a for a in legal_actions if a.get("type") == "hora"]
        if hora_actions:
            chosen = hora_actions[0]
            scored_candidates = self._score_all_candidates(
                legal_actions, chosen=chosen, default=-100.0, chosen_score=1000.0
            )
            return chosen, scored_candidates

        reach_actions = [a for a in legal_actions if a.get("type") == "reach"]
        if reach_actions:
            chosen = reach_actions[0]
            scored_candidates = self._score_all_candidates(
                legal_actions, chosen=chosen, default=-50.0, chosen_score=900.0
            )
            return chosen, scored_candidates

        if runtime_snap.get("reached", [False, False, False, False])[actor]:
            last_tsumo = (runtime_snap.get("last_tsumo") or [None, None, None, None])[
                actor
            ]
            if last_tsumo:
                tsumogiri = next(
                    (
                        action
                        for action in legal_actions
                        if action.get("type") == "dahai"
                        and normalize_tile(action.get("pai", ""))
                        == normalize_tile(last_tsumo)
                        and action.get("tsumogiri")
                    ),
                    None,
                )
                if tsumogiri is not None:
                    chosen = tsumogiri
                    scored_candidates = self._score_all_candidates(
                        legal_actions,
                        chosen=chosen,
                        default=-20.0,
                        chosen_score=800.0,
                    )
                    return chosen, scored_candidates

        meld_actions = [
            a
            for a in legal_actions
            if a.get("type") in {"pon", "chi", "daiminkan"}
        ]
        if meld_actions:
            chosen_meld = self._choose_meld(runtime_snap, meld_actions)
            if chosen_meld is not None:
                chosen = chosen_meld
                scored_candidates = self._score_meld_candidates(
                    runtime_snap, legal_actions, chosen
                )
                return chosen, scored_candidates

            none_action = next(
                (action for action in legal_actions if action.get("type") == "none"),
                None,
            )
            if none_action is not None:
                chosen = none_action
                scored_candidates = self._score_meld_candidates(
                    runtime_snap, legal_actions, chosen
                )
                return chosen, scored_candidates

        dahai_actions = [a for a in legal_actions if a.get("type") == "dahai"]
        if dahai_actions:
            chosen = self._choose_discard(runtime_snap, dahai_actions)
            scored_candidates = self._score_discard_candidates(
                runtime_snap, legal_actions, chosen
            )
            return chosen, scored_candidates

        none_action = next(
            (action for action in legal_actions if action.get("type") == "none"),
            {"type": "none", "actor": actor},
        )
        chosen = none_action
        scored_candidates = self._score_all_candidates(
            legal_actions, chosen=chosen, default=-1.0, chosen_score=10.0
        )
        return chosen, scored_candidates

    def _choose_discard(self, runtime_snap: dict, actions: list[dict]) -> dict:
        best_action = actions[0]
        best_key: tuple[float, float, float] = (-999.0, -999.0, -999.0)
        current_hand = list(runtime_snap.get("hand", []))
        current_melds = (runtime_snap.get("melds") or [[], [], [], []])[self.player_id]

        for action in actions:
            pai = action.get("pai", "")
            next_hand = _remove_one_tile(current_hand, pai)
            next_shanten, next_waits, _waits_tiles, _ = _calc_shanten_waits(
                next_hand, current_melds
            )
            # 优先更低向听；其次更高进张；再偏好非摸切，避免无意义摸切。
            key = (-float(next_shanten), float(next_waits), 0.0 if action.get("tsumogiri") else 1.0)
            if key > best_key:
                best_key = key
                best_action = action
        return best_action

    def _choose_meld(self, runtime_snap: dict, actions: list[dict]) -> Optional[dict]:
        actor = self.player_id
        current_shanten = int(runtime_snap.get("shanten", 8))
        current_melds = (runtime_snap.get("melds") or [[], [], [], []])[actor]
        opened = bool(current_melds)
        best_action: Optional[dict] = None
        best_key: tuple[int, int] | None = None

        for action in actions:
            action_type = action.get("type", "")
            if action_type == "pon" and not (
                self._is_yakuhai(runtime_snap, action.get("pai", "")) or opened
            ):
                continue
            if action_type == "chi" and not opened:
                continue

            next_shanten, next_waits = self._simulate_action_shanten(runtime_snap, action)
            if next_shanten > current_shanten:
                continue
            if next_shanten == current_shanten and next_waits <= 0:
                continue

            key = (-next_shanten, next_waits)
            if best_key is None or key > best_key:
                best_key = key
                best_action = action

        return best_action

    def _simulate_action_shanten(self, runtime_snap: dict, action: dict) -> tuple[int, int]:
        sim_state = copy.deepcopy(self.game_state)
        apply_event(sim_state, action)
        sim_snap = sim_state.snapshot(self.player_id)
        hand = sim_snap.get("hand", [])
        melds = (sim_snap.get("melds") or [[], [], [], []])[self.player_id]
        shanten, waits_count, _waits_tiles, _ = _calc_shanten_waits(hand, melds)
        return int(shanten), int(waits_count)

    def _is_yakuhai(self, runtime_snap: dict, tile: str) -> bool:
        if tile in _DRAGON_TILES:
            return True
        if tile not in _WIND_ORDER:
            return False
        bakaze = runtime_snap.get("bakaze", "E")
        oya = int(runtime_snap.get("oya", 0))
        jikaze = _WIND_ORDER[(self.player_id - oya) % 4]
        return tile == bakaze or tile == jikaze

    def _score_discard_candidates(
        self,
        runtime_snap: dict,
        legal_actions: list[dict],
        chosen: dict,
    ) -> list[ScoredCandidate]:
        current_hand = list(runtime_snap.get("hand", []))
        current_melds = (runtime_snap.get("melds") or [[], [], [], []])[self.player_id]
        scores: list[ScoredCandidate] = []
        for action in legal_actions:
            action_type = action.get("type")
            if action_type != "dahai":
                penalty = -200.0 if action_type == "none" else -500.0
                scores.append(
                    ScoredCandidate(
                        action=action,
                        logit=penalty,
                        final_score=penalty,
                    )
                )
                continue
            next_hand = _remove_one_tile(current_hand, action.get("pai", ""))
            next_shanten, next_waits, _waits_tiles, _ = _calc_shanten_waits(
                next_hand, current_melds
            )
            score = -100.0 * float(next_shanten) + float(next_waits)
            if action.get("tsumogiri"):
                score -= 0.1
            if self._same_action(action, chosen):
                score += 1000.0
            scores.append(ScoredCandidate(action=action, logit=score, final_score=score))
        return sorted(scores, key=lambda c: c.final_score, reverse=True)

    def _score_meld_candidates(
        self,
        runtime_snap: dict,
        legal_actions: list[dict],
        chosen: dict,
    ) -> list[ScoredCandidate]:
        current_shanten = int(runtime_snap.get("shanten", 8))
        scores: list[ScoredCandidate] = []
        for action in legal_actions:
            if action.get("type") in {"pon", "chi", "daiminkan"}:
                next_shanten, next_waits = self._simulate_action_shanten(
                    runtime_snap, action
                )
                score = (
                    20.0 * float(current_shanten - next_shanten)
                    + float(next_waits)
                )
            elif action.get("type") == "none":
                score = 0.0
            else:
                score = -200.0
            if self._same_action(action, chosen):
                score += 1000.0
            scores.append(ScoredCandidate(action=action, logit=score, final_score=score))
        return sorted(scores, key=lambda c: c.final_score, reverse=True)

    @staticmethod
    def _score_all_candidates(
        legal_actions: list[dict],
        *,
        chosen: dict,
        default: float,
        chosen_score: float,
    ) -> list[ScoredCandidate]:
        return [
            ScoredCandidate(
                action=action,
                logit=chosen_score if RulebaseBot._same_action(action, chosen) else default,
                final_score=chosen_score if RulebaseBot._same_action(action, chosen) else default,
            )
            for action in legal_actions
        ]

    @staticmethod
    def _same_action(a: dict, b: dict) -> bool:
        if a.get("type") != b.get("type"):
            return False
        if a.get("type") == "dahai":
            return normalize_tile(a.get("pai", "")) == normalize_tile(
                b.get("pai", "")
            ) and bool(a.get("tsumogiri")) == bool(b.get("tsumogiri"))
        if a.get("type") in {"pon", "chi", "daiminkan", "ankan", "kakan"}:
            return (
                normalize_tile(a.get("pai", "")) == normalize_tile(b.get("pai", ""))
                and Counter(normalize_tile(x) for x in a.get("consumed", []))
                == Counter(normalize_tile(x) for x in b.get("consumed", []))
            )
        return True
