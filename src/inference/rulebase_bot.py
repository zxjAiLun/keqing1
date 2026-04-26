from __future__ import annotations

from typing import Optional

import keqing_core

from inference import (
    DefaultDecisionContextBuilder,
    DefaultRuntimeReviewExporter,
    same_action,
)
from inference.contracts import DecisionResult, ModelAuxOutputs, ScoredCandidate
from inference.runtime_bot import enumerate_legal_actions, inject_shanten_waits
from mahjong_env.state import GameState


class RulebaseBot:
    """Rust rulebase bot wrapper.

    决策逻辑完全下沉到 `keqing_core`，Python 这里只负责上下文构建与 review 日志。
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

        chosen = keqing_core.choose_rulebase_action(
            ctx.runtime_snap,
            actor,
            legal_actions,
        )
        if chosen is None:
            chosen = legal_actions[0]

        decision = DecisionResult(
            chosen=chosen,
            candidates=self._score_candidates(legal_actions, chosen),
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

    @staticmethod
    def _score_candidates(
        legal_actions: list[dict],
        chosen: dict,
    ) -> list[ScoredCandidate]:
        scores: list[ScoredCandidate] = []
        for action in legal_actions:
            if same_action(action, chosen):
                score = 1000.0
            elif action.get("type") == "none":
                score = -10.0
            else:
                score = 0.0
            scores.append(ScoredCandidate(action=action, logit=score, final_score=score))
        return scores
