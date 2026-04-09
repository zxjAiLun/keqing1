from __future__ import annotations

from collections import Counter
from typing import Protocol

import numpy as np

from inference.contracts import (
    DecisionContext,
    DecisionResult,
    ModelAuxOutputs,
    ScoredCandidate,
)
from inference.keqing_adapter import KeqingModelAdapter
from keqingv1.action_space import NONE_IDX, action_to_idx, build_legal_mask
from mahjong_env.legal_actions import _reach_discard_candidates
from mahjong_env.tiles import normalize_tile


def _aux_bonus(
    aux_outputs: ModelAuxOutputs,
    score_delta_lambda: float,
    win_prob_lambda: float,
    dealin_prob_lambda: float,
) -> float:
    return (
        score_delta_lambda * aux_outputs.score_delta
        + win_prob_lambda * aux_outputs.win_prob
        - dealin_prob_lambda * aux_outputs.dealin_prob
    )


def _legal_score(
    policy_logits: np.ndarray,
    action: dict,
    value: float = 0.0,
    style_lambda: float = 0.0,
    aux_bonus: float = 0.0,
) -> float:
    idx = action_to_idx(action)
    score = float(policy_logits[NONE_IDX if idx == NONE_IDX else idx])
    if action.get("type") != "none":
        score += style_lambda * value
        score += aux_bonus
    return score


def _candidate_final_score(
    policy_logits: np.ndarray,
    action: dict,
    beam_value_scores: dict[int, float],
    value: float,
    style_lambda: float,
    aux_bonus: float,
) -> float:
    idx = action_to_idx(action)
    if idx in beam_value_scores:
        return float(beam_value_scores[idx])
    return float(
        _legal_score(
            policy_logits,
            action,
            value=value,
            style_lambda=style_lambda,
            aux_bonus=aux_bonus,
        )
    )


def _find_best_legal(
    policy_logits: np.ndarray,
    legal_actions: list,
    value: float = 0.0,
    style_lambda: float = 0.0,
    aux_bonus: float = 0.0,
) -> dict:
    best_score = -1e18
    best_action = legal_actions[0]
    for a in legal_actions:
        score = _legal_score(
            policy_logits,
            a,
            value=value,
            style_lambda=style_lambda,
            aux_bonus=aux_bonus,
        )
        if score > best_score:
            best_score = score
            best_action = a
    return best_action


def _eval_snapshot_outputs(
    adapter: KeqingModelAdapter,
    snap: dict,
    actor: int,
) -> tuple[float, ModelAuxOutputs]:
    result = adapter.forward(snap, actor)
    return result.value, result.aux


def _simulate_discard_snapshot(
    snap: dict,
    actor: int,
    pai: str,
) -> dict:
    hand = list(snap.get("hand", []))
    removed = False
    new_hand = []
    norm_pai = normalize_tile(pai)
    for t in hand:
        if not removed and normalize_tile(t) == norm_pai:
            removed = True
        else:
            new_hand.append(t)

    fake_snap = dict(snap)
    fake_snap["hand"] = new_hand if removed else hand
    discards = [list(d) for d in snap.get("discards", [[], [], [], []])]
    discards[actor] = discards[actor] + [pai]
    fake_snap["discards"] = discards
    fake_snap["tsumo_pai"] = None
    return fake_snap


def _dahai_beam_search(
    adapter: KeqingModelAdapter,
    snap: dict,
    actor: int,
    policy_logits: np.ndarray,
    legal_dahai: list,
    beam_k: int,
    beam_lambda: float,
    score_delta_lambda: float,
    win_prob_lambda: float,
    dealin_prob_lambda: float,
) -> tuple[dict, dict[int, float]]:
    sorted_dahai = sorted(
        legal_dahai,
        key=lambda a: policy_logits[action_to_idx(a)],
        reverse=True,
    )[:beam_k]

    best_score = -1e18
    best_action = sorted_dahai[0]
    value_scores: dict[int, float] = {}

    for a in sorted_dahai:
        pai = a.get("pai", "")
        fake_snap = _simulate_discard_snapshot(snap, actor, pai)

        value, aux = _eval_snapshot_outputs(adapter, fake_snap, actor)
        score = (
            float(policy_logits[action_to_idx(a)])
            + beam_lambda * value
            + _aux_bonus(aux, score_delta_lambda, win_prob_lambda, dealin_prob_lambda)
        )
        value_scores[action_to_idx(a)] = score
        if score > best_score:
            best_score = score
            best_action = a

    return best_action, value_scores


def _meld_value_eval(
    adapter: KeqingModelAdapter,
    snap: dict,
    actor: int,
    policy_logits: np.ndarray,
    meld_actions: list,
    none_actions: list,
    beam_lambda: float,
    score_delta_lambda: float,
    win_prob_lambda: float,
    dealin_prob_lambda: float,
) -> tuple[dict, dict[int, float]]:
    best_score = -1e18
    best_action = none_actions[0] if none_actions else meld_actions[0]
    value_scores: dict[int, float] = {}

    if none_actions:
        v_none, aux_none = _eval_snapshot_outputs(adapter, snap, actor)
        none_bonus = _aux_bonus(aux_none, score_delta_lambda, win_prob_lambda, dealin_prob_lambda)
        for a in none_actions:
            s = float(policy_logits[action_to_idx(a)]) + beam_lambda * v_none
            if a.get("type") != "none":
                s += none_bonus
            value_scores[action_to_idx(a)] = s
            if s > best_score:
                best_score = s
                best_action = a

    for a in meld_actions:
        meld_type = a.get("type", "")
        consumed = a.get("consumed", [])
        pai = a.get("pai", "")

        fake_snap = dict(snap)
        new_hand = list(snap.get("hand", []))
        for c in consumed:
            norm_c = normalize_tile(c)
            for i, t in enumerate(new_hand):
                if normalize_tile(t) == norm_c:
                    new_hand.pop(i)
                    break
        fake_snap["hand"] = new_hand

        melds = [list(m) for m in snap.get("melds", [[], [], [], []])]
        melds[actor] = melds[actor] + [
            {
                "type": meld_type,
                "pai": normalize_tile(pai),
                "consumed": [normalize_tile(c) for c in consumed],
            }
        ]
        fake_snap["melds"] = melds

        value, aux = _eval_snapshot_outputs(adapter, fake_snap, actor)
        score = (
            float(policy_logits[action_to_idx(a)])
            + beam_lambda * value
            + _aux_bonus(aux, score_delta_lambda, win_prob_lambda, dealin_prob_lambda)
        )
        value_scores[action_to_idx(a)] = score
        if score > best_score:
            best_score = score
            best_action = a

    return best_action, value_scores


def _reach_value_eval(
    adapter: KeqingModelAdapter,
    snap: dict,
    actor: int,
    policy_logits: np.ndarray,
    reach_action: dict,
    other_actions: list,
    beam_lambda: float,
    score_delta_lambda: float,
    win_prob_lambda: float,
    dealin_prob_lambda: float,
) -> tuple[dict, dict[int, float], dict[int, dict]]:
    reach_idx = action_to_idx(reach_action)
    hand = Counter(snap.get("hand", []))
    last_tsumo_all = list(snap.get("last_tsumo", [None, None, None, None]))
    last_tsumo_raw_all = list(snap.get("last_tsumo_raw", [None, None, None, None]))
    last_tsumo = last_tsumo_all[actor] if actor < len(last_tsumo_all) else None
    last_tsumo_raw = last_tsumo_raw_all[actor] if actor < len(last_tsumo_raw_all) else None
    reach_discards = _reach_discard_candidates(hand, last_tsumo, last_tsumo_raw)

    best_decl_action: dict | None = None
    best_reach_score = -1e18
    for pai_out, tsumogiri in reach_discards:
        decl_action = {
            "type": "dahai",
            "actor": actor,
            "pai": pai_out,
            "tsumogiri": tsumogiri,
        }
        fake_snap = _simulate_discard_snapshot(snap, actor, pai_out)
        reached = list(snap.get("reached", [False, False, False, False]))
        if actor < len(reached):
            reached[actor] = True
        fake_snap["reached"] = reached
        pending_reach = list(snap.get("pending_reach", [False, False, False, False]))
        if actor < len(pending_reach):
            pending_reach[actor] = False
        fake_snap["pending_reach"] = pending_reach

        value, aux = _eval_snapshot_outputs(adapter, fake_snap, actor)
        decl_idx = action_to_idx(decl_action)
        score = (
            float(policy_logits[reach_idx])
            + float(policy_logits[decl_idx])
            + beam_lambda * value
            + _aux_bonus(aux, score_delta_lambda, win_prob_lambda, dealin_prob_lambda)
        )
        if score > best_reach_score:
            best_reach_score = score
            best_decl_action = decl_action

    if best_decl_action is None:
        fake_snap = dict(snap)
        reached = list(snap.get("reached", [False, False, False, False]))
        if actor < len(reached):
            reached[actor] = True
        fake_snap["reached"] = reached
        reach_value, reach_aux = _eval_snapshot_outputs(adapter, fake_snap, actor)
        best_reach_score = (
            float(policy_logits[reach_idx])
            + beam_lambda * reach_value
            + _aux_bonus(reach_aux, score_delta_lambda, win_prob_lambda, dealin_prob_lambda)
        )

    value_scores: dict[int, float] = {reach_idx: best_reach_score}
    reach_meta: dict[int, dict] = {}
    if best_decl_action is not None:
        reach_meta[reach_idx] = {"reach_discard": best_decl_action}

    best_score = best_reach_score
    best_action = reach_action
    for a in other_actions:
        s = float(policy_logits[action_to_idx(a)])
        value_scores[action_to_idx(a)] = s
        if s > best_score:
            best_score = s
            best_action = a

    return best_action, value_scores, reach_meta


class ActionScorer(Protocol):
    def score(self, ctx: DecisionContext) -> DecisionResult:
        ...


class DefaultActionScorer:
    def __init__(
        self,
        *,
        adapter: KeqingModelAdapter,
        beam_k: int,
        beam_lambda: float,
        style_lambda: float,
        score_delta_lambda: float,
        win_prob_lambda: float,
        dealin_prob_lambda: float,
    ):
        self.adapter = adapter
        self.beam_k = beam_k
        self.beam_lambda = beam_lambda
        self.style_lambda = style_lambda
        self.score_delta_lambda = score_delta_lambda
        self.win_prob_lambda = win_prob_lambda
        self.dealin_prob_lambda = dealin_prob_lambda

    def score(self, ctx: DecisionContext) -> DecisionResult:
        model_snap = dict(ctx.model_snap)
        model_snap["legal_actions"] = ctx.legal_actions
        model_result = self.adapter.forward(model_snap, ctx.actor)
        logits_np = np.asarray(model_result.policy_logits, dtype=np.float32)
        value_scalar = float(model_result.value)
        aux_bonus = _aux_bonus(
            model_result.aux,
            self.score_delta_lambda,
            self.win_prob_lambda,
            self.dealin_prob_lambda,
        )

        legal_dicts = ctx.legal_actions
        mask = np.array(build_legal_mask(legal_dicts), dtype=np.float32)
        logits_np = np.where(mask > 0, logits_np, -1e9)

        legal_dahai = [a for a in legal_dicts if a.get("type") == "dahai"]
        legal_meld = [
            a
            for a in legal_dicts
            if a.get("type") in ("chi", "pon", "daiminkan", "ankan", "kakan")
        ]
        legal_reach = [a for a in legal_dicts if a.get("type") == "reach"]
        legal_none = [a for a in legal_dicts if a.get("type") == "none"]

        beam_value_scores: dict[int, float] = {}
        beam_meta: dict[int, dict] = {}
        if self.beam_k > 0 and legal_meld:
            non_meld = [
                a
                for a in legal_dicts
                if a.get("type") not in ("chi", "pon", "daiminkan", "ankan", "kakan")
            ]
            chosen, beam_value_scores = _meld_value_eval(
                self.adapter,
                ctx.runtime_snap,
                ctx.actor,
                logits_np,
                legal_meld,
                legal_none,
                beam_lambda=self.beam_lambda,
                score_delta_lambda=self.score_delta_lambda,
                win_prob_lambda=self.win_prob_lambda,
                dealin_prob_lambda=self.dealin_prob_lambda,
            )
            if non_meld:
                fallback = _find_best_legal(
                    logits_np,
                    non_meld,
                    value=value_scalar,
                    style_lambda=self.style_lambda,
                    aux_bonus=aux_bonus,
                )
                if _legal_score(logits_np, fallback, value_scalar, self.style_lambda, aux_bonus) > _legal_score(
                    logits_np, chosen, value_scalar, self.style_lambda, aux_bonus
                ):
                    chosen = fallback
        elif self.beam_k > 0 and legal_reach:
            non_reach = [a for a in legal_dicts if a.get("type") != "reach"]
            chosen, beam_value_scores, beam_meta = _reach_value_eval(
                self.adapter,
                ctx.runtime_snap,
                ctx.actor,
                logits_np,
                legal_reach[0],
                non_reach,
                beam_lambda=self.beam_lambda,
                score_delta_lambda=self.score_delta_lambda,
                win_prob_lambda=self.win_prob_lambda,
                dealin_prob_lambda=self.dealin_prob_lambda,
            )
        elif self.beam_k > 0 and len(legal_dahai) > 1:
            chosen, beam_value_scores = _dahai_beam_search(
                self.adapter,
                ctx.runtime_snap,
                ctx.actor,
                logits_np,
                legal_dahai,
                beam_k=self.beam_k,
                beam_lambda=self.beam_lambda,
                score_delta_lambda=self.score_delta_lambda,
                win_prob_lambda=self.win_prob_lambda,
                dealin_prob_lambda=self.dealin_prob_lambda,
            )
        else:
            chosen = _find_best_legal(
                logits_np,
                legal_dicts,
                value=value_scalar,
                style_lambda=self.style_lambda,
                aux_bonus=aux_bonus,
            )

        candidates = sorted(
            [
                ScoredCandidate(
                    action=a,
                    logit=float(logits_np[action_to_idx(a)]),
                    final_score=_candidate_final_score(
                        logits_np,
                        a,
                        beam_value_scores,
                        value_scalar,
                        self.style_lambda,
                        aux_bonus,
                    ),
                    beam_score=(
                        float(beam_value_scores[action_to_idx(a)])
                        if action_to_idx(a) in beam_value_scores
                        else None
                    ),
                    meta=beam_meta.get(action_to_idx(a), {}),
                )
                for a in legal_dicts
            ],
            key=lambda x: x.logit,
            reverse=True,
        )

        return DecisionResult(
            chosen=chosen,
            candidates=candidates,
            model_value=value_scalar,
            model_aux=model_result.aux,
        )
