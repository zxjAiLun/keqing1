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
from keqing_core import (
    aggregate_keqingv4_continuation_scores as _rust_aggregate_keqingv4_continuation_scores,
    enumerate_keqingv4_live_draw_weights as _rust_enumerate_keqingv4_live_draw_weights,
    enumerate_keqingv4_post_meld_discards as _rust_enumerate_keqingv4_post_meld_discards,
    enumerate_keqingv4_reach_discards as _rust_enumerate_keqingv4_reach_discards,
    enumerate_legal_action_specs_structural as _rust_enumerate_legal_action_specs_structural,
    resolve_keqingv4_continuation_scenarios as _rust_resolve_keqingv4_continuation_scenarios,
    is_missing_rust_capability_error as _is_missing_rust_capability_error,
    project_keqingv4_call_snapshot as _rust_project_keqingv4_call_snapshot,
    project_keqingv4_discard_snapshot as _rust_project_keqingv4_discard_snapshot,
    resolve_keqingv4_post_meld_followup as _rust_resolve_keqingv4_post_meld_followup,
    resolve_keqingv4_reach_followup as _rust_resolve_keqingv4_reach_followup,
    resolve_keqingv4_rinshan_followup as _rust_resolve_keqingv4_rinshan_followup,
    score_keqingv4_continuation_scenario as _rust_score_keqingv4_continuation_scenario,
    project_keqingv4_reach_snapshot as _rust_project_keqingv4_reach_snapshot,
    project_keqingv4_rinshan_draw_snapshot as _rust_project_keqingv4_rinshan_draw_snapshot,
)
from mahjong_env.action_space import NONE_IDX, action_to_idx, build_legal_mask
from mahjong_env.legal_actions import _reach_discard_candidates, enumerate_legal_actions
from mahjong_env.tiles import normalize_tile, tile_to_34

_TILE34_STR = (
    "1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m",
    "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p", "9p",
    "1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s",
    "E", "S", "W", "N", "P", "F", "C",
)
_SPECIAL_ACTION_SLOT = {"reach": 0, "hora": 1, "ryukyoku": 2}


def _validate_continuation_score_payload(
    payload: object,
    *,
    continuation_kind: str,
) -> tuple[dict | None, float]:
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"keqingv4 continuation scoring contract drift for {continuation_kind}: payload must be a dict"
        )
    if "score" not in payload:
        raise RuntimeError(
            f"keqingv4 continuation scoring contract drift for {continuation_kind}: missing score"
        )
    best_action = payload.get("best_action")
    if best_action is not None and not isinstance(best_action, dict):
        raise RuntimeError(
            f"keqingv4 continuation scoring contract drift for {continuation_kind}: best_action must be a dict or null"
        )
    return best_action, float(payload["score"])


def _validate_continuation_aggregation_payload(
    payload: object,
    *,
    action: dict,
) -> tuple[float, dict]:
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"keqingv4 continuation aggregation contract drift for action={action.get('type')}: payload must be a dict"
        )
    if "final_score" not in payload:
        raise RuntimeError(
            f"keqingv4 continuation aggregation contract drift for action={action.get('type')}: missing final_score"
        )
    meta = payload.get("meta", {})
    if not isinstance(meta, dict):
        raise RuntimeError(
            f"keqingv4 continuation aggregation contract drift for action={action.get('type')}: meta must be a dict"
        )
    return float(payload["final_score"]), meta


def _validate_continuation_scenario_payload(scenarios: object) -> list[dict]:
    if not isinstance(scenarios, list):
        raise RuntimeError("keqingv4 continuation scenario contract drift: payload must be a list")
    validated: list[dict] = []
    for idx, scenario in enumerate(scenarios):
        if not isinstance(scenario, dict):
            raise RuntimeError(
                f"keqingv4 continuation scenario contract drift at index {idx}: scenario must be a dict"
            )
        projected_snapshot = scenario.get("projected_snapshot", {})
        if not isinstance(projected_snapshot, dict):
            raise RuntimeError(
                f"keqingv4 continuation scenario contract drift at index {idx}: projected_snapshot must be a dict"
            )
        legal_actions = scenario.get("legal_actions", [])
        if not isinstance(legal_actions, list):
            raise RuntimeError(
                f"keqingv4 continuation scenario contract drift at index {idx}: legal_actions must be a list"
            )
        continuation_kind = scenario.get("continuation_kind", "")
        if not isinstance(continuation_kind, str):
            raise RuntimeError(
                f"keqingv4 continuation scenario contract drift at index {idx}: continuation_kind must be a string"
            )
        declaration_action = scenario.get("declaration_action")
        if declaration_action is not None and not isinstance(declaration_action, dict):
            raise RuntimeError(
                f"keqingv4 continuation scenario contract drift at index {idx}: declaration_action must be a dict or null"
            )
        validated.append(scenario)
    return validated


def _xmodel1_special_type_for_action(action: dict) -> int | None:
    from xmodel1.schema import (
        XMODEL1_SPECIAL_TYPE_ANKAN,
        XMODEL1_SPECIAL_TYPE_CHI_HIGH,
        XMODEL1_SPECIAL_TYPE_CHI_LOW,
        XMODEL1_SPECIAL_TYPE_CHI_MID,
        XMODEL1_SPECIAL_TYPE_DAIMINKAN,
        XMODEL1_SPECIAL_TYPE_HORA,
        XMODEL1_SPECIAL_TYPE_KAKAN,
        XMODEL1_SPECIAL_TYPE_NONE,
        XMODEL1_SPECIAL_TYPE_PON,
        XMODEL1_SPECIAL_TYPE_REACH,
        XMODEL1_SPECIAL_TYPE_RYUKYOKU,
    )

    action_type = action.get("type")
    if action_type == "reach":
        return XMODEL1_SPECIAL_TYPE_REACH
    if action_type == "hora":
        return XMODEL1_SPECIAL_TYPE_HORA
    if action_type == "pon":
        return XMODEL1_SPECIAL_TYPE_PON
    if action_type == "daiminkan":
        return XMODEL1_SPECIAL_TYPE_DAIMINKAN
    if action_type == "ankan":
        return XMODEL1_SPECIAL_TYPE_ANKAN
    if action_type == "kakan":
        return XMODEL1_SPECIAL_TYPE_KAKAN
    if action_type == "ryukyoku":
        return XMODEL1_SPECIAL_TYPE_RYUKYOKU
    if action_type == "none":
        return XMODEL1_SPECIAL_TYPE_NONE
    if action_type != "chi":
        return None
    pai = normalize_tile(action.get("pai", ""))
    pai_rank = int(pai[0]) if len(pai) == 2 and pai[0].isdigit() else 0
    consumed = sorted(
        int(tile[0]) for tile in [normalize_tile(value) for value in action.get("consumed", [])] if len(tile) == 2 and tile[0].isdigit()
    )
    if len(consumed) < 2 or pai_rank < consumed[0]:
        return XMODEL1_SPECIAL_TYPE_CHI_LOW
    if pai_rank < consumed[1]:
        return XMODEL1_SPECIAL_TYPE_CHI_MID
    return XMODEL1_SPECIAL_TYPE_CHI_HIGH


def _xmodel1_candidate_components(
    adapter: KeqingModelAdapter,
    snap: dict,
    actor: int,
    legal_actions: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    candidate_feat, candidate_tile_id, candidate_mask, _candidate_flags = adapter._runtime_candidate_builder(
        snap,
        actor,
        legal_actions,
    )
    if adapter._runtime_special_candidate_builder is not None:
        special_feat, special_type_id, special_mask = adapter._runtime_special_candidate_builder(
            snap,
            actor,
            legal_actions,
        )
    else:
        special_feat = np.zeros((0, 25), dtype=np.float32)
        special_type_id = np.zeros((0,), dtype=np.int16)
        special_mask = np.zeros((0,), dtype=np.uint8)
    return candidate_feat, candidate_tile_id, candidate_mask, special_feat, special_type_id, special_mask


def _score_xmodel1_candidates(
    adapter: KeqingModelAdapter,
    ctx: DecisionContext,
    model_result: ModelForwardResult,
) -> DecisionResult:
    payload = model_result.xmodel1
    assert payload is not None
    legal_actions = ctx.legal_actions
    candidate_feat, candidate_tile_id, candidate_mask, special_feat, special_type_id, special_mask = _xmodel1_candidate_components(
        adapter,
        dict(ctx.model_snap, legal_actions=legal_actions),
        ctx.actor,
        legal_actions,
    )
    composed_ev = payload.win_prob * payload.pts_given_win - payload.dealin_prob * payload.pts_given_dealin
    candidates: list[ScoredCandidate] = []
    chosen_action = legal_actions[0]
    best_score = -1e18
    legal_dahai = [action for action in legal_actions if action.get("type") == "dahai"]
    best_reach_discard = None
    if legal_dahai:
        def _reach_discard_score(action: dict) -> float:
            tile34 = tile_to_34(normalize_tile(action.get("pai", "")))
            slot_mask = (candidate_mask > 0) & (candidate_tile_id == tile34)
            if not np.any(slot_mask):
                return -1e9
            return float(np.max(payload.discard_logits[slot_mask]))

        best_reach_discard = max(legal_dahai, key=_reach_discard_score)
    for action in legal_actions:
        if action.get("type") == "dahai":
            tile34 = tile_to_34(normalize_tile(action.get("pai", "")))
            slot_mask = (candidate_mask > 0) & (candidate_tile_id == tile34)
            if not np.any(slot_mask):
                logit = -1e9
                final = -1e9
            else:
                slot = int(np.where(slot_mask)[0][np.argmax(payload.discard_logits[slot_mask])])
                risk = float(np.dot(payload.opp_tenpai_probs, candidate_feat[slot, 11:14]))
                logit = float(payload.discard_logits[slot])
                final = logit + 0.3 * composed_ev - 0.3 * risk
            meta = {}
        else:
            special_type = _xmodel1_special_type_for_action(action)
            slot_mask = (special_mask > 0) & (special_type_id == special_type) if special_type is not None else np.zeros_like(special_mask, dtype=bool)
            if not np.any(slot_mask):
                logit = -1e9
                final = -1e9
            else:
                slot = int(np.where(slot_mask)[0][np.argmax(payload.special_logits[slot_mask])])
                risk = float(np.dot(payload.opp_tenpai_probs, special_feat[slot, 14:17]))
                logit = float(payload.special_logits[slot])
                final = logit + 0.3 * composed_ev - 0.3 * risk
            meta = {"xmodel1_special_type": int(special_type) if special_type is not None else None}
            if action.get("type") == "reach" and best_reach_discard is not None:
                meta["reach_discard"] = best_reach_discard
        candidates.append(
            ScoredCandidate(
                action=action,
                logit=logit,
                final_score=final,
                beam_score=final,
                meta=meta,
            )
        )
        if final > best_score:
            best_score = final
            chosen_action = action
    candidates.sort(key=lambda item: item.final_score, reverse=True)
    return DecisionResult(
        chosen=chosen_action,
        candidates=candidates,
        model_value=float(model_result.value),
        model_aux=model_result.aux,
    )


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
    special_bonus: float = 0.0,
) -> float:
    idx = action_to_idx(action)
    score = float(policy_logits[NONE_IDX if idx == NONE_IDX else idx])
    if action.get("type") != "none":
        score += style_lambda * value
        score += aux_bonus
        score += special_bonus
    return score


def _candidate_final_score(
    policy_logits: np.ndarray,
    action: dict,
    beam_value_scores: dict[int, float],
    value: float,
    style_lambda: float,
    aux_bonus: float,
    special_bonus_scores: dict[int, float] | None = None,
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
            special_bonus=(special_bonus_scores or {}).get(idx, 0.0),
        )
    )


def _special_meta_from_summary(action: dict, summary_vec: np.ndarray) -> dict:
    action_type = action.get("type", "")
    meta: dict[str, object] = {
        "special_semantics": action_type,
        "special_summary_bias": float(summary_vec[12]),
        "special_summary_present": float(summary_vec[13]),
    }
    if action_type == "reach":
        meta.update(
            {
                "reach_decl_tenpai": float(summary_vec[1]),
                "reach_decl_waits_ratio": float(summary_vec[2]),
                "reach_decl_ukeire_live": float(summary_vec[4]),
                "reach_decl_improvement_live": float(summary_vec[6]),
            }
        )
    elif action_type == "hora":
        meta.update(
            {
                "hora_tsumo": float(summary_vec[2]),
                "hora_ron": float(summary_vec[3]),
                "hora_rinshan": float(summary_vec[4]),
                "hora_chankan": float(summary_vec[5]),
                "hora_haitei": float(summary_vec[6]),
                "hora_houtei": float(summary_vec[7]),
            }
        )
    elif action_type == "ryukyoku":
        meta.update(
            {
                "ryukyoku_yaochu_ratio": float(summary_vec[2]),
                "ryukyoku_abortive_flag": float(summary_vec[3]),
                "ryukyoku_abortive_pressure": float(summary_vec[4]),
            }
        )
    return meta


def _build_v4_special_meta(
    adapter: KeqingModelAdapter,
    snap: dict,
    actor: int,
    legal_actions: list[dict],
) -> dict[int, dict]:
    if getattr(adapter, "model_version", None) != "keqingv4":
        return {}
    try:
        _discard_summary, _call_summary, special_summary = adapter.resolve_runtime_v4_summaries(
            snap,
            actor,
            legal_actions,
        )
    except RuntimeError as exc:
        if not _is_missing_rust_capability_error(exc):
            raise
        return {}
    out: dict[int, dict] = {}
    for action in legal_actions:
        action_type = action.get("type", "")
        slot = _SPECIAL_ACTION_SLOT.get(action_type)
        if slot is None:
            continue
        out[action_to_idx(action)] = _special_meta_from_summary(action, special_summary[slot])
    return out


def _special_action_calibration_bonus(meta: dict) -> float:
    semantics = meta.get("special_semantics")
    if semantics == "reach":
        return float(
            0.20 * meta.get("special_summary_bias", 0.0)
            + 0.25 * meta.get("reach_decl_tenpai", 0.0)
            + 0.30 * meta.get("reach_decl_waits_ratio", 0.0)
            + 0.20 * meta.get("reach_decl_ukeire_live", 0.0)
            + 0.10 * meta.get("reach_decl_improvement_live", 0.0)
        )
    if semantics == "hora":
        return float(
            0.80 * meta.get("special_summary_bias", 0.0)
            + 0.25 * meta.get("hora_tsumo", 0.0)
            + 0.15 * meta.get("hora_rinshan", 0.0)
            + 0.10 * meta.get("hora_chankan", 0.0)
            + 0.08 * meta.get("hora_haitei", 0.0)
            + 0.08 * meta.get("hora_houtei", 0.0)
        )
    if semantics == "ryukyoku":
        return float(
            0.20 * meta.get("special_summary_bias", 0.0)
            + 0.45 * meta.get("ryukyoku_abortive_flag", 0.0)
            + 0.20 * meta.get("ryukyoku_yaochu_ratio", 0.0)
            + 0.10 * meta.get("ryukyoku_abortive_pressure", 0.0)
        )
    return 0.0


def _find_best_legal(
    policy_logits: np.ndarray,
    legal_actions: list,
    value: float = 0.0,
    style_lambda: float = 0.0,
    aux_bonus: float = 0.0,
    special_bonus_scores: dict[int, float] | None = None,
) -> dict:
    best_score = -1e18
    best_action = legal_actions[0]
    for a in legal_actions:
        idx = action_to_idx(a)
        score = _legal_score(
            policy_logits,
            a,
            value=value,
            style_lambda=style_lambda,
            aux_bonus=aux_bonus,
            special_bonus=(special_bonus_scores or {}).get(idx, 0.0),
        )
        if score > best_score:
            best_score = score
            best_action = a
    return best_action


def _eval_snapshot_outputs(
    adapter: KeqingModelAdapter,
    snap: dict,
    actor: int,
    legal_actions: list[dict] | None = None,
) -> tuple[float, ModelAuxOutputs]:
    resolved_snap = snap if legal_actions is None else dict(snap, legal_actions=legal_actions)
    result = adapter.forward(resolved_snap, actor)
    return result.value, result.aux


def _resolve_runtime_legal_actions(snap: dict, actor: int) -> list[dict]:
    try:
        return _rust_enumerate_legal_action_specs_structural(snap, actor)
    except RuntimeError as exc:
        if not _is_missing_rust_capability_error(exc):
            raise
        return [
            action.to_mjai() if hasattr(action, "to_mjai") else dict(action)
            for action in enumerate_legal_actions(snap, actor)
        ]


def _simulate_discard_snapshot(
    snap: dict,
    actor: int,
    pai: str,
) -> dict:
    try:
        return _rust_project_keqingv4_discard_snapshot(snap, actor, pai)
    except RuntimeError as exc:
        if not _is_missing_rust_capability_error(exc):
            raise

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


def _simulate_meld_snapshot(
    snap: dict,
    actor: int,
    action: dict,
) -> dict:
    try:
        projected = _rust_project_keqingv4_call_snapshot(snap, actor, action)
        if projected is not None:
            return projected
    except RuntimeError as exc:
        if not _is_missing_rust_capability_error(exc):
            raise

    meld_type = action.get("type", "")
    consumed = list(action.get("consumed", []))
    pai = action.get("pai", "")

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
            "pai": normalize_tile(pai) if pai else pai,
            "consumed": [normalize_tile(c) for c in consumed],
            "target": action.get("target"),
        }
    ]
    fake_snap["melds"] = melds
    fake_snap["last_discard"] = None
    fake_snap["last_kakan"] = None
    fake_snap["actor_to_move"] = actor
    fake_snap["tsumo_pai"] = None
    return fake_snap


def _visible_counts34_from_snap(snap: dict) -> tuple[int, ...]:
    visible = [0] * 34
    for tile in snap.get("hand", []):
        idx = tile_to_34(normalize_tile(tile))
        if idx >= 0:
            visible[idx] += 1
    tsumo_tile = snap.get("tsumo_pai")
    if tsumo_tile:
        idx = tile_to_34(normalize_tile(tsumo_tile))
        if idx >= 0:
            visible[idx] += 1
    for meld_group in snap.get("melds", [[], [], [], []]):
        for meld in meld_group:
            for tile in meld.get("consumed", []):
                idx = tile_to_34(normalize_tile(tile))
                if idx >= 0:
                    visible[idx] += 1
            pai = meld.get("pai")
            if pai:
                idx = tile_to_34(normalize_tile(pai))
                if idx >= 0:
                    visible[idx] += 1
    for disc_group in snap.get("discards", [[], [], [], []]):
        for discard in disc_group:
            pai = discard.get("pai") if isinstance(discard, dict) else discard
            if not pai:
                continue
            idx = tile_to_34(normalize_tile(pai))
            if idx >= 0:
                visible[idx] += 1
    for marker in snap.get("dora_markers", []):
        idx = tile_to_34(normalize_tile(marker))
        if idx >= 0:
            visible[idx] += 1
    return tuple(visible)


def _live_draw_tile_weights(snap: dict) -> list[tuple[str, int]]:
    try:
        return _rust_enumerate_keqingv4_live_draw_weights(snap)
    except RuntimeError as exc:
        if not _is_missing_rust_capability_error(exc):
            raise

    visible = _visible_counts34_from_snap(snap)
    result: list[tuple[str, int]] = []
    for tile34, seen in enumerate(visible):
        live = max(0, 4 - int(seen))
        if live > 0:
            result.append((_TILE34_STR[tile34], live))
    return result


def _simulate_rinshan_draw_snapshot(
    snap: dict,
    actor: int,
    pai: str,
) -> dict:
    try:
        return _rust_project_keqingv4_rinshan_draw_snapshot(snap, actor, pai)
    except RuntimeError as exc:
        if not _is_missing_rust_capability_error(exc):
            raise

    fake_snap = dict(snap)
    hand = list(snap.get("hand", []))
    hand.append(pai)
    fake_snap["hand"] = hand
    fake_snap["tsumo_pai"] = pai
    last_tsumo = list(snap.get("last_tsumo", [None, None, None, None]))
    while len(last_tsumo) <= actor:
        last_tsumo.append(None)
    last_tsumo[actor] = normalize_tile(pai)
    fake_snap["last_tsumo"] = last_tsumo
    last_tsumo_raw = list(snap.get("last_tsumo_raw", [None, None, None, None]))
    while len(last_tsumo_raw) <= actor:
        last_tsumo_raw.append(None)
    last_tsumo_raw[actor] = pai
    fake_snap["last_tsumo_raw"] = last_tsumo_raw
    fake_snap["actor_to_move"] = actor
    fake_snap["last_discard"] = None
    fake_snap["last_kakan"] = None
    return fake_snap


def _post_meld_discard_actions(
    snap: dict,
    actor: int,
) -> list[dict]:
    try:
        return _rust_enumerate_keqingv4_post_meld_discards(snap, actor)
    except RuntimeError as exc:
        if not _is_missing_rust_capability_error(exc):
            raise

    seen: set[str] = set()
    actions: list[dict] = []
    for tile in snap.get("hand", []):
        key = normalize_tile(tile)
        if key in seen:
            continue
        seen.add(key)
        actions.append(
            {
                "type": "dahai",
                "actor": actor,
                "pai": tile,
                "tsumogiri": False,
            }
        )
    return actions


def _resolve_post_meld_followup_context(
    snap: dict,
    actor: int,
    action: dict,
) -> tuple[dict, list[dict]]:
    try:
        projected, legal_actions = _rust_resolve_keqingv4_post_meld_followup(snap, actor, action)
        if projected is not None:
            return projected, legal_actions
    except RuntimeError as exc:
        if not _is_missing_rust_capability_error(exc):
            raise
    projected = _simulate_meld_snapshot(snap, actor, action)
    return projected, _post_meld_discard_actions(projected, actor)


def _resolve_rinshan_followup_context(
    snap: dict,
    actor: int,
    pai: str,
) -> tuple[dict, list[dict]]:
    try:
        return _rust_resolve_keqingv4_rinshan_followup(snap, actor, pai)
    except RuntimeError as exc:
        if not _is_missing_rust_capability_error(exc):
            raise
    projected = _simulate_rinshan_draw_snapshot(snap, actor, pai)
    return projected, _resolve_runtime_legal_actions(projected, actor)


def _resolve_reach_followup_context(
    snap: dict,
    actor: int,
    pai: str,
) -> tuple[dict, list[dict]]:
    try:
        return _rust_resolve_keqingv4_reach_followup(snap, actor, pai)
    except RuntimeError as exc:
        if not _is_missing_rust_capability_error(exc):
            raise
    fake_snap = _simulate_discard_snapshot(snap, actor, pai)
    reached = list(snap.get("reached", [False, False, False, False]))
    if actor < len(reached):
        reached[actor] = True
    fake_snap["reached"] = reached
    pending_reach = list(snap.get("pending_reach", [False, False, False, False]))
    if actor < len(pending_reach):
        pending_reach[actor] = False
    fake_snap["pending_reach"] = pending_reach
    return fake_snap, _resolve_runtime_legal_actions(fake_snap, actor)


def _build_python_continuation_scenarios(
    snap: dict,
    actor: int,
    action: dict,
) -> list[dict]:
    action_type = action.get("type", "")
    if action_type in {"chi", "pon"}:
        projected, legal_actions = _resolve_post_meld_followup_context(snap, actor, action)
        if projected is None:
            return []
        return [
            {
                "projected_snapshot": projected,
                "legal_actions": legal_actions,
                "weight": 1.0,
                "continuation_kind": "post_meld_followup",
                "declaration_action": None,
            }
        ]
    if action_type in {"daiminkan", "ankan", "kakan"}:
        post_meld_snapshot = _simulate_meld_snapshot(snap, actor, action)
        scenarios: list[dict] = []
        for draw_tile, weight in _live_draw_tile_weights(post_meld_snapshot):
            projected, legal_actions = _resolve_rinshan_followup_context(post_meld_snapshot, actor, draw_tile)
            scenarios.append(
                {
                    "projected_snapshot": projected,
                    "legal_actions": legal_actions,
                    "weight": float(weight),
                    "continuation_kind": "rinshan_followup",
                    "declaration_action": None,
                }
            )
        if scenarios:
            return scenarios
        return [
            {
                "projected_snapshot": post_meld_snapshot,
                "legal_actions": _resolve_runtime_legal_actions(post_meld_snapshot, actor),
                "weight": 1.0,
                "continuation_kind": "state_value",
                "declaration_action": None,
            }
        ]
    if action_type == "reach":
        hand = Counter(snap.get("hand", []))
        last_tsumo_all = list(snap.get("last_tsumo", [None, None, None, None]))
        last_tsumo_raw_all = list(snap.get("last_tsumo_raw", [None, None, None, None]))
        last_tsumo = last_tsumo_all[actor] if actor < len(last_tsumo_all) else None
        last_tsumo_raw = last_tsumo_raw_all[actor] if actor < len(last_tsumo_raw_all) else None
        try:
            reach_discards = _rust_enumerate_keqingv4_reach_discards(snap, actor)
        except RuntimeError as exc:
            if not _is_missing_rust_capability_error(exc):
                raise
            reach_discards = _reach_discard_candidates(hand, last_tsumo, last_tsumo_raw)
        scenarios = []
        for pai_out, tsumogiri in reach_discards:
            projected, legal_actions = _resolve_reach_followup_context(snap, actor, pai_out)
            scenarios.append(
                {
                    "projected_snapshot": projected,
                    "legal_actions": legal_actions,
                    "weight": 1.0,
                    "continuation_kind": "reach_declaration",
                    "declaration_action": {
                        "type": "dahai",
                        "actor": actor,
                        "pai": pai_out,
                        "tsumogiri": tsumogiri,
                    },
                }
            )
        if scenarios:
            return scenarios
        reached_snapshot = dict(snap)
        reached = list(snap.get("reached", [False, False, False, False]))
        if actor < len(reached):
            reached[actor] = True
        reached_snapshot["reached"] = reached
        return [
            {
                "projected_snapshot": reached_snapshot,
                "legal_actions": _resolve_runtime_legal_actions(reached_snapshot, actor),
                "weight": 1.0,
                "continuation_kind": "state_value",
                "declaration_action": None,
            }
        ]
    return []


def _resolve_continuation_scenarios(
    snap: dict,
    actor: int,
    action: dict,
) -> list[dict]:
    try:
        scenarios = _rust_resolve_keqingv4_continuation_scenarios(snap, actor, action)
    except RuntimeError as exc:
        if not _is_missing_rust_capability_error(exc):
            raise
    else:
        return _validate_continuation_scenario_payload(scenarios)
    return _build_python_continuation_scenarios(snap, actor, action)


def _best_followup_action_score(
    adapter: KeqingModelAdapter,
    snap: dict,
    actor: int,
    legal_actions: list[dict],
    *,
    score_delta_lambda: float,
    win_prob_lambda: float,
    dealin_prob_lambda: float,
) -> tuple[dict | None, float]:
    if not legal_actions:
        return None, -1e18

    result = adapter.forward(dict(snap, legal_actions=legal_actions), actor)
    try:
        payload = _rust_score_keqingv4_continuation_scenario(
            "post_meld_followup",
            np.asarray(result.policy_logits, dtype=np.float32),
            legal_actions,
            value=float(result.value),
            score_delta=float(result.aux.score_delta),
            win_prob=float(result.aux.win_prob),
            dealin_prob=float(result.aux.dealin_prob),
            beam_lambda=1.0,
            score_delta_lambda=score_delta_lambda,
            win_prob_lambda=win_prob_lambda,
            dealin_prob_lambda=dealin_prob_lambda,
        )
        best_action, score = _validate_continuation_score_payload(
            payload,
            continuation_kind="post_meld_followup",
        )
        return best_action, score
    except RuntimeError as exc:
        if not _is_missing_rust_capability_error(exc):
            raise

    cont_logits = np.asarray(result.policy_logits, dtype=np.float32)
    mask = np.array(build_legal_mask(legal_actions), dtype=np.float32)
    cont_logits = np.where(mask > 0, cont_logits, -1e9)
    cont_aux_bonus = _aux_bonus(
        result.aux,
        score_delta_lambda,
        win_prob_lambda,
        dealin_prob_lambda,
    )
    best_action = _find_best_legal(
        cont_logits,
        legal_actions,
        value=float(result.value),
        style_lambda=0.0,
        aux_bonus=cont_aux_bonus,
    )
    best_score = _legal_score(
        cont_logits,
        best_action,
        value=float(result.value),
        style_lambda=0.0,
        aux_bonus=cont_aux_bonus,
    )
    return best_action, best_score


def _score_continuation_result_payload(
    continuation_kind: str,
    result: ModelForwardResult,
    legal_actions: list[dict],
    *,
    beam_lambda: float,
    score_delta_lambda: float,
    win_prob_lambda: float,
    dealin_prob_lambda: float,
) -> tuple[dict | None, float]:
    try:
        payload = _rust_score_keqingv4_continuation_scenario(
            continuation_kind,
            np.asarray(result.policy_logits, dtype=np.float32),
            legal_actions,
            value=float(result.value),
            score_delta=float(result.aux.score_delta),
            win_prob=float(result.aux.win_prob),
            dealin_prob=float(result.aux.dealin_prob),
            beam_lambda=beam_lambda,
            score_delta_lambda=score_delta_lambda,
            win_prob_lambda=win_prob_lambda,
            dealin_prob_lambda=dealin_prob_lambda,
        )
        return _validate_continuation_score_payload(
            payload,
            continuation_kind=continuation_kind,
        )
    except RuntimeError as exc:
        if not _is_missing_rust_capability_error(exc):
            raise

    if continuation_kind in {"reach_declaration", "state_value"}:
        continuation_score = beam_lambda * float(result.value) + _aux_bonus(
            result.aux,
            score_delta_lambda,
            win_prob_lambda,
            dealin_prob_lambda,
        )
        return None, continuation_score

    if not legal_actions:
        return None, -1e18
    cont_logits = np.asarray(result.policy_logits, dtype=np.float32)
    mask = np.array(build_legal_mask(legal_actions), dtype=np.float32)
    cont_logits = np.where(mask > 0, cont_logits, -1e9)
    cont_aux_bonus = _aux_bonus(
        result.aux,
        score_delta_lambda,
        win_prob_lambda,
        dealin_prob_lambda,
    )
    best_action = _find_best_legal(
        cont_logits,
        legal_actions,
        value=float(result.value),
        style_lambda=0.0,
        aux_bonus=cont_aux_bonus,
    )
    best_score = _legal_score(
        cont_logits,
        best_action,
        value=float(result.value),
        style_lambda=0.0,
        aux_bonus=cont_aux_bonus,
    )
    return best_action, best_score


def _score_continuation_scenarios(
    adapter: KeqingModelAdapter,
    actor: int,
    policy_logits: np.ndarray,
    action: dict,
    scenarios: list[dict],
    *,
    beam_lambda: float,
    score_delta_lambda: float,
    win_prob_lambda: float,
    dealin_prob_lambda: float,
) -> tuple[float, dict]:
    if not scenarios:
        return float(policy_logits[action_to_idx(action)]), {}

    scenario_scores_payload: list[dict] = []
    legacy_scenario_scores: list[tuple[float, float, dict]] = []
    for scenario in scenarios:
        projected_snapshot = dict(scenario.get("projected_snapshot", {}))
        legal_actions = list(scenario.get("legal_actions", []))
        continuation_kind = str(scenario.get("continuation_kind", ""))
        declaration_action = scenario.get("declaration_action")
        weight = float(scenario.get("weight", 1.0) or 0.0)
        resolved_snapshot = projected_snapshot if not legal_actions else dict(projected_snapshot, legal_actions=legal_actions)
        result = adapter.forward(resolved_snapshot, actor)
        _best_action, continuation_score = _score_continuation_result_payload(
            continuation_kind,
            result,
            legal_actions,
            beam_lambda=beam_lambda,
            score_delta_lambda=score_delta_lambda,
            win_prob_lambda=win_prob_lambda,
            dealin_prob_lambda=dealin_prob_lambda,
        )
        scenario_scores_payload.append(
            {
                "weight": weight,
                "score": continuation_score,
                "continuation_kind": continuation_kind,
                "declaration_action": declaration_action if isinstance(declaration_action, dict) else None,
            }
        )
        legacy_meta: dict[str, object] = {}
        if action.get("type", "") == "reach" and isinstance(declaration_action, dict):
            legacy_meta["reach_discard"] = declaration_action
        legacy_scenario_scores.append((weight, continuation_score, legacy_meta))

    try:
        payload = _rust_aggregate_keqingv4_continuation_scores(
            policy_logits,
            action,
            scenario_scores_payload,
        )
        return _validate_continuation_aggregation_payload(payload, action=action)
    except RuntimeError as exc:
        if not _is_missing_rust_capability_error(exc):
            raise

    action_idx = action_to_idx(action)
    action_type = action.get("type", "")
    if action_type == "reach":
        _, best_score, best_meta = max(
            legacy_scenario_scores,
            key=lambda item: item[1] + float(policy_logits[action_to_idx(item[2]["reach_discard"])]) if "reach_discard" in item[2] else item[1],
        )
        if "reach_discard" in best_meta:
            best_score += float(policy_logits[action_to_idx(best_meta["reach_discard"])])
        return float(policy_logits[action_idx]) + best_score, best_meta

    total_weight = sum(weight for weight, _score, _meta in legacy_scenario_scores)
    if total_weight <= 0.0:
        _, best_score, best_meta = max(legacy_scenario_scores, key=lambda item: item[1])
        return float(policy_logits[action_idx]) + best_score, best_meta
    weighted_score = sum(weight * score for weight, score, _meta in legacy_scenario_scores) / total_weight
    return float(policy_logits[action_idx]) + weighted_score, {}


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
        scenarios = _resolve_continuation_scenarios(snap, actor, a)
        if scenarios:
            score, _meta = _score_continuation_scenarios(
                adapter,
                actor,
                policy_logits,
                a,
                scenarios,
                beam_lambda=beam_lambda,
                score_delta_lambda=score_delta_lambda,
                win_prob_lambda=win_prob_lambda,
                dealin_prob_lambda=dealin_prob_lambda,
            )
        else:
            fake_snap = _simulate_meld_snapshot(snap, actor, a)
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
    special_bonus_scores: dict[int, float] | None = None,
) -> tuple[dict, dict[int, float], dict[int, dict]]:
    reach_idx = action_to_idx(reach_action)
    scenarios = _resolve_continuation_scenarios(snap, actor, reach_action)
    best_reach_score, reach_meta = _score_continuation_scenarios(
        adapter,
        actor,
        policy_logits,
        reach_action,
        scenarios,
        beam_lambda=beam_lambda,
        score_delta_lambda=score_delta_lambda,
        win_prob_lambda=win_prob_lambda,
        dealin_prob_lambda=dealin_prob_lambda,
    )
    best_reach_score += (special_bonus_scores or {}).get(reach_idx, 0.0)

    value_scores: dict[int, float] = {reach_idx: best_reach_score}
    reach_meta_by_idx: dict[int, dict] = {}
    if reach_meta:
        reach_meta_by_idx[reach_idx] = reach_meta

    best_score = best_reach_score
    best_action = reach_action
    for a in other_actions:
        idx = action_to_idx(a)
        s = float(policy_logits[idx]) + (special_bonus_scores or {}).get(idx, 0.0)
        value_scores[idx] = s
        if s > best_score:
            best_score = s
            best_action = a

    return best_action, value_scores, reach_meta_by_idx


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
        special_calibration_scale: float = 1.0,
    ):
        self.adapter = adapter
        self.beam_k = beam_k
        self.beam_lambda = beam_lambda
        self.style_lambda = style_lambda
        self.score_delta_lambda = score_delta_lambda
        self.win_prob_lambda = win_prob_lambda
        self.dealin_prob_lambda = dealin_prob_lambda
        self.special_calibration_scale = special_calibration_scale

    def score(self, ctx: DecisionContext) -> DecisionResult:
        model_snap = dict(ctx.model_snap)
        model_snap["legal_actions"] = ctx.legal_actions
        model_result = self.adapter.forward(model_snap, ctx.actor)
        if model_result.xmodel1 is not None:
            return _score_xmodel1_candidates(self.adapter, ctx, model_result)
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
        special_meta = _build_v4_special_meta(self.adapter, ctx.model_snap, ctx.actor, legal_dicts)
        special_bonus_scores = {
            idx: self.special_calibration_scale * _special_action_calibration_bonus(meta)
            for idx, meta in special_meta.items()
        }
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
                    special_bonus_scores=special_bonus_scores,
                )
                chosen_score = _candidate_final_score(
                    logits_np,
                    chosen,
                    beam_value_scores,
                    value_scalar,
                    self.style_lambda,
                    aux_bonus,
                    special_bonus_scores,
                )
                fallback_score = _legal_score(
                    logits_np,
                    fallback,
                    value_scalar,
                    self.style_lambda,
                    aux_bonus,
                    special_bonus=special_bonus_scores.get(action_to_idx(fallback), 0.0),
                )
                if fallback_score > chosen_score:
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
                special_bonus_scores=special_bonus_scores,
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
                special_bonus_scores=special_bonus_scores,
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
                        special_bonus_scores,
                    ),
                    beam_score=(
                        float(beam_value_scores[action_to_idx(a)])
                        if action_to_idx(a) in beam_value_scores
                        else None
                    ),
                    meta={
                        **special_meta.get(action_to_idx(a), {}),
                        **beam_meta.get(action_to_idx(a), {}),
                    },
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
