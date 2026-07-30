from __future__ import annotations

from inference.review import same_action
from mahjong_env.tiles import normalize_tile

_RESPONSE_ACTION_TYPES = {"chi", "pon", "daiminkan", "ankan", "kakan", "hora"}
_RESPONSE_PRIORITY = {
    "none": 0,
    "chi": 1,
    "pon": 2,
    "daiminkan": 2,
    "ankan": 2,
    "kakan": 2,
    "hora": 3,
}


def _same_response_source(left: dict, right: dict) -> bool:
    """Whether two response actions refer to the same discard source."""
    if left.get("target") is not None and right.get("target") is not None:
        if int(left["target"]) != int(right["target"]):
            return False
    left_pai = left.get("pai")
    right_pai = right.get("pai")
    if left_pai is None or right_pai is None:
        return True
    return normalize_tile(str(left_pai)) == normalize_tile(str(right_pai))


def _higher_priority_response_intercepted(
    pending_action: dict,
    current_action: dict,
    player_id: int | None,
) -> bool:
    """Detect a response window closed by another player's higher-priority call."""
    if not pending_action or not current_action:
        return False
    if current_action.get("actor") == player_id:
        return False
    pending_type = str(pending_action.get("type", ""))
    current_type = str(current_action.get("type", ""))
    if pending_type not in {"chi", "pon"}:
        return False
    if current_type not in {"pon", "daiminkan", "hora"}:
        return False
    if _RESPONSE_PRIORITY.get(current_type, 0) <= _RESPONSE_PRIORITY.get(pending_type, 0):
        return False
    return _same_response_source(pending_action, current_action)


def normalize_replay_decisions(decisions: dict, meta: dict | None = None) -> dict:
    if not isinstance(decisions, dict):
        return decisions

    log = decisions.get("log", [])
    player_id = decisions.get("player_id")
    pending_idx: int | None = None
    for idx, entry in enumerate(log):
        if not entry.get("is_obs") and entry.get("gt_action") is None:
            pending_idx = idx
        if pending_idx is None or idx == pending_idx:
            continue
        pending = log[pending_idx]
        if pending.get("gt_action") is not None:
            pending_idx = None
            continue
        chosen = pending.get("chosen") or {}
        candidates = pending.get("candidates", [])
        current_action = entry.get("gt_action") or entry.get("chosen") or {}
        has_none_candidate = any(
            c.get("action", {}).get("type") == "none" for c in candidates
        )
        has_non_none_candidate = any(
            c.get("action", {}).get("type") != "none" for c in candidates
        )
        if chosen.get("type") == "none" and has_non_none_candidate:
            pending["gt_action"] = {"type": "none", "actor": player_id}
            pending_idx = None
        elif chosen.get("type") in _RESPONSE_ACTION_TYPES and has_none_candidate:
            if _higher_priority_response_intercepted(chosen, current_action, player_id):
                # The player had a real response opportunity, but the same
                # discard was consumed by a higher-priority pon/kan/ron from
                # another seat before this response could execute.  Keep the
                # actual action as pass for board reconstruction, but exclude
                # this decision from mistake and match accounting.
                pending["comparison_exempt"] = "response_preempted"
                pending["comparison_exempt_by"] = dict(current_action)
                pending["gt_action"] = {"type": "none", "actor": player_id}
                pending_idx = None
                continue
            # 仅在后续条目明确确认了相同副露/和牌时，才把响应动作补成 chosen。
            # 否则保守地视为错过该响应窗口（实际为 none），避免把“可碰但没碰”
            # 误标成“实际碰了”，导致后续手牌/副露状态和动作标签互相矛盾。
            if same_action(current_action, chosen):
                pending["gt_action"] = {
                    **current_action,
                    "actor": current_action.get("actor", chosen.get("actor", player_id)),
                }
            else:
                pending["gt_action"] = {"type": "none", "actor": player_id}
            pending_idx = None

    own_log = [e for e in log if not e.get("is_obs")]
    comparable_log = [entry for entry in own_log if not entry.get("comparison_exempt")]
    total_ops = len(comparable_log)
    match_count = sum(
        1 for e in comparable_log if same_action(e.get("chosen"), e.get("gt_action"))
    )
    return {
        **decisions,
        "log": log,
        "total_ops": total_ops,
        "match_count": match_count,
        "bot_type": (meta or {}).get("bot_type", decisions.get("bot_type")),
        "player_names": decisions.get("player_names") or (meta or {}).get("player_names"),
        "external_review_links": decisions.get("external_review_links") or (meta or {}).get("external_review_links") or {},
    }


__all__ = ["normalize_replay_decisions"]
