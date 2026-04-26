from __future__ import annotations

from inference.review import same_action

_RESPONSE_ACTION_TYPES = {"chi", "pon", "daiminkan", "ankan", "kakan", "hora"}


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
    total_ops = len(own_log)
    match_count = sum(
        1 for e in own_log if same_action(e.get("chosen"), e.get("gt_action"))
    )
    return {
        **decisions,
        "log": log,
        "total_ops": total_ops,
        "match_count": match_count,
        "bot_type": (meta or {}).get("bot_type", decisions.get("bot_type")),
        "player_names": decisions.get("player_names") or (meta or {}).get("player_names"),
    }


__all__ = ["normalize_replay_decisions"]
