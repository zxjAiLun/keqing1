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


def _repair_kakan_snapshots(decisions: dict, events: list[dict] | None) -> None:
    """Repair cached snapshots written by a native runtime without kakan_accepted."""
    if not events:
        return

    accepted: list[tuple[tuple[str, int, int], int, str]] = []
    current_kyoku: tuple[str, int, int] | None = None
    for event in events:
        if event.get("type") == "start_kyoku":
            current_kyoku = (
                str(event.get("bakaze", "")),
                int(event.get("kyoku", 0)),
                int(event.get("honba", 0)),
            )
        elif event.get("type") == "kakan_accepted" and current_kyoku is not None:
            accepted.append((current_kyoku, int(event.get("actor", -1)), str(event.get("pai", ""))))

    consumed_accepts: set[int] = set()
    active: tuple[tuple[str, int, int], str, int] | None = None
    for entry in decisions.get("log", []):
        key_data = entry.get("kyoku_key") or {}
        entry_key = (
            str(key_data.get("bakaze", entry.get("bakaze", ""))),
            int(key_data.get("kyoku", entry.get("kyoku", 0))),
            int(key_data.get("honba", entry.get("honba", 0))),
        )
        if active is not None and active[0] != entry_key:
            active = None

        if active is not None:
            _, added_tile, actor = active
            hand = list(entry.get("hand") or [])
            for index, tile in enumerate(hand):
                if normalize_tile(str(tile)) == normalize_tile(added_tile):
                    hand.pop(index)
                    break
            entry["hand"] = hand

            melds = entry.get("melds")
            if isinstance(melds, list) and 0 <= actor < len(melds):
                for meld in melds[actor] or []:
                    if (
                        isinstance(meld, dict)
                        and meld.get("type") == "pon"
                        and normalize_tile(str(meld.get("pai", ""))) == normalize_tile(added_tile)
                    ):
                        meld["type"] = "kakan"
                        consumed = list(meld.get("consumed") or [])
                        if len(consumed) < 4:
                            consumed.append(added_tile)
                        meld["consumed"] = consumed

            entry["candidates"] = [
                candidate
                for candidate in (entry.get("candidates") or [])
                if not (
                    candidate.get("action", {}).get("type") in {"dahai", "kakan"}
                    and candidate.get("action", {}).get("pai") is not None
                    and normalize_tile(str(candidate["action"]["pai"])) == normalize_tile(added_tile)
                    and not any(normalize_tile(str(tile)) == normalize_tile(added_tile) for tile in hand)
                )
            ]

        gt_action = entry.get("gt_action") or {}
        if gt_action.get("type") != "kakan":
            continue
        for index, (accepted_key, accepted_actor, accepted_tile) in enumerate(accepted):
            if index in consumed_accepts:
                continue
            if (
                accepted_key == entry_key
                and accepted_actor == int(gt_action.get("actor", -1))
                and normalize_tile(accepted_tile) == normalize_tile(str(gt_action.get("pai", "")))
            ):
                consumed_accepts.add(index)
                active = (entry_key, accepted_tile, accepted_actor)
                break


def normalize_replay_decisions(
    decisions: dict,
    meta: dict | None = None,
    events: list[dict] | None = None,
) -> dict:
    if not isinstance(decisions, dict):
        return decisions

    _repair_kakan_snapshots(decisions, events)
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
                # discard was consumed by a higher-priority call first.
                pending["comparison_exempt"] = "response_preempted"
                pending["comparison_exempt_by"] = dict(current_action)
                pending["gt_action"] = {"type": "none", "actor": player_id}
            elif same_action(current_action, chosen):
                pending["gt_action"] = {
                    **current_action,
                    "actor": current_action.get("actor", chosen.get("actor", player_id)),
                }
            else:
                pending["gt_action"] = {"type": "none", "actor": player_id}
            pending_idx = None

    # Cached decisions may already contain an explicit none gt_action from an
    # older normalizer. Re-check those response windows for historical replays.
    for idx, pending in enumerate(log):
        if pending.get("is_obs") or pending.get("comparison_exempt"):
            continue
        chosen = pending.get("chosen") or {}
        if chosen.get("type") not in {"chi", "pon"}:
            continue
        if not any(
            (candidate.get("action") or {}).get("type") == "none"
            for candidate in pending.get("candidates", [])
        ):
            continue
        if idx + 1 >= len(log):
            continue
        current_action = log[idx + 1].get("gt_action") or log[idx + 1].get("chosen") or {}
        if _higher_priority_response_intercepted(chosen, current_action, player_id):
            pending["comparison_exempt"] = "response_preempted"
            pending["comparison_exempt_by"] = dict(current_action)
            pending["gt_action"] = {"type": "none", "actor": player_id}

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
