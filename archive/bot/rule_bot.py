from __future__ import annotations

from typing import Dict, List


def fallback_action(legal_actions: List[Dict], actor: int) -> Dict:
    if not legal_actions:
        return {"type": "none"}
    for a in legal_actions:
        if a["type"] == "dahai":
            return a
    for a in legal_actions:
        if a["type"] != "none":
            return a
    out = legal_actions[0]
    if "actor" not in out:
        out = dict(out)
        out["actor"] = actor
    return out

