from __future__ import annotations

from typing import Dict, List, Tuple

from mahjong_env.tiles import all_discardable_tiles_with_aka

FUURO_TYPES = ["chi", "pon", "daiminkan", "ankan", "kakan"]


def build_action_vocab() -> Tuple[List[str], Dict[str, int]]:
    actions: List[str] = ["none", "reach", "hora", "ryukyoku"]
    for t in all_discardable_tiles_with_aka():
        actions.append(f"dahai:{t}")
    for fuuro_type in FUURO_TYPES:
        for t in all_discardable_tiles_with_aka():
            actions.append(f"{fuuro_type}:{t}")
    stoi = {a: i for i, a in enumerate(actions)}
    return actions, stoi


def action_to_token(action: dict) -> str:
    at = action["type"]
    if at == "dahai":
        return f"dahai:{action['pai']}"
    if at in FUURO_TYPES:
        return f"{at}:{action['pai']}"
    return at

