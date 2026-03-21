from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional


MjaiEvent = Dict[str, Any]


@dataclass
class Action:
    type: str
    actor: int
    pai: Optional[str] = None
    consumed: Optional[List[str]] = None
    target: Optional[int] = None
    tsumogiri: Optional[bool] = None

    def to_mjai(self) -> Dict[str, Any]:
        # mjai protocol: "none" must not include "actor".
        if self.type == "none":
            return {"type": "none"}

        out: Dict[str, Any] = {"type": self.type, "actor": self.actor}
        if self.pai is not None:
            out["pai"] = self.pai
        if self.consumed is not None:
            out["consumed"] = self.consumed
        if self.target is not None:
            out["target"] = self.target
        if self.tsumogiri is not None:
            out["tsumogiri"] = self.tsumogiri
        return out

