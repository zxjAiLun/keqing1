"""Style-context contract for KeqingRL-Lite."""

from __future__ import annotations

from dataclasses import dataclass

import torch


STYLE_CONTEXT_DIM = 5
STYLE_CONTEXT_VERSION = "keqingrl_style_context_v1"


@dataclass(frozen=True)
class StyleContext:
    call_bias: float = 0.0
    defense_bias: float = 0.0
    menzen_bias: float = 0.0
    push_bias: float = 0.0
    riichi_bias: float = 0.0

    def to_tensor(self) -> torch.Tensor:
        return torch.tensor(
            [
                float(self.call_bias),
                float(self.defense_bias),
                float(self.menzen_bias),
                float(self.push_bias),
                float(self.riichi_bias),
            ],
            dtype=torch.float32,
        )


DEFAULT_STYLE_CONTEXT = StyleContext()


def build_style_context(
    *,
    call_bias: float = 0.0,
    defense_bias: float = 0.0,
    menzen_bias: float = 0.0,
    push_bias: float = 0.0,
    riichi_bias: float = 0.0,
) -> torch.Tensor:
    return StyleContext(
        call_bias=call_bias,
        defense_bias=defense_bias,
        menzen_bias=menzen_bias,
        push_bias=push_bias,
        riichi_bias=riichi_bias,
    ).to_tensor()


__all__ = [
    "DEFAULT_STYLE_CONTEXT",
    "STYLE_CONTEXT_DIM",
    "STYLE_CONTEXT_VERSION",
    "StyleContext",
    "build_style_context",
]
