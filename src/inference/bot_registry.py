from __future__ import annotations

from pathlib import Path
from typing import Any

from inference.rulebase_bot import RulebaseBot
from inference.runtime_bot import RuntimeBot

SUPPORTED_BOT_NAMES = {"keqingv1", "keqingv2", "keqingv3", "keqingv31", "xmodel1", "rulebase"}


def create_runtime_bot(
    *,
    bot_name: str,
    player_id: int,
    project_root: str | Path,
    model_path: str | Path | None = None,
    device: str = "cuda",
    verbose: bool = False,
    beam_k: int = 3,
    beam_lambda: float = 1.0,
) -> Any:
    if bot_name == "rulebase":
        return RulebaseBot(player_id=player_id, verbose=verbose)
    if bot_name not in SUPPORTED_BOT_NAMES:
        raise ValueError(f"Unsupported bot name: {bot_name}")
    resolved_model_path = (
        Path(model_path)
        if model_path is not None
        else Path(project_root) / "artifacts" / "models" / bot_name / "best.pth"
    )
    return RuntimeBot(
        player_id=player_id,
        model_path=resolved_model_path,
        device=device,
        verbose=verbose,
        beam_k=beam_k,
        beam_lambda=beam_lambda,
        model_version=bot_name,
    )
