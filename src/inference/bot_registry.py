from __future__ import annotations

from pathlib import Path
from typing import Any

from inference.rulebase_bot import RulebaseBot
from inference.mortal_bot import MortalReviewBot

SUPPORTED_BOT_NAMES = {"rulebase", "mortal"}


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
    rank_pt_lambda: float = 0.0,
) -> Any:
    if bot_name == "rulebase":
        return RulebaseBot(player_id=player_id, verbose=verbose)
    if bot_name == "mortal":
        resolved_model_path = (
            Path(model_path)
            if model_path is not None
            else Path(project_root) / "artifacts" / "mortal_serving" / "mortal.pth"
        )
        return MortalReviewBot(
            player_id=player_id,
            model_path=resolved_model_path,
            mortal_root=Path(project_root) / "third_party" / "Mortal",
            device=device,
            verbose=verbose,
        )
    raise ValueError(f"Unsupported bot name: {bot_name}")
