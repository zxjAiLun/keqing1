from __future__ import annotations

import torch

from keqingrl.rewards import (
    build_rule_context,
    normalize_pt_map,
    pt_map_scale,
    terminal_rank_rewards,
)


def test_terminal_rank_rewards_normalize_pt_map_by_first_place_scale() -> None:
    pt_map = (90.0, 45.0, 0.0, -135.0)
    scores = (32000, 28000, 21000, 19000)

    rewards = terminal_rank_rewards(scores, pt_map, initial_oya=0, normalize=True)

    assert rewards == (1.0, 0.5, 0.0, -1.5)
    assert pt_map_scale(pt_map) == 90.0
    assert normalize_pt_map(pt_map) == (1.0, 0.5, 0.0, -1.5)


def test_build_rule_context_uses_normalized_pt_map_and_mode_flag() -> None:
    ctx = build_rule_context((90.0, 45.0, 0.0, -135.0), rank_score_scale=0.25, is_hanchan=False)

    assert ctx.shape == (6,)
    assert torch.allclose(
        ctx,
        torch.tensor([1.0, 0.5, 0.0, -1.5, 0.25, 0.0], dtype=torch.float32),
    )
