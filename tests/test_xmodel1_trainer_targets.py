from __future__ import annotations

import pytest
import torch

from xmodel1.schema import (
    XMODEL1_SPECIAL_TYPE_CHI_LOW,
    XMODEL1_SPECIAL_TYPE_DAMA,
    XMODEL1_SPECIAL_TYPE_NONE,
    XMODEL1_SPECIAL_TYPE_PON,
    XMODEL1_SPECIAL_TYPE_REACH,
)
from xmodel1.trainer import (
    _chosen_action_targets,
    _masked_mean,
    _resolve_pts_given_targets,
    _special_comparison_losses,
)


def test_chosen_action_targets_maps_selected_candidate_tiles_to_dahai_space():
    candidate_tile_id = torch.tensor(
        [
            [4, 9, -1],
            [27, 2, -1],
        ],
        dtype=torch.long,
    )
    chosen_idx = torch.tensor([1, 0], dtype=torch.long)

    targets = _chosen_action_targets(candidate_tile_id, chosen_idx)

    assert targets.tolist() == [9, 27]


def test_chosen_action_targets_rejects_invalid_selected_padding_tile():
    candidate_tile_id = torch.tensor([[4, -1, -1]], dtype=torch.long)
    chosen_idx = torch.tensor([1], dtype=torch.long)

    with pytest.raises(ValueError):
        _chosen_action_targets(candidate_tile_id, chosen_idx)


def test_special_comparison_losses_reward_reach_over_dama_and_pon_over_none():
    special_logits = torch.tensor(
        [
            [2.0, 0.5, -1e4, -1e4, -1e4, -1e4],
            [-1e4, -1e4, -1e4, 1.5, -1e4, 0.25],
        ],
        dtype=torch.float32,
    )
    special_type_id = torch.tensor(
        [
            [XMODEL1_SPECIAL_TYPE_REACH, XMODEL1_SPECIAL_TYPE_DAMA, -1, -1, -1, -1],
            [-1, -1, -1, XMODEL1_SPECIAL_TYPE_PON, -1, XMODEL1_SPECIAL_TYPE_NONE],
        ],
        dtype=torch.long,
    )
    special_mask = torch.tensor(
        [
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 1],
        ],
        dtype=torch.uint8,
    )
    chosen_special_idx = torch.tensor([0, 3], dtype=torch.long)
    hard_bad = torch.zeros_like(special_logits)

    reach_loss, call_loss = _special_comparison_losses(
        special_logits,
        special_type_id,
        special_mask,
        chosen_special_idx,
        hard_bad,
        margin=0.25,
    )

    assert float(reach_loss) == 0.0
    assert float(call_loss) == 0.0


def test_special_comparison_losses_penalize_reach_margin_violation():
    special_logits = torch.tensor([[0.1, 0.3, -1e4]], dtype=torch.float32)
    special_type_id = torch.tensor(
        [[XMODEL1_SPECIAL_TYPE_REACH, XMODEL1_SPECIAL_TYPE_DAMA, -1]],
        dtype=torch.long,
    )
    special_mask = torch.tensor([[1, 1, 0]], dtype=torch.uint8)
    chosen_special_idx = torch.tensor([0], dtype=torch.long)
    hard_bad = torch.zeros_like(special_logits)

    reach_loss, call_loss = _special_comparison_losses(
        special_logits,
        special_type_id,
        special_mask,
        chosen_special_idx,
        hard_bad,
        margin=0.25,
    )

    assert float(reach_loss) > 0.0
    assert float(call_loss) == 0.0


def test_special_comparison_losses_penalize_none_only_when_best_call_is_hard_bad():
    special_logits = torch.tensor([[-1e4, -1e4, 1.0, -1e4, -1e4, 0.2]], dtype=torch.float32)
    special_type_id = torch.tensor([[-1, -1, XMODEL1_SPECIAL_TYPE_CHI_LOW, -1, -1, XMODEL1_SPECIAL_TYPE_NONE]], dtype=torch.long)
    special_mask = torch.tensor([[0, 0, 1, 0, 0, 1]], dtype=torch.uint8)
    chosen_special_idx = torch.tensor([5], dtype=torch.long)
    hard_bad = torch.tensor([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)

    _, call_loss = _special_comparison_losses(
        special_logits,
        special_type_id,
        special_mask,
        chosen_special_idx,
        hard_bad,
        margin=0.25,
    )

    assert float(call_loss) > 0.0


def test_masked_mean_ignores_invalid_rows():
    values = torch.tensor([2.0, 4.0, 100.0])
    mask = torch.tensor([1, 1, 0], dtype=torch.bool)
    assert float(_masked_mean(values, mask)) == 3.0


def test_resolve_pts_given_targets_prefers_exported_true_labels():
    batch = {
        "score_delta_target": torch.tensor([0.3, -0.7], dtype=torch.float32),
        "win_target": torch.tensor([1.0, 0.0], dtype=torch.float32),
        "dealin_target": torch.tensor([0.0, 1.0], dtype=torch.float32),
        "pts_given_win_target": torch.tensor([0.25, 0.0], dtype=torch.float32),
        "pts_given_dealin_target": torch.tensor([0.0, 0.5], dtype=torch.float32),
    }

    pts_win, pts_dealin = _resolve_pts_given_targets(batch)

    assert pts_win.tolist() == [0.25, 0.0]
    assert pts_dealin.tolist() == [0.0, 0.5]


def test_resolve_pts_given_targets_rejects_old_cache_batches():
    batch = {
        "score_delta_target": torch.tensor([0.3, -0.7, 0.2], dtype=torch.float32),
        "win_target": torch.tensor([1.0, 0.0, 0.0], dtype=torch.float32),
        "dealin_target": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32),
    }

    with pytest.raises(ValueError):
        _resolve_pts_given_targets(batch)
