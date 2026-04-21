from __future__ import annotations

import pytest
import torch

from xmodel1.trainer import (
    _chosen_action_targets,
    _masked_mean,
    _response_candidate_ce_loss,
    _response_post_teacher_ce_loss,
    _resolve_pts_given_targets,
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


def test_response_candidate_ce_ignores_rows_without_valid_choice():
    logits = torch.tensor([[1.5, -1e4], [0.1, 0.3]], dtype=torch.float32)
    chosen = torch.tensor([0, -1], dtype=torch.long)
    mask = torch.tensor([[1, 0], [0, 0]], dtype=torch.uint8)

    loss = _response_candidate_ce_loss(logits, chosen, mask)

    assert float(loss) >= 0.0


def test_response_post_teacher_ce_uses_chosen_response_slot_teacher():
    response_post_logits = torch.tensor(
        [
            [
                [1.2, 0.1, -1e4],
                [0.0, 2.0, 0.1],
            ]
        ],
        dtype=torch.float32,
    )
    chosen_response_idx = torch.tensor([1], dtype=torch.long)
    response_teacher_discard_idx = torch.tensor([[0, 1]], dtype=torch.long)
    response_action_mask = torch.tensor([[1, 1]], dtype=torch.uint8)
    response_post_candidate_mask = torch.tensor(
        [
            [
                [1, 1, 0],
                [1, 1, 1],
            ]
        ],
        dtype=torch.uint8,
    )

    loss = _response_post_teacher_ce_loss(
        response_post_logits,
        chosen_response_idx,
        response_teacher_discard_idx,
        response_action_mask,
        response_post_candidate_mask,
    )

    assert float(loss) > 0.0


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
