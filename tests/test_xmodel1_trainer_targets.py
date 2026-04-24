from __future__ import annotations

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from xmodel1.model import (
    AuxiliaryPolicyOutput,
    PlacementPolicyOutput,
    PolicyOutput,
    ResponsePolicyOutput,
)
from xmodel1.trainer import (
    _make_xmodel1_task,
    _chosen_action_targets,
    _masked_mean,
    _response_candidate_ce_loss,
    _response_post_human_ce_loss,
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


def test_response_post_human_ce_prefers_human_continuation_target():
    response_post_logits = torch.tensor(
        [
            [
                [0.1, 3.5, -1e4],
                [1.0, 0.0, -1e4],
            ]
        ],
        dtype=torch.float32,
    )
    chosen_response_idx = torch.tensor([0], dtype=torch.long)
    response_teacher_discard_idx = torch.tensor([[0, 0]], dtype=torch.long)
    response_human_discard_idx = torch.tensor([[1, 0]], dtype=torch.long)
    response_action_mask = torch.tensor([[1, 1]], dtype=torch.uint8)
    response_post_candidate_mask = torch.tensor(
        [
            [
                [1, 1, 0],
                [1, 1, 0],
            ]
        ],
        dtype=torch.uint8,
    )

    teacher_loss = _response_post_teacher_ce_loss(
        response_post_logits,
        chosen_response_idx,
        response_teacher_discard_idx,
        response_action_mask,
        response_post_candidate_mask,
    )
    human_loss = _response_post_human_ce_loss(
        response_post_logits,
        chosen_response_idx,
        response_human_discard_idx,
        response_action_mask,
        response_post_candidate_mask,
    )

    assert float(human_loss) < float(teacher_loss)


class _FakeXmodel1TrainWrapper:
    def __init__(self, output: PolicyOutput) -> None:
        self._output = output

    def get_last_xmodel1_output(self) -> PolicyOutput:
        return self._output


def _make_policy_output(
    *,
    response_post_logits: torch.Tensor,
    response_logits: torch.Tensor | None = None,
) -> PolicyOutput:
    batch = response_post_logits.shape[0]
    discard_logits = torch.zeros((batch, 14), dtype=torch.float32)
    action_logits = torch.zeros((batch, 45), dtype=torch.float32)
    action_logits[:, 34] = 1.0
    if response_logits is None:
        response_logits = torch.zeros((batch, 8), dtype=torch.float32)
        response_logits[:, 0] = 1.0
    return PolicyOutput(
        discard_logits=discard_logits,
        action_logits=action_logits,
        response=ResponsePolicyOutput(
            response_logits=response_logits,
            response_post_logits=response_post_logits,
        ),
        auxiliary=AuxiliaryPolicyOutput(
            win_logit=torch.zeros((batch, 1), dtype=torch.float32),
            dealin_logit=torch.zeros((batch, 1), dtype=torch.float32),
            pts_given_win=torch.zeros((batch, 1), dtype=torch.float32),
            pts_given_dealin=torch.zeros((batch, 1), dtype=torch.float32),
            opp_tenpai_logits=torch.zeros((batch, 3), dtype=torch.float32),
            placement=PlacementPolicyOutput(
                rank_logits=torch.zeros((batch, 4), dtype=torch.float32),
                final_score_delta=torch.zeros((batch, 1), dtype=torch.float32),
            ),
        ),
    )


def test_xmodel1_task_response_post_loss_mixes_human_ce_and_heuristic_rank():
    task = _make_xmodel1_task(
        {
            "ce_loss_weight": 0.0,
            "response_ce_loss_weight": 0.0,
            "response_post_ce_loss_weight": 2.0,
            "response_post_rank_loss_weight": 3.0,
            "rank_loss_weight": 0.0,
            "hard_bad_loss_weight": 0.0,
            "win_loss_weight": 0.0,
            "dealin_loss_weight": 0.0,
            "pts_win_loss_weight": 0.0,
            "pts_dealin_loss_weight": 0.0,
            "opp_tenpai_loss_weight": 0.0,
            "final_rank_loss_weight": 0.0,
            "final_score_delta_loss_weight": 0.0,
            "rank_pt_value_loss_weight": 0.0,
        }
    )
    response_post_logits = torch.full((1, 8, 14), -1e4, dtype=torch.float32)
    response_post_logits[0, 0, 0] = 0.0
    response_post_logits[0, 0, 1] = 2.0
    model = _FakeXmodel1TrainWrapper(_make_policy_output(response_post_logits=response_post_logits))
    batch_data = {
        "sample_type": torch.tensor([1], dtype=torch.long),
        "action_idx": torch.tensor([34], dtype=torch.long),
        "mask": torch.nn.functional.one_hot(torch.tensor([34]), num_classes=45).to(torch.uint8),
        "chosen_candidate_idx": torch.tensor([0], dtype=torch.long),
        "candidate_quality_score": torch.zeros((1, 14), dtype=torch.float32),
        "candidate_mask": torch.zeros((1, 14), dtype=torch.uint8),
        "candidate_hard_bad_flag": torch.zeros((1, 14), dtype=torch.float32),
        "chosen_response_action_idx": torch.tensor([0], dtype=torch.long),
        "response_action_mask": torch.tensor([[1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.uint8),
        "response_post_candidate_mask": torch.zeros((1, 8, 14), dtype=torch.uint8),
        "response_post_candidate_quality_score": torch.zeros((1, 8, 14), dtype=torch.float32),
        "response_human_discard_idx": torch.full((1, 8), -1, dtype=torch.long),
        "response_teacher_discard_idx": torch.full((1, 8), -1, dtype=torch.long),
        "win_target": torch.zeros((1,), dtype=torch.float32),
        "dealin_target": torch.zeros((1,), dtype=torch.float32),
        "pts_given_win_target": torch.zeros((1,), dtype=torch.float32),
        "pts_given_dealin_target": torch.zeros((1,), dtype=torch.float32),
        "opp_tenpai_target": torch.zeros((1, 3), dtype=torch.float32),
        "final_rank_target": torch.zeros((1,), dtype=torch.long),
        "final_score_delta_points_target": torch.zeros((1,), dtype=torch.int32),
    }
    batch_data["response_post_candidate_mask"][0, 0, :2] = 1
    batch_data["response_post_candidate_quality_score"][0, 0, :2] = torch.tensor([0.5, 1.5], dtype=torch.float32)
    batch_data["response_human_discard_idx"][0, 0] = 1
    batch_data["response_teacher_discard_idx"][0, 0] = 0

    extra_loss, metrics = task.compute_extra_loss(model, torch.device("cpu"), batch_data, True, 0)

    expected = 2.0 * metrics["response_post_ce"] + 3.0 * metrics["response_post_rank"]
    assert float(extra_loss.item()) == pytest.approx(expected, rel=1e-6, abs=1e-6)
    assert metrics["response_post_ce"] > 0.0
    assert metrics["response_post_rank"] > 0.0


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
