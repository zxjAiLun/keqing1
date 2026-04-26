from __future__ import annotations

import numpy as np

from evals.xmodel1.slice_report import (
    build_slice_masks,
    build_slice_report,
    chosen_special_type_ids,
    merge_slice_reports,
)
from xmodel1.schema import (
    XMODEL1_SPECIAL_TYPE_CHI_LOW,
    XMODEL1_SPECIAL_TYPE_CHI_MID,
    XMODEL1_SPECIAL_TYPE_HORA,
    XMODEL1_SPECIAL_TYPE_NONE,
    XMODEL1_SPECIAL_TYPE_REACH,
)


def test_chosen_special_type_ids_extracts_selected_slot_types():
    type_ids = np.array(
        [
            [XMODEL1_SPECIAL_TYPE_REACH, -1, -1],
            [XMODEL1_SPECIAL_TYPE_CHI_LOW, XMODEL1_SPECIAL_TYPE_NONE, -1],
            [XMODEL1_SPECIAL_TYPE_HORA, -1, -1],
        ],
        dtype=np.int16,
    )
    chosen = np.array([0, 1, 0], dtype=np.int16)

    out = chosen_special_type_ids(type_ids, chosen)

    assert out.tolist() == [
        XMODEL1_SPECIAL_TYPE_REACH,
        XMODEL1_SPECIAL_TYPE_NONE,
        XMODEL1_SPECIAL_TYPE_HORA,
    ]


def test_build_slice_masks_freezes_xmodel1_special_slice_definitions():
    action_targets = np.array([34, 35, 44, 42, 36], dtype=np.int16)
    type_ids = np.array(
        [
            [XMODEL1_SPECIAL_TYPE_REACH, -1],
            [XMODEL1_SPECIAL_TYPE_CHI_LOW, -1],
            [XMODEL1_SPECIAL_TYPE_NONE, -1],
            [XMODEL1_SPECIAL_TYPE_HORA, -1],
            [XMODEL1_SPECIAL_TYPE_CHI_MID, -1],
        ],
        dtype=np.int16,
    )
    chosen = np.array([0, 0, 0, 0, 0], dtype=np.int16)

    masks = build_slice_masks(
        action_idx_target=action_targets,
        special_candidate_type_id=type_ids,
        chosen_special_candidate_idx=chosen,
    )

    assert masks["reach"].tolist() == [True, False, False, False, False]
    assert masks["call"].tolist() == [False, True, False, False, True]
    assert masks["none"].tolist() == [False, False, True, False, False]
    assert masks["hora"].tolist() == [False, False, False, True, False]
    assert masks["chi_low"].tolist() == [False, True, False, False, False]
    assert masks["chi_mid"].tolist() == [False, False, False, False, True]
    assert masks["chi_high"].tolist() == [False, False, False, False, False]


def test_build_slice_report_computes_per_slice_accuracy():
    report = build_slice_report(
        action_idx_target=np.array([34, 35, 44, 42], dtype=np.int16),
        action_idx_pred=np.array([34, 36, 44, 41], dtype=np.int16),
        special_candidate_type_id=np.array(
            [
                [XMODEL1_SPECIAL_TYPE_REACH, -1],
                [XMODEL1_SPECIAL_TYPE_CHI_LOW, -1],
                [XMODEL1_SPECIAL_TYPE_NONE, -1],
                [XMODEL1_SPECIAL_TYPE_HORA, -1],
            ],
            dtype=np.int16,
        ),
        chosen_special_candidate_idx=np.array([0, 0, 0, 0], dtype=np.int16),
    )

    assert report["reach"].count == 1
    assert report["reach"].accuracy == 1.0
    assert report["call"].count == 1
    assert report["call"].accuracy == 0.0
    assert report["none"].count == 1
    assert report["none"].accuracy == 1.0
    assert report["hora"].count == 1
    assert report["hora"].accuracy == 0.0


def test_merge_slice_reports_accumulates_counts_across_batches():
    first = build_slice_report(
        action_idx_target=np.array([34, 35], dtype=np.int16),
        action_idx_pred=np.array([34, 36], dtype=np.int16),
        special_candidate_type_id=np.array(
            [
                [XMODEL1_SPECIAL_TYPE_REACH, -1],
                [XMODEL1_SPECIAL_TYPE_CHI_LOW, -1],
            ],
            dtype=np.int16,
        ),
        chosen_special_candidate_idx=np.array([0, 0], dtype=np.int16),
    )
    second = build_slice_report(
        action_idx_target=np.array([44, 42], dtype=np.int16),
        action_idx_pred=np.array([44, 41], dtype=np.int16),
        special_candidate_type_id=np.array(
            [
                [XMODEL1_SPECIAL_TYPE_NONE, -1],
                [XMODEL1_SPECIAL_TYPE_HORA, -1],
            ],
            dtype=np.int16,
        ),
        chosen_special_candidate_idx=np.array([0, 0], dtype=np.int16),
    )

    merged = merge_slice_reports(first, second)

    assert merged["reach"].count == 1
    assert merged["reach"].accuracy == 1.0
    assert merged["call"].count == 1
    assert merged["call"].accuracy == 0.0
    assert merged["none"].count == 1
    assert merged["none"].accuracy == 1.0
    assert merged["hora"].count == 1
    assert merged["hora"].accuracy == 0.0
