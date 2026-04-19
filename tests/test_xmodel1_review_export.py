from pathlib import Path

from xmodel1.review_export import (
    ReviewCandidate,
    ReviewRecord,
    export_review_records,
    special_type_to_action_label,
    tile34_to_action_label,
    topk_candidates_from_row,
    topk_special_candidates_from_row,
)


def test_tile34_to_action_label_formats_discard():
    assert tile34_to_action_label(0).startswith("dahai:")


def test_special_type_to_action_label_formats_special_candidates():
    assert special_type_to_action_label(0) == "reach"


def test_topk_candidates_from_row_filters_padding_and_sorts():
    items = topk_candidates_from_row(
        scores=[0.2, 0.8, -1.0],
        candidate_tile_ids=[0, 1, -1],
        candidate_mask=[1, 1, 0],
        quality_scores=[0.1, 0.9, 0.0],
        rank_buckets=[1, 3, 0],
        hard_bad_flags=[0, 0, 0],
        k=2,
    )
    assert len(items) == 2
    assert items[0].score >= items[1].score
    assert items[0].action.startswith("dahai:")


def test_topk_special_candidates_from_row_filters_padding_and_sorts():
    items = topk_special_candidates_from_row(
        scores=[0.1, 0.9, -1.0],
        special_type_ids=[0, 11, -1],
        special_mask=[1, 1, 0],
        quality_scores=[0.2, 0.8, 0.0],
        rank_buckets=[1, 3, 0],
        hard_bad_flags=[0, 0, 0],
        k=2,
    )
    assert len(items) == 2
    assert items[0].score >= items[1].score
    assert items[0].action in {"reach", "none"}


def test_export_review_records_writes_jsonl(tmp_path: Path):
    out = tmp_path / "review.jsonl"
    export_review_records(
        [
            ReviewRecord(
                sample_id="s1",
                replay_id="r1",
                category="smoke",
                chosen_action="dahai:1m",
                top_k=(ReviewCandidate(action="dahai:1m", score=1.0),),
            )
        ],
        out,
    )
    lines = out.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"sample_id": "s1"' in lines[0]
