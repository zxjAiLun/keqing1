from __future__ import annotations

from pathlib import Path

from evals.xmodel1.runner import evaluate_case_records, load_case_records, record_from_case


def test_load_case_records_and_evaluate(tmp_path: Path):
    path = tmp_path / "cases.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"case_id":"c1","category":"tenpai","preferred_actions":["5m"],"acceptable_actions":["5m","2p"],"unacceptable_actions":["E"]}',
                '{"case_id":"c2","category":"yakuhai","preferred_actions":["2p"],"acceptable_actions":["2p"],"unacceptable_actions":["P"]}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rows = load_case_records(path)
    scored = [
        record_from_case(rows[0], chosen_action="5m"),
        record_from_case(rows[1], chosen_action="P"),
    ]
    summary = evaluate_case_records(scored)
    assert summary.total == 2
    assert summary.preferred == 1
    assert summary.blunder == 1
    assert summary.preferred_rate == 0.5
    assert summary.blunder_rate == 0.5
