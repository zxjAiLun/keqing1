from __future__ import annotations

import json

from scripts.mortal import prepare_reviewer_teacher_probe
from tools.mjai_jsonl_to_tenhou6 import convert_mjai_jsonl_to_tenhou6


def test_mjai_jsonl_to_tenhou6_converts_core_events() -> None:
    events = [
        {"type": "start_game", "names": ["A", "B", "C", "D"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "4m",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5mr", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "E"],
                ["1s"] * 13,
                ["2s"] * 13,
                ["3s"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "6p"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": False},
        {"type": "tsumo", "actor": 1, "pai": "9p"},
        {"type": "dahai", "actor": 1, "pai": "9p", "tsumogiri": True},
        {"type": "pon", "actor": 3, "target": 1, "pai": "9p", "consumed": ["9p", "9p"]},
        {"type": "dahai", "actor": 3, "pai": "3s", "tsumogiri": False},
        {"type": "hora", "actor": 0, "target": 3, "deltas": [8000, 0, 0, -8000], "ura_markers": ["1p"]},
        {"type": "end_kyoku"},
        {"type": "end_game"},
    ]

    tenhou6 = convert_mjai_jsonl_to_tenhou6(events)
    kyoku = tenhou6["log"][0]

    assert tenhou6["name"] == ["A", "B", "C", "D"]
    assert kyoku[0] == [0, 0, 0]
    assert kyoku[2] == [14]
    assert kyoku[3] == [21]
    assert kyoku[4][4] == 51
    assert "r41" in kyoku[6]
    assert 60 in kyoku[9]
    assert kyoku[-1] == [["和了", [8000, 0, 0, -8000], []]]


def test_prepare_reviewer_teacher_probe_dry_run(tmp_path) -> None:
    log = tmp_path / "game.jsonl"
    log.write_text(
        "\n".join(
            json.dumps(event)
            for event in [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {"type": "end_game"},
            ]
        ),
        encoding="utf-8",
    )

    summary = prepare_reviewer_teacher_probe.prepare_probe(
        logs=[str(tmp_path / "*.jsonl")],
        output_root=tmp_path / "out",
        experiment_id="R0_test",
        limit=1,
        target_players=[0, 2],
        target_player_name=None,
        networks=["3.0", "4.1b"],
        validate_convlog=False,
        dry_run=True,
    )

    assert summary["experiment_id"] == "R0_test"
    assert summary["source_count"] == 1
    assert summary["rows"][0]["target_players"] == [0, 2]
    assert summary["rows"][0]["networks"] == ["3.0", "4.1b"]
    assert not (tmp_path / "out").exists()


def test_prepare_reviewer_teacher_probe_can_target_unique_player_name(tmp_path) -> None:
    log = tmp_path / "game.jsonl"
    events = [
        {"type": "start_game", "names": ["champion", "challenger", "champion", "champion"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [["1m"] * 13, ["2m"] * 13, ["3m"] * 13, ["4m"] * 13],
        },
        {"type": "ryukyoku", "deltas": [0, 0, 0, 0]},
        {"type": "end_kyoku"},
    ]
    log.write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")

    summary = prepare_reviewer_teacher_probe.prepare_probe(
        logs=[str(log)],
        output_root=tmp_path / "out",
        experiment_id="R0_test",
        limit=1,
        target_players=[0],
        target_player_name="challenger",
        networks=["4.1b"],
        validate_convlog=False,
        dry_run=False,
    )

    manifest = (tmp_path / "out" / "R0_test" / "manifest.jsonl").read_text(encoding="utf-8")
    row = json.loads(manifest)
    assert row["target_players"] == [1]
    assert row["target_player_name"] == "challenger"
    assert summary["source_count"] == 1
