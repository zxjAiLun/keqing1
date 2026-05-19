from __future__ import annotations

import json

from scripts.mortal import archive_reviewer_reports
from scripts.mortal import parse_reviewer_teacher_reports
from scripts.mortal import prepare_reviewer_teacher_probe
from scripts.mortal import submit_reviewer_teacher_probe
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


def test_archive_reviewer_report_downloads_json_and_appends_manifest(tmp_path) -> None:
    source = tmp_path / "input" / "0001_game.tenhou6.json"
    source.parent.mkdir()
    source.write_text('{"name":["A","B","C","D"],"log":[]}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "source_log": "game.json.gz",
                "tenhou6_path": str(source),
                "target_players": [0],
                "networks": ["3.0"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    seen_urls: list[str] = []

    def fake_downloader(url: str) -> bytes:
        seen_urls.append(url)
        return json.dumps(
            {
                "review": [
                    {
                        "actual_action": "dahai",
                        "details": [{"action": "dahai", "q_value": 0.25, "score": 0.7}],
                    }
                ]
            }
        ).encode("utf-8")

    row = archive_reviewer_reports.archive_report(
        source_manifest=manifest,
        source_index=0,
        source_tenhou6=None,
        target_player=0,
        network="3.0",
        report="https://mjai.ekyu.moe/report/abc123",
        output_dir=tmp_path / "external",
        dry_run=False,
        downloader=fake_downloader,
    )

    assert seen_urls == ["https://mjai.ekyu.moe/report/abc123.json"]
    report_path = tmp_path / "external" / "reports" / "0001_game__3.0__p0.json"
    assert report_path.exists()
    assert row["report_id"] == "abc123"
    assert row["report_json_path"] == str(report_path)
    assert row["report_schema_summary"]["has_detail_like_key"] is True
    assert row["report_schema_summary"]["has_q_or_score_like_key"] is True
    assert row["report_schema_summary"]["has_action_like_key"] is True

    report_manifest = tmp_path / "external" / "report_manifest.jsonl"
    manifest_row = json.loads(report_manifest.read_text(encoding="utf-8"))
    assert manifest_row["network"] == "3.0"
    assert manifest_row["target_player"] == 0


def test_archive_reviewer_report_dry_run_accepts_json_url(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"source_log": "g", "tenhou6_path": "x.tenhou6.json"}) + "\n", encoding="utf-8")

    row = archive_reviewer_reports.archive_report(
        source_manifest=manifest,
        source_index=0,
        source_tenhou6=None,
        target_player=2,
        network="4.1b",
        report="https://mjai.ekyu.moe/report/report-id_42.json",
        output_dir=tmp_path / "external",
        dry_run=True,
    )

    assert row["dry_run"] is True
    assert row["report_id"] == "report-id_42"
    assert row["report_page_url"] == "https://mjai.ekyu.moe/report/report-id_42"
    assert row["report_json_url"] == "https://mjai.ekyu.moe/report/report-id_42.json"


def test_archive_reviewer_report_dry_run_accepts_killerducky_url(tmp_path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"source_log": "g", "tenhou6_path": "x.tenhou6.json"}) + "\n", encoding="utf-8")

    row = archive_reviewer_reports.archive_report(
        source_manifest=manifest,
        source_index=0,
        source_tenhou6=None,
        target_player=0,
        network="4.1b",
        report="https://mjai.ekyu.moe/killerducky/?data=/report/44cbfc905f0667fd.json",
        output_dir=tmp_path / "external",
        dry_run=True,
    )

    assert row["report_id"] == "44cbfc905f0667fd"
    assert row["report_json_path"] == str(tmp_path / "external" / "reports" / "x__4.1b__p0.json")


def _fake_report(*, expected_pai: str, expected_prob: float, actual_prob: float, is_equal: bool) -> dict:
    actual = {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False}
    expected = {"type": "dahai", "actor": 0, "pai": expected_pai, "tsumogiri": False}
    details = [
        {"action": expected, "q_value": 0.5, "prob": expected_prob},
        {"action": actual, "q_value": 0.2, "prob": actual_prob},
    ]
    return {
        "engine": "Mortal",
        "game_length": "Hanchan",
        "version": "1.5.10",
        "player_id": 0,
        "review": {
            "total_reviewed": 1,
            "total_matches": 1 if is_equal else 0,
            "temperature": 0.1,
            "kyokus": [
                {
                    "kyoku": 0,
                    "honba": 0,
                    "entries": [
                        {
                            "junme": 3,
                            "tiles_left": 60,
                            "expected": expected,
                            "actual": actual,
                            "is_equal": is_equal,
                            "details": details,
                            "shanten": 2,
                            "at_self_chi_pon": False,
                            "at_self_riichi": False,
                        }
                    ],
                }
            ],
        },
    }


def test_parse_reviewer_teacher_reports_outputs_decision_and_alignment_tables(tmp_path) -> None:
    source = tmp_path / "input" / "0001_game.tenhou6.json"
    source.parent.mkdir()
    source.write_text("{}", encoding="utf-8")
    reports = tmp_path / "reports"
    reports.mkdir()
    report_30 = reports / "game__3.0__p0.json"
    report_41b = reports / "game__4.1b__p0.json"
    report_30.write_text(json.dumps(_fake_report(expected_pai="1m", expected_prob=0.9, actual_prob=0.9, is_equal=True)))
    report_41b.write_text(json.dumps(_fake_report(expected_pai="2m", expected_prob=0.85, actual_prob=0.10, is_equal=False)))

    manifest = tmp_path / "report_manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "source_log": "game.json.gz",
                    "source_tenhou6_path": str(source),
                    "target_player": 0,
                    "network": "3.0",
                    "report_id": "r30",
                    "report_json_path": str(report_30),
                },
                {
                    "source_log": "game.json.gz",
                    "source_tenhou6_path": str(source),
                    "target_player": 0,
                    "network": "4.1b",
                    "report_id": "r41b",
                    "report_json_path": str(report_41b),
                },
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = parse_reviewer_teacher_reports.parse_reports(
        report_manifest=manifest,
        output_dir=tmp_path / "out",
        high_confidence_prob=0.6,
        high_confidence_margin=0.2,
        top_k=10,
    )

    assert summary["decision_count"] == 2
    assert summary["network_summaries"]["3.0"]["match_rate"] == 1.0
    assert summary["network_summaries"]["4.1b"]["match_rate"] == 0.0
    assert summary["network_summaries"]["4.1b"]["high_confidence_disagreement_count"] == 1
    assert summary["aligned_summary"]["aligned_entry_count"] == 1
    assert summary["aligned_summary"]["teacher_agreement_rate"] == 0.0
    assert summary["aligned_summary"]["match_pattern_counts"] == {"matches_3_0_only": 1}

    decisions = [
        json.loads(line)
        for line in (tmp_path / "out" / "decision_table.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    aligned = json.loads((tmp_path / "out" / "aligned_decisions.jsonl").read_text(encoding="utf-8"))
    top = json.loads((tmp_path / "out" / "top_disagreements.jsonl").read_text(encoding="utf-8"))
    assert len(decisions) == 2
    assert aligned["match_actual_3_0"] is True
    assert aligned["match_actual_4_1b"] is False
    assert aligned["teacher_agree"] is False
    assert top["network"] == "4.1b"
    assert top["expected_prob"] == 0.85


def test_submit_reviewer_teacher_probe_parses_curl_and_dry_runs(tmp_path) -> None:
    tenhou6 = tmp_path / "0001_game.tenhou6.json"
    tenhou6.write_text('{"name":["A","B","C","D"],"log":[]}\n', encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "source_log": "game.json.gz",
                "tenhou6_path": str(tenhou6),
                "target_players": [0],
                "networks": ["3.0", "4.1b"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    curl_file = tmp_path / "submit.curl"
    curl_file.write_text(
        "curl 'https://mjai.ekyu.moe/review' "
        "-H 'User-Agent: Mozilla/5.0' "
        "-H 'Cookie: session=abc' "
        "--data-raw 'input-method=tenhou6&player-id=0&engine=mortal&mortal-model-tag=3.0&ui=killerducky&lang=en&cf-turnstile-response=tok'",
        encoding="utf-8",
    )

    parsed = submit_reviewer_teacher_probe.parse_curl_file(curl_file)
    assert parsed["url"] == "https://mjai.ekyu.moe/review"
    assert parsed["headers"]["Cookie"] == "session=abc"
    assert parsed["form"]["cf-turnstile-response"] == "tok"

    summary = submit_reviewer_teacher_probe.submit_probe(
        input_manifest=manifest,
        output_dir=tmp_path / "out",
        submit_curl_file=curl_file,
        networks_override=["4.1b"],
        source_index=None,
        start_index=0,
        limit=1,
        sleep_seconds=0,
        poll_seconds=0,
        poll_attempts=1,
        skip_existing=True,
        dry_run=True,
    )

    assert summary["result_counts"] == {"dry_run": 1}
    row = summary["results"][0]
    assert row["network"] == "4.1b"
    assert row["target_player"] == 0
    assert row["has_captcha_response"] is True
    assert (tmp_path / "out" / "submit_manifest.jsonl").exists()


def test_submit_reviewer_teacher_probe_parses_windows_curl_and_start_index(tmp_path) -> None:
    first = tmp_path / "0001.tenhou6.json"
    second = tmp_path / "0002.tenhou6.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"source_log": "a", "tenhou6_path": str(first), "target_players": [0], "networks": ["3.0"]},
                {"source_log": "b", "tenhou6_path": str(second), "target_players": [1], "networks": ["3.0"]},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    curl_file = tmp_path / "submit_windows.curl"
    curl_file.write_text(
        'curl ^"https://mjai.ekyu.moe/review^" ^\n'
        '  -H ^"accept: */*^" ^\n'
        '  --data-raw ^"input-method=tenhou6^&player-id=0^&engine=mortal^&mortal-model-tag=3.0^&cf-turnstile-response=tok^"',
        encoding="utf-8",
    )

    parsed = submit_reviewer_teacher_probe.parse_curl_file(curl_file)
    assert parsed["url"] == "https://mjai.ekyu.moe/review"
    assert parsed["headers"]["accept"] == "*/*"
    assert parsed["form"]["cf-turnstile-response"] == "tok"

    summary = submit_reviewer_teacher_probe.submit_probe(
        input_manifest=manifest,
        output_dir=tmp_path / "out",
        submit_curl_file=curl_file,
        networks_override=None,
        source_index=None,
        start_index=1,
        limit=1,
        sleep_seconds=0,
        poll_seconds=0,
        poll_attempts=1,
        skip_existing=True,
        dry_run=True,
    )

    assert summary["result_counts"] == {"dry_run": 1}
    assert summary["results"][0]["source_index"] == 1
    assert summary["results"][0]["source_tenhou6_path"] == str(second)
    assert summary["results"][0]["target_player"] == 1


def test_submit_reviewer_teacher_probe_source_index_selects_original_manifest_row(tmp_path) -> None:
    first = tmp_path / "0001.tenhou6.json"
    second = tmp_path / "0002.tenhou6.json"
    first.write_text("{}", encoding="utf-8")
    second.write_text("{}", encoding="utf-8")
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"source_log": "a", "tenhou6_path": str(first), "target_players": [0], "networks": ["3.0"]},
                {"source_log": "b", "tenhou6_path": str(second), "target_players": [2], "networks": ["4.1b"]},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    curl_file = tmp_path / "submit.curl"
    curl_file.write_text("curl 'https://mjai.ekyu.moe/review' --data-raw 'cf-turnstile-response=tok'", encoding="utf-8")

    summary = submit_reviewer_teacher_probe.submit_probe(
        input_manifest=manifest,
        output_dir=tmp_path / "out",
        submit_curl_file=curl_file,
        networks_override=None,
        source_index=1,
        start_index=0,
        limit=None,
        sleep_seconds=0,
        poll_seconds=0,
        poll_attempts=1,
        skip_existing=True,
        dry_run=True,
    )

    assert len(summary["results"]) == 1
    assert summary["results"][0]["source_index"] == 1
    assert summary["results"][0]["source_tenhou6_path"] == str(second)
    assert summary["results"][0]["network"] == "4.1b"
    assert summary["results"][0]["target_player"] == 2


def test_submit_reviewer_teacher_probe_extracts_redirect_report_id() -> None:
    class Headers(dict):
        def get(self, key, default=None):  # type: ignore[override]
            return super().get(key, default)

    assert (
        submit_reviewer_teacher_probe.extract_report_id_from_response(
            Headers({"Location": "/killerducky/?data=/report/44cbfc905f0667fd.json"}),
            "",
            "https://mjai.ekyu.moe/review",
        )
        == "44cbfc905f0667fd"
    )
