from replay.external_reports import (
    build_mortal_teacher_report,
    build_naga_teacher_reports,
    resolve_external_report_url,
)
from replay.server import _attach_teacher_report_overlays


def _discard_entry(step: int, pai: str) -> dict:
    return {
        "step": step,
        "bakaze": "E",
        "kyoku": 1,
        "honba": 0,
        "gt_action": {"type": "dahai", "actor": 0, "pai": pai, "tsumogiri": False},
        "chosen": {"type": "dahai", "actor": 0, "pai": pai, "tsumogiri": False},
        "candidates": [
            {"action": {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False}},
            {"action": {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": False}},
        ],
    }


def test_resolves_viewer_urls_to_report_json() -> None:
    assert resolve_external_report_url(
        "https://naga.dmv.nico/htmls/report_viewer.html?report_id=abc&tw=0",
        "naga",
    ) == "https://naga.dmv.nico/reports/abc.json"
    assert resolve_external_report_url(
        "https://mjai.ekyu.moe/killerducky/?data=/report/abc.json",
        "mortal",
    ) == "https://mjai.ekyu.moe/report/abc.json"


def test_naga_report_becomes_two_step_aligned_teacher_models() -> None:
    decisions = {
        "log": [
            _discard_entry(4, "2m"),
            {
                "step": 12,
                "bakaze": "E",
                "kyoku": 1,
                "honba": 0,
                "gt_action": {"type": "none", "actor": 0},
                "chosen": {"type": "none"},
                "candidates": [
                    {"action": {"type": "none"}},
                    {
                        "action": {
                            "type": "chi",
                            "actor": 0,
                            "target": 3,
                            "pai": "1m",
                            "consumed": ["2m", "3m"],
                        }
                    },
                ],
            },
        ]
    }
    tile_probs_a = [0] * 34
    tile_probs_b = [0] * 34
    tile_probs_a[0], tile_probs_a[1] = 2000, 8000
    tile_probs_b[0], tile_probs_b[1] = 7000, 3000
    raw = {
        "naga_types": {"0": "ニシキ", "1": "カガシ"},
        "pred": [[
            {"info": {"msg": {
                "type": "start_kyoku",
                "bakaze": "E",
                "kyoku": 1,
                "honba": 0,
                "scores": [],
                "tehais": [[], [], [], []],
            }}},
            {
                "info": {"msg": {"type": "tsumo", "actor": 0, "real_dahai": "2m"}},
                "dahai_pred": [tile_probs_a, tile_probs_b],
            },
            {
                "info": {"msg": {"type": "dahai", "actor": 3, "pai": "1m"}},
                "huro": {"0": [{"0": 6000, "1": 4000}, {"0": 1000, "1": 9000}]},
            },
        ]],
    }

    reports = build_naga_teacher_reports(raw, decisions, player_id=0, replay_id="replay_test")

    assert [item["review"]["model_tag"] for item in reports] == ["NAGA ニシキ", "NAGA カガシ"]
    first_entries = reports[0]["review"]["kyokus"][0]["entries"]
    second_entries = reports[1]["review"]["kyokus"][0]["entries"]
    assert [entry["step"] for entry in first_entries] == [4, 12]
    assert first_entries[0]["expected"]["pai"] == "2m"
    assert second_entries[0]["expected"]["pai"] == "1m"
    assert first_entries[1]["expected"]["type"] == "none"
    assert second_entries[1]["expected"]["type"] == "chi"


def test_mortal_report_keeps_review_and_uses_explicit_label() -> None:
    raw = {
        "player_id": 0,
        "review": {"model_tag": "4.1c", "kyokus": [{"entries": []}]},
    }

    decisions = {
        "log": [{
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "scores": [25000] * 4,
            "hand": ["1m"] * 13,
        }]
    }
    report = build_mortal_teacher_report(raw, decisions, player_id=0, replay_id="replay_test")

    assert report["review"]["model_tag"] == "Mortal 4.1c"
    assert report["source"] == "mortal"


def test_naga_reach_probability_expands_to_tile_dama_and_tile_reach(tmp_path) -> None:
    hand = ["1s", "4p", "4p", "5p", "5pr", "5s", "6m", "6m", "7p", "7p", "8m", "8p", "8p"]
    decisions = {
        "player_id": 3,
        "log": [
            {
                "step": 417,
                "bakaze": "S",
                "kyoku": 2,
                "honba": 0,
                "scores": [25300, 10500, 40000, 24200],
                "dora_markers": ["9m"],
                "hand": hand,
                "tsumo_pai": "5s",
                "gt_action": {"type": "reach", "actor": 3},
                "chosen": {"type": "dahai", "actor": 3, "pai": "8m", "tsumogiri": False},
                "candidates": [
                    {"action": {"type": "dahai", "actor": 3, "pai": "8m", "tsumogiri": False}},
                    {"action": {"type": "reach", "actor": 3}},
                    {"action": {"type": "dahai", "actor": 3, "pai": "1s", "tsumogiri": False}},
                ],
            },
            {
                "step": 418,
                "bakaze": "S",
                "kyoku": 2,
                "honba": 0,
                "gt_action": {"type": "dahai", "actor": 3, "pai": "1s", "tsumogiri": False},
                "chosen": {"type": "dahai", "actor": 3, "pai": "8m", "tsumogiri": False},
                "candidates": [],
            },
        ],
    }
    nishiki = [0] * 34
    kagashi = [0] * 34
    nishiki[7], nishiki[18], nishiki[22] = 2393, 7605, 1
    kagashi[7], kagashi[18], kagashi[22] = 4224, 5774, 1
    raw = {
        "naga_types": {"0": "nishiki", "1": "kagashi"},
        "pred": [[
            {"info": {"msg": {
                "type": "start_kyoku", "bakaze": "S", "kyoku": 2, "honba": 0,
                "scores": [25300, 10500, 40000, 24200], "dora_marker": "9m",
                "tehais": [[], [], [], hand],
            }}},
            {
                "info": {"msg": {
                    "type": "tsumo", "actor": 3, "pai": "5s", "real_dahai": "1s",
                    "next_tsumogiri": False,
                }},
                "dahai_pred": [nishiki, kagashi],
                "reach": [3064, 3388],
            },
            {"info": {"msg": {"type": "reach", "actor": 3}}},
            {"info": {"msg": {"type": "dahai", "actor": 3, "pai": "1s"}}},
        ]],
    }

    reports = build_naga_teacher_reports(raw, decisions, player_id=3, replay_id="replay_test")
    entry = reports[0]["review"]["kyokus"][0]["entries"][0]

    assert entry["step"] == 417
    assert entry["display_mode"] == "joint_reach_dahai"
    assert entry["actual"] == {"type": "reach", "actor": 3, "pai": "1s", "tsumogiri": False}
    assert [(item["action"]["type"], item["action"]["pai"]) for item in entry["details"]] == [
        ("dahai", "1s"), ("reach", "1s"), ("dahai", "8m"), ("reach", "8m"),
    ]
    probs = {(item["action"]["type"], item["action"]["pai"]): item["prob"] for item in entry["details"]}
    assert probs[("dahai", "8m")] == 0.2393 * 0.3064
    assert probs[("reach", "8m")] == 0.2393 * 0.6936
    assert probs[("dahai", "1s")] == 0.7605 * 0.3064
    assert probs[("reach", "1s")] == 0.7605 * 0.6936

    report_path = tmp_path / "naga.json"
    report_path.write_text(__import__("json").dumps(reports[0]), encoding="utf-8")
    attached = _attach_teacher_report_overlays(decisions, [report_path])
    root = attached["log"][0]
    eight_man = next(item for item in root["candidates"] if item["action"].get("pai") == "8m")
    reach = next(item for item in root["candidates"] if item["action"].get("type") == "reach")
    assert eight_man["teachers"][0]["prob"] == 0.2393
    assert round(reach["teachers"][0]["prob"], 4) == 0.6935
