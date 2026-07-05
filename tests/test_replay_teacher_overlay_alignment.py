import json

from replay.server import _attach_teacher_report_overlays


def test_teacher_overlay_aligns_by_global_step(tmp_path) -> None:
    decisions = {
        "player_id": 1,
        "log": [
            {
                "step": 1,
                "chosen": {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
                "gt_action": {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
                "candidates": [
                    {"action": {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False}}
                ],
            },
            {
                "step": 72,
                "chosen": {"type": "none"},
                "gt_action": {"type": "none", "actor": 1},
                "candidates": [
                    {"action": {"type": "none"}},
                    {
                        "action": {
                            "type": "chi",
                            "actor": 1,
                            "target": 0,
                            "pai": "2m",
                            "consumed": ["1m", "3m"],
                        }
                    },
                ],
            },
        ],
    }
    report = {
        "player_id": 1,
        "review": {
            "model_tag": "teacher",
            "kyokus": [
                {
                    "entries": [
                        {
                            "step": 72,
                            "actual": {"type": "none", "actor": 1},
                            "expected": {"type": "none"},
                            "details": [
                                {"action": {"type": "none"}, "q_value": 0.2, "prob": 0.9},
                                {
                                    "action": {
                                        "type": "chi",
                                        "actor": 1,
                                        "target": 0,
                                        "pai": "2m",
                                        "consumed": ["1m", "3m"],
                                    },
                                    "q_value": -2.0,
                                    "prob": 0.1,
                                },
                            ],
                        },
                        {
                            "step": 1,
                            "actual": {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
                            "expected": {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
                            "details": [
                                {
                                    "action": {"type": "dahai", "actor": 1, "pai": "1m", "tsumogiri": False},
                                    "q_value": 0.4,
                                    "prob": 1.0,
                                }
                            ],
                        },
                    ]
                }
            ],
        },
    }
    report_path = tmp_path / "teacher.json"
    report_path.write_text(json.dumps(report), encoding="utf-8")

    attached = _attach_teacher_report_overlays(decisions, [report_path])

    assert attached["teacher_review_overlays"][0]["attached_decision_count"] == 2
    assert attached["log"][0]["teacher_reviews"][0]["step"] == 1
    assert attached["log"][1]["teacher_reviews"][0]["step"] == 72
    assert attached["log"][1]["candidates"][1]["teachers"][0]["prob"] == 0.1
