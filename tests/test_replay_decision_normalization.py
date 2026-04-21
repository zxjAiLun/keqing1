from replay.normalize import normalize_replay_decisions


def test_normalize_replay_decisions_fills_explicit_none_gt_action():
    decisions = {
        "player_id": 2,
        "log": [
            {
                "is_obs": False,
                "chosen": {"type": "none"},
                "gt_action": None,
                "candidates": [
                    {"action": {"type": "none"}},
                    {"action": {"type": "pon", "actor": 2, "target": 1, "pai": "P", "consumed": ["P", "P"]}},
                ],
            },
            {
                "is_obs": True,
                "chosen": {"type": "dahai", "actor": 1, "pai": "P"},
                "gt_action": {"type": "dahai", "actor": 1, "pai": "P"},
            },
        ],
    }

    normalized = normalize_replay_decisions(decisions)

    assert normalized["log"][0]["gt_action"] == {"type": "none", "actor": 2}
    assert normalized["total_ops"] == 1
    assert normalized["match_count"] == 1


def test_normalize_replay_decisions_fills_response_gt_action_from_confirmed_action():
    decisions = {
        "player_id": 1,
        "log": [
            {
                "is_obs": False,
                "chosen": {
                    "type": "chi",
                    "actor": 1,
                    "target": 0,
                    "pai": "4p",
                    "consumed": ["5p", "6p"],
                },
                "gt_action": None,
                "candidates": [
                    {"action": {"type": "none"}},
                    {"action": {"type": "chi", "actor": 1, "target": 0, "pai": "4p", "consumed": ["5pr", "6p"]}},
                ],
            },
            {
                "is_obs": True,
                "chosen": {"type": "chi", "actor": 1, "target": 0, "pai": "4p", "consumed": ["5pr", "6p"]},
                "gt_action": {"type": "chi", "actor": 1, "target": 0, "pai": "4p", "consumed": ["5pr", "6p"]},
            },
        ],
    }

    normalized = normalize_replay_decisions(decisions, meta={"bot_type": "xmodel1"})

    assert normalized["log"][0]["gt_action"]["type"] == "chi"
    assert normalized["bot_type"] == "xmodel1"
    assert normalized["match_count"] == 1


def test_normalize_replay_decisions_marks_unconfirmed_response_as_none():
    decisions = {
        "player_id": 2,
        "log": [
            {
                "is_obs": False,
                "chosen": {
                    "type": "pon",
                    "actor": 2,
                    "target": 1,
                    "pai": "9s",
                    "consumed": ["9s", "9s"],
                },
                "gt_action": None,
                "last_discard": {"actor": 1, "pai": "9s"},
                "candidates": [
                    {"action": {"type": "pon", "actor": 2, "target": 1, "pai": "9s", "consumed": ["9s", "9s"]}},
                    {"action": {"type": "none"}},
                ],
            },
            {
                "is_obs": False,
                "chosen": {"type": "dahai", "actor": 2, "pai": "9s", "tsumogiri": False},
                "gt_action": {"type": "dahai", "actor": 2, "pai": "2m", "tsumogiri": False},
                "tsumo_pai": "7s",
            },
        ],
    }

    normalized = normalize_replay_decisions(decisions)

    assert normalized["log"][0]["gt_action"] == {"type": "none", "actor": 2}
    assert normalized["match_count"] == 0
