from inference.review import same_action as _same_action
from replay.legacy_render import render_candidates_logit


def test_replay_same_action_ignores_dahai_tsumogiri_difference():
    chosen = {"type": "dahai", "actor": 1, "pai": "3p", "tsumogiri": False}
    gt = {"type": "dahai", "actor": 1, "pai": "3p", "tsumogiri": True}

    assert _same_action(chosen, gt) is True


def test_replay_same_action_matches_aka_equivalent_chi():
    chosen = {
        "type": "chi",
        "actor": 1,
        "target": 0,
        "pai": "4p",
        "consumed": ["5p", "6p"],
    }
    gt = {
        "type": "chi",
        "actor": 1,
        "target": 0,
        "pai": "4p",
        "consumed": ["5pr", "6p"],
    }

    assert _same_action(chosen, gt) is True


def test_render_candidates_logit_marks_equivalent_gt_action():
    candidate_action = {
        "type": "chi",
        "actor": 1,
        "target": 0,
        "pai": "4p",
        "consumed": ["5pr", "6p"],
    }
    chosen = {
        "type": "chi",
        "actor": 1,
        "target": 0,
        "pai": "4p",
        "consumed": ["5p", "6p"],
    }
    gt = {
        "type": "chi",
        "actor": 1,
        "target": 0,
        "pai": "4p",
        "consumed": ["5p", "6p"],
    }
    html = render_candidates_logit(
        [{"action": candidate_action, "logit": 1.25}],
        chosen,
        gt,
    )

    assert "✓Bot" in html
    assert "★玩家" in html
