from mahjong_env.replay import (
    PendingValueSample,
    ReplaySample,
    _finalize_aux_targets,
    _label_matches_legal,
)
from mahjong_env.replay_normalizer import (
    normalize_replay_events,
)


# =============================================================================
# 辅助目标回填
# =============================================================================

def _sample(actor: int) -> PendingValueSample:
    return PendingValueSample(
        sample=ReplaySample(
            state={},
            actor=actor,
            actor_name=f"p{actor}",
            label_action={"type": "dahai", "actor": actor},
            legal_actions=[],
            value_target=0.0,
        ),
        round_step_index=0,
    )


def test_finalize_aux_targets_hora_round():
    pending = [_sample(i) for i in range(4)]
    terminal = {"type": "hora", "actor": 1, "target": 2, "deltas": [-1000, 7700, -7700, 1000]}

    _finalize_aux_targets(pending, terminal)

    assert pending[1].sample.win_target == 1.0
    assert pending[2].sample.dealin_target == 1.0
    assert pending[0].sample.win_target == 0.0
    assert pending[0].sample.dealin_target == 0.0
    assert pending[1].sample.score_delta_target == 7700 / 30000.0


def test_finalize_aux_targets_no_terminal():
    pending = [_sample(i) for i in range(2)]

    _finalize_aux_targets(pending, None)

    for p in pending:
        assert p.sample.score_delta_target == 0.0
        assert p.sample.win_target == 0.0
        assert p.sample.dealin_target == 0.0
        assert p.sample.ryukyoku_tenpai_target == 0.0


def test_finalize_aux_targets_ryukyoku_tracks_tenpai_players():
    pending = [_sample(i) for i in range(4)]
    terminal = {"type": "ryukyoku", "deltas": [0, 0, 0, 0], "tenpai_players": [1, 3]}

    _finalize_aux_targets(pending, terminal)

    assert pending[0].sample.ryukyoku_tenpai_target == 0.0
    assert pending[1].sample.ryukyoku_tenpai_target == 1.0
    assert pending[2].sample.ryukyoku_tenpai_target == 0.0
    assert pending[3].sample.ryukyoku_tenpai_target == 1.0


# =============================================================================
# replay label 与 legal set 的语义匹配
# 这些回归对应外部牌谱字段口径与 battle canonical action 不一致的兼容层。
# =============================================================================

def test_label_matches_legal_allows_ankan_label_without_pai_field():
    label = {
        "type": "ankan",
        "actor": 3,
        "consumed": ["1p", "1p", "1p", "1p"],
    }
    legal_actions = [
        {
            "type": "ankan",
            "actor": 3,
            "pai": "1p",
            "consumed": ["1p", "1p", "1p", "1p"],
        }
    ]

    assert _label_matches_legal(label, legal_actions) is True


def test_label_matches_legal_allows_hora_label_without_pai_field():
    label = {
        "type": "hora",
        "actor": 0,
        "target": 1,
        "deltas": [3900, -2900, 0, 0],
        "ura_markers": [],
    }
    legal_actions = [
        {
            "type": "hora",
            "actor": 0,
            "pai": "8m",
            "target": 1,
        }
    ]

    assert _label_matches_legal(label, legal_actions) is True


def test_label_matches_legal_hora_pai_mismatch_allowed():
    """hora label 的 pai 与 legal 的 pai 不同，但 target 相同时应允许（label 可能缺 pai）。"""
    label = {
        "type": "hora",
        "actor": 0,
        "target": 1,
    }
    legal_actions = [
        {
            "type": "hora",
            "actor": 0,
            "pai": "5mr",
            "target": 1,
        }
    ]

    assert _label_matches_legal(label, legal_actions) is True


def test_label_matches_legal_chi_consumed_aka_family_equivalent():
    label = {
        "type": "chi",
        "actor": 1,
        "target": 0,
        "pai": "4p",
        "consumed": ["5p", "6p"],
    }
    legal_actions = [
        {
            "type": "chi",
            "actor": 1,
            "target": 0,
            "pai": "4p",
            "consumed": ["5pr", "6p"],
        }
    ]

    assert _label_matches_legal(label, legal_actions) is True


def test_label_matches_legal_dahai_tsumogiri_mismatch_allowed():
    label = {
        "type": "dahai",
        "actor": 1,
        "pai": "3p",
        "tsumogiri": False,
    }
    legal_actions = [
        {
            "type": "dahai",
            "actor": 1,
            "pai": "3p",
            "tsumogiri": True,
        }
    ]

    assert _label_matches_legal(label, legal_actions) is True


def test_label_matches_legal_ankan_pai_aka_family_equivalent():
    label = {
        "type": "ankan",
        "actor": 3,
        "consumed": ["5m", "5m", "5m", "5mr"],
    }
    legal_actions = [
        {
            "type": "ankan",
            "actor": 3,
            "pai": "5mr",
            "consumed": ["5mr", "5m", "5m", "5m"],
        }
    ]

    assert _label_matches_legal(label, legal_actions) is True


def test_label_matches_legal_kakan_pai_aka_family_equivalent():
    label = {
        "type": "kakan",
        "actor": 2,
        "pai": "5p",
        "consumed": ["5p", "5p", "5p"],
    }
    legal_actions = [
        {
            "type": "kakan",
            "actor": 2,
            "pai": "5pr",
            "consumed": ["5p", "5p", "5p"],
        }
    ]

    assert _label_matches_legal(label, legal_actions) is True


def test_label_matches_legal_reach_requires_explicit_reach_action():
    label = {
        "type": "reach",
        "actor": 2,
    }
    legal_actions = [
        {"type": "reach", "actor": 2},
        {"type": "dahai", "actor": 2, "pai": "6s", "tsumogiri": True},
    ]

    assert _label_matches_legal(label, legal_actions) is True


def test_label_matches_legal_reach_does_not_match_plain_discard_set():
    label = {
        "type": "reach",
        "actor": 2,
    }
    legal_actions = [
        {"type": "hora", "actor": 2, "pai": "6s", "target": 2},
        {"type": "dahai", "actor": 2, "pai": "6s", "tsumogiri": True},
        {"type": "dahai", "actor": 2, "pai": "9p", "tsumogiri": False},
    ]

    assert _label_matches_legal(label, legal_actions) is False


def test_normalize_replay_events_enriches_self_hora_with_haitei_flag_and_pai(monkeypatch):
    from mahjong_env.state import apply_event as real_apply_event

    def patched_apply_event(state, event):
        real_apply_event(state, event)
        if event.get("type") == "tsumo" and event.get("actor") == 1:
            state.remaining_wall = 0

    monkeypatch.setattr("mahjong_env.replay_normalizer.apply_event", patched_apply_event)

    events = [
        {"type": "start_game", "names": ["P0", "P1", "P2", "P3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "names": ["P0", "P1", "P2", "P3"],
            "dora_marker": "1m",
            "tehais": [
                ["1m"] * 13,
                ["2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "4m"],
                ["3m"] * 13,
                ["4m"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 1, "pai": "6m"},
        {"type": "hora", "actor": 1, "target": 1, "deltas": [0, 1000, -500, -500], "ura_markers": []},
    ]

    normalized = normalize_replay_events(events)
    hora = normalized[-1]
    assert hora["type"] == "hora"
    assert hora["pai"] == "6m"
    assert hora["is_haitei"] is True


def test_normalize_replay_events_enriches_discard_hora_with_houtei_flag_and_pai(monkeypatch):
    from mahjong_env.state import apply_event as real_apply_event

    def patched_apply_event(state, event):
        real_apply_event(state, event)
        if event.get("type") == "dahai" and event.get("actor") == 1:
            state.remaining_wall = 0

    monkeypatch.setattr("mahjong_env.replay_normalizer.apply_event", patched_apply_event)

    events = [
        {"type": "start_game", "names": ["P0", "P1", "P2", "P3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "names": ["P0", "P1", "P2", "P3"],
            "dora_marker": "1m",
            "tehais": [
                ["1m"] * 13,
                ["2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "4m"],
                ["3m"] * 13,
                ["4m"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 1, "pai": "6m"},
        {"type": "dahai", "actor": 1, "pai": "4m", "tsumogiri": False},
        {"type": "hora", "actor": 2, "target": 1, "deltas": [0, -1000, 1000, 0], "ura_markers": []},
    ]

    normalized = normalize_replay_events(events)
    hora = normalized[-1]
    assert hora["type"] == "hora"
    assert hora["pai"] == "4m"
    assert hora["is_houtei"] is True


def test_normalize_replay_events_does_not_mark_houtei_when_remaining_wall_not_zero():
    events = [
        {"type": "start_game", "names": ["P0", "P1", "P2", "P3"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "names": ["P0", "P1", "P2", "P3"],
            "dora_marker": "1m",
            "tehais": [
                ["1m"] * 13,
                ["2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "2m", "4m"],
                ["3m"] * 13,
                ["4m"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 1, "pai": "6m"},
        {"type": "dahai", "actor": 1, "pai": "4m", "tsumogiri": False},
        {"type": "hora", "actor": 2, "target": 1, "deltas": [0, -1000, 1000, 0], "ura_markers": []},
    ]

    normalized = normalize_replay_events(events)
    hora = normalized[-1]
    assert hora["type"] == "hora"
    assert hora["pai"] == "4m"
    assert "is_houtei" not in hora
