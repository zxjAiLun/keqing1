import inference.scoring as inference_scoring
import numpy as np
import pytest
from keqing_core import (
    aggregate_keqingv4_continuation_scores,
    score_keqingv4_continuation_scenario,
)
from inference.default_context import DefaultDecisionContextBuilder
from training.cache_schema import KEQINGV4_SUMMARY_DIM
from inference import (
    DecisionContext,
    DecisionResult,
    DefaultActionScorer,
    DefaultRuntimeReviewExporter,
    action_label,
    action_primary_tile,
    ModelAuxOutputs,
    ModelForwardResult,
    ScoredCandidate,
    same_action,
    summarize_decision_matches,
    summarize_reach_followup,
    candidate_to_log_dict,
    Xmodel1RuntimeOutputs,
)
from mahjong_env.action_space import action_to_idx
from mahjong_env.event_history import compute_event_history
from mahjong_env.state import GameState


def _inject_shanten_waits_stub(snap, *, hand_list, melds_list, model_version):
    del hand_list, melds_list, model_version
    snap["shanten"] = 1
    snap["waits_count"] = 0
    snap["waits_tiles"] = [0] * 34


def test_inference_contract_dataclasses_roundtrip():
    aux = ModelAuxOutputs(
        score_delta=0.2,
        win_prob=0.4,
        dealin_prob=0.1,
        rank_probs=(0.1, 0.2, 0.3, 0.4),
        final_score_delta=-0.25,
        rank_pt_value=-0.4,
    )
    xmodel1 = Xmodel1RuntimeOutputs(
        discard_logits=np.array([1.0], dtype=np.float32),
        candidate_tile_id=np.array([0], dtype=np.int16),
        candidate_mask=np.array([1], dtype=np.uint8),
        response_logits=np.array([0.5], dtype=np.float32),
        response_action_idx=np.array([11], dtype=np.int16),
        response_action_mask=np.array([1], dtype=np.uint8),
        response_post_candidate_feat=np.zeros((1, 1), dtype=np.float32),
        response_post_candidate_mask=np.array([1], dtype=np.uint8),
        response_teacher_discard_idx=np.array([0], dtype=np.int16),
        win_prob=0.4,
        dealin_prob=0.1,
        pts_given_win=0.3,
        pts_given_dealin=0.2,
        opp_tenpai_probs=np.array([0.2, 0.1, 0.0], dtype=np.float32),
    )
    forward = ModelForwardResult(policy_logits=[1.0, 2.0], value=0.3, aux=aux, xmodel1=xmodel1)
    ctx = DecisionContext(
        actor=1,
        event={"type": "tsumo", "actor": 1, "pai": "5mr"},
        runtime_snap={"hand": ["1m"] * 14},
        model_snap={"hand": ["1m"] * 13, "tsumo_pai": "5mr"},
        legal_actions=[{"type": "reach", "actor": 1}],
    )
    candidate = ScoredCandidate(
        action={"type": "reach", "actor": 1},
        logit=-1.0,
        final_score=0.5,
        beam_score=0.5,
        meta={"source": "test"},
    )
    result = DecisionResult(
        chosen={"type": "reach", "actor": 1},
        candidates=[candidate],
        model_value=forward.value,
        model_aux=forward.aux,
    )

    assert ctx.actor == 1
    assert ctx.model_snap["tsumo_pai"] == "5mr"
    assert result.candidates[0].final_score == 0.5
    assert result.model_aux.win_prob == 0.4
    assert result.model_aux.rank_probs == (0.1, 0.2, 0.3, 0.4)
    assert forward.xmodel1 is not None


def test_default_context_builder_injects_keqingv4_event_history():
    builder = DefaultDecisionContextBuilder(
        model_version="keqingv4",
        riichi_state=None,
        inject_shanten_waits=_inject_shanten_waits_stub,
    )
    state = GameState()
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
            "dora_marker": "1m",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "2p", "3p", "4p"],
                ["1s"] * 13,
                ["2s"] * 13,
                ["3s"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 1, "pai": "6m"},
        {"type": "dahai", "actor": 1, "pai": "6m", "tsumogiri": True},
    ]
    assert builder.build(state, 0, events[0]) is None
    assert builder.build(state, 0, events[1]) is None
    assert builder.build(state, 0, events[2]) is None
    ctx = builder.build(state, 0, events[3])
    assert ctx is not None
    expected = compute_event_history(events, 3)
    assert np.array_equal(ctx.runtime_snap["event_history"], expected)
    assert np.array_equal(ctx.model_snap["event_history"], expected)


class _FakeAdapter:
    def __init__(self, forwards):
        self.forwards = list(forwards)
        self.calls = []

    def forward(self, snap: dict, actor: int):
        self.calls.append((snap, actor))
        return self.forwards.pop(0)


class _FakeV4Adapter(_FakeAdapter):
    model_version = "keqingv4"

    def resolve_runtime_v4_summaries(self, snap, actor, legal_actions=None):
        del snap, actor, legal_actions
        return (
            np.zeros((34, KEQINGV4_SUMMARY_DIM), dtype=np.float32),
            np.zeros((8, KEQINGV4_SUMMARY_DIM), dtype=np.float32),
            np.zeros((3, KEQINGV4_SUMMARY_DIM), dtype=np.float32),
        )


class _FakeXmodel1Adapter(_FakeAdapter):
    model_version = "xmodel1"


def _policy_with_scores(scores: dict[tuple[str, str | None], float]):
    import numpy as np

    logits = np.full((45,), -1e9, dtype=np.float32)
    for (atype, pai), score in scores.items():
        action = {"type": atype}
        if pai is not None:
            action["pai"] = pai
        if atype == "dahai":
            action["tsumogiri"] = False
        if atype == "reach":
            action["actor"] = 0
        logits[action_to_idx(action)] = score
    return logits


def _empty_policy():
    import numpy as np

    return np.full((45,), -1e9, dtype=np.float32)


def test_rust_continuation_scenario_scorer_matches_followup_formula():
    logits = _empty_policy()
    legal_actions = [
        {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": False},
        {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False},
    ]
    logits[action_to_idx(legal_actions[0])] = -0.5
    logits[action_to_idx(legal_actions[1])] = 1.25

    payload = score_keqingv4_continuation_scenario(
        "post_meld_followup",
        logits,
        legal_actions,
        value=0.0,
        score_delta=0.4,
        win_prob=0.2,
        dealin_prob=0.1,
        rank_pt_value=1.5,
        beam_lambda=1.0,
        score_delta_lambda=0.5,
        win_prob_lambda=0.25,
        dealin_prob_lambda=0.75,
        rank_pt_lambda=0.2,
    )

    expected = 1.25 + 0.5 * 0.4 + 0.25 * 0.2 - 0.75 * 0.1 + 0.2 * 1.5
    assert payload["best_action"]["pai"] == "9m"
    assert float(payload["score"]) == pytest.approx(expected)


@pytest.mark.parametrize("continuation_kind", ["reach_declaration", "state_value"])
def test_rust_continuation_scenario_state_value_path_includes_placement_bonus(continuation_kind: str):
    payload = score_keqingv4_continuation_scenario(
        continuation_kind,
        _empty_policy(),
        [],
        value=0.8,
        score_delta=0.2,
        win_prob=0.1,
        dealin_prob=0.4,
        rank_pt_value=2.0,
        beam_lambda=0.5,
        score_delta_lambda=1.0,
        win_prob_lambda=0.5,
        dealin_prob_lambda=0.25,
        rank_pt_lambda=0.3,
    )

    expected = 0.5 * 0.8 + 1.0 * 0.2 + 0.5 * 0.1 - 0.25 * 0.4 + 0.3 * 2.0
    assert payload["best_action"] is None
    assert float(payload["score"]) == pytest.approx(expected)


def test_rust_continuation_aggregation_matches_reach_and_weighted_paths():
    logits = _empty_policy()
    logits[action_to_idx({"type": "reach", "actor": 0})] = 0.8
    logits[action_to_idx({"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": False})] = 0.1
    logits[action_to_idx({"type": "daiminkan", "actor": 0, "pai": "5m", "consumed": ["5m", "5m", "5m"]})] = 0.2

    reach_payload = aggregate_keqingv4_continuation_scores(
        logits,
        {"type": "reach", "actor": 0},
        [
            {
                "weight": 1.0,
                "score": 0.4,
                "continuation_kind": "reach_declaration",
                "declaration_action": {"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": False},
            }
        ],
    )
    assert float(reach_payload["final_score"]) == pytest.approx(1.3)
    assert reach_payload["meta"]["reach_discard"]["pai"] == "4m"

    weighted_payload = aggregate_keqingv4_continuation_scores(
        logits,
        {"type": "daiminkan", "actor": 0, "pai": "5m", "consumed": ["5m", "5m", "5m"]},
        [
            {"weight": 2.0, "score": 1.0, "continuation_kind": "rinshan_followup", "declaration_action": None},
            {"weight": 1.0, "score": -0.5, "continuation_kind": "rinshan_followup", "declaration_action": None},
        ],
    )
    expected = 0.2 + (2.0 * 1.0 + 1.0 * -0.5) / 3.0
    assert float(weighted_payload["final_score"]) == pytest.approx(expected)


def test_default_action_scorer_post_meld_propagates_rust_continuation_scenario_schema_drift(monkeypatch):
    monkeypatch.setattr(
        inference_scoring,
        "_rust_resolve_keqingv4_continuation_scenarios",
        lambda snap, actor, action: [
            {
                "projected_snapshot": [],
                "legal_actions": [
                    {"type": "dahai", "actor": actor, "pai": "9m", "tsumogiri": False},
                ],
                "weight": 1.0,
                "continuation_kind": "post_meld_followup",
                "declaration_action": None,
            }
        ],
    )
    monkeypatch.setattr(
        inference_scoring,
        "_build_python_continuation_scenarios",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("python fallback should stay unused")),
    )

    primary_logits = _empty_policy()
    primary_logits[action_to_idx({"type": "chi", "actor": 0, "target": 3, "pai": "5m", "consumed": ["6m", "7m"]})] = 0.2
    adapter = _FakeAdapter(
        [
            ModelForwardResult(
                policy_logits=primary_logits,
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "dahai", "actor": 3, "pai": "5m", "tsumogiri": False},
        runtime_snap={
            "marker": "runtime",
            "hand": ["5mr", "6m", "7m", "9m"],
            "discards": [[], [], [], []],
            "melds": [[], [], [], []],
            "last_discard": {"actor": 3, "pai": "5m"},
        },
        model_snap={"marker": "model"},
        legal_actions=[
            {"type": "chi", "actor": 0, "target": 3, "pai": "5m", "consumed": ["6m", "7m"]},
        ],
    )

    with pytest.raises(RuntimeError, match="projected_snapshot must be a dict"):
        scorer.score(ctx)


def test_default_action_scorer_post_meld_propagates_rust_continuation_scoring_schema_drift(monkeypatch):
    monkeypatch.setattr(
        inference_scoring,
        "_rust_resolve_keqingv4_continuation_scenarios",
        lambda snap, actor, action: [
            {
                "projected_snapshot": {
                    **snap,
                    "projected_by": "rust",
                    "actor_to_move": actor,
                    "last_discard": None,
                },
                "legal_actions": [
                    {"type": "dahai", "actor": actor, "pai": "9m", "tsumogiri": False},
                ],
                "weight": 1.0,
                "continuation_kind": "post_meld_followup",
                "declaration_action": None,
            }
        ],
    )
    monkeypatch.setattr(
        inference_scoring,
        "_build_python_continuation_scenarios",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("python fallback should stay unused")),
    )
    monkeypatch.setattr(
        inference_scoring,
        "_find_best_legal",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("python continuation scoring fallback should stay unused")),
    )
    monkeypatch.setattr(
        inference_scoring,
        "_rust_score_keqingv4_continuation_scenario",
        lambda *args, **kwargs: {"best_action": None},
    )

    primary_logits = _empty_policy()
    primary_logits[action_to_idx({"type": "chi", "actor": 0, "target": 3, "pai": "5m", "consumed": ["6m", "7m"]})] = 0.2
    adapter = _FakeAdapter(
        [
            ModelForwardResult(
                policy_logits=primary_logits,
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_policy_with_scores({("dahai", "9m"): 0.8}),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "dahai", "actor": 3, "pai": "5m", "tsumogiri": False},
        runtime_snap={
            "marker": "runtime",
            "hand": ["5mr", "6m", "7m", "9m"],
            "discards": [[], [], [], []],
            "melds": [[], [], [], []],
            "last_discard": {"actor": 3, "pai": "5m"},
        },
        model_snap={"marker": "model"},
        legal_actions=[
            {"type": "chi", "actor": 0, "target": 3, "pai": "5m", "consumed": ["6m", "7m"]},
        ],
    )

    with pytest.raises(RuntimeError, match="continuation scoring contract drift"):
        scorer.score(ctx)


def test_default_action_scorer_post_meld_propagates_rust_continuation_aggregation_schema_drift(monkeypatch):
    monkeypatch.setattr(
        inference_scoring,
        "_rust_resolve_keqingv4_continuation_scenarios",
        lambda snap, actor, action: [
            {
                "projected_snapshot": {
                    **snap,
                    "projected_by": "rust",
                    "actor_to_move": actor,
                    "last_discard": None,
                },
                "legal_actions": [
                    {"type": "dahai", "actor": actor, "pai": "9m", "tsumogiri": False},
                ],
                "weight": 1.0,
                "continuation_kind": "post_meld_followup",
                "declaration_action": None,
            }
        ],
    )
    monkeypatch.setattr(
        inference_scoring,
        "_build_python_continuation_scenarios",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("python fallback should stay unused")),
    )
    monkeypatch.setattr(
        inference_scoring,
        "_rust_score_keqingv4_continuation_scenario",
        lambda *args, **kwargs: {
            "best_action": {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False},
            "score": 0.8,
        },
    )
    monkeypatch.setattr(
        inference_scoring,
        "_rust_aggregate_keqingv4_continuation_scores",
        lambda *args, **kwargs: {"meta": {}},
    )

    primary_logits = _empty_policy()
    primary_logits[action_to_idx({"type": "chi", "actor": 0, "target": 3, "pai": "5m", "consumed": ["6m", "7m"]})] = 0.2
    adapter = _FakeAdapter(
        [
            ModelForwardResult(
                policy_logits=primary_logits,
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_policy_with_scores({("dahai", "9m"): 0.8}),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "dahai", "actor": 3, "pai": "5m", "tsumogiri": False},
        runtime_snap={
            "marker": "runtime",
            "hand": ["5mr", "6m", "7m", "9m"],
            "discards": [[], [], [], []],
            "melds": [[], [], [], []],
            "last_discard": {"actor": 3, "pai": "5m"},
        },
        model_snap={"marker": "model"},
        legal_actions=[
            {"type": "chi", "actor": 0, "target": 3, "pai": "5m", "consumed": ["6m", "7m"]},
        ],
    )

    with pytest.raises(RuntimeError, match="continuation aggregation contract drift"):
        scorer.score(ctx)


def test_default_action_scorer_uses_model_snap_for_primary_forward():
    adapter = _FakeAdapter(
        [
            ModelForwardResult(
                policy_logits=_policy_with_scores({("reach", None): -1.0, ("dahai", "1m"): -0.5}),
                value=0.4,
                aux=ModelAuxOutputs(score_delta=0.2, win_prob=0.1, dealin_prob=0.0),
            )
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=0,
        beam_lambda=1.0,
        style_lambda=0.5,
        score_delta_lambda=0.2,
        win_prob_lambda=0.2,
        dealin_prob_lambda=0.25,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "1m"},
        runtime_snap={"marker": "runtime"},
        model_snap={"marker": "model", "tsumo_pai": "1m"},
        legal_actions=[
            {"type": "reach", "actor": 0},
            {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
        ],
    )

    result = scorer.score(ctx)

    assert adapter.calls[0][0]["marker"] == "model"
    assert result.chosen["type"] == "dahai"
    assert result.model_aux.score_delta == 0.2


def test_default_action_scorer_dahai_beam_uses_runtime_snap_followups():
    adapter = _FakeAdapter(
        [
            ModelForwardResult(
                policy_logits=_policy_with_scores({("dahai", "1m"): 0.5, ("dahai", "2m"): 0.4}),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_policy_with_scores({}),
                value=0.1,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_policy_with_scores({}),
                value=0.9,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=2,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "2m"},
        runtime_snap={"marker": "runtime", "hand": ["1m", "2m"], "discards": [[], [], [], []]},
        model_snap={"marker": "model", "hand": ["1m"], "tsumo_pai": "2m"},
        legal_actions=[
            {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
            {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": False},
        ],
    )

    result = scorer.score(ctx)

    assert adapter.calls[0][0]["marker"] == "model"
    assert adapter.calls[1][0]["marker"] == "runtime"
    assert adapter.calls[2][0]["marker"] == "runtime"
    assert result.chosen["pai"] == "2m"
    log_dict = candidate_to_log_dict(result.candidates[0])
    assert "final_score" in log_dict


def test_default_action_scorer_reach_uses_best_declaration_discard(monkeypatch):
    monkeypatch.setattr(
        inference_scoring,
        "_reach_discard_candidates",
        lambda hand, last_tsumo, last_tsumo_raw: [("4m", False), ("5mr", True)],
    )

    adapter = _FakeAdapter(
        [
            ModelForwardResult(
                policy_logits=_policy_with_scores(
                    {
                        ("reach", None): -4.0,
                        ("dahai", "4m"): -0.1,
                        ("dahai", "5mr"): -0.2,
                    }
                ),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_policy_with_scores({}),
                value=4.6,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_policy_with_scores({}),
                value=0.5,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "5mr"},
        runtime_snap={
            "marker": "runtime",
            "hand": ["4m", "5mr"],
            "discards": [[], [], [], []],
            "reached": [False, False, False, False],
            "last_tsumo": ["5m", None, None, None],
            "last_tsumo_raw": ["5mr", None, None, None],
        },
        model_snap={"marker": "model", "hand": ["4m"], "tsumo_pai": "5mr"},
        legal_actions=[
            {"type": "reach", "actor": 0},
            {"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": False},
            {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": True},
        ],
    )

    result = scorer.score(ctx)

    assert adapter.calls[0][0]["marker"] == "model"
    assert adapter.calls[1][0]["marker"] == "runtime"
    assert adapter.calls[1][0]["reached"][0] is True
    assert result.chosen["type"] == "reach"
    reach_candidate = next(c for c in result.candidates if c.action["type"] == "reach")
    assert reach_candidate.meta["reach_discard"]["pai"] == "4m"
    assert reach_candidate.beam_score == reach_candidate.final_score


def test_default_action_scorer_meld_beam_uses_post_meld_best_discard_value():
    primary_logits = _empty_policy()
    primary_logits[action_to_idx({"type": "chi", "pai": "5m", "consumed": ["6m", "7m"]})] = 0.2
    primary_logits[action_to_idx({"type": "none"})] = 0.1
    adapter = _FakeAdapter(
        [
            ModelForwardResult(
                policy_logits=primary_logits,
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_policy_with_scores({}),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_policy_with_scores({("dahai", "9m"): 1.2, ("dahai", "5mr"): -2.0}),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=3,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "dahai", "actor": 3, "pai": "5m", "tsumogiri": False},
        runtime_snap={
            "marker": "runtime",
            "hand": ["5mr", "6m", "7m", "9m"],
            "discards": [[], [], [], []],
            "melds": [[], [], [], []],
            "last_discard": {"actor": 3, "pai": "5m"},
        },
        model_snap={"marker": "model"},
        legal_actions=[
            {"type": "chi", "actor": 0, "target": 3, "pai": "5m", "consumed": ["6m", "7m"]},
            {"type": "none"},
        ],
    )

    result = scorer.score(ctx)

    assert result.chosen["type"] == "chi"
    assert adapter.calls[2][0]["last_discard"] is None
    assert adapter.calls[2][0]["actor_to_move"] == 0
    assert adapter.calls[2][0]["legal_actions"] == [
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": False},
        {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False},
    ]
    chi_candidate = next(c for c in result.candidates if c.action["type"] == "chi")
    none_candidate = next(c for c in result.candidates if c.action["type"] == "none")
    assert chi_candidate.final_score > none_candidate.final_score


def test_default_action_scorer_post_meld_eval_prefers_rust_followup_bridge(monkeypatch):
    monkeypatch.setattr(
        inference_scoring,
        "_rust_resolve_keqingv4_continuation_scenarios",
        lambda snap, actor, action: [
            {
                "projected_snapshot": {
                    **snap,
                    "projected_by": "rust",
                    "actor_to_move": actor,
                    "last_discard": None,
                },
                "legal_actions": [
                    {"type": "dahai", "actor": actor, "pai": "5mr", "tsumogiri": False},
                    {"type": "dahai", "actor": actor, "pai": "9m", "tsumogiri": False},
                ],
                "weight": 1.0,
                "continuation_kind": "post_meld_followup",
                "declaration_action": None,
            }
        ],
    )
    monkeypatch.setattr(
        inference_scoring,
        "_build_python_continuation_scenarios",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("python continuation scenario fallback should not be used")),
    )
    monkeypatch.setattr(
        inference_scoring,
        "_resolve_post_meld_followup_context",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("legacy post-meld followup helper should not be used")),
    )

    primary_logits = _empty_policy()
    primary_logits[action_to_idx({"type": "chi", "pai": "5m", "consumed": ["6m", "7m"]})] = 0.2
    primary_logits[action_to_idx({"type": "none"})] = 0.1
    followup_logits = _empty_policy()
    followup_logits[action_to_idx({"type": "dahai", "pai": "9m", "tsumogiri": False})] = 1.2
    followup_logits[action_to_idx({"type": "dahai", "pai": "5mr", "tsumogiri": False})] = -2.0
    adapter = _FakeAdapter(
        [
            ModelForwardResult(
                policy_logits=primary_logits,
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_policy_with_scores({}),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=followup_logits,
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=3,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "dahai", "actor": 3, "pai": "5m", "tsumogiri": False},
        runtime_snap={
            "marker": "runtime",
            "hand": ["5mr", "6m", "7m", "9m"],
            "discards": [[], [], [], []],
            "melds": [[], [], [], []],
            "last_discard": {"actor": 3, "pai": "5m"},
        },
        model_snap={"marker": "model"},
        legal_actions=[
            {"type": "chi", "actor": 0, "target": 3, "pai": "5m", "consumed": ["6m", "7m"]},
            {"type": "none"},
        ],
    )

    result = scorer.score(ctx)

    assert result.chosen["type"] == "chi"
    assert adapter.calls[2][0]["projected_by"] == "rust"
    assert adapter.calls[2][0]["legal_actions"] == [
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": False},
        {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False},
    ]


def test_default_action_scorer_post_meld_bridge_falls_back_on_missing_capability(monkeypatch):
    monkeypatch.setattr(
        inference_scoring,
        "_rust_resolve_keqingv4_continuation_scenarios",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Rust keqingv4 continuation scenario capability is not available")
        ),
    )
    monkeypatch.setattr(
        inference_scoring,
        "_build_python_continuation_scenarios",
        lambda snap, actor, action: [
            {
                "projected_snapshot": {
                    **snap,
                    "projected_by": "python",
                    "actor_to_move": actor,
                    "last_discard": None,
                },
                "legal_actions": [
                    {"type": "dahai", "actor": actor, "pai": "5mr", "tsumogiri": False},
                    {"type": "dahai", "actor": actor, "pai": "9m", "tsumogiri": False},
                ],
                "weight": 1.0,
                "continuation_kind": "post_meld_followup",
                "declaration_action": None,
            }
        ],
    )

    primary_logits = _empty_policy()
    primary_logits[action_to_idx({"type": "chi", "pai": "5m", "consumed": ["6m", "7m"]})] = 0.2
    primary_logits[action_to_idx({"type": "none"})] = 0.1
    followup_logits = _empty_policy()
    followup_logits[action_to_idx({"type": "dahai", "pai": "9m", "tsumogiri": False})] = 1.2
    followup_logits[action_to_idx({"type": "dahai", "pai": "5mr", "tsumogiri": False})] = -2.0
    adapter = _FakeAdapter(
        [
            ModelForwardResult(
                policy_logits=primary_logits,
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_policy_with_scores({}),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=followup_logits,
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=3,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "dahai", "actor": 3, "pai": "5m", "tsumogiri": False},
        runtime_snap={
            "marker": "runtime",
            "hand": ["5mr", "6m", "7m", "9m"],
            "discards": [[], [], [], []],
            "melds": [[], [], [], []],
            "last_discard": {"actor": 3, "pai": "5m"},
        },
        model_snap={"marker": "model"},
        legal_actions=[
            {"type": "chi", "actor": 0, "target": 3, "pai": "5m", "consumed": ["6m", "7m"]},
            {"type": "none"},
        ],
    )

    result = scorer.score(ctx)

    assert result.chosen["type"] == "chi"
    assert adapter.calls[2][0]["projected_by"] == "python"
    assert adapter.calls[2][0]["legal_actions"] == [
        {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": False},
        {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False},
    ]


def test_default_action_scorer_kan_beam_uses_rinshan_weighted_followups(monkeypatch):
    monkeypatch.setattr(
        inference_scoring,
        "_live_draw_tile_weights",
        lambda snap: [("4m", 2), ("5m", 1)],
    )
    monkeypatch.setattr(
        inference_scoring,
        "_rust_resolve_keqingv4_continuation_scenarios",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("Rust keqingv4 continuation scenario capability is not available")
        ),
    )

    primary_logits = _empty_policy()
    primary_logits[action_to_idx({"type": "daiminkan", "pai": "5m", "consumed": ["5m", "5m", "5m"]})] = 0.2
    primary_logits[action_to_idx({"type": "none"})] = 0.1
    adapter = _FakeAdapter(
        [
            ModelForwardResult(
                policy_logits=primary_logits,
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_policy_with_scores({}),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_policy_with_scores({("dahai", "9m"): 1.0, ("dahai", "4m"): -1.0}),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_policy_with_scores({("dahai", "9m"): -0.6, ("dahai", "5m"): -2.0}),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=3,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "dahai", "actor": 3, "pai": "5m", "tsumogiri": False},
        runtime_snap={
            "marker": "runtime",
            "hand": ["5m", "5m", "5m", "9m"],
            "discards": [[], [], [], []],
            "melds": [[], [], [], []],
            "dora_markers": [],
            "last_discard": {"actor": 3, "pai": "5m"},
        },
        model_snap={"marker": "model"},
        legal_actions=[
            {"type": "daiminkan", "actor": 0, "target": 3, "pai": "5m", "consumed": ["5m", "5m", "5m"]},
            {"type": "none"},
        ],
    )

    result = scorer.score(ctx)

    assert result.chosen["type"] == "daiminkan"
    kan_candidate = next(c for c in result.candidates if c.action["type"] == "daiminkan")
    none_candidate = next(c for c in result.candidates if c.action["type"] == "none")
    assert kan_candidate.final_score > none_candidate.final_score
    assert adapter.calls[2][0]["tsumo_pai"] == "4m"
    assert adapter.calls[3][0]["tsumo_pai"] == "5m"
    assert "legal_actions" in adapter.calls[2][0]
    assert "legal_actions" in adapter.calls[3][0]


def test_default_action_scorer_kan_beam_allows_rinshan_hora_followup(monkeypatch):
    monkeypatch.setattr(
        inference_scoring,
        "_rust_resolve_keqingv4_continuation_scenarios",
        lambda snap, actor, action: [
            {
                "projected_snapshot": {
                    **snap,
                    "tsumo_pai": "4p",
                    "actor_to_move": actor,
                    "projected_by": "rust",
                },
                "legal_actions": [{"type": "hora", "pai": "4p"}, {"type": "dahai", "actor": actor, "pai": "9m", "tsumogiri": False}],
                "weight": 1.0,
                "continuation_kind": "rinshan_followup",
                "declaration_action": None,
            }
        ],
    )

    primary_logits = _empty_policy()
    primary_logits[action_to_idx({"type": "daiminkan", "pai": "5m", "consumed": ["5m", "5m", "5m"]})] = 0.2
    primary_logits[action_to_idx({"type": "none"})] = 0.1
    followup_logits = _empty_policy()
    followup_logits[action_to_idx({"type": "hora", "pai": "4p"})] = 3.0
    followup_logits[action_to_idx({"type": "dahai", "pai": "9m", "tsumogiri": False})] = -1.0
    adapter = _FakeAdapter(
        [
            ModelForwardResult(
                policy_logits=primary_logits,
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_policy_with_scores({}),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=followup_logits,
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=3,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "dahai", "actor": 3, "pai": "5m", "tsumogiri": False},
        runtime_snap={
            "marker": "runtime",
            "hand": ["5m", "5m", "5m", "9m"],
            "discards": [[], [], [], []],
            "melds": [[], [], [], []],
            "dora_markers": [],
            "last_discard": {"actor": 3, "pai": "5m"},
            "_force_hora_legal": True,
        },
        model_snap={"marker": "model"},
        legal_actions=[
            {"type": "daiminkan", "actor": 0, "target": 3, "pai": "5m", "consumed": ["5m", "5m", "5m"]},
            {"type": "none"},
        ],
    )

    result = scorer.score(ctx)

    assert result.chosen["type"] == "daiminkan"
    assert adapter.calls[2][0]["tsumo_pai"] == "4p"
    assert adapter.calls[2][0]["projected_by"] == "rust"
    assert adapter.calls[2][0]["legal_actions"] == [
        {"type": "hora", "pai": "4p"},
        {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False},
    ]


def test_default_action_scorer_keqingv4_reach_candidate_carries_special_meta(monkeypatch):
    monkeypatch.setattr(
        inference_scoring,
        "_rust_resolve_keqingv4_continuation_scenarios",
        lambda snap, actor, action: [
            {
                "projected_snapshot": {
                    **snap,
                    "projected_by": "rust",
                    "reached": [True, False, False, False],
                },
                "legal_actions": [{"type": "none"}],
                "weight": 1.0,
                "continuation_kind": "reach_declaration",
                "declaration_action": {
                    "type": "dahai",
                    "actor": actor,
                    "pai": "4m",
                    "tsumogiri": False,
                },
            }
        ],
    )

    class _FakeV4Adapter(_FakeAdapter):
        model_version = "keqingv4"

        def resolve_runtime_v4_summaries(self, snap, actor, legal_actions=None):
            special = np.zeros((3, KEQINGV4_SUMMARY_DIM), dtype=np.float32)
            special[0, 1] = 1.0
            special[0, 2] = 0.25
            special[0, 4] = 0.5
            special[0, 6] = 0.125
            special[0, 12] = 1.0
            special[0, 13] = 1.0
            return (
                np.zeros((34, KEQINGV4_SUMMARY_DIM), dtype=np.float32),
                np.zeros((8, KEQINGV4_SUMMARY_DIM), dtype=np.float32),
                special,
            )

    adapter = _FakeV4Adapter(
        [
            ModelForwardResult(
                policy_logits=_policy_with_scores(
                    {
                        ("reach", None): 0.8,
                        ("dahai", "4m"): 0.1,
                    }
                ),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_policy_with_scores({}),
                value=0.4,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "4m"},
        runtime_snap={
            "marker": "runtime",
            "hand": ["4m"],
            "discards": [[], [], [], []],
            "reached": [False, False, False, False],
            "last_tsumo": ["4m", None, None, None],
            "last_tsumo_raw": ["4m", None, None, None],
        },
        model_snap={
            "marker": "model",
            "hand": [],
            "tsumo_pai": "4m",
            "reached": [False, False, False, False],
            "last_tsumo": ["4m", None, None, None],
            "last_tsumo_raw": ["4m", None, None, None],
        },
        legal_actions=[
            {"type": "reach", "actor": 0},
            {"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": True},
        ],
    )

    result = scorer.score(ctx)

    reach_candidate = next(c for c in result.candidates if c.action["type"] == "reach")
    assert reach_candidate.meta["special_semantics"] == "reach"
    assert reach_candidate.meta["reach_decl_tenpai"] == 1.0
    assert reach_candidate.meta["reach_decl_waits_ratio"] == 0.25
    assert reach_candidate.meta["reach_discard"]["pai"] == "4m"


def test_default_action_scorer_non_keqingv4_path_does_not_require_private_adapter_attr():
    adapter = _FakeAdapter(
        [
            ModelForwardResult(
                policy_logits=_policy_with_scores(
                    {
                        ("dahai", "9m"): 0.9,
                        ("dahai", "1m"): 0.1,
                    }
                ),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=0,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "9m"},
        runtime_snap={"marker": "runtime"},
        model_snap={"marker": "model"},
        legal_actions=[
            {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": True},
            {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
        ],
    )

    result = scorer.score(ctx)

    assert result.chosen["pai"] == "9m"


def test_review_exporter_surfaces_special_candidate_meta():
    exporter = DefaultRuntimeReviewExporter()
    ctx = DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "4m"},
        runtime_snap={"hand": ["4m"]},
        model_snap={"hand": [], "tsumo_pai": "4m"},
        legal_actions=[{"type": "reach", "actor": 0}],
    )
    decision = DecisionResult(
        chosen={"type": "reach", "actor": 0},
        candidates=[
            ScoredCandidate(
                action={"type": "reach", "actor": 0},
                logit=0.8,
                final_score=1.2,
                beam_score=1.2,
                meta={
                    "special_semantics": "reach",
                    "reach_decl_tenpai": 1.0,
                    "reach_discard": {"type": "dahai", "pai": "4m", "tsumogiri": False},
                },
            )
        ],
        model_value=0.1,
        model_aux=ModelAuxOutputs(),
    )

    entry = exporter.build_decision_entry(
        step=0,
        ctx=ctx,
        decision=decision,
        gt_action=None,
        actor=0,
    )

    assert entry["candidates"][0]["special_semantics"] == "reach"
    assert entry["candidates"][0]["reach_decl_tenpai"] == 1.0
    assert entry["candidates"][0]["reach_discard"]["pai"] == "4m"


def test_default_action_scorer_keqingv4_ryukyoku_can_gain_calibration_bonus():
    class _FakeV4Adapter(_FakeAdapter):
        model_version = "keqingv4"

        def resolve_runtime_v4_summaries(self, snap, actor, legal_actions=None):
            special = np.zeros((3, KEQINGV4_SUMMARY_DIM), dtype=np.float32)
            special[2, 2] = 1.0
            special[2, 3] = 1.0
            special[2, 4] = 1.4
            special[2, 12] = 1.0
            special[2, 13] = 1.0
            return (
                np.zeros((34, KEQINGV4_SUMMARY_DIM), dtype=np.float32),
                np.zeros((8, KEQINGV4_SUMMARY_DIM), dtype=np.float32),
                special,
            )

    adapter = _FakeV4Adapter(
        [
            ModelForwardResult(
                policy_logits=_policy_with_scores(
                    {
                        ("ryukyoku", None): 0.10,
                        ("dahai", "9m"): 0.45,
                    }
                ),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "5m"},
        runtime_snap={"hand": ["1m", "9m"], "tsumo_pai": "5m"},
        model_snap={"hand": ["1m", "9m"], "tsumo_pai": "5m"},
        legal_actions=[
            {"type": "ryukyoku"},
            {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False},
        ],
    )

    result = scorer.score(ctx)

    assert result.chosen["type"] == "ryukyoku"
    ryukyoku_candidate = next(c for c in result.candidates if c.action["type"] == "ryukyoku")
    assert ryukyoku_candidate.meta["special_semantics"] == "ryukyoku"
    assert ryukyoku_candidate.final_score > 0.45


def test_default_action_scorer_meld_fallback_compares_final_beam_score(monkeypatch):
    monkeypatch.setattr(
        inference_scoring,
        "_meld_value_eval",
        lambda *args, **kwargs: (
            {"type": "chi", "actor": 0, "target": 3, "pai": "5m", "consumed": ["6m", "7m"]},
            {action_to_idx({"type": "chi", "pai": "5m", "consumed": ["6m", "7m"]}): 5.0},
        ),
    )

    primary_logits = _empty_policy()
    primary_logits[action_to_idx({"type": "chi", "pai": "5m", "consumed": ["6m", "7m"]})] = 0.1
    primary_logits[action_to_idx({"type": "dahai", "pai": "9m", "tsumogiri": False})] = 0.9
    adapter = _FakeAdapter(
        [
            ModelForwardResult(
                policy_logits=primary_logits,
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=3,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "dahai", "actor": 3, "pai": "5m", "tsumogiri": False},
        runtime_snap={"marker": "runtime"},
        model_snap={"marker": "model"},
        legal_actions=[
            {"type": "chi", "actor": 0, "target": 3, "pai": "5m", "consumed": ["6m", "7m"]},
            {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False},
        ],
    )

    result = scorer.score(ctx)

    assert result.chosen["type"] == "chi"


def test_default_action_scorer_reach_eval_prefers_rust_reach_projection(monkeypatch):
    monkeypatch.setattr(
        inference_scoring,
        "_build_python_continuation_scenarios",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("python continuation scenario fallback should not be used")),
    )
    monkeypatch.setattr(
        inference_scoring,
        "_rust_resolve_keqingv4_continuation_scenarios",
        lambda snap, actor, action: [
            {
                "projected_snapshot": {
                    **snap,
                    "projected_by": "rust",
                    "reached": [True, False, False, False],
                    "pending_reach": [False, False, False, False],
                },
                "legal_actions": [{"type": "none"}],
                "weight": 1.0,
                "continuation_kind": "reach_declaration",
                "declaration_action": {"type": "dahai", "actor": actor, "pai": "1p", "tsumogiri": False},
            }
        ],
    )

    primary_logits = _empty_policy()
    primary_logits[action_to_idx({"type": "reach"})] = 0.5
    primary_logits[action_to_idx({"type": "dahai", "pai": "1p", "tsumogiri": False})] = 0.2
    followup_logits = _empty_policy()
    followup_logits[action_to_idx({"type": "none"})] = 0.0
    adapter = _FakeAdapter(
        [
            ModelForwardResult(policy_logits=primary_logits, value=0.0, aux=ModelAuxOutputs()),
            ModelForwardResult(policy_logits=followup_logits, value=0.3, aux=ModelAuxOutputs()),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "5s"},
        runtime_snap={
            "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
            "tsumo_pai": "5s",
            "last_tsumo": ["5s", None, None, None],
            "last_tsumo_raw": ["5s", None, None, None],
            "reached": [False, False, False, False],
            "pending_reach": [False, False, False, False],
        },
        model_snap={"hand": [], "tsumo_pai": None},
        legal_actions=[
            {"type": "reach", "actor": 0},
            {"type": "dahai", "actor": 0, "pai": "1p", "tsumogiri": False},
        ],
    )

    result = scorer.score(ctx)

    assert result.chosen["type"] == "reach"
    assert adapter.calls[1][0]["projected_by"] == "rust"
    assert adapter.calls[1][0]["legal_actions"] == [{"type": "none"}]


def test_default_action_scorer_reach_eval_injects_structural_legal_actions_into_followup(monkeypatch):
    monkeypatch.setattr(
        inference_scoring,
        "_rust_resolve_keqingv4_continuation_scenarios",
        lambda snap, actor, action: [
            {
                "projected_snapshot": {
                    **snap,
                    "projected_by": "rust",
                    "reached": [True, False, False, False],
                    "pending_reach": [False, False, False, False],
                },
                "legal_actions": [{"type": "none"}, {"type": "hora", "actor": actor, "pai": "1p"}],
                "weight": 1.0,
                "continuation_kind": "reach_declaration",
                "declaration_action": {"type": "dahai", "actor": actor, "pai": "1p", "tsumogiri": False},
            }
        ],
    )
    monkeypatch.setattr(
        inference_scoring,
        "_build_python_continuation_scenarios",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("python continuation scenario fallback should not be used")),
    )

    primary_logits = _empty_policy()
    primary_logits[action_to_idx({"type": "reach"})] = 0.5
    primary_logits[action_to_idx({"type": "dahai", "pai": "1p", "tsumogiri": False})] = 0.2
    adapter = _FakeAdapter(
        [
            ModelForwardResult(policy_logits=primary_logits, value=0.0, aux=ModelAuxOutputs()),
            ModelForwardResult(policy_logits=_empty_policy(), value=0.3, aux=ModelAuxOutputs()),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "5s"},
        runtime_snap={
            "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
            "tsumo_pai": "5s",
            "last_tsumo": ["5s", None, None, None],
            "last_tsumo_raw": ["5s", None, None, None],
            "reached": [False, False, False, False],
            "pending_reach": [False, False, False, False],
        },
        model_snap={"hand": [], "tsumo_pai": None},
        legal_actions=[
            {"type": "reach", "actor": 0},
            {"type": "dahai", "actor": 0, "pai": "1p", "tsumogiri": False},
        ],
    )

    result = scorer.score(ctx)

    assert result.chosen["type"] == "reach"
    assert adapter.calls[1][0]["legal_actions"] == [
        {"type": "none"},
        {"type": "hora", "actor": 0, "pai": "1p"},
    ]


def test_default_action_scorer_reach_eval_propagates_rust_projection_errors(monkeypatch):
    monkeypatch.setattr(
        inference_scoring,
        "_rust_resolve_keqingv4_continuation_scenarios",
        lambda snap, actor, action: (_ for _ in ()).throw(RuntimeError("reach projection drift")),
    )
    monkeypatch.setattr(
        inference_scoring,
        "_build_python_continuation_scenarios",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("python fallback should stay unused")),
    )

    primary_logits = _empty_policy()
    primary_logits[action_to_idx({"type": "reach"})] = 0.5
    primary_logits[action_to_idx({"type": "dahai", "pai": "1p", "tsumogiri": False})] = 0.2
    adapter = _FakeAdapter(
        [
            ModelForwardResult(policy_logits=primary_logits, value=0.0, aux=ModelAuxOutputs()),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "5s"},
        runtime_snap={
            "hand": ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
            "tsumo_pai": "5s",
            "last_tsumo": ["5s", None, None, None],
            "last_tsumo_raw": ["5s", None, None, None],
            "reached": [False, False, False, False],
            "pending_reach": [False, False, False, False],
        },
        model_snap={"hand": [], "tsumo_pai": None},
        legal_actions=[
            {"type": "reach", "actor": 0},
            {"type": "dahai", "actor": 0, "pai": "1p", "tsumogiri": False},
        ],
    )

    with pytest.raises(RuntimeError, match="reach projection drift"):
        scorer.score(ctx)


def test_runtime_review_exporter_builds_decision_entry():
    exporter = DefaultRuntimeReviewExporter()
    ctx = DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "5mr"},
        runtime_snap={"hand": ["4m", "5mr"]},
        model_snap={
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "hand": ["4m"],
            "tsumo_pai": "5mr",
            "actor_to_move": 0,
        },
        legal_actions=[{"type": "reach", "actor": 0}],
    )
    decision = DecisionResult(
        chosen={"type": "reach", "actor": 0},
        candidates=[
            ScoredCandidate(
                action={"type": "reach", "actor": 0},
                logit=-1.2,
                final_score=0.8,
                beam_score=0.8,
                meta={"reach_discard": {"type": "dahai", "pai": "4m", "tsumogiri": False}},
            )
        ],
        model_value=0.3,
        model_aux=ModelAuxOutputs(
            score_delta=0.1,
            win_prob=0.2,
            dealin_prob=0.05,
            rank_probs=(0.5, 0.3, 0.15, 0.05),
            final_score_delta=0.2,
            rank_pt_value=0.65,
        ),
    )

    entry = exporter.build_decision_entry(
        step=7,
        ctx=ctx,
        decision=decision,
        gt_action={"type": "reach", "actor": 0},
        actor=0,
    )

    assert entry["step"] == 7
    assert entry["tsumo_pai"] == "5mr"
    assert entry["chosen"]["type"] == "reach"
    assert entry["candidates"][0]["reach_discard"]["pai"] == "4m"
    assert entry["aux_outputs"]["win_prob"] == 0.2
    assert entry["aux_outputs"]["rank_probs"] == [0.5, 0.3, 0.15, 0.05]
    assert entry["aux_outputs"]["final_score_delta"] == 0.2
    assert entry["aux_outputs"]["rank_pt_value"] == 0.65


def test_runtime_review_exporter_candidate_score_priority():
    exporter = DefaultRuntimeReviewExporter()

    assert exporter.candidate_score({"final_score": 1.5, "beam_score": 0.7, "logit": -1.0}) == 1.5
    assert exporter.candidate_score({"beam_score": 0.7, "logit": -1.0}) == 0.7
    assert exporter.candidate_score({"logit": -1.0}) == -1.0
    assert exporter.candidate_score({}) is None
    assert exporter.candidate_sort_key({"beam_score": 0.7, "logit": -1.0}) == 0.7


def test_same_action_normalizes_pass_and_aka_equivalence():
    assert same_action(
        {"type": "pass"},
        {"type": "none", "actor": 0},
    ) is True


def test_summarize_decision_matches_ignores_obs_steps():
    total_ops, match_count = summarize_decision_matches(
        [
            {
                "is_obs": False,
                "chosen": {"type": "none"},
                "gt_action": {"type": "pass"},
            },
            {
                "is_obs": False,
                "chosen": {"type": "dahai", "pai": "1m"},
                "gt_action": {"type": "dahai", "pai": "2m"},
            },
            {
                "is_obs": True,
                "chosen": {"type": "dahai", "pai": "3m"},
                "gt_action": {"type": "dahai", "pai": "3m"},
            },
        ]
    )

    assert total_ops == 2
    assert match_count == 1


def test_runtime_review_exporter_compute_rating_prefers_final_then_beam_then_logit():
    exporter = DefaultRuntimeReviewExporter()
    rating = exporter.compute_rating(
        [
            {
                "chosen": {"type": "dahai", "pai": "1m"},
                "gt_action": {"type": "dahai", "pai": "2m"},
                "candidates": [
                    {
                        "action": {"type": "dahai", "pai": "1m"},
                        "final_score": 1.0,
                        "beam_score": 0.0,
                        "logit": -1.0,
                    },
                    {
                        "action": {"type": "dahai", "pai": "2m"},
                        "beam_score": 0.5,
                        "logit": -2.0,
                    },
                ],
            }
        ],
        alpha=0.5,
    )

    assert rating == 77.9


def test_review_action_helpers_format_and_extract_primary_tile():
    assert action_label({"type": "dahai", "pai": "5mr", "tsumogiri": True}) == "打 5mr (摸切)"
    assert action_label({"type": "reach"}) == "立直"
    assert action_primary_tile({"type": "chi", "pai": "4p"}) == "4p"
    assert action_primary_tile({"type": "reach"}) is None


def test_summarize_reach_followup_skips_obs_and_returns_next_discard():
    summary = summarize_reach_followup(
        [
            {"chosen": {"type": "reach", "actor": 0}},
            {"is_obs": True, "chosen": {"type": "dahai", "pai": "9p"}},
            {
                "chosen": {"type": "dahai", "pai": "4m"},
                "gt_action": {"type": "dahai", "pai": "5m"},
            },
        ],
        0,
    )

    assert summary is not None
    assert summary["bot_action"]["pai"] == "4m"
    assert summary["gt_action"]["pai"] == "5m"


def test_summarize_reach_followup_returns_none_for_non_reach():
    assert summarize_reach_followup([{"chosen": {"type": "dahai", "pai": "1m"}}], 0) is None
    assert same_action(
        {
            "type": "chi",
            "actor": 1,
            "target": 0,
            "pai": "4p",
            "consumed": ["5pr", "6p"],
        },
        {
            "type": "chi",
            "actor": 1,
            "target": 0,
            "pai": "4p",
            "consumed": ["5p", "6p"],
        },
    ) is True


def test_default_action_scorer_rank_pt_disabled_preserves_dahai_behavior(monkeypatch):
    monkeypatch.setattr(
        inference_scoring,
        "_score_actions_with_runtime_placement",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("placement path must stay disabled")),
    )

    adapter = _FakeV4Adapter(
        [
            ModelForwardResult(
                policy_logits=_policy_with_scores({("dahai", "1m"): 0.2, ("dahai", "2m"): 0.1}),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_empty_policy(),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
        rank_pt_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "2m"},
        runtime_snap={"hand": ["1m", "2m"], "discards": [[], [], [], []], "melds": [[], [], [], []]},
        model_snap={"hand": ["1m"], "tsumo_pai": "2m"},
        legal_actions=[
            {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
            {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": True},
        ],
    )

    result = scorer.score(ctx)

    assert result.chosen["pai"] == "1m"
    assert [candidate.action["pai"] for candidate in result.candidates] == ["1m", "2m"]
    assert [candidate.final_score for candidate in result.candidates] == pytest.approx([0.2, 0.1])


def test_default_action_scorer_rank_pt_disabled_preserves_meld_none_behavior(monkeypatch):
    chi_action = {"type": "chi", "actor": 0, "target": 3, "pai": "5m", "consumed": ["6m", "7m"]}
    none_action = {"type": "none"}
    monkeypatch.setattr(
        inference_scoring,
        "_score_actions_with_runtime_placement",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("placement path must stay disabled")),
    )
    monkeypatch.setattr(
        inference_scoring,
        "_resolve_continuation_scenarios",
        lambda snap, actor, action: [
            {
                "projected_snapshot": {"marker": "chi_followup"},
                "legal_actions": [{"type": "dahai", "actor": actor, "pai": "9m", "tsumogiri": False}],
                "weight": 1.0,
                "continuation_kind": "post_meld_followup",
                "declaration_action": None,
            }
        ]
        if action == chi_action
        else [],
    )

    logits = _empty_policy()
    logits[action_to_idx(chi_action)] = 0.2
    logits[action_to_idx(none_action)] = 0.3
    adapter = _FakeV4Adapter(
        [
            ModelForwardResult(policy_logits=logits, value=0.0, aux=ModelAuxOutputs()),
            ModelForwardResult(policy_logits=_empty_policy(), value=0.0, aux=ModelAuxOutputs()),
            ModelForwardResult(
                policy_logits=_policy_with_scores({("dahai", "9m"): 0.0}),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
        rank_pt_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "dahai", "actor": 3, "pai": "5m"},
        runtime_snap={"hand": ["5mr", "6m", "7m", "9m"], "discards": [[], [], [], []], "melds": [[], [], [], []]},
        model_snap={"hand": ["5mr", "6m", "7m", "9m"]},
        legal_actions=[chi_action, none_action],
    )

    result = scorer.score(ctx)

    assert result.chosen["type"] == "none"
    assert [candidate.action["type"] for candidate in result.candidates] == ["none", "chi"]
    assert [candidate.final_score for candidate in result.candidates] == pytest.approx([0.3, 0.2])


def test_default_action_scorer_rank_pt_disabled_preserves_reach_discard_behavior(monkeypatch):
    reach_action = {"type": "reach", "actor": 0}
    discard_reach = {"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": False}
    discard_alt = {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False}
    monkeypatch.setattr(
        inference_scoring,
        "_score_actions_with_runtime_placement",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("placement path must stay disabled")),
    )
    monkeypatch.setattr(
        inference_scoring,
        "_resolve_continuation_scenarios",
        lambda snap, actor, action: [
            {
                "projected_snapshot": {"marker": "reach_followup"},
                "legal_actions": [{"type": "none"}],
                "weight": 1.0,
                "continuation_kind": "reach_declaration",
                "declaration_action": discard_reach,
            }
        ]
        if action == reach_action
        else [],
    )

    adapter = _FakeV4Adapter(
        [
            ModelForwardResult(
                policy_logits=_policy_with_scores(
                    {("reach", None): 0.1, ("dahai", "4m"): 0.05, ("dahai", "9m"): 0.2}
                ),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_empty_policy(),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
        rank_pt_lambda=0.0,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "9m"},
        runtime_snap={"hand": ["4m", "9m"], "discards": [[], [], [], []], "melds": [[], [], [], []]},
        model_snap={"hand": ["4m"], "tsumo_pai": "9m"},
        legal_actions=[reach_action, discard_reach, discard_alt],
    )

    result = scorer.score(ctx)

    assert result.chosen["pai"] == "9m"
    assert [candidate.action.get("pai", candidate.action["type"]) for candidate in result.candidates] == ["9m", "reach", "4m"]
    assert [candidate.final_score for candidate in result.candidates] == pytest.approx([0.2, 0.15, 0.05])
    reach_candidate = next(candidate for candidate in result.candidates if candidate.action["type"] == "reach")
    assert reach_candidate.meta["reach_discard"]["pai"] == "4m"


def test_default_action_scorer_runtime_placement_reranks_all_discards():
    ctx = DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "2m"},
        runtime_snap={"hand": ["1m", "2m"], "discards": [[], [], [], []], "melds": [[], [], [], []]},
        model_snap={"hand": ["1m"], "tsumo_pai": "2m"},
        legal_actions=[
            {"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": False},
            {"type": "dahai", "actor": 0, "pai": "2m", "tsumogiri": True},
        ],
    )

    base_adapter = _FakeV4Adapter(
        [
            ModelForwardResult(
                policy_logits=_policy_with_scores({("dahai", "1m"): 0.2, ("dahai", "2m"): 0.1}),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(policy_logits=_empty_policy(), value=0.0, aux=ModelAuxOutputs()),
        ]
    )
    base_scorer = DefaultActionScorer(
        adapter=base_adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
        rank_pt_lambda=0.0,
    )
    base_result = base_scorer.score(ctx)

    placement_adapter = _FakeV4Adapter(
        [
            ModelForwardResult(
                policy_logits=_policy_with_scores({("dahai", "1m"): 0.2, ("dahai", "2m"): 0.1}),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
            ModelForwardResult(
                policy_logits=_empty_policy(),
                value=0.0,
                aux=ModelAuxOutputs(rank_pt_value=0.0),
            ),
            ModelForwardResult(
                policy_logits=_empty_policy(),
                value=0.0,
                aux=ModelAuxOutputs(rank_pt_value=5.0),
            ),
        ]
    )
    placement_scorer = DefaultActionScorer(
        adapter=placement_adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
        rank_pt_lambda=0.1,
    )
    placement_result = placement_scorer.score(ctx)

    assert base_result.chosen["pai"] == "1m"
    assert placement_result.chosen["pai"] == "2m"
    assert [candidate.action["pai"] for candidate in placement_result.candidates] == ["2m", "1m"]
    for candidate in placement_result.candidates:
        assert candidate.meta["placement_source"] == "after_discard"
        assert "placement_rank_pt_value" in candidate.meta
        assert "placement_bonus" in candidate.meta


def test_default_action_scorer_runtime_placement_reranks_meld_over_none(monkeypatch):
    chi_action = {"type": "chi", "actor": 0, "target": 3, "pai": "5m", "consumed": ["6m", "7m"]}
    none_action = {"type": "none"}
    monkeypatch.setattr(
        inference_scoring,
        "_resolve_continuation_scenarios",
        lambda snap, actor, action: [
            {
                "projected_snapshot": {"marker": "chi_followup"},
                "legal_actions": [{"type": "dahai", "actor": actor, "pai": "9m", "tsumogiri": False}],
                "weight": 1.0,
                "continuation_kind": "post_meld_followup",
                "declaration_action": None,
            }
        ]
        if action == chi_action
        else [],
    )
    logits = _empty_policy()
    logits[action_to_idx(chi_action)] = 0.2
    logits[action_to_idx(none_action)] = 0.3
    ctx = DecisionContext(
        actor=0,
        event={"type": "dahai", "actor": 3, "pai": "5m"},
        runtime_snap={"hand": ["5mr", "6m", "7m", "9m"], "discards": [[], [], [], []], "melds": [[], [], [], []]},
        model_snap={"hand": ["5mr", "6m", "7m", "9m"]},
        legal_actions=[chi_action, none_action],
    )

    base_adapter = _FakeV4Adapter(
        [
            ModelForwardResult(policy_logits=logits, value=0.0, aux=ModelAuxOutputs()),
            ModelForwardResult(policy_logits=_empty_policy(), value=0.0, aux=ModelAuxOutputs()),
            ModelForwardResult(
                policy_logits=_policy_with_scores({("dahai", "9m"): 0.0}),
                value=0.0,
                aux=ModelAuxOutputs(rank_pt_value=0.0),
            ),
        ]
    )
    base_result = DefaultActionScorer(
        adapter=base_adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
        rank_pt_lambda=0.0,
    ).score(ctx)

    placement_adapter = _FakeV4Adapter(
        [
            ModelForwardResult(policy_logits=logits, value=0.0, aux=ModelAuxOutputs(rank_pt_value=0.0)),
            ModelForwardResult(
                policy_logits=_policy_with_scores({("dahai", "9m"): 0.0}),
                value=0.0,
                aux=ModelAuxOutputs(rank_pt_value=5.0),
            ),
        ]
    )
    placement_result = DefaultActionScorer(
        adapter=placement_adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
        rank_pt_lambda=0.1,
    ).score(ctx)

    assert base_result.chosen["type"] == "none"
    assert placement_result.chosen["type"] == "chi"
    chi_candidate = next(candidate for candidate in placement_result.candidates if candidate.action["type"] == "chi")
    assert chi_candidate.meta["placement_source"] == "meld_continuation"
    assert chi_candidate.meta["placement_bonus"] == pytest.approx(0.5)


def test_default_action_scorer_runtime_placement_reranks_reach_over_discard(monkeypatch):
    reach_action = {"type": "reach", "actor": 0}
    discard_reach = {"type": "dahai", "actor": 0, "pai": "4m", "tsumogiri": False}
    discard_alt = {"type": "dahai", "actor": 0, "pai": "9m", "tsumogiri": False}
    monkeypatch.setattr(
        inference_scoring,
        "_resolve_continuation_scenarios",
        lambda snap, actor, action: [
            {
                "projected_snapshot": {"marker": "reach_followup"},
                "legal_actions": [{"type": "none"}],
                "weight": 1.0,
                "continuation_kind": "reach_declaration",
                "declaration_action": discard_reach,
            }
        ]
        if action == reach_action
        else [],
    )
    primary_logits = _policy_with_scores(
        {("reach", None): 0.1, ("dahai", "4m"): 0.05, ("dahai", "9m"): 0.2}
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "9m"},
        runtime_snap={"hand": ["4m", "9m"], "discards": [[], [], [], []], "melds": [[], [], [], []]},
        model_snap={"hand": ["4m"], "tsumo_pai": "9m"},
        legal_actions=[reach_action, discard_reach, discard_alt],
    )

    base_adapter = _FakeV4Adapter(
        [
            ModelForwardResult(policy_logits=primary_logits, value=0.0, aux=ModelAuxOutputs()),
            ModelForwardResult(policy_logits=_empty_policy(), value=0.0, aux=ModelAuxOutputs(rank_pt_value=0.0)),
        ]
    )
    base_result = DefaultActionScorer(
        adapter=base_adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
        rank_pt_lambda=0.0,
    ).score(ctx)

    placement_adapter = _FakeV4Adapter(
        [
            ModelForwardResult(policy_logits=primary_logits, value=0.0, aux=ModelAuxOutputs(rank_pt_value=0.0)),
            ModelForwardResult(policy_logits=_empty_policy(), value=0.0, aux=ModelAuxOutputs(rank_pt_value=0.0)),
            ModelForwardResult(policy_logits=_empty_policy(), value=0.0, aux=ModelAuxOutputs(rank_pt_value=0.0)),
            ModelForwardResult(policy_logits=_empty_policy(), value=0.0, aux=ModelAuxOutputs(rank_pt_value=5.0)),
        ]
    )
    placement_result = DefaultActionScorer(
        adapter=placement_adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
        rank_pt_lambda=0.1,
    ).score(ctx)

    assert base_result.chosen["pai"] == "9m"
    assert placement_result.chosen["type"] == "reach"
    reach_candidate = next(candidate for candidate in placement_result.candidates if candidate.action["type"] == "reach")
    assert reach_candidate.meta["reach_discard"]["pai"] == "4m"
    assert reach_candidate.meta["placement_source"] == "reach_continuation"
    assert reach_candidate.meta["placement_bonus"] == pytest.approx(0.5)


def test_default_action_scorer_runtime_placement_fails_closed_for_xmodel1():
    adapter = _FakeXmodel1Adapter(
        [
            ModelForwardResult(
                policy_logits=_policy_with_scores({("dahai", "1m"): 0.2}),
                value=0.0,
                aux=ModelAuxOutputs(),
            ),
        ]
    )
    scorer = DefaultActionScorer(
        adapter=adapter,
        beam_k=1,
        beam_lambda=1.0,
        style_lambda=0.0,
        score_delta_lambda=0.0,
        win_prob_lambda=0.0,
        dealin_prob_lambda=0.0,
        rank_pt_lambda=0.1,
    )
    ctx = DecisionContext(
        actor=0,
        event={"type": "tsumo", "actor": 0, "pai": "1m"},
        runtime_snap={"hand": ["1m"], "discards": [[], [], [], []], "melds": [[], [], [], []]},
        model_snap={"hand": [], "tsumo_pai": "1m"},
        legal_actions=[{"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": True}],
    )

    with pytest.raises(RuntimeError, match="only supported for keqingv4"):
        scorer.score(ctx)
