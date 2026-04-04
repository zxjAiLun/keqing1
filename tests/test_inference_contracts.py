import inference.scoring as inference_scoring
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
)
from keqingv1.action_space import action_to_idx


def test_inference_contract_dataclasses_roundtrip():
    aux = ModelAuxOutputs(score_delta=0.2, win_prob=0.4, dealin_prob=0.1)
    forward = ModelForwardResult(policy_logits=[1.0, 2.0], value=0.3, aux=aux)
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


class _FakeAdapter:
    def __init__(self, forwards):
        self.forwards = list(forwards)
        self.calls = []

    def forward(self, snap: dict, actor: int):
        self.calls.append((snap, actor))
        return self.forwards.pop(0)


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
        model_aux=ModelAuxOutputs(score_delta=0.1, win_prob=0.2, dealin_prob=0.05),
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
