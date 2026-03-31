from mahjong_env.replay import PendingValueSample, ReplaySample, _finalize_aux_targets


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
