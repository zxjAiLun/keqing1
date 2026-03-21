from mahjong_env.replay import build_supervised_samples, read_mjai_jsonl, replay_validate_label_legal


def test_replay_and_legality() -> None:
    events = read_mjai_jsonl("log.jsonl")
    samples = build_supervised_samples(events)
    assert len(samples) > 0
    errors = replay_validate_label_legal(samples)
    assert errors == []

