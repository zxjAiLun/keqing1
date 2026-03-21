from mahjong_env.replay import build_supervised_samples, read_mjai_jsonl


def test_actor_name_filter_single_view() -> None:
    events = read_mjai_jsonl("log.jsonl")
    samples = build_supervised_samples(events, actor_name_filter={"p2"})
    assert len(samples) > 0
    assert all(s.actor_name == "p2" for s in samples)

