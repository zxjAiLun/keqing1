from mahjong_env.replay import build_supervised_samples, read_mjai_jsonl
from train.dataset import OBS_DIM, vectorize_state


def test_obs_dim_and_values() -> None:
    events = read_mjai_jsonl("log.jsonl")
    samples = build_supervised_samples(events)
    assert samples
    s = samples[0]
    obs = vectorize_state(s.state, s.actor)
    assert len(obs) == OBS_DIM
    assert obs.sum() > 0

