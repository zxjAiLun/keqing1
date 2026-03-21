from convert.libriichi_bridge import convert_raw_to_mjai
from mahjong_env.replay import build_supervised_samples, read_mjai_jsonl


def test_value_target_from_score_delta() -> None:
    out = "artifacts/converted/value_target_test.jsonl"
    convert_raw_to_mjai("1.json", out, libriichi_bin=None)
    events = read_mjai_jsonl(out)
    samples = build_supervised_samples(events, actor_name_filter={"私"})
    assert samples
    vals = [s.value_target for s in samples]
    assert max(vals) - min(vals) < 1e-6  # same actor in one kyoku
    assert abs(vals[0]) > 0.0

