from pathlib import Path

from convert.libriichi_bridge import convert_raw_to_mjai
from convert.validate_mjai import validate_mjai_jsonl
from mahjong_env.replay import build_supervised_samples, read_mjai_jsonl


def test_convert_and_replay_on_1json() -> None:
    out = "artifacts/converted/test_1.jsonl"
    convert_raw_to_mjai("1.json", out, libriichi_bin=None)
    assert Path(out).exists()
    errs = validate_mjai_jsonl(out)
    assert errs == []
    events = read_mjai_jsonl(out)
    samples = build_supervised_samples(events)
    assert len(samples) > 0

