from __future__ import annotations

import argparse
import json

from mahjong_env.replay import read_mjai_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-path", required=True)
    args = parser.parse_args()

    try:
        import riichi  # type: ignore
    except Exception as e:
        raise RuntimeError(f"riichi module import failed: {e}") from e

    states = [riichi.state.PlayerState(pid) for pid in range(4)]
    ok = 0
    skipped = 0
    for event in read_mjai_jsonl(args.log_path):
        et = event.get("type")
        if et in {"hora", "ryukyoku", "end_kyoku", "end_game"}:
            skipped += 1
            continue
        payload = json.dumps(event, ensure_ascii=False)
        for s in states:
            s.update(payload)
        ok += 1
    print(json.dumps({"riichi_validate_ok_events": ok, "riichi_validate_skipped_events": skipped}, ensure_ascii=False))


if __name__ == "__main__":
    main()

