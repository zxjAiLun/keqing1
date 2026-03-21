from __future__ import annotations

import argparse
import json
from pathlib import Path

from convert.libriichi_bridge import convert_raw_to_mjai
from convert.validate_mjai import summarize_mjai, validate_mjai_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Raw tenhou/majsoul json file path")
    parser.add_argument("--output", default="artifacts/converted/log.jsonl")
    parser.add_argument("--libriichi-bin", default=None)
    args = parser.parse_args()

    result = convert_raw_to_mjai(args.input, args.output, args.libriichi_bin)
    errors = validate_mjai_jsonl(args.output)
    if errors:
        raise RuntimeError(f"conversion validation failed: {errors}")
    summary = summarize_mjai(args.output)
    out = {"convert": result, "summary": summary, "output_exists": Path(args.output).exists()}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

