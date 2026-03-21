from __future__ import annotations

import argparse
import json
from pathlib import Path

from convert.libriichi_bridge import convert_raw_to_mjai
from convert.validate_mjai import validate_mjai_jsonl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/converted")
    parser.add_argument("--libriichi-bin", default=None)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    converted = []
    for src in sorted(input_dir.glob("*.json")):
        out = output_dir / f"{src.stem}.jsonl"
        info = convert_raw_to_mjai(str(src), str(out), args.libriichi_bin)
        errs = validate_mjai_jsonl(out)
        if errs:
            raise RuntimeError(f"{src.name} failed validation: {errs}")
        converted.append(info)
    print(json.dumps({"count": len(converted), "files": converted}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

