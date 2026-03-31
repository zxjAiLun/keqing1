#!/usr/bin/env python3
"""统一预处理入口：选择 cache adapter 和 value strategy。"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.preprocess import (
    BasePreprocessAdapter,
    MeldRankPreprocessAdapter,
    V3PreprocessAdapter,
    run_preprocess,
)


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--task", choices=["base", "meld_rank", "v3_base"], default="base")
    parser.add_argument("--output-dir-default", default=None)
    known, remaining = parser.parse_known_args()

    sys.argv = [sys.argv[0], *remaining]
    if known.task == "meld_rank":
        adapter = MeldRankPreprocessAdapter()
        encode_module = "keqingv1.features"
        default_value_strategy = "heuristic"
    elif known.task == "v3_base":
        adapter = V3PreprocessAdapter()
        encode_module = "keqingv3.features"
        default_value_strategy = "mc_return"
    else:
        adapter = BasePreprocessAdapter()
        encode_module = "keqingv1.features"
        default_value_strategy = "heuristic"
    default_output_dir = known.output_dir_default or (
        "processed_v3" if known.task == "v3_base"
        else ("processed" if known.task == "base" else "processed_v2")
    )
    run_preprocess(
        default_output_dir=default_output_dir,
        adapter=adapter,
        default_value_strategy=default_value_strategy,
        encode_module=encode_module,
    )


if __name__ == "__main__":
    main()
