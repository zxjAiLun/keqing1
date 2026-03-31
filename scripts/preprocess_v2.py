#!/usr/bin/env python3
"""keqingv2 预处理入口：复用共享 preprocess runner + meld rank adapter。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.preprocess import MeldRankPreprocessAdapter, run_preprocess


def main():
    run_preprocess(
        default_output_dir="processed_v2",
        adapter=MeldRankPreprocessAdapter(),
    )


if __name__ == "__main__":
    main()
