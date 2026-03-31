#!/usr/bin/env python3
"""keqingv1 预处理入口：复用共享 preprocess runner。"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from training.preprocess import BasePreprocessAdapter, run_preprocess


def main():
    run_preprocess(
        default_output_dir="processed",
        adapter=BasePreprocessAdapter(),
    )


if __name__ == "__main__":
    main()
