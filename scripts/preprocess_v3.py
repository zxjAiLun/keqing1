#!/usr/bin/env python3
"""keqingv3 预处理薄封装。"""

from __future__ import annotations

import sys
from pathlib import Path


def main():
    root = Path(__file__).parent.parent
    sys.path.insert(0, str(root / "src"))

    from training.preprocess import V3PreprocessAdapter, run_preprocess
    import sys as _sys

    argv = _sys.argv[:]
    if "--config" not in argv:
        argv.extend(["--config", "configs/keqingv3_preprocess.yaml"])
    _sys.argv = argv

    run_preprocess(
        default_output_dir="processed_v3",
        adapter=V3PreprocessAdapter(),
        default_value_strategy="mc_return",
        encode_module="keqingv3.features",
    )


if __name__ == "__main__":
    main()
