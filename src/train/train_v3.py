"""keqingv3 训练薄封装。"""

from __future__ import annotations

import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "src"))

    from train.train_task import main as train_task_main

    argv = sys.argv[:]
    if "--task" not in argv:
        argv.extend(["--task", "v3_base"])
    if "--config" not in argv:
        argv.extend(["--config", "configs/keqingv3_default.yaml"])
    sys.argv = argv
    train_task_main()


if __name__ == "__main__":
    main()
