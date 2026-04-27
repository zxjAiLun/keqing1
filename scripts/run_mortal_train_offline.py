#!/usr/bin/env python3
"""Run Mortal offline training with lazy test-play baseline loading."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run third_party/Mortal offline train.py")
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--config", type=Path, required=True)
    return parser.parse_args()


def install_lazy_test_player(mortal_python_dir: Path) -> None:
    mortal_python_dir = mortal_python_dir.resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))

    import player as mortal_player  # noqa: PLC0415

    original_test_player = mortal_player.TestPlayer

    class LazyTestPlayer:
        def __init__(self) -> None:
            self._inner = None

        def test_play(self, *args, **kwargs):
            if self._inner is None:
                self._inner = original_test_player()
            return self._inner.test_play(*args, **kwargs)

    mortal_player.TestPlayer = LazyTestPlayer


def main() -> None:
    args = _parse_args()
    mortal_root = args.mortal_root.resolve()
    mortal_python_dir = mortal_root / "mortal"
    config_path = args.config.resolve()
    if not mortal_python_dir.exists():
        raise FileNotFoundError(f"Mortal python directory does not exist: {mortal_python_dir}")
    if not config_path.exists():
        raise FileNotFoundError(f"Mortal config does not exist: {config_path}")

    os.environ["MORTAL_CFG"] = str(config_path)
    install_lazy_test_player(mortal_python_dir)
    os.chdir(mortal_python_dir)

    import train as mortal_train  # noqa: PLC0415

    mortal_train.train()


if __name__ == "__main__":
    main()
