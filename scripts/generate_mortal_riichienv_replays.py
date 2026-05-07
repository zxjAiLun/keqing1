#!/usr/bin/env python3
"""Deprecated wrapper for Mortal RiichiEnv selfplay replay generation.

Use scripts/mortal/generate_riichienv_selfplay_replays.py instead.
"""

import sys

from scripts.mortal import generate_riichienv_selfplay_replays as _impl


if __name__ == "__main__":
    _impl.main()
else:
    sys.modules[__name__] = _impl
