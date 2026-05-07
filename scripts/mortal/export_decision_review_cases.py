#!/usr/bin/env python3
"""Deprecated KeqingRL decision-review exporter.

The previous implementation depended on the archived KeqingRL imitation stack.
Use ``materialize_replay_sidecars.py`` for the active Mortal replay review
pipeline.
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(description="Deprecated KeqingRL decision-review exporter")
    parser.parse_args()
    raise SystemExit(
        "scripts/mortal/export_decision_review_cases.py is archived: "
        "KeqingRL imitation code has been removed. Use "
        "scripts/mortal/materialize_replay_sidecars.py for active Mortal review."
    )


if __name__ == "__main__":
    main()
