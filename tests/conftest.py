from __future__ import annotations

from pathlib import Path

import pytest


ARCHIVE_TEST_PREFIXES = (
    "test_keqingv4_",
    "test_xmodel1_",
    "test_xmodel2_",
)
ARCHIVE_TEST_FILES = {
    "test_selfplay_preprocess_alignment.py",
}


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-archive",
        action="store_true",
        default=False,
        help="Run archived Keqingv4/XModel/old SL preprocess tests.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-archive"):
        return
    skip_archive = pytest.mark.skip(
        reason="archived Keqingv4/XModel/old SL preprocess test; use --run-archive to run"
    )
    for item in items:
        path = Path(str(item.fspath))
        name = path.name
        if name in ARCHIVE_TEST_FILES or name.startswith(ARCHIVE_TEST_PREFIXES):
            item.add_marker(pytest.mark.archive)
            item.add_marker(skip_archive)
