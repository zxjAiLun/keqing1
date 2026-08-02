from argparse import Namespace
from pathlib import Path
from unittest.mock import patch

import pytest

from scripts.mortal.ladder_publish_hook import LadderPublishHook, hook_from_args


def test_disabled_hook_does_not_publish() -> None:
    hook = LadderPublishHook(registry_path=None, log_dirs=(Path("logs"),), mortal_root=Path("mortal"))

    assert hook.publish(25) is None
    assert hook.metadata()["enabled"] is False


def test_cadence_requires_registry() -> None:
    with pytest.raises(ValueError, match="requires --ladder-registry"):
        LadderPublishHook(
            registry_path=None,
            log_dirs=(Path("logs"),),
            mortal_root=Path("mortal"),
            every_games=25,
        )


@patch("scripts.mortal.publish_ladder_snapshot.publish_snapshot")
def test_game_cadence_passes_complete_log_set(mock_publish) -> None:
    mock_publish.return_value = {"snapshot_dir": "snapshot-25", "games": 25}
    hook = LadderPublishHook(
        registry_path=Path("registry.json"),
        log_dirs=(Path("logs-a"), Path("logs-b")),
        mortal_root=Path("mortal"),
        every_games=25,
    )

    assert hook.publish(24) is None
    result = hook.publish(25)
    assert result == {"snapshot_dir": "snapshot-25", "games": 25}
    mock_publish.assert_called_once_with(
        registry_path=Path("registry.json"),
        log_dirs=[Path("logs-a"), Path("logs-b")],
        snapshot_root=None,
        mortal_root=Path("mortal"),
        platform_model_label=None,
        interleave_log_dirs=False,
    )
    assert hook.publish(25, force=True) is None


@patch("scripts.mortal.publish_ladder_snapshot.publish_snapshot")
def test_hook_from_args_preserves_final_only_opt_in(mock_publish) -> None:
    mock_publish.return_value = {"snapshot_dir": "snapshot-10", "games": 10}
    args = Namespace(ladder_registry=Path("registry.json"))
    hook = hook_from_args(args, log_dirs=(Path("logs"),), mortal_root=Path("mortal"))

    assert hook.enabled is True
    assert hook.publish(10, force=True)["games"] == 10
    mock_publish.assert_called_once()
