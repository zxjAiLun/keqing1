"""Opt-in periodic publishing of complete native-runner logs to Live Ladder."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
import time
from typing import Any, Sequence


def add_ladder_publish_args(parser: argparse.ArgumentParser) -> None:
    """Add the common opt-in publisher flags to a native runner."""
    parser.add_argument(
        "--ladder-registry",
        type=Path,
        default=None,
        help="external dynamic season registry; omit to disable publishing",
    )
    parser.add_argument(
        "--ladder-publish-every-games",
        type=int,
        default=0,
        help="publish after at least N newly completed hanchans (0 disables game cadence)",
    )
    parser.add_argument(
        "--ladder-publish-every-seconds",
        type=float,
        default=0.0,
        help="publish after at least N seconds since the last publish (0 disables time cadence)",
    )
    parser.add_argument(
        "--ladder-snapshot-root",
        type=Path,
        default=None,
        help="optional external snapshot root passed to publish_ladder_snapshot.py",
    )
    parser.add_argument(
        "--ladder-platform-model-label",
        default=None,
        help="optional forced account model label for the publisher",
    )
    parser.add_argument(
        "--ladder-interleave-log-dirs",
        action="store_true",
        help="forward interleaving when a runner supplies multiple complete log directories",
    )
    parser.add_argument(
        "--ladder-publish-best-effort",
        action="store_true",
        help="log publisher errors and continue the native run instead of failing fast",
    )


@dataclass
class LadderPublishHook:
    """Publish a full log set at explicit game/time boundaries.

    The hook never passes an incremental subset: every call forwards the complete
    log directory list received at construction time.  A registry is required for
    the hook to be enabled; without it native runners retain their old behavior.
    """

    registry_path: Path | None
    log_dirs: Sequence[Path]
    mortal_root: Path
    every_games: int = 0
    every_seconds: float = 0.0
    snapshot_root: Path | None = None
    platform_model_label: str | None = None
    interleave_log_dirs: bool = False
    best_effort: bool = False
    last_published_games: int = 0
    last_published_at: float = field(default_factory=time.monotonic)

    def __post_init__(self) -> None:
        if self.every_games < 0:
            raise ValueError("ladder publish every_games cannot be negative")
        if self.every_seconds < 0:
            raise ValueError("ladder publish every_seconds cannot be negative")
        if self.registry_path is None:
            if self.every_games or self.every_seconds:
                raise ValueError("ladder publish cadence requires --ladder-registry")
            return
        if not self.log_dirs:
            raise ValueError("ladder publisher requires at least one complete log directory")

    @property
    def enabled(self) -> bool:
        return self.registry_path is not None

    def metadata(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "registry": str(self.registry_path) if self.registry_path else None,
            "log_dirs": [str(path) for path in self.log_dirs],
            "every_games": int(self.every_games),
            "every_seconds": float(self.every_seconds),
            "snapshot_root": str(self.snapshot_root) if self.snapshot_root else None,
            "best_effort": bool(self.best_effort),
            "publishes_complete_log_set": True,
        }

    def _due(self, completed_games: int, *, force: bool) -> bool:
        if not self.enabled or completed_games <= self.last_published_games:
            return False
        if force:
            return True
        game_due = self.every_games > 0 and completed_games - self.last_published_games >= self.every_games
        time_due = self.every_seconds > 0 and time.monotonic() - self.last_published_at >= self.every_seconds
        return game_due or time_due

    def publish(self, completed_games: int, *, force: bool = False) -> dict[str, Any] | None:
        """Publish if due and return the publisher result."""
        if not self._due(completed_games, force=force):
            return None
        from scripts.mortal.publish_ladder_snapshot import publish_snapshot

        print(
            f"[ladder] publishing complete log set at games={completed_games} "
            f"registry={self.registry_path}",
            flush=True,
        )
        try:
            result = publish_snapshot(
                registry_path=self.registry_path,
                log_dirs=list(self.log_dirs),
                snapshot_root=self.snapshot_root,
                mortal_root=self.mortal_root,
                platform_model_label=self.platform_model_label,
                interleave_log_dirs=self.interleave_log_dirs,
            )
        except Exception as exc:
            if not self.best_effort:
                raise
            print(f"[ladder] publish failed (continuing): {exc}", flush=True)
            return None
        self.last_published_games = int(completed_games)
        self.last_published_at = time.monotonic()
        print(f"[ladder] published snapshot={result.get('snapshot_dir')}", flush=True)
        return result


def hook_from_args(args: argparse.Namespace, *, log_dirs: Sequence[Path], mortal_root: Path) -> LadderPublishHook:
    """Construct a hook from either CLI args or a test Namespace."""
    return LadderPublishHook(
        registry_path=getattr(args, "ladder_registry", None),
        log_dirs=tuple(log_dirs),
        mortal_root=mortal_root,
        every_games=int(getattr(args, "ladder_publish_every_games", 0) or 0),
        every_seconds=float(getattr(args, "ladder_publish_every_seconds", 0.0) or 0.0),
        snapshot_root=getattr(args, "ladder_snapshot_root", None),
        platform_model_label=getattr(args, "ladder_platform_model_label", None),
        interleave_log_dirs=bool(getattr(args, "ladder_interleave_log_dirs", False)),
        best_effort=bool(getattr(args, "ladder_publish_best_effort", False)),
    )
