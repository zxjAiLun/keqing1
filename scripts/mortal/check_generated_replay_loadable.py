#!/usr/bin/env python3
"""Check whether generated MJAI .json.gz replays are loadable by Mortal/libriichi."""
# ruff: noqa: E402

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
import sys
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-check generated replays with libriichi native loaders")
    parser.add_argument("paths", nargs="+", help="Replay files, directories, or glob patterns")
    parser.add_argument("--mortal-root", type=Path, default=Path("third_party/Mortal"))
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--oracle", action="store_true")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--allow-empty", action="store_true")
    return parser.parse_args()


def expand_replay_paths(values: Sequence[str | Path]) -> list[str]:
    files: list[Path] = []
    for value in values:
        text = str(value)
        path = Path(text)
        if path.is_dir():
            files.extend(sorted(path.glob("**/*.json.gz")))
            continue
        matches = [Path(match) for match in sorted(glob.glob(text))]
        if matches:
            for match in matches:
                if match.is_dir():
                    files.extend(sorted(match.glob("**/*.json.gz")))
                elif match.name.endswith(".json.gz"):
                    files.append(match)
            continue
        if path.name.endswith(".json.gz"):
            files.append(path)
    unique = sorted({str(path) for path in files})
    if not unique:
        raise FileNotFoundError(f"no .json.gz replay files found from: {list(values)}")
    return unique


def check_loadable(
    replay_files: Sequence[str],
    *,
    mortal_root: str | Path = Path("third_party/Mortal"),
    version: int = 4,
    oracle: bool = False,
) -> dict[str, Any]:
    mortal_python_dir = (Path(mortal_root) / "mortal").resolve()
    if str(mortal_python_dir) not in sys.path:
        sys.path.insert(0, str(mortal_python_dir))

    from libriichi.dataset import GameplayLoader, Grp  # noqa: PLC0415

    files = [str(path) for path in replay_files]
    loader = GameplayLoader(version=int(version), oracle=bool(oracle))
    gameplay_files = loader.load_gz_log_files(files)
    grp_games = Grp.load_gz_log_files(files)

    gameplay_group_count = 0
    obs_rows = 0
    action_rows = 0
    mask_rows = 0
    player_ids: list[int] = []
    for gameplay_file in gameplay_files:
        for game in gameplay_file:
            gameplay_group_count += 1
            obs = game.take_obs()
            actions = game.take_actions()
            masks = game.take_masks()
            player_ids.append(int(game.take_player_id()))
            obs_rows += int(len(obs))
            action_rows += int(len(actions))
            mask_rows += int(len(masks))

    grp_rows = 0
    for grp in grp_games:
        feature = grp.take_feature()
        grp_rows += int(feature.shape[0])

    return {
        "schema": "keqing.mortal.generated_replay_loadable.v1",
        "files": files,
        "file_count": len(files),
        "version": int(version),
        "oracle": bool(oracle),
        "gameplay_group_count": int(gameplay_group_count),
        "obs_rows": int(obs_rows),
        "action_rows": int(action_rows),
        "mask_rows": int(mask_rows),
        "grp_game_count": int(len(grp_games)),
        "grp_rows": int(grp_rows),
        "player_ids": sorted(set(player_ids)),
        "loadable": bool(obs_rows > 0 and action_rows > 0 and mask_rows > 0 and grp_rows > 0),
    }


def main() -> None:
    args = _parse_args()
    replay_files = expand_replay_paths(args.paths)
    report = check_loadable(
        replay_files,
        mortal_root=args.mortal_root,
        version=int(args.version),
        oracle=bool(args.oracle),
    )
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    if not report["loadable"] and not args.allow_empty:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
