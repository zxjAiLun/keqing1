from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _run(cmd: list[str]) -> dict:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-json", default=None)
    parser.add_argument("--raw-dir", default=None)
    parser.add_argument("--converted-out", default="artifacts/converted/pipeline_log.jsonl")
    parser.add_argument("--converted-dir", default="artifacts/converted/pipeline")
    parser.add_argument("--checkpoint-dir", default="artifacts/sl")
    parser.add_argument("--player-id", type=int, default=0)
    parser.add_argument("--view-mode", choices=["all", "single", "me"], default="all")
    parser.add_argument("--view-name", default="keqing1")
    parser.add_argument("--libriichi-bin", default=None)
    args = parser.parse_args()
    if not args.raw_json and not args.raw_dir:
        raise RuntimeError("either --raw-json or --raw-dir is required")

    Path("artifacts/converted").mkdir(parents=True, exist_ok=True)
    replay_log = args.converted_out
    if args.raw_dir:
        candidates = sorted(Path(args.raw_dir).glob("*.json"))
        if not candidates:
            raise RuntimeError(f"no json files in {args.raw_dir}")
        replay_log = f"{args.converted_dir}/{candidates[0].stem}.jsonl"
    steps = []
    if args.raw_json:
        steps.append(
            _run(
                [
                    sys.executable,
                    "-m",
                    "tools.convert_one",
                    "--input",
                    args.raw_json,
                    "--output",
                    args.converted_out,
                ]
                + (["--libriichi-bin", args.libriichi_bin] if args.libriichi_bin else [])
            )
        )
    else:
        steps.append(
            _run(
                [
                    sys.executable,
                    "-m",
                    "tools.convert_dir",
                    "--input-dir",
                    args.raw_dir,
                    "--output-dir",
                    args.converted_dir,
                ]
                + (["--libriichi-bin", args.libriichi_bin] if args.libriichi_bin else [])
            )
        )
    steps.append(
        _run(
            [
                sys.executable,
                "-m",
                "train.train_sl",
                "--view-mode",
                args.view_mode,
                "--view-name",
                args.view_name,
                "--out-dir",
                args.checkpoint_dir,
            ]
            + (["--raw-json", args.raw_json, "--converted-out", args.converted_out] if args.raw_json else [])
            + (["--raw-dir", args.raw_dir, "--converted-dir", args.converted_dir] if args.raw_dir else [])
            + (["--libriichi-bin", args.libriichi_bin] if args.libriichi_bin else [])
        )
    )
    steps.append(
        _run(
            [
                sys.executable,
                "-m",
                "train.eval_offline",
                "--view-mode",
                args.view_mode,
                "--view-name",
                args.view_name,
                "--checkpoint",
                f"{args.checkpoint_dir}/best.npz",
            ]
            + (["--log-path", args.converted_out] if args.raw_json else [])
            + (["--raw-dir", args.raw_dir, "--converted-dir", args.converted_dir] if args.raw_dir else [])
        )
    )
    steps.append(
        _run(
            [
                sys.executable,
                "-m",
                "replay.bot",
                "--log-path",
                replay_log,
                "--checkpoint",
                f"{args.checkpoint_dir}/best.npz",
                "--player-id",
                str(args.player_id),
            ]
        )
    )
    steps.append(
        _run(
            [
                sys.executable,
                "-m",
                "tools.validate_riichi",
                "--log-path",
                replay_log,
            ]
        )
    )

    failed = any(s["returncode"] != 0 for s in steps)
    print(json.dumps({"failed": failed, "steps": steps}, ensure_ascii=False, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

