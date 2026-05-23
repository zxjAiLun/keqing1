#!/usr/bin/env python3
"""Watch for checkpoint archives and auto-stop online training when targets reached."""
import os
import signal
import sys
import time
from pathlib import Path


def main():
    if len(sys.argv) < 3:
        print("usage: python watch_trainer.py <checkpoints_dir> <target_step> [target_step ...]", flush=True)
        print("  example: python watch_trainer.py .../checkpoints 70400 70800 71200", flush=True)
        sys.exit(1)

    checkpoints_dir = Path(sys.argv[1])
    targets = set(int(s) for s in sys.argv[2:])

    print(f"watching {checkpoints_dir} for steps: {sorted(targets)}", flush=True)

    while True:
        archived = set()
        if checkpoints_dir.exists():
            for f in checkpoints_dir.iterdir():
                if f.suffix == ".pth" and not f.name.endswith(".tmp"):
                    name = f.stem
                    # extract step from filename like mortal_online_70400
                    parts = name.split("_")
                    for p in parts:
                        try:
                            step = int(p)
                            archived.add(step)
                        except ValueError:
                            pass

        remaining = targets - archived
        if not remaining:
            print(f"all targets archived: {sorted(archived)}", flush=True)
            # Kill server, trainer, client
            _kill_by_name("server.py")
            _kill_by_name("train.py")
            _kill_by_name("client.py")
            _kill_by_name("archive_online")
            print("done - all processes killed", flush=True)
            sys.exit(0)

        if archived:
            print(f"  archived: {sorted(archived)}, waiting for: {sorted(remaining)}", flush=True)

        time.sleep(10)


def _kill_by_name(name: str):
    try:
        os.system(f"pkill -f '{name}' 2>/dev/null")
    except Exception:
        pass


if __name__ == "__main__":
    main()
