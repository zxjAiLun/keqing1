#!/usr/bin/env python3
"""Batch convert Tenhou6 JSON to MJAI JSONL using mjai-reviewer's convlog"""

import subprocess
import os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

CONVLOG_BIN = Path(__file__).parent.parent / "third_party" / "mjai-reviewer" / "target" / "release" / "convlog"


def convert_file(input_path: Path, output_path: Path) -> tuple[bool, str]:
    """Convert a single file using convlog. Returns (success, error_msg)."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if output_path.exists():
        return True, ""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(CONVLOG_BIN), str(input_path), str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        return False, result.stderr.strip()
    return True, ""


def process_directory(input_dir: Path, output_dir: Path, workers: int = 0) -> tuple[int, int, int]:
    """Convert all JSON files in input_dir to output_dir.
    Returns (success, skipped, failed)."""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        return 0, 0, 0

    pairs = [(f, output_dir / (f.stem + ".mjson")) for f in json_files]
    already_done = sum(1 for _, o in pairs if o.exists())
    pending = [(i, o) for i, o in pairs if not o.exists()]

    print(f"  {len(json_files)} files: {already_done} already done, {len(pending)} to convert")

    if not pending:
        return 0, already_done, 0

    n_workers = workers if workers > 0 else min(os.cpu_count() or 4, len(pending))
    success = 0
    failed = 0

    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(convert_file, inp, out): inp for inp, out in pending}
        done_count = 0
        for fut in as_completed(futures):
            ok, err = fut.result()
            done_count += 1
            if ok:
                success += 1
            else:
                failed += 1
                print(f"  ERROR {futures[fut].name}: {err}")
            if done_count % 100 == 0:
                print(f"  Progress: {done_count}/{len(pending)}")

    print(f"  Done: {success} converted, {already_done} skipped, {failed} failed")
    return success, already_done, failed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch convert Tenhou6 JSON to MJAI JSONL")
    parser.add_argument("--workers", type=int, default=6,
                        help="Number of parallel workers (default: auto = cpu_count)")
    parser.add_argument("--input", type=str, default="dataset/tenhou6",
                        help="Base input directory containing ds* subdirs")
    parser.add_argument("--output", type=str, default="artifacts/converted_mjai",
                        help="Base output directory")
    args = parser.parse_args()

    if not CONVLOG_BIN.exists():
        print(f"ERROR: convlog binary not found at {CONVLOG_BIN}")
        print("Build it with: cd third_party/mjai-reviewer && cargo build --release")
        raise SystemExit(1)

    base_input = Path(args.input)
    base_output = Path(args.output)

    total_success = total_skipped = total_failed = 0
    for ds_dir in sorted(base_input.glob("ds*")):
        ds_name = ds_dir.name
        output_ds = base_output / ds_name
        print(f"\n=== {ds_name} ===")
        s, sk, f = process_directory(ds_dir, output_ds, workers=6)
        total_success += s
        total_skipped += sk
        total_failed += f

    print(f"\n=== Total: {total_success} converted, {total_skipped} skipped, {total_failed} failed ===")


if __name__ == "__main__":
    main()
