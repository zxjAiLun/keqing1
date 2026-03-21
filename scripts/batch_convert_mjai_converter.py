#!/usr/bin/env python3
"""Batch convert Tenhou6 JSON to MJAI JSONL using mjai-reviewer's convlog"""

import subprocess
import sys
import os
from pathlib import Path

CONVLOG_BIN = Path(__file__).parent.parent / "third_party" / "mjai-reviewer" / "target" / "release" / "convlog"

def convert_file(input_path: Path, output_path: Path):
    """Convert a single file using convlog"""
    cmd = [str(CONVLOG_BIN), str(input_path), str(output_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ERROR: {result.stderr}")
        return False
    return True

def process_directory(input_dir: Path, output_dir: Path):
    """Convert all JSON files in input_dir to output_dir"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(input_dir.glob("*.json"))
    print(f"Converting {len(json_files)} files from {input_dir} -> {output_dir}")

    success = 0
    failed = 0
    for i, json_file in enumerate(json_files):
        output_file = output_dir / (json_file.stem + ".mjson")
        if convert_file(json_file, output_file):
            success += 1
        else:
            failed += 1
        if (i + 1) % 100 == 0:
            print(f"  Progress: {i + 1}/{len(json_files)}")

    print(f"Done: {success} succeeded, {failed} failed")
    return success, failed

def main():
    base_input = Path("dataset/tenhou6")
    base_output = Path("artifacts/converted_mjai")

    for ds_dir in sorted(base_input.glob("ds*")):
        ds_name = ds_dir.name
        output_ds = base_output / ds_name
        print(f"\n=== Processing {ds_name} ===")
        process_directory(ds_dir, output_ds)

if __name__ == "__main__":
    main()
