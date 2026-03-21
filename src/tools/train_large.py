from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List


def _chunk(items: List[Path], n: int) -> List[List[Path]]:
    return [items[i : i + n] for i in range(0, len(items), n)]


def _run(cmd: List[str]) -> dict:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return {"cmd": cmd, "returncode": p.returncode, "stdout": p.stdout, "stderr": p.stderr}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", required=True, help="Directory containing tenhou6 json files")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--out-root", default="artifacts/sl_large")
    parser.add_argument("--tmp-root", default="artifacts/tmp_chunks")
    parser.add_argument("--view-mode", choices=["all", "single", "me"], default="all")
    parser.add_argument("--view-name", default="keqing1")
    parser.add_argument("--config", default="configs/mvp.yaml")
    parser.add_argument("--libriichi-bin", default=None)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    files = sorted(raw_dir.glob("*.json"))
    if not files:
        raise RuntimeError(f"no tenhou6 json found in {raw_dir}")

    out_root = Path(args.out_root)
    tmp_root = Path(args.tmp_root)
    out_root.mkdir(parents=True, exist_ok=True)
    if tmp_root.exists():
        shutil.rmtree(tmp_root)
    tmp_root.mkdir(parents=True, exist_ok=True)

    chunks = _chunk(files, args.chunk_size)
    runs = []
    best_loss = math.inf
    best_chunk = -1
    for i, chunk in enumerate(chunks):
        chunk_dir = tmp_root / f"chunk_{i:04d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        for src in chunk:
            dst = chunk_dir / src.name
            try:
                dst.symlink_to(src.resolve())
            except OSError:
                shutil.copy2(src, dst)

        out_dir = out_root / f"chunk_{i:04d}"
        cmd = [
            sys.executable,
            "-m",
            "train.train_sl",
            "--raw-dir",
            str(chunk_dir),
            "--converted-dir",
            str(tmp_root / f"converted_{i:04d}"),
            "--view-mode",
            args.view_mode,
            "--view-name",
            args.view_name,
            "--config",
            args.config,
            "--out-dir",
            str(out_dir),
        ]
        if args.libriichi_bin:
            cmd.extend(["--libriichi-bin", args.libriichi_bin])
        result = _run(cmd)
        metrics = {}
        metrics_path = out_dir / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            loss = float(metrics.get("best_val_loss", math.inf))
            if loss < best_loss:
                best_loss = loss
                best_chunk = i
        runs.append({"chunk_index": i, "num_files": len(chunk), "result": result, "metrics": metrics})
        if result["returncode"] != 0:
            break

    summary = {
        "num_total_files": len(files),
        "num_chunks": len(chunks),
        "chunk_size": args.chunk_size,
        "view_mode": args.view_mode,
        "view_name": args.view_name,
        "best_chunk": best_chunk,
        "best_val_loss": best_loss if best_loss < math.inf else None,
        "runs": runs,
    }
    (out_root / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    if runs and runs[-1]["result"]["returncode"] != 0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

