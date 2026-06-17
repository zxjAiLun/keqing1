#!/usr/bin/env python3
"""Run one_vs_three_smoke.py in chunks with cool-down between chunks."""
import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--challenger", type=Path, required=True)
    p.add_argument("--champion", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--require-cuda", action="store_true")
    p.add_argument("--seed-start", type=int, required=True)
    p.add_argument("--seed-key", type=int, default=8192)
    p.add_argument("--seed-count", type=int, required=True)
    p.add_argument("--chunk-size", type=int, default=25)
    p.add_argument("--cool-down-sec", type=int, default=300)
    p.add_argument("--resume", action="store_true", help="skip chunks with complete metrics.json")
    p.add_argument("--allow-partial", action="store_true", help="aggregate successful chunks instead of failing on the first chunk error")
    p.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="pass through to one_vs_three_smoke.py; 0 keeps each chunk as one native batch",
    )
    p.add_argument("--challenger-label", default="challenger")
    p.add_argument("--champion-label", default="champion")
    return p.parse_args()


def cuda_preflight(require_cuda: bool) -> None:
    if not require_cuda:
        return
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required but torch.cuda.is_available() is False")
    print(f"[cuda] {torch.cuda.get_device_name(0)}", flush=True)


def _now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_progress(output_root, payload):
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "progress.json").write_text(
        json.dumps({"updated_at": _now_iso(), **payload}, ensure_ascii=False, indent=2) + "\n"
    )


def _read_chunk_metrics(path):
    metrics_path = path / "metrics.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text())


def _chunk_summary(chunk_idx, seed_start, seed_count, result, status, returncode=0, elapsed_sec=0.0):
    run = result.get("run", {}) if result else {}
    metrics = result.get("metrics", {}).get("challenger", {}) if result else {}
    return {
        "chunk": chunk_idx,
        "status": status,
        "seed_start": seed_start,
        "seed_count": seed_count,
        "device": run.get("device"),
        "returncode": returncode,
        "elapsed_sec": round(float(elapsed_sec), 3),
        "games": metrics.get("games", 0),
        "rank_counts": metrics.get("rank_counts", [0, 0, 0, 0]),
        "avg_rank_pt_tenhou_reference": metrics.get("avg_rank_pt_tenhou_reference"),
    }


def run_chunk(args, chunk_idx, seed_start, seed_count, output_dir):
    cmd = [
        sys.executable,
        str(REPO / "scripts/mortal/one_vs_three_smoke.py"),
        "--challenger", str(args.challenger),
        "--champion", str(args.champion),
        "--device", str(args.device),
        "--seed-start", str(seed_start),
        "--seed-key", str(args.seed_key),
        "--seed-count", str(seed_count),
        "--challenger-label", args.challenger_label,
        "--champion-label", args.champion_label,
        "--output-dir", str(output_dir),
    ]
    if args.progress_every > 0:
        cmd.extend(["--progress-every", str(args.progress_every)])
    print(f"[chunk {chunk_idx}] seeds {seed_start}-{seed_start + seed_count - 1}", flush=True)
    started_at = time.monotonic()
    proc = subprocess.Popen(
        cmd,
        cwd=str(REPO),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stderr_tail = []
    assert proc.stderr is not None
    for line in proc.stderr:
        stderr_tail.append(line)
        stderr_tail = stderr_tail[-20:]
        print(f"  {line.rstrip()}", flush=True)
    assert proc.stdout is not None
    stdout = proc.stdout.read()
    returncode = proc.wait()
    elapsed = time.monotonic() - started_at
    if returncode != 0:
        print(f"  FAIL after {elapsed:.1f}s: {''.join(stderr_tail)[-500:]}", flush=True)
        return None, _chunk_summary(chunk_idx, seed_start, seed_count, None, "failed", returncode=returncode, elapsed_sec=elapsed)
    print(f"  done in {elapsed:.1f}s", flush=True)
    print(f"  {stdout.strip()}", flush=True)
    parsed = json.loads(stdout)
    metrics = _read_chunk_metrics(output_dir)
    return parsed, _chunk_summary(chunk_idx, seed_start, seed_count, metrics, "completed", returncode=0, elapsed_sec=elapsed)


def main():
    args = parse_args()
    cuda_preflight(bool(args.require_cuda))
    total = args.seed_count
    chunk_size = args.chunk_size
    num_chunks = (total + chunk_size - 1) // chunk_size
    output_root = Path(args.output_dir)

    all_results = []
    chunk_summaries = []
    for i in range(num_chunks):
        start = args.seed_start + i * chunk_size
        remaining = total - i * chunk_size
        count = min(chunk_size, remaining)
        chunk_dir = output_root / f"chunk_{i:03d}"
        if args.resume:
            metrics = _read_chunk_metrics(chunk_dir)
            if metrics is not None:
                actual_run = metrics.get("run", {})
                if (
                    int(actual_run.get("seed_start", -1)) == start
                    and int(actual_run.get("seed_count", -1)) == count
                    and int(actual_run.get("seed_key", -1)) == int(args.seed_key)
                    and str(actual_run.get("device")) == str(args.device)
                ):
                    print(f"[chunk {i}] resume skip complete chunk {chunk_dir}", flush=True)
                    all_results.append(metrics["metrics"])
                    chunk_summaries.append(_chunk_summary(i, start, count, metrics, "skipped"))
                    continue
                raise SystemExit(f"resume chunk metadata mismatch: {chunk_dir / 'metrics.json'}")
        write_progress(
            output_root,
            {
                "status": "running",
                "chunks_completed": len(all_results),
                "chunks_total": num_chunks,
                "current_chunk": i,
                "current_seed_start": start,
                "current_seed_count": count,
                "seed_start": args.seed_start,
                "seed_key": args.seed_key,
                "seed_count": total,
            },
        )
        result, chunk_summary = run_chunk(args, i, start, count, chunk_dir)
        chunk_summaries.append(chunk_summary)
        if result is not None:
            all_results.append(result)
        elif not args.allow_partial:
            write_progress(
                output_root,
                {
                    "status": "failed",
                    "chunks_completed": len(all_results),
                    "chunks_total": num_chunks,
                    "failed_chunk": i,
                    "chunk_summaries": chunk_summaries,
                    "seed_start": args.seed_start,
                    "seed_key": args.seed_key,
                    "seed_count": total,
                },
            )
            sys.exit(1)
        write_progress(
            output_root,
            {
                "status": "running",
                "chunks_completed": len(all_results),
                "chunks_total": num_chunks,
                "last_finished_chunk": i,
                "seed_start": args.seed_start,
                "seed_key": args.seed_key,
                "seed_count": total,
                "chunk_summaries": chunk_summaries,
            },
        )
        if i < num_chunks - 1:
            print(f"  cooling down {args.cool_down_sec}s...", flush=True)
            time.sleep(args.cool_down_sec)

    # Aggregate
    total_games = 0
    all_ranks = [0, 0, 0, 0]
    for r in all_results:
        rc = r["challenger"]["rank_counts"]
        for j in range(4):
            all_ranks[j] += rc[j]
        total_games += r["challenger"]["games"]

    if total_games == 0:
        print("no results to aggregate", flush=True)
        sys.exit(1)

    avg_rank = sum((j + 1) * all_ranks[j] for j in range(4)) / total_games
    pts = [90, 45, 0, -135]
    avg_pt = sum(p * all_ranks[j] for j, p in enumerate(pts)) / total_games

    summary = {
        "chunked": True,
        "chunks_completed": len(all_results),
        "chunks_total": num_chunks,
        "complete": len(all_results) == num_chunks,
        "seed_start": args.seed_start,
        "seed_key": args.seed_key,
        "seed_count": total,
        "device": str(args.device),
        "chunk_summaries": chunk_summaries,
        "challenger": {
            "games": total_games,
            "rank_counts": all_ranks,
            "avg_rank": avg_rank,
            "avg_rank_pt_tenhou_reference": avg_pt,
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "aggregated_metrics.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )
    if len(all_results) != num_chunks and not args.allow_partial:
        write_progress(
            output_root,
            {
                "status": "failed",
                "chunks_completed": len(all_results),
                "chunks_total": num_chunks,
                "seed_start": args.seed_start,
                "seed_key": args.seed_key,
                "seed_count": total,
                "aggregated_metrics": str(output_root / "aggregated_metrics.json"),
                "chunk_summaries": chunk_summaries,
            },
        )
        sys.exit(1)
    write_progress(
        output_root,
        {
            "status": "completed" if len(all_results) == num_chunks else "partial",
            "chunks_completed": len(all_results),
            "chunks_total": num_chunks,
            "seed_start": args.seed_start,
            "seed_key": args.seed_key,
            "seed_count": total,
            "aggregated_metrics": str(output_root / "aggregated_metrics.json"),
            "chunk_summaries": chunk_summaries,
        },
    )
    print(f"\n=== AGGREGATED ===", flush=True)
    print(f"  games: {total_games}", flush=True)
    print(f"  ranks: {all_ranks}", flush=True)
    print(f"  avg_rank: {avg_rank:.4f}", flush=True)
    print(f"  tenhou_pt: {avg_pt:+.4f}", flush=True)
    print(f"  saved: {output_root / 'aggregated_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
