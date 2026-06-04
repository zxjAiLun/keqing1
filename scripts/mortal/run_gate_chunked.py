#!/usr/bin/env python3
"""Run one_vs_three_smoke.py in chunks with cool-down between chunks."""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--challenger", type=Path, required=True)
    p.add_argument("--champion", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--seed-start", type=int, required=True)
    p.add_argument("--seed-key", type=int, default=8192)
    p.add_argument("--seed-count", type=int, required=True)
    p.add_argument("--chunk-size", type=int, default=25)
    p.add_argument("--cool-down-sec", type=int, default=300)
    p.add_argument("--challenger-label", default="challenger")
    p.add_argument("--champion-label", default="champion")
    return p.parse_args()


def run_chunk(args, chunk_idx, seed_start, seed_count, output_dir):
    cmd = [
        "uv", "run", "python",
        str(REPO / "scripts/mortal/one_vs_three_smoke.py"),
        "--challenger", str(args.challenger),
        "--champion", str(args.champion),
        "--seed-start", str(seed_start),
        "--seed-key", str(args.seed_key),
        "--seed-count", str(seed_count),
        "--challenger-label", args.challenger_label,
        "--champion-label", args.champion_label,
        "--output-dir", str(output_dir),
    ]
    print(f"[chunk {chunk_idx}] seeds {seed_start}-{seed_start + seed_count - 1}", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    if result.returncode != 0:
        print(f"  FAIL: {result.stderr[-300:]}", flush=True)
        return None
    print(f"  {result.stdout.strip()}", flush=True)
    return json.loads(result.stdout)


def main():
    args = parse_args()
    total = args.seed_count
    chunk_size = args.chunk_size
    num_chunks = (total + chunk_size - 1) // chunk_size
    output_root = Path(args.output_dir)

    all_results = []
    for i in range(num_chunks):
        start = args.seed_start + i * chunk_size
        remaining = total - i * chunk_size
        count = min(chunk_size, remaining)
        chunk_dir = output_root / f"chunk_{i:03d}"
        result = run_chunk(args, i, start, count, chunk_dir)
        if result is not None:
            all_results.append(result)
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
        "seed_start": args.seed_start,
        "seed_key": args.seed_key,
        "seed_count": total,
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
    print(f"\n=== AGGREGATED ===", flush=True)
    print(f"  games: {total_games}", flush=True)
    print(f"  ranks: {all_ranks}", flush=True)
    print(f"  avg_rank: {avg_rank:.4f}", flush=True)
    print(f"  tenhou_pt: {avg_pt:+.4f}", flush=True)
    print(f"  saved: {output_root / 'aggregated_metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
