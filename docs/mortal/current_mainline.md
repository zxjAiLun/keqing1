# Mortal Mainline

## Data

`artifacts/experiments/model_pool_2026_07/V2_data/` contains the active V2 corpus: 6,000 unique native four-player hanchans and one `model_v4` training seat per game. Its final audit requires 6,000 files, 6,000 canonical unique logs, zero malformed logs, and zero seed/key overlaps.

## Training

Run `scripts/mortal/run_v2_population_mixed_warmstart.ps1 -RunTraining` from Windows. The runner warm-starts from 70k weights with a fresh optimizer and data stream, archives steps 72,000 and 74,000, and trains only the DQN, CQL, and next-rank objectives.

## Evaluation

Use `scripts/mortal/four_player_native.py` for CUDA native random-seat arenas and `scripts/mortal/build_platform_account_report.py` for per-account Pt/R and behavior summaries. Prefer average rank points, rank distribution, and behavior metrics over one-direction 1v3 gates.
