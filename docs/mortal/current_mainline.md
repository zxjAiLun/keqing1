# Mortal Mainline

## Data

`artifacts/experiments/model_pool_2026_07/V2_data/` contains the active corpus: 6,000 unique native four-player hanchans and one `ext_mortal` training seat per game. Its final audit requires 6,000 files, 6,000 canonical unique logs, zero malformed logs, and zero seed/key overlaps.

## Naming

The external checkpoint is named `ext_mortal` in project code and reports. This avoids colliding with future project-generated `V` series names.

## Training

Run `scripts/mortal/run_v2_population_mixed_warmstart.ps1 -RunTraining` from Windows. The runner warm-starts from 70k weights with a fresh optimizer and data stream, archives steps 72,000 and 74,000, and trains only the DQN, CQL, and next-rank objectives.

V2 used the legacy sparse `terminal_rank` reward. New training uses the project-owned `reward.mode = "final_rank_mc"`, which assigns centered final-rank returns `[+3,+1,-1,-3]` to every decision in a hanchan.

V3 used the same 6,000-hanchan corpus and `reward.mode = "final_rank_mc"`. It was trained from the 70k anchor to steps 72,000 and 74,000. The full result is recorded in [`v3_final_rank_mc_2026_07.md`](v3_final_rank_mc_2026_07.md).

## Evaluation

Use `scripts/mortal/four_player_native.py` for CUDA native random-seat arenas and `scripts/mortal/build_platform_account_report.py` for per-account Pt/R and behavior summaries. Prefer average rank points, rank distribution, and behavior metrics over one-direction 1v3 gates.

The promotion gate is a balanced model-pool league, not an automatic replay regeneration step. A new replay pool is created only when the next hypothesis changes the data distribution or training objective. The V3 1,000-hanchan league used five 200-hanchan lineups so each model family received exactly 1,000 seat-hanchans.

The GRP-versus-final-rank matched A/B is documented in [`reward_ab_2026_07.md`](reward_ab_2026_07.md). After 3,000 paired hanchans, GRP delta-Pt remains mixed across training seeds and is not promoted to the default reward. The project-owned `keqing_grp_v1` remains frozen; it was trained only on an independent 2,000-hanchan corpus outside the formal 6,000-hanchan reward corpus. Add new matched F/G training seeds before introducing Adam-moment, LR, CQL, or architecture variables.
