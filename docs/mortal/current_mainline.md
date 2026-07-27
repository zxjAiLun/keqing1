# Mortal Mainline

## Data

`artifacts/experiments/model_pool_2026_07/V2_data/` contains the active corpus: 6,000 unique native four-player hanchans and one `ext_mortal` training seat per game. Its final audit requires 6,000 files, 6,000 canonical unique logs, zero malformed logs, and zero seed/key overlaps.

## Naming

The external checkpoint is named `ext_mortal` in project code and reports. This avoids colliding with future project-generated `V` series names.

## Training

Run `scripts/mortal/run_v2_population_mixed_warmstart.ps1 -RunTraining` from Windows for a new V2 continuation. The runner warm-starts from the 70k weights **and the 70k Adam state**, while keeping scheduler, scaler, and data stream fresh; it archives steps 72,000 and 74,000 and trains only the DQN, CQL, and next-rank objectives. The V3 preparation/runner follows the same contract.

### 70k continuation contract

For any new legacy continuation from the 70k parent, the command must explicitly pass:

```text
--initialize-from <70k.pth>
--initialize-optimizer-from <same-70k.pth>
--initial-steps 70000
```

The training runner verifies that the two checkpoint SHA256 values are identical. This is a weights-plus-optimizer warm-start only: scheduler, AMP scaler, data cursor, and RNG/data stream are fresh. Existing historical V2/V3 state files retain the initialization provenance recorded when they were trained and are not silently rewritten.

V2 used the legacy sparse `terminal_rank` reward. New training uses the project-owned `reward.mode = "final_rank_mc"`, which assigns centered final-rank returns `[+3,+1,-1,-3]` to every decision in a hanchan.

V3 used the same 6,000-hanchan corpus and `reward.mode = "final_rank_mc"`. It was trained from the 70k anchor to steps 72,000 and 74,000. The full result is recorded in [`v3_final_rank_mc_2026_07.md`](v3_final_rank_mc_2026_07.md).

## Evaluation

Use `scripts/mortal/four_player_native.py` for CUDA native random-seat arenas and `scripts/mortal/build_platform_account_report.py` for per-account Pt/R and behavior summaries. Prefer average rank points, rank distribution, and behavior metrics over one-direction 1v3 gates.

The promotion gate is a balanced model-pool league, not an automatic replay regeneration step. A new replay pool is created only when the next hypothesis changes the data distribution or training objective. The V3 1,000-hanchan league used five 200-hanchan lineups so each model family received exactly 1,000 seat-hanchans.

The GRP-versus-final-rank matched A/B is documented in [`reward_ab_2026_07.md`](reward_ab_2026_07.md). GRP delta-Pt remains mixed across training seeds and is not promoted to the default reward. The six-seed optimizer A/B is documented in [`optimizer_ab_2026_07.md`](optimizer_ab_2026_07.md): preserved Adam is now the default initialization for 70k legacy continuation only, not a checkpoint promotion or a general recipe claim. The project-owned `keqing_grp_v1` remains frozen; it was trained only on an independent corpus outside the formal 6,000-hanchan reward corpus. Do not open new optimizer, LR, CQL, or reward grids; the next research phase is data distribution and project-owned lineage.

The mixed-versus-pure data route is documented in [`data_route_ab_2026_07.md`](data_route_ab_2026_07.md). Three matched training seeds and 1,000-hanchan native evaluations per seed did not support replacing the mixed corpus with pure `ext_mortal` selfplay: S0 minus M0 averaged `-2.325` Pt with an equal-seed hierarchical interval crossing zero. Freeze this route rather than generating another pure pool solely for replication.

The final-rank MC objective audit is documented in [`objective_learnability_2026_07.md`](objective_learnability_2026_07.md). Across the retained M0 and S0 corpora, the target standard deviation is about 2.2 Pt and the 70k parent explains only 3.4--6.3% of sampled target variance. Both 72k candidates show material common-Q-offset and margin drift. This closes pure-selfplay expansion and moves the next hypothesis to one controlled, project-owned objective variant; do not open a broad reward, optimizer, LR, CQL, or data grid.
