# D2 Project-Owned Descendant-View Mix

## Status

Training is complete for all three preregistered matched seeds. Strength evaluation is pending; this document does not promote a checkpoint.

## Data Contract

- Experiment: `D2_project_owned_descendant_view_mix_2026_08`
- Source corpus: the existing 6,000 D1 hanchans; no new selfplay generation
- Trainable perspective: exactly one perspective per hanchan
- Assignment: 3,000 `V2_74000` and 3,000 `V3_74000`, assigned by canonical-hash order and fixed 50/50 alternation
- `K0_70k` and `ext_mortal` remain environment opponents only
- D2 audit: 0 malformed hanchans; 971,035 decisions; parent-greedy agreement 90.9627%; mean parent-Q behavior rank 1.1081; mean parent-Q regret 0.0390341

## Training Contract

- Parent weights: `mortal_default_70k_promoted_candidate.pth`, step `70000`
- Optimizer initialization: preserved Adam from the same parent checkpoint
- Scheduler, scaler, data stream, and RNG: fresh per seed
- Objective: `behavior_action_mc`
- Reward: `final_rank_mc`
- Target: `72000`
- Batch size: `512`
- Device: CUDA, NVIDIA GeForce RTX 4060 Laptop GPU
- Training commit recorded in each exposure report: `572ee31e9c69a35c01a3f0a6174e0eea49cd0991`
- All training exposure reports record `git_dirty=false`

## Completed Runs

| Seed | Final step | Batches | Samples | Archive steps | Resume note |
| ---: | ---: | ---: | ---: | --- | --- |
| `20260806` | `72000` | `2000` | `1,024,000` | `70001, 70010, 70100, 70500, 71000, 72000` | fresh |
| `20260807` | `72000` | `2000` | `1,024,000` | `70001, 70010, 70100, 70500, 71000, 72000` | resumed from `70500`; skipped `500` previously delivered batches |
| `20260808` | `72000` | `2000` | `1,024,000` | `70001, 70010, 70100, 70500, 71000, 72000` | fresh |

Final checkpoints are stored locally under the corresponding ignored artifact run directories. Each final checkpoint loads with `steps == 72000`; the parent and optimizer checkpoint SHA prefixes are identical: `6c0e70058644`.

## Next Step

Run the fixed B250 random-seat evaluation for each seed with the lineup:

```text
70k / ext_mortal / M0_seed / D2_seed
```

Report `D2-M0`, `D2-70k`, and `M0-70k` separately with complete-hanchan bootstrap and equal-seed hierarchical bootstrap. Do not select a seed or checkpoint from training metrics before the evaluation is complete.
