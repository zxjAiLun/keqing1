# Reward / PT Table Experiment Report

Status: R0/R1/R2 short training complete; 100 half-game quick smoke complete.

## Fixed Matrix

Mortal reward uses `E[PT]_{t+1} - E[PT]_t`, so adding a constant to every rank does not change training. `[6,4,2,0]` and `[3,1,-1,-3]` are the same training objective; `zero_sum_balanced` is intentionally not a separate experiment.

| Experiment | Parent | Reward profile | PT table | GRP | Target steps |
| --- | --- | --- | --- | --- | --- |
| R0_mortal_default | `artifacts/mortal_training/mortal.pth@65000` | `mortal_default` | `[6, 4, 2, 0]` | old GRP | 70000 |
| R1_avoid4_norm | `artifacts/mortal_training/mortal.pth@65000` | `avoid4_norm` | `[2.142857, 1.285714, 0.428571, -3.857143]` | old GRP | 70000 |
| R2_top1_norm | `artifacts/mortal_training/mortal.pth@65000` | `top1_norm` | `[3.3, 0.3, -0.9, -2.7]` | old GRP | 70000 |

Optional engineering variants `avoid4_raw=[4,3,2,-3]` and `top1_raw=[8,3,1,-2]` remain available in tooling, but they change both utility shape and reward scale.

## Preparation

Generate isolated configs:

```bash
uv run python scripts/mortal/prepare_reward_pt_experiments.py \
  --base-config artifacts/mortal_training/config.toml \
  --parent-checkpoint artifacts/mortal_training/mortal.pth \
  --output-root artifacts/experiments/reward_pt_2026_05 \
  --train-steps 5000
```

Before training, either copy the parent checkpoint manually to each experiment `state_file`, or rerun preparation with `--copy-parent-checkpoint`.

## Evaluation Commands

Quick smoke uses 25 seeds, which yields 100 half-games in `OneVsThree`:

```bash
uv run python scripts/mortal/one_vs_three_smoke.py \
  --challenger artifacts/experiments/reward_pt_2026_05/R0_mortal_default/mortal.pth \
  --champion artifacts/mortal_training/mortal.pth \
  --seed-count 25 \
  --rank-points-profile mortal_default \
  --output-dir artifacts/eval/reward_pt_2026_05/R0_mortal_default_quick
```

Gate uses 250 seeds for 1000 half-games. Final A/B uses `scripts/mortal/ab_match.py` with both one-vs-three directions.

## Result Table

Quick smoke uses each 70k checkpoint as `challenger` against the 65k parent as `champion`, with 25 `OneVsThree` seeds (100 half-games). Gate and A/B are not run yet.

| Experiment | quick avg rank | quick pt(profile) | quick pt(Tenhou) | gate avg rank | 1st | 4th | agari | houjuu | fuuro | riichi | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R0_mortal_default | 2.48 | 3.04 | -3.60 | pending | 27 | 30 | 0.2243 | 0.1214 | 0.2948 | 0.1832 | quick smoke only |
| R1_avoid4_norm | 2.49 | -0.1286 | -3.15 | pending | 27 | 29 | 0.2178 | 0.1270 | 0.2959 | 0.1904 | quick smoke only |
| R2_top1_norm | 2.41 | 0.1620 | 1.35 | pending | 29 | 28 | 0.2202 | 0.1204 | 0.2984 | 0.1859 | quick smoke only |

## Artifacts

| Experiment | checkpoint | quick metrics |
| --- | --- | --- |
| R0_mortal_default | `artifacts/experiments/reward_pt_2026_05/R0_mortal_default/mortal.pth` | `artifacts/experiments/reward_pt_2026_05/R0_mortal_default/eval/quick_smoke_100h_v2/metrics.json` |
| R1_avoid4_norm | `artifacts/experiments/reward_pt_2026_05/R1_avoid4_norm/mortal.pth` | `artifacts/experiments/reward_pt_2026_05/R1_avoid4_norm/eval/quick_smoke_100h/metrics.json` |
| R2_top1_norm | `artifacts/experiments/reward_pt_2026_05/R2_top1_norm/mortal.pth` | `artifacts/experiments/reward_pt_2026_05/R2_top1_norm/eval/quick_smoke_100h/metrics.json` |

Notes:

- The parent checkpoint was already at 65000 local steps, so this run targets 70000.
- Training used `--num-workers 0` because this environment returned `OSError: [Errno 95] Operation not supported` from multiprocessing DataLoader sockets.
- R0's first quick smoke wrote logs but failed while formatting detailed stats for float rank points; the fixed full run is under `quick_smoke_100h_v2`.
