# Reward / PT Table Experiment Report

Status: scaffolded, experiments not yet run.

## Fixed Matrix

Mortal reward uses `E[PT]_{t+1} - E[PT]_t`, so adding a constant to every rank does not change training. `[6,4,2,0]` and `[3,1,-1,-3]` are the same training objective; `zero_sum_balanced` is intentionally not a separate experiment.

| Experiment | Parent | Reward profile | PT table | GRP | Target steps |
| --- | --- | --- | --- | --- | --- |
| R0_mortal_default | `artifacts/mortal_training/mortal.pth@60000` | `mortal_default` | `[6, 4, 2, 0]` | old GRP | 65000 |
| R1_avoid4_norm | `artifacts/mortal_training/mortal.pth@60000` | `avoid4_norm` | `[2.142857, 1.285714, 0.428571, -3.857143]` | old GRP | 65000 |
| R2_top1_norm | `artifacts/mortal_training/mortal.pth@60000` | `top1_norm` | `[3.3, 0.3, -0.9, -2.7]` | old GRP | 65000 |

Optional engineering variants `avoid4_raw=[4,3,2,-3]` and `top1_raw=[8,3,1,-2]` remain available in tooling, but they change both utility shape and reward scale.

## Preparation

Generate isolated configs:

```bash
uv run python scripts/mortal/prepare_reward_pt_experiments.py \
  --base-config artifacts/mortal_training/config.toml \
  --parent-checkpoint artifacts/mortal_training/mortal.pth \
  --output-root artifacts/experiments/reward_pt_2026_05 \
  --target-steps 65000 \
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

Fill after runs complete.

| Experiment | quick avg rank | quick pt(profile) | quick pt(Tenhou) | gate avg rank | 1st | 4th | agari | houjuu | fuuro | riichi | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R0_mortal_default | | | | | | | | | | | |
| R1_avoid4_norm | | | | | | | | | | | |
| R2_top1_norm | | | | | | | | | | | |
