# Reward / PT Table Experiment Report

Status: scaffolded, experiments not yet run.

## Fixed Matrix

| Experiment | Parent | Reward profile | PT table | GRP | Target steps |
| --- | --- | --- | --- | --- | --- |
| R0_base | `artifacts/mortal_training/mortal.pth@60000` | `base` | `[6, 4, 2, 0]` | old GRP | 65000 |
| R1_avoid4_strong | `artifacts/mortal_training/mortal.pth@60000` | `avoid4_strong` | `[4, 3, 2, -3]` | old GRP | 65000 |
| R2_top1_heavy | `artifacts/mortal_training/mortal.pth@60000` | `top1_heavy` | `[8, 3, 1, -2]` | old GRP | 65000 |
| R3_zero_sum_balanced | `artifacts/mortal_training/mortal.pth@60000` | `zero_sum_balanced` | `[3, 1, -1, -3]` | old GRP | 65000 |

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
  --challenger artifacts/experiments/reward_pt_2026_05/R0_base/mortal.pth \
  --champion artifacts/mortal_training/mortal.pth \
  --seed-count 25 \
  --rank-points-profile base \
  --output-dir artifacts/eval/reward_pt_2026_05/R0_base_quick
```

Gate uses 250 seeds for 1000 half-games. Final A/B uses `scripts/mortal/ab_match.py` with both one-vs-three directions.

## Result Table

Fill after runs complete.

| Experiment | quick avg rank | quick pt(profile) | quick pt(Tenhou) | gate avg rank | 1st | 4th | agari | houjuu | fuuro | riichi | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R0_base | | | | | | | | | | | |
| R1_avoid4_strong | | | | | | | | | | | |
| R2_top1_heavy | | | | | | | | | | | |
| R3_zero_sum_balanced | | | | | | | | | | | |
