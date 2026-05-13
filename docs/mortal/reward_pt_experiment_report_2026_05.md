# Reward / PT Table Experiment Report

Status: R0/R1/R2 short training complete; 100 half-game quick smoke, 1000 half-game gates, and R1 final two-direction A/B complete.

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

Gate uses 250 seeds for 1000 half-games. Final A/B uses `scripts/mortal/one_vs_three_smoke.py` with both one-vs-three directions and 1250 seeds per direction.

## Result Table

Quick smoke uses each 70k checkpoint as `challenger` against the 65k parent as `champion`, with 25 `OneVsThree` seeds (100 half-games). Gate A uses the same pairing with 250 seeds. Gate B compares reward variants against R0 70k with 250 seeds.

| Experiment | quick avg rank | quick pt(profile) | quick pt(Tenhou) | Gate A avg rank | Gate A pt(Tenhou) | 1st | 4th | agari | houjuu | fuuro | riichi | Decision |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R0_mortal_default | 2.48 | 3.04 | -3.60 | 2.491 | -0.315 | 243 | 258 | 0.2177 | 0.1304 | 0.2908 | 0.1886 | baseline gate complete |
| R1_avoid4_norm | 2.49 | -0.1286 | -3.15 | 2.510 | -1.350 | 242 | 260 | 0.2191 | 0.1301 | 0.2928 | 0.1886 | Gate B positive vs R0 |
| R2_top1_norm | 2.41 | 0.1620 | 1.35 | 2.486 | 1.080 | 242 | 245 | 0.2185 | 0.1290 | 0.2945 | 0.1875 | Gate A positive vs parent |

## Gate B

Gate B compares each variant against the same-step R0 70k checkpoint, so it isolates reward shape more cleanly than parent-only Gate A.

| Match | challenger | champion | avg rank | pt(profile) | pt(Tenhou) | 1st | 4th | agari | houjuu | fuuro | riichi | Read |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| GateB_R1_vs_R0_1000h | R1_avoid4_norm | R0_mortal_default | 2.474 | 0.0600 | 2.160 | 257 | 239 | 0.2189 | 0.1319 | 0.2922 | 0.1911 | avoid4 has a positive 1000h same-step signal |
| GateB_R2_vs_R0_1000h | R2_top1_norm | R0_mortal_default | 2.493 | 0.0204 | 0.765 | 255 | 245 | 0.2184 | 0.1309 | 0.2933 | 0.1898 | top1 remains mildly positive, weaker than quick smoke |

## Final A/B

R1 was the only candidate advanced to final A/B. R2 remains parked because its same-step Gate B signal was weaker. A short RiichiEnv `ab_match.py` run was started first, but it was stopped after confirming throughput was unsuitable for 5000h; the recorded final results below use the same libriichi arena backend as quick smoke and gates.

| Direction | challenger | champion | games | avg rank | pt(profile) | pt(Tenhou) | 1st | 4th | agari | houjuu | fuuro | riichi | Read |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| R1_vs_3R0 | R1_avoid4_norm | R0_mortal_default | 5000 | 2.5078 | -0.0122 | -0.495 | 1244 | 1258 | 0.2154 | 0.1341 | 0.2930 | 0.1843 | R1 does not reproduce Gate B advantage as single challenger |
| R0_vs_3R1 | R0_mortal_default | R1_avoid4_norm | 5000 | 2.4842 | 3.0316 | 1.305 | 1256 | 1217 | 0.2189 | 0.1328 | 0.2891 | 0.1861 | R0 is positive as single challenger against R1 |

Final A/B does not support promoting `R1_avoid4_norm` over `R0_mortal_default` as the next main checkpoint. The 1000h Gate B signal appears to have been noise or not robust to the higher-sample two-direction setup.

## Decision

This reward/PT first-pass line is closed for the current 70k checkpoints.

- `R0_mortal_default` remains the default mainline candidate.
- `R1_avoid4_norm` is rejected for mainline promotion. It showed a positive 1000h same-step gate signal, but failed to reproduce in 5000h two-direction final A/B.
- `R2_top1_norm` is parked. It had early positive signal, but its same-step gate advantage was weaker than R1, so it was not advanced to final A/B.

The experimental chain itself is valid: configurable PT table -> reward scalarization -> continued training -> quick smoke -> gate -> final A/B. The result is a negative promotion decision for these normalized reward variants, not a failure of the tooling. Further PT shaping should wait until default training plateaus, GRP audit shows a relevant mismatch, or a platform/style-specific objective becomes the primary goal.

## Artifacts

| Experiment | checkpoint | quick metrics | Gate A metrics |
| --- | --- | --- | --- |
| R0_mortal_default | `artifacts/experiments/reward_pt_2026_05/R0_mortal_default/mortal.pth` | `artifacts/experiments/reward_pt_2026_05/R0_mortal_default/eval/quick_smoke_100h_v2/metrics.json` | `artifacts/experiments/reward_pt_2026_05/gates/GateA_R0_vs_parent_1000h/metrics.json` |
| R1_avoid4_norm | `artifacts/experiments/reward_pt_2026_05/R1_avoid4_norm/mortal.pth` | `artifacts/experiments/reward_pt_2026_05/R1_avoid4_norm/eval/quick_smoke_100h/metrics.json` | `artifacts/experiments/reward_pt_2026_05/gates/GateA_R1_vs_parent_1000h/metrics.json` |
| R2_top1_norm | `artifacts/experiments/reward_pt_2026_05/R2_top1_norm/mortal.pth` | `artifacts/experiments/reward_pt_2026_05/R2_top1_norm/eval/quick_smoke_100h/metrics.json` | `artifacts/experiments/reward_pt_2026_05/gates/GateA_R2_vs_parent_1000h/metrics.json` |

Gate B artifacts:

- `artifacts/experiments/reward_pt_2026_05/gates/GateB_R1_vs_R0_1000h/metrics.json`
- `artifacts/experiments/reward_pt_2026_05/gates/GateB_R2_vs_R0_1000h/metrics.json`

Final A/B artifacts:

- `artifacts/experiments/reward_pt_2026_05/final_ab/arena_R1_vs_3R0_5000h/metrics.json`
- `artifacts/experiments/reward_pt_2026_05/final_ab/arena_R0_vs_3R1_5000h/metrics.json`

Notes:

- The parent checkpoint was already at 65000 local steps, so this run targets 70000.
- Training used `--num-workers 0` because this environment returned `OSError: [Errno 95] Operation not supported` from multiprocessing DataLoader sockets.
- R0's first quick smoke wrote logs but failed while formatting detailed stats for float rank points; the fixed full run is under `quick_smoke_100h_v2`.
- The incomplete RiichiEnv final A/B trial under `final_ab/R1_vs_3R0_5000h` is not used in the report.
