# Default Mortal Mainline Progress

Status: 80k default checkpoint evaluated against 70k default baseline; 80k is not promoted. Four-domain GRP audit is complete and does not support GRP retraining yet.

## Checkpoints

| Label | Path | Step | Role |
| --- | --- | ---: | --- |
| `mortal_default_70k` | `artifacts/experiments/reward_pt_2026_05/R0_mortal_default/mortal.pth` | 70000 | Current default promotion candidate |
| `mortal_default_80k` | `artifacts/mortal_training/mortal.pth` | 80000 | Evaluated, not promoted |

Stable local archive copies:

- `artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth`
- `artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth`

## Gate Setup

Both gates use `scripts/mortal/one_vs_three_smoke.py`, 250 seeds, and 1000 half-games. Rank point reporting should use Tenhou reference `[90,45,0,-135]` for readability; the `mortal_default` training point table `[6,4,2,0]` is only the training reward scalarization.

| Gate | Challenger | Champion | Purpose |
| --- | --- | --- | --- |
| GateA_80k_vs_70k_1000h | `mortal_default_80k` | `mortal_default_70k` | Check whether 80k beats 70k as single challenger |
| GateB_70k_vs_80k_1000h | `mortal_default_70k` | `mortal_default_80k` | Reverse direction check |

## Results

| Gate | Challenger | Avg rank | Tenhou avg pt | Rank counts | Agari | Houjuu | Fuuro | Riichi |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| GateA_80k_vs_70k_1000h | 80k | 2.508 | -0.81 | `[255,237,253,255]` | 0.2232 | 0.1455 | 0.3171 | 0.2080 |
| GateB_70k_vs_80k_1000h | 70k | 2.458 | 4.32 | `[244,277,256,223]` | 0.2209 | 0.1303 | 0.2976 | 0.1825 |

In GateA, 80k is slightly negative against 70k. In GateB, 70k is clearly positive against 80k. This does not support promoting the 80k checkpoint.

## Behavior Drift

Compared with the 70k baseline in the reverse gate, 80k shows a more aggressive profile:

| Model context | Agari | Houjuu | Fuuro | Riichi |
| --- | ---: | ---: | ---: | ---: |
| 80k as challenger vs 70k | 0.2232 | 0.1455 | 0.3171 | 0.2080 |
| 70k as challenger vs 80k | 0.2209 | 0.1303 | 0.2976 | 0.1825 |
| 80k as champions vs 70k | 0.2243 | 0.1432 | 0.3171 | 0.1991 |

The 80k checkpoint wins and calls more, but also deals in substantially more. The higher houjuu rate is the main warning sign: default offline training from 70k to 80k appears to have pushed the policy toward a more aggressive style without robust strength gain.

## Decision

- Keep `mortal_default_70k` as the current default promotion candidate.
- Mark `mortal_default_80k` as rejected at the 1000h gate.
- Do not attribute the 80k rejection primarily to GRP expected-PT error or reward-delta variance degradation.
- Do not blindly continue default offline training to 100k without a checkpoint retention and evaluation plan.
- If default training continues, compare retained checkpoints against the 70k candidate and monitor behavior drift.

## GRP Audit Follow-Up

The four-domain audit in `docs/mortal/grp_audit_report_2026_05.md` compared original logs, 70k selfplay, 80k selfplay, and style selfplay stress logs.

Key result under the active `mortal_default` audit profile:

| Domain | rank CE | ECE | Mortal MAE | Mortal RMSE | Reward-delta var |
| --- | ---: | ---: | ---: | ---: | ---: |
| original sample5000 | 2.6148 | 0.0546 | 1.5499 | 1.8628 | 0.2855 |
| 70k default selfplay | 2.6099 | 0.0542 | 1.5625 | 1.8703 | 0.2934 |
| 80k default selfplay | 2.5705 | 0.1114 | 1.5158 | 1.8320 | 0.2760 |

80k shows worse calibration, but not worse expected-PT error or reward-delta variance. GRP retraining is therefore not recommended yet.

## Next Recommended Work

1. Diagnose 70k vs 80k behavior drift, especially riichi/call choices and houjuu concentration.
2. If default training continues, archive every 5k or 10k checkpoint and compare against the 70k candidate, not only the latest `mortal.pth`.
3. Keep GRP unchanged unless future selfplay audits show both calibration degradation and expected-PT or reward-delta degradation.

The 80k-vs-65k run under `artifacts/experiments/default_mainline_2026_05/gates/GateA_80k_vs_65k_1000h` was produced by a mistaken baseline choice and is not used for this decision.
