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

## L1 Behavior Diff

The all-same 500h arena selfplay logs were compared with `scripts/mortal/compare_checkpoint_stat_reports.py`.

Artifacts:

- `artifacts/experiments/default_mainline_2026_05/behavior_diff/stat_70k.json`
- `artifacts/experiments/default_mainline_2026_05/behavior_diff/stat_80k.json`
- `artifacts/experiments/default_mainline_2026_05/behavior_diff/stat_delta.md`

Key deltas, 80k minus 70k:

| Metric | 70k | 80k | Delta |
| --- | ---: | ---: | ---: |
| Win rate | 21.69% | 22.28% | +0.59pp |
| Deal-in rate | 13.13% | 14.66% | +1.53pp |
| Call rate | 28.71% | 31.82% | +3.11pp |
| Riichi rate | 18.84% | 19.90% | +1.07pp |
| Winning rate after call | 33.73% | 32.24% | -1.50pp |
| Deal-in rate after call | 14.91% | 15.78% | +0.87pp |
| Winning rate after riichi | 49.09% | 49.31% | +0.21pp |
| Deal-in rate after riichi | 14.95% | 14.64% | -0.31pp |
| Avg winning delta score | 6236.14 | 5814.60 | -421.54 |
| Chasing riichi rate | 16.97% | 20.94% | +3.97pp |
| Riichi chased rate | 18.28% | 22.55% | +4.27pp |

L1 diagnosis: the 80k drift is more consistent with call/fuuro quality degradation than with riichi quality degradation. 80k calls much more often, wins less often after calls, and deals in more often after calls. Riichi is also more frequent, but post-riichi win/deal-in rates do not show the same deterioration in this aggregate view.

## L2 Behavior Slices

The same 500h logs were then parsed directly with `scripts/mortal/analyze_checkpoint_behavior_slices.py`. Unlike the L1 `stat_report.py` artifact, this pass includes all four seats from the all-same arena logs.

Artifacts:

- `artifacts/experiments/default_mainline_2026_05/behavior_slices/behavior_slices_left.json`
- `artifacts/experiments/default_mainline_2026_05/behavior_slices/behavior_slices_right.json`
- `artifacts/experiments/default_mainline_2026_05/behavior_slices/slice_metrics.csv`
- `artifacts/experiments/default_mainline_2026_05/behavior_slices/slice_diff.csv`
- `artifacts/experiments/default_mainline_2026_05/behavior_slices/behavior_slice_report.md`

Overall all-seat deltas, 80k minus 70k:

| Metric | Delta |
| --- | ---: |
| Agari rate | +0.60pp |
| Houjuu rate | +1.61pp |
| Fuuro rate | +2.86pp |
| Riichi rate | +1.07pp |
| After-fuuro agari rate | -1.29pp |
| After-fuuro houjuu rate | +1.23pp |
| After-riichi houjuu rate | -0.41pp |

Main slices:

| Slice | Fuuro delta | Houjuu delta | After-fuuro agari delta | After-fuuro houjuu delta |
| --- | ---: | ---: | ---: | ---: |
| dealer | +4.00pp | +2.20pp | -3.09pp | +3.97pp |
| nondealer | +2.48pp | +1.42pp | -0.64pp | +0.17pp |
| start rank 1 | +2.83pp | +1.41pp | -0.80pp | -0.96pp |
| start rank 2 | +2.62pp | +2.09pp | -1.88pp | +5.17pp |
| start rank 3 | +3.54pp | +1.99pp | +0.12pp | +0.27pp |
| start rank 4 | +2.45pp | +0.97pp | -2.64pp | +0.23pp |

L2 diagnosis: 80k is not merely becoming more aggressive when behind. The call-rate increase appears in every start-rank bucket, including rank 1, and the dealer slice is especially risky. The worst localized signal is dealer call quality: after-fuuro agari falls while after-fuuro houjuu rises sharply. The next diagnostic pass should focus on call windows, especially dealer calls and start-rank 2 calls.

## L3 Decision Drift

The same arena logs were analyzed with `scripts/mortal/analyze_checkpoint_decision_drift.py`, using per-event Mortal Q metadata from the arena mjai logs.

Artifacts:

- `artifacts/experiments/default_mainline_2026_05/decision_drift/decision_drift_left.json`
- `artifacts/experiments/default_mainline_2026_05/decision_drift/decision_drift_right.json`
- `artifacts/experiments/default_mainline_2026_05/decision_drift/decision_metrics.csv`
- `artifacts/experiments/default_mainline_2026_05/decision_drift/decision_diff.csv`
- `artifacts/experiments/default_mainline_2026_05/decision_drift/decision_drift_report.md`

This L3 pass can measure chosen action margins such as `Q(call)-Q(pass)` and `Q(reach)-Q(discard)`. It cannot fully count pass-over-call opportunities because arena logs do not emit pass events; a sidecar-rich RiichiEnv sample is still needed for full PASS-vs-CALL opportunity analysis.

Chosen-call deltas, 80k minus 70k:

| Slice | Count delta | Avg margin delta | Agari delta | Houjuu delta |
| --- | ---: | ---: | ---: | ---: |
| all | +1282 | +0.3930 | -1.48pp | +0.55pp |
| dealer | +457 | +0.4017 | -2.26pp | +4.62pp |
| nondealer | +825 | +0.3855 | -1.18pp | -1.09pp |
| start rank 2 | +279 | +0.4054 | -1.20pp | +4.58pp |
| start rank 4 | +209 | +0.3019 | -5.78pp | -0.79pp |

Chosen-riichi deltas:

| Slice | Count delta | Avg margin delta | Agari delta | Houjuu delta |
| --- | ---: | ---: | ---: | ---: |
| all | +167 | -0.0150 | +0.22pp | -0.41pp |
| dealer | -14 | -0.1183 | +0.46pp | +0.93pp |
| nondealer | +181 | +0.0289 | -0.02pp | -0.85pp |

L3 diagnosis: the 80k problem is concentrated in calls, not riichi. The surprising signal is that 80k chosen calls have a higher average `Q(call)-Q(pass)` margin, yet worse outcomes. Dealer calls and start-rank 2 calls are the clearest bad pockets. This points away from "ambiguous marginal calls flipped by noise" and toward a value-estimation or policy-drift issue where 80k is more confident in calls whose downstream results are worse.

## L4 Replay Casebook Export

Representative call cases were exported with `scripts/mortal/export_behavior_replay_cases.py`.

Artifacts:

- `artifacts/experiments/default_mainline_2026_05/behavior_replay_cases/manifest.jsonl`
- `artifacts/experiments/default_mainline_2026_05/behavior_replay_cases/manifest.json`
- `artifacts/experiments/default_mainline_2026_05/behavior_replay_cases/cases/*.mjson`
- `artifacts/experiments/default_mainline_2026_05/behavior_replay_cases/review_payloads/*.review.json`

Case kinds:

| Case kind | Count | Purpose |
| --- | ---: | --- |
| `80k_dealer_call_bad` | 20 | High-margin 80k dealer calls with bad downstream outcomes |
| `80k_rank2_call_bad` | 20 | High-margin 80k start-rank 2 calls with bad downstream outcomes |
| `70k_dealer_call_good` | 20 | High-margin 70k dealer-call good controls |
| `70k_rank2_call_good` | 20 | High-margin 70k start-rank 2 call good controls |

Each manifest row includes:

- `mjson_path`: replay file loadable by the existing mjai replay path
- `focus_event_index` / `focus_step`: the key call decision
- `checkpoint_path`: archived checkpoint for the labeled model, when generated by the current exporter
- `slice_tags`: dealer/start-rank/outcome tags
- `margin`: original arena-log `Q(call)-Q(pass)`
- `outcome`: `agari`, `houjuu`, `ryukyoku`, or `not_agari`
- `review_payload_path`: compact precomputed focus-event/Q payload

This is the MVP bridge from statistics to GUI review. The current GUI can manually load a case `.mjson`; manifest focus fields identify where to jump. A later GUI case-bundle viewer can consume `manifest.jsonl` directly and auto-load the focused replay step.

The integrated service currently imports a case by re-running review with archived checkpoints:

- `70k`: `artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth`
- `80k`: `artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth`

This prevents the `80k` casebook from drifting if `artifacts/mortal_training/mortal.pth` later becomes 90k or 100k. The replay view is therefore a fresh review under the archived checkpoint, not a zero-recompute rendering of the original L3 arena-Q payload. The original arena focus Q is preserved in `review_payload_path` and `margin`.

The import API returns `focus_resolution`:

- `exact`: replay decision log contained the original `focus_event_index`
- `nearest`: replay decision log did not contain the exact source event, so the nearest step was used
- `missing`: no usable focus step was found

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

1. Review the L4 casebook in GUI, starting with `80k_dealer_call_bad` and its 70k dealer-call controls.
2. Build a GUI case-bundle viewer only if manual manifest-guided review is too slow.
3. Generate a smaller sidecar-rich RiichiEnv sample only if full PASS-vs-CALL opportunity analysis is needed.
4. Focus any policy diagnostics on dealer call windows and start-rank 2 call windows.
5. If default training continues, archive every 5k or 10k checkpoint and compare against the 70k candidate, not only the latest `mortal.pth`.
6. Keep GRP unchanged unless future selfplay audits show both calibration degradation and expected-PT or reward-delta degradation.

The 80k-vs-65k run under `artifacts/experiments/default_mainline_2026_05/gates/GateA_80k_vs_65k_1000h` was produced by a mistaken baseline choice and is not used for this decision.
