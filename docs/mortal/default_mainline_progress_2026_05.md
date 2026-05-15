# Default Mortal Mainline Progress

Status: 80k default checkpoint was screened against the 70k default reference. The gate blocks automatic mainline promotion of 80k, but it is not a final strength verdict. Four-domain GRP audit is complete and does not support GRP retraining yet.

## Checkpoints

| Label | Path | Step | Role |
| --- | --- | ---: | --- |
| `mortal_default_70k` | `artifacts/experiments/reward_pt_2026_05/R0_mortal_default/mortal.pth` | 70000 | Standard reference / balanced anchor |
| `mortal_default_80k` | `artifacts/mortal_training/mortal.pth` | 80000 | Aggressive / fuuro-heavy behavior anchor; not auto-promoted |

Stable local archive copies:

- `artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth`
- `artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth`

Naming note: the archive filenames reflect the earlier promotion-gate workflow. The current research framing is more neutral: 70k is the standard anchor, and 80k is an aggressive behavior anchor that did not pass this screening gate as an automatic replacement.

## Gate Setup

Both gates use `scripts/mortal/one_vs_three_smoke.py`, 250 seed blocks, 4 seat rotations per block, and therefore 1000 half-games. Rank point reporting should use Tenhou reference `[90,45,0,-135]` for readability; the `mortal_default` training point table `[6,4,2,0]` is only the training reward scalarization.

| Gate | Challenger | Champion | Purpose |
| --- | --- | --- | --- |
| GateA_80k_vs_70k_1000h | `mortal_default_80k` | `mortal_default_70k` | Check whether 80k beats 70k as single challenger |
| GateB_70k_vs_80k_1000h | `mortal_default_70k` | `mortal_default_80k` | Reverse direction check |

## Results

| Gate | Challenger | Avg rank | Tenhou avg pt | Rank counts | Agari | Houjuu | Fuuro | Riichi |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: |
| GateA_80k_vs_70k_1000h | 80k | 2.508 | -0.81 | `[255,237,253,255]` | 0.2232 | 0.1455 | 0.3171 | 0.2080 |
| GateB_70k_vs_80k_1000h | 70k | 2.458 | 4.32 | `[244,277,256,223]` | 0.2209 | 0.1303 | 0.2976 | 0.1825 |

In GateA, 80k is slightly negative against 70k. In GateB, 70k is positive against 80k. This is sufficient to block automatic 80k promotion, but because the setup is 250 independent seed blocks with rotations, it should be treated as a screening result rather than a final 70k > 80k strength theorem.

## Behavior Drift

Compared with the 70k baseline in the reverse gate, 80k shows a more aggressive profile:

| Model context | Agari | Houjuu | Fuuro | Riichi |
| --- | ---: | ---: | ---: | ---: |
| 80k as challenger vs 70k | 0.2232 | 0.1455 | 0.3171 | 0.2080 |
| 70k as challenger vs 80k | 0.2209 | 0.1303 | 0.2976 | 0.1825 |
| 80k as champions vs 70k | 0.2243 | 0.1432 | 0.3171 | 0.1991 |

The 80k checkpoint wins and calls more, but also deals in substantially more. The higher houjuu rate is the main warning sign for automatic promotion. For research use, this same drift makes 80k useful as an aggressive / fuuro-heavy behavior source.

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

The initial single-case export remains available, but it is now treated as an exploratory reference rather than the main diagnostic artifact. Its labels have been made neutral in the GUI:

| Old interpretation | Current neutral meaning |
| --- | --- |
| `80k_dealer_call_bad` | `80k` dealer call drift sample |
| `80k_rank2_call_bad` | `80k` start-rank 2 call drift sample |
| `70k_dealer_call_good` | `70k` dealer call reference sample |
| `70k_rank2_call_good` | `70k` start-rank 2 call reference sample |

The main L4 artifact is now a paired divergence casebook generated by `scripts/mortal/export_paired_behavior_divergence_cases.py`.

Paired artifacts:

- `artifacts/experiments/default_mainline_2026_05/paired_behavior_cases/paired_manifest.jsonl`
- `artifacts/experiments/default_mainline_2026_05/paired_behavior_cases/paired_manifest.json`
- `artifacts/experiments/default_mainline_2026_05/paired_behavior_cases/pairs/70k/*.mjson`
- `artifacts/experiments/default_mainline_2026_05/paired_behavior_cases/pairs/80k/*.mjson`

Export summary from the current 500h same-seed arena logs:

| Case kind | Count | Purpose |
| --- | ---: | --- |
| `paired_call_divergence` | 8 | First divergence is `70k` pass/reference path vs `80k` call |
| `paired_dealer_call_divergence` | 8 | Same, in dealer context |
| `paired_rank2_call_divergence` | 8 | Same, in start-rank 2 context |
| `paired_discard_divergence` | 20 | First divergence is a discard difference |
| `paired_first_divergence` | 4 | Other first-divergence cases |

Each paired manifest row includes:

- `left_model` / `right_model`: currently `70k` and `80k`
- `left_mjson_path` / `right_mjson_path`: the two replay files
- `left_focus_event_index` / `right_focus_event_index`: focus points around the first divergence
- `prefix_match_event_count`: number of shared mjai events before the first divergence
- `divergence_kind`: e.g. `left_pass_vs_right_pon`
- `left_action` / `right_action`: neutral action summary, e.g. `pass` vs `pon`
- `downstream_summary`: actor hanchan rank and kyoku outcome after the split
- `slice_tags`: dealer/start-rank/round/score/outcome tags

This paired view is stricter than outcome-labeled single cases: it asks where the same seed/split/seat trajectory first diverges, then lets the reviewer inspect how the subsequent game path separates. It is still not a perfect counterfactual, but it is much closer to the intended 70k-vs-80k behavior comparison.

The integrated `/behavior-casebook` page now shows the paired casebook first. Each paired card exposes two buttons:

- `打开 70k`: import the left replay under `mortal_default_70k_promoted_candidate.pth`
- `打开 80k`: import the right replay under `mortal_default_80k_rejected_gate.pth`

Both imports re-run review under the archived checkpoint and jump to the side-specific focus step. A future dual-pane viewer can synchronize the two sides, but the current two-button MVP is enough to inspect the same divergence from both model paths.

Legacy single-case artifacts:

Artifacts:

- `artifacts/experiments/default_mainline_2026_05/behavior_replay_cases/manifest.jsonl`
- `artifacts/experiments/default_mainline_2026_05/behavior_replay_cases/manifest.json`
- `artifacts/experiments/default_mainline_2026_05/behavior_replay_cases/cases/*.mjson`
- `artifacts/experiments/default_mainline_2026_05/behavior_replay_cases/review_payloads/*.review.json`

Single-case kinds:

| Case kind | Count | Purpose |
| --- | ---: | --- |
| `80k_dealer_call_bad` | 20 | High-margin 80k dealer call drift samples |
| `80k_rank2_call_bad` | 20 | High-margin 80k start-rank 2 call drift samples |
| `70k_dealer_call_good` | 20 | High-margin 70k dealer call reference samples |
| `70k_rank2_call_good` | 20 | High-margin 70k start-rank 2 call reference samples |

Each manifest row includes:

- `mjson_path`: replay file loadable by the existing mjai replay path
- `focus_event_index` / `focus_step`: the key call decision
- `checkpoint_path`: archived checkpoint for the labeled model, when generated by the current exporter
- `slice_tags`: dealer/start-rank/outcome tags
- `margin`: original arena-log `Q(call)-Q(pass)`
- `outcome`: `agari`, `houjuu`, `ryukyoku`, or `not_agari`
- `review_payload_path`: compact precomputed focus-event/Q payload

The integrated service currently imports a case by re-running review with archived checkpoints:

- `70k`: `artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth`
- `80k`: `artifacts/mortal_training/checkpoints/mortal_default_80k_rejected_gate.pth`

This prevents the `80k` casebook from drifting if `artifacts/mortal_training/mortal.pth` later becomes 90k or 100k. The replay view is therefore a fresh review under the archived checkpoint, not a zero-recompute rendering of the original L3 arena-Q payload. The original arena focus Q is preserved in `review_payload_path` and `margin`.

The import API returns `focus_resolution`:

- `exact`: replay decision log contained the original `focus_event_index`
- `nearest`: replay decision log did not contain the exact source event, so the nearest step was used
- `missing`: no usable focus step was found

## Decision

- Use `mortal_default_70k` as the standard reference / balanced anchor.
- Use `mortal_default_80k` as an aggressive / fuuro-heavy behavior anchor. It is not auto-promoted over 70k by this gate, but it remains valuable as a style source.
- Do not interpret the 1000h bidirectional gate as a final strong claim that 70k is absolutely stronger than 80k.
- Do not attribute the 80k non-promotion primarily to GRP expected-PT error or reward-delta variance degradation.
- Do not blindly continue default offline training to 100k without a checkpoint retention and evaluation plan.
- If default training continues, compare retained checkpoints against the 70k anchor and monitor behavior drift.

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

1. Start Selfplay Fine-tune Phase 1 with 70k as the standard anchor and 80k as the aggressive behavior anchor.
2. Review paired L4 cases in GUI as an explanatory tool, starting with `paired_dealer_call_divergence` and `paired_rank2_call_divergence`.
3. Use the legacy single-case section only as supporting context, not as the main good/bad judgment.
4. Add dual-pane synchronized replay only if the two-button paired workflow is too slow.
5. Generate a smaller sidecar-rich RiichiEnv sample only if full PASS-vs-CALL opportunity analysis is needed.
6. Focus any policy diagnostics on dealer call windows and start-rank 2 call windows.
7. If default training continues, archive every 5k or 10k checkpoint and compare against the 70k anchor, not only the latest `mortal.pth`.
8. Keep GRP unchanged unless future selfplay audits show both calibration degradation and expected-PT or reward-delta degradation.

The 80k-vs-65k run under `artifacts/experiments/default_mainline_2026_05/gates/GateA_80k_vs_65k_1000h` was produced by a mistaken baseline choice and is not used for this decision.

Selfplay Fine-tune Phase 1 is specified in `docs/mortal/selfplay_finetune_phase1_2026_05.md`.
