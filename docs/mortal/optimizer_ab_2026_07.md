# Fresh Adam vs Preserved Adam

## Scope

This phase tests only optimizer-state transfer during a controlled continuation from the 70k parent:

- reward: `final_rank_mc`
- parent: `mortal_default_70k_promoted_candidate.pth`, step `70000`
- target: step `72000`
- groups: fresh Adam versus preserved Adam moments
- training seeds: `20260724`, `20260725`, `20260726`
- evaluation: 1000 hanchans per seed, native four-model random-seat lineup
- lineup: `70k`, external Mortal, the matched fresh checkpoint, and the matched preserved checkpoint
- rank points: `[90, 45, 0, -135]`

The two A/B checkpoints were trained from the same parent, data stream, model seed, and training configuration. The only intended variable was whether Adam state was initialized empty or restored from the legacy 70k parent. All six training contracts recorded `git_dirty=false`.

## Evaluation Result

`delta_pt` is calculated per complete hanchan as `Pt(preserved) - Pt(fresh)`. The three training-seed means were:

| Training seed | Fresh avg Pt | Preserved avg Pt | Preserved - fresh | Preserved ahead rate |
| ---: | ---: | ---: | ---: | ---: |
| 20260724 | -0.315 | +4.230 | +4.545 | 50.9% |
| 20260725 | -4.140 | -2.475 | +1.665 | 50.7% |
| 20260726 | -7.335 | -1.800 | +5.535 | 51.4% |

All three seed-level averages favor preserved Adam, but the effect is not large enough to treat a single seed as decisive.

- Mean of seed means: `+3.915 Pt/局`.
- Median of seed means: `+4.545 Pt/局`.
- Pooled complete-hanchan bootstrap 95% CI: `[-1.080, +8.910] Pt/局`.
- Equal-seed hierarchical bootstrap 95% CI: `[-1.455, +9.135] Pt/局`.
- Exact one-sided seed-direction sign test: `3/3`, `p=0.125`.

The intervals include zero. The hanchan interval measures arena uncertainty conditional on these checkpoints; the hierarchical interval additionally resamples the three training seeds equally. Neither interval proves a general optimizer improvement.

## Against the 70k Anchor

The same 3000 hanchans also allow a paired comparison against the 70k seat in each lineup:

| Training seed | Preserved - 70k Pt | Fresh - 70k Pt | Preserved - 70k rank | Fresh - 70k rank |
| ---: | ---: | ---: | ---: | ---: |
| 20260724 | +9.855 | +5.310 | -0.115 | -0.060 |
| 20260725 | +3.060 | +1.395 | -0.044 | -0.037 |
| 20260726 | +0.630 | -4.905 | +0.044 | +0.095 |

Pooled preserved-minus-70k is `+4.515 Pt/局`, with hanchan bootstrap CI `[-0.555, +9.555]` and equal-seed hierarchical CI `[-1.785, +11.115]`. Pooled fresh-minus-70k is `+0.600 Pt/局`, with hanchan bootstrap CI `[-4.575, +5.581]` and equal-seed hierarchical CI `[-6.481, +7.230]`.

This is evidence that preserved Adam may reduce continuation degradation relative to the 70k anchor, but it does not yet establish a stable win over 70k.

## Behavior Readout

The optimizer effect is not a simple global behavior shift. Across the three seed lineups, preserved Adam has:

- lower or similar houjuu than fresh in seeds `20260724` and `20260726`, but slightly higher houjuu in `20260725`;
- similar riichi and fuuro rates overall;
- small, seed-dependent changes in after-riichi and after-fuuro outcomes.

The detailed per-seed behavior table is in the local summary artifact. The control models are included to anchor the lineup, but this experiment is not a replacement for the final model-pool evaluation.

## Decision

Do not promote preserved Adam as the project-wide optimizer default solely from this result. The result is a reproducible positive direction across the three matched seeds, and the exact sign test is `p=0.125`; this is enough to justify one pre-registered replication batch, but not a recipe promotion.

For the next research decision:

1. Keep `final_rank_mc` as the operational reward and keep fresh Adam as the reproducibility baseline.
2. Preserve the optimizer-state implementation and this A/B result as evidence that optimizer state can affect continuation strength.
3. Do not start an LR/CQL/reward grid on the basis of this result.
4. Run exactly one replication batch of three new matched seeds, keeping reward, LR, CQL, architecture, data index, target step, and evaluation protocol fixed. Stop optimizer expansion after the six-seed result.
5. If the six-seed result is `5/6` or `6/6` positive with mean/median still around `+3 Pt/局`, promote preserved Adam only as the default optimizer initialization for 70k legacy continuation. Do not automatically promote any checkpoint.
6. If the effect shrinks or becomes mixed, stop optimizer expansion and return to data distribution/project-owned lineage rather than adding another local tuning variable.

## Reproducibility

The full machine-generated summary is kept locally at:

`artifacts/experiments/model_pool_2026_07/optimizer_ab_2026_07_epoch1/eval_1000h/summary/optimizer_ab_eval_1000h_summary.json`

and:

`artifacts/experiments/model_pool_2026_07/optimizer_ab_2026_07_epoch1/eval_1000h/summary/optimizer_ab_eval_1000h_summary.md`

The raw logs and checkpoints remain local artifacts and are intentionally not added to Git.
