# KeqingRL Step 3.9C/D Low-Rank And Weak-Margin Penalty Screen

source_config_id: `93`
base_config: `cfg10-like: scale=0.25, temp=1.25, lr=0.003, update_epochs=4, clip=0.2, rule_kl=0.005, delta_l2=0.001`
gate_role: `adaptive_recovery is adaptive_movement_diagnostic; movement_quality_gate is the selector`
eval_strength_note: `screening only; skipped eval when movement quality/fresh gate failed`

## Runs

Each compressed run used `2*8=16` rollout half-hands plus `8` fresh validation half-hands. All compressed runs failed the train/fresh movement gate and skipped eval, so each run used `24` script half-hands.

| label | train_top1 | fresh_top1 | changed_rank | rank_ge5 | margin_p50 | qualified_eval | elapsed_s |
|---|---:|---:|---:|---:|---:|---|---:|
| baseline_no_penalty | 0 | 0 | 0 | 0 | 0 | false | 25.64 |
| c_k3_coef001 | 0 | 0 | 0 | 0 | 0 | false | 26.41 |
| c_k3_coef005 | 0 | 0 | 0 | 0 | 0 | false | 25.59 |
| c_k5_coef001 | 0 | 0 | 0 | 0 | 0 | false | 25.79 |
| c_k5_coef005 | 0 | 0 | 0 | 0 | 0 | false | 24.02 |
| d_weak_margin_coef001 | 0 | 0 | 0 | 0 | 0 | false | 26.52 |
| d_weak_margin_coef005 | 0 | 0 | 0 | 0 | 0 | false | 26.27 |
| cd_k3_lr005_wm005 | 0 | 0 | 0 | 0 | 0 | false | 24.78 |

## Full Seed Check

`k5_coef001` was rerun on the full seed `202604340000` shape: `3*16=48` rollout half-hands plus `16` fresh validation half-hands. With movement quality gate enabled it was rolled back to `top1=0`, skipped eval, and used `64` script half-hands in `63.82s`.

The same full seed with movement quality gate disabled used `48` rollout + `16` fresh + `16` eval = `80` script half-hands in `82.54s`, and produced:

| label | train_top1 | fresh_top1 | changed_rank | rank_ge5 | margin_p50 | eval_fourth | deal_in |
|---|---:|---:|---:|---:|---:|---:|---:|
| k5_coef001_no_quality_gate | 0.004016 | 0.009091 | 10.0 | 1.0 | 3.008 | 0.4375 | 0.0625 |

## Interpretation

- The implemented penalties are wired correctly and appear in loss/report/checkpoint config metadata.
- The compressed screen itself was too short to produce movement: even no-penalty baseline ended at `top1=0`.
- The full-seed diagnostic shows `low_rank K=5 coef=0.001` does not solve direction quality. It mostly suppresses movement, and the remaining flips are still rank-10 / rank>=5 / high-margin flips.
- Do not expand to `coef=0.005/0.01` full registry yet; `0.001` already fails to produce useful top2/top3 movement here.

Next useful variant is not a stronger penalty. It should be a gated objective that only applies the PPO policy loss on weak-margin states, or weights advantages by weak-margin support, so training pressure is redirected before the regularizer has to suppress bad flips.
