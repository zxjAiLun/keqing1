# KeqingRL Step 3.13C Per-Iteration Fresh Early-Stop

scope: `diagnostic early-stop selector; not strength evidence`
source_config_id: `93`
support_policy_mode: `support-only-topk`
teacher_source: `rule-component-v1`
teacher_ce_coef: `0.03`
teacher_confidence_gate: `entropy<=1.0, margin>=0.2`
rule_score_scale: `0.25`
seeds: `202604340000,202604350000,202604360000,202604370000`
run_count: `432 half-hands; 449.57s`

## Aggregate

- qualified: `3/4`
- train_gate: `4/4`
- fresh_gate: `3/4`
- selected_iterations: `{'202604340000': 1, '202604350000': 0, '202604360000': 0, '202604370000': 1}`
- top1_mean/max: `0.0594975` / `0.0938967`
- fresh_mean/max: `0.0322449` / `0.0762332`
- changed_rank_mean: `2.40238`
- rank_ge5_max: `0`
- teacher_keep_mean: `0.273173`
- teacher_prior_agree_mean: `0.667186`
- eval_fourth_mean qualified-only: `0.375`
- eval_deal_in_mean qualified-only: `0.125`

## Per Seed

- seed=202604340000 qualified=True selected_iter=1 train=True fresh=True top1=0.0673077 fresh_top1=0.0762332 changed_rank=2.14286 rank_ge5=0 teacher_keep=0.307692 eval_fourth=0.5 deal_in=0.25
- seed=202604350000 qualified=False selected_iter=0 train=True fresh=False top1=0.0267857 fresh_top1=0.00480769 changed_rank=2.5 rank_ge5=0 teacher_keep=0.276786 eval_fourth=nan deal_in=nan
- seed=202604360000 qualified=True selected_iter=0 train=True fresh=True top1=0.0938967 fresh_top1=0.0327103 changed_rank=2.55 rank_ge5=0 teacher_keep=0.258216 eval_fourth=0.375 deal_in=0.0625
- seed=202604370000 qualified=True selected_iter=1 train=True fresh=True top1=0.05 fresh_top1=0.0152284 changed_rank=2.41667 rank_ge5=0 teacher_keep=0.25 eval_fourth=0.25 deal_in=0.0625

## Conclusion

- Per-iteration fresh early-stop improves the gate result from the previous `1/4` for the meaningful confidence-gated teacher config to `3/4`.
- The remaining failed seed under-moves on fresh (`fresh_top1=0.00480769`), not because of topK-outside flips.
- `support-only-topK=3` remains intact: `rank_ge5_max=0`.
- This is still diagnostic selection, not strength evidence; do not claim strength from these eval smoke rows.
- The next useful check is a paired seat-rotation sanity/strength-proxy eval for the selected early-stop checkpoints, while keeping the diagnostic caveat.

## Artifacts

- `step313c_early_stop_summary.csv`
- `step313c_early_stop_summary.json`
- `seed_*/iterations.csv` with per-iteration fresh fields
- `seed_*/summary.csv` with selected iteration fields
