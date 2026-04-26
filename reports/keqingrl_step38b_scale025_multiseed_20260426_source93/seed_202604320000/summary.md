# KeqingRL Tempered-Ratio PPO Diagnostic

source_type: `checkpoint`
ratio_mode: `tempered_current_logits`
candidate_summary: `reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/summary.csv`
source_config_ids: `93`
episodes: `16`
iterations: `3`
rule_score_scales: `0.25`
rule_score_scale_version: `keqingrl_rule_score_scale_v1`
temperatures: `1.25`
lrs: `0.001`
update_epochs_values: `8`
clip_eps_values: `0.2`
rule_kl_coef_values: `0.001`
entropy_coef: `0.005`
pass_criteria: `{'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'max_eval_fourth': 0.5, 'max_eval_deal_in': 0.25}`
eval_seed_registry_id: `base=202604320000:stride=1:count=16`
eval_seed_hash: `720865d483c78a30`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=0 source=93 scale=0.25 temp=1.25 lr=0.001 epochs=8 clip=0.2 rule_kl=0.001 pass=False non_top1=200 non_top1_pos=120 top1_changed=0 effective_margin=0.539212 scaled_prior_margin=0.750036 t_kl=0.00345338 t_clip=0.0378151 u_kl=0.00755464 u_clip=0.130252 delta_max=0.489102 eval_fourth=0.25 deal_in=0.0625

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
