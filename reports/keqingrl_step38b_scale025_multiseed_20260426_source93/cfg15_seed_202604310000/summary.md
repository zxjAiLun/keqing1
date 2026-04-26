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
lrs: `0.003`
update_epochs_values: `8`
clip_eps_values: `0.2`
rule_kl_coef_values: `0.005`
entropy_coef: `0.005`
pass_criteria: `{'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'max_eval_fourth': 0.5, 'max_eval_deal_in': 0.25}`
eval_seed_registry_id: `base=202604310000:stride=1:count=16`
eval_seed_hash: `0a023780fd4aa5d1`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=0 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.005 pass=True non_top1=207 non_top1_pos=180 top1_changed=0.141079 effective_margin=0.483732 scaled_prior_margin=0.750029 t_kl=0.0123282 t_clip=0.219917 u_kl=0.0193551 u_clip=0.307054 delta_max=1.97481 eval_fourth=0.4375 deal_in=0.0625

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
