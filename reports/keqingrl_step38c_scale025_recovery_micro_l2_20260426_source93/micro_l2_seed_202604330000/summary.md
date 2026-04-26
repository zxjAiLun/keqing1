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
update_epochs_values: `4`
clip_eps_values: `0.2`
rule_kl_coef_values: `0.005`
delta_l2_coef_values: `0.0003,0.0005`
delta_clip_values: `0.0`
delta_clip_coef_values: `0.0`
entropy_coef: `0.005`
pass_criteria: `{'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'max_eval_fourth': 0.5, 'max_eval_deal_in': 0.25}`
eval_seed_registry_id: `base=202604330000:stride=1:count=16`
eval_seed_hash: `cace0c7e064285d7`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=0 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=4 clip=0.2 rule_kl=0.005 delta_l2=0.0003 delta_clip=0/0 pass=False non_top1=210 non_top1_pos=135 top1_changed=0.0772358 effective_margin=0.50804 scaled_prior_margin=0.750057 t_kl=0.0191521 t_clip=0.325203 u_kl=0.0359963 u_clip=0.479675 delta_max=1.11189 eval_fourth=0.3125 deal_in=0
- cfg=1 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=4 clip=0.2 rule_kl=0.005 delta_l2=0.0005 delta_clip=0/0 pass=True non_top1=212 non_top1_pos=141 top1_changed=0.1 effective_margin=0.421826 scaled_prior_margin=0.75005 t_kl=0.00895457 t_clip=0.129167 u_kl=0.0163384 u_clip=0.204167 delta_max=0.891514 eval_fourth=0.3125 deal_in=0.0625

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
