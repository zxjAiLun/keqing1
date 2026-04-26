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
lrs: `0.001,0.003`
update_epochs_values: `4,8`
clip_eps_values: `0.1,0.2`
rule_kl_coef_values: `0.001,0.005`
entropy_coef: `0.005`
pass_criteria: `{'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'max_eval_fourth': 0.5, 'max_eval_deal_in': 0.25}`
eval_seed_registry_id: `base=202604310000:stride=1:count=16`
eval_seed_hash: `0a023780fd4aa5d1`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=0 source=93 scale=0.25 temp=1.25 lr=0.001 epochs=4 clip=0.1 rule_kl=0.001 pass=False non_top1=169 non_top1_pos=74 top1_changed=0 effective_margin=0.743828 scaled_prior_margin=0.750038 t_kl=5.52681e-06 t_clip=0 u_kl=0.00138738 u_clip=0.146465 delta_max=0.0165684 eval_fourth=0.375 deal_in=0.125
- cfg=1 source=93 scale=0.25 temp=1.25 lr=0.001 epochs=4 clip=0.1 rule_kl=0.005 pass=False non_top1=207 non_top1_pos=121 top1_changed=0 effective_margin=0.740278 scaled_prior_margin=0.750037 t_kl=3.98144e-06 t_clip=0 u_kl=0.00105003 u_clip=0.1 delta_max=0.0340933 eval_fourth=0.375 deal_in=0.125
- cfg=2 source=93 scale=0.25 temp=1.25 lr=0.001 epochs=4 clip=0.2 rule_kl=0.001 pass=False non_top1=194 non_top1_pos=152 top1_changed=0 effective_margin=0.728705 scaled_prior_margin=0.750038 t_kl=1.89988e-05 t_clip=0 u_kl=0.00142341 u_clip=0 delta_max=0.120392 eval_fourth=0.375 deal_in=0.125
- cfg=3 source=93 scale=0.25 temp=1.25 lr=0.001 epochs=4 clip=0.2 rule_kl=0.005 pass=False non_top1=191 non_top1_pos=102 top1_changed=0 effective_margin=0.729275 scaled_prior_margin=0.750046 t_kl=2.51549e-05 t_clip=0 u_kl=0.00130674 u_clip=0 delta_max=0.0727227 eval_fourth=0.375 deal_in=0.125
- cfg=4 source=93 scale=0.25 temp=1.25 lr=0.001 epochs=8 clip=0.1 rule_kl=0.001 pass=False non_top1=199 non_top1_pos=116 top1_changed=0 effective_margin=0.577063 scaled_prior_margin=0.750034 t_kl=0.00304969 t_clip=0.185022 u_kl=0.0063963 u_clip=0.352423 delta_max=0.490187 eval_fourth=0.375 deal_in=0.125
- cfg=5 source=93 scale=0.25 temp=1.25 lr=0.001 epochs=8 clip=0.1 rule_kl=0.005 pass=False non_top1=189 non_top1_pos=106 top1_changed=0 effective_margin=0.688835 scaled_prior_margin=0.750036 t_kl=0.000198398 t_clip=0 u_kl=0.00171854 u_clip=0.12844 delta_max=0.186003 eval_fourth=0.375 deal_in=0.125
- cfg=6 source=93 scale=0.25 temp=1.25 lr=0.001 epochs=8 clip=0.2 rule_kl=0.001 pass=True non_top1=197 non_top1_pos=105 top1_changed=0.113537 effective_margin=0.443151 scaled_prior_margin=0.750024 t_kl=0.00455194 t_clip=0.0305677 u_kl=0.0102113 u_clip=0.152838 delta_max=1.08436 eval_fourth=0.375 deal_in=0.125
- cfg=7 source=93 scale=0.25 temp=1.25 lr=0.001 epochs=8 clip=0.2 rule_kl=0.005 pass=True non_top1=191 non_top1_pos=93 top1_changed=0.113122 effective_margin=0.37286 scaled_prior_margin=0.750028 t_kl=0.0149963 t_clip=0.230769 u_kl=0.0279743 u_clip=0.371041 delta_max=0.819078 eval_fourth=0.375 deal_in=0
- cfg=8 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=4 clip=0.1 rule_kl=0.001 pass=False non_top1=199 non_top1_pos=124 top1_changed=0.0045045 effective_margin=0.414435 scaled_prior_margin=0.750036 t_kl=0.00365366 t_clip=0.247748 u_kl=0.00823166 u_clip=0.477477 delta_max=0.736604 eval_fourth=0.375 deal_in=0.125
- cfg=9 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=4 clip=0.1 rule_kl=0.005 pass=False non_top1=190 non_top1_pos=112 top1_changed=0 effective_margin=0.679868 scaled_prior_margin=0.750028 t_kl=0.000613628 t_clip=0.00454545 u_kl=0.00201165 u_clip=0.127273 delta_max=0.344157 eval_fourth=0.375 deal_in=0.125
- cfg=10 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=4 clip=0.2 rule_kl=0.001 pass=True non_top1=182 non_top1_pos=145 top1_changed=0.108225 effective_margin=0.433713 scaled_prior_margin=0.750039 t_kl=0.00399069 t_clip=0.0519481 u_kl=0.0140187 u_clip=0.147186 delta_max=1.27036 eval_fourth=0.4375 deal_in=0.125
- cfg=11 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=4 clip=0.2 rule_kl=0.005 pass=True non_top1=210 non_top1_pos=157 top1_changed=0.217213 effective_margin=0.39447 scaled_prior_margin=0.750038 t_kl=0.00713977 t_clip=0.0778688 u_kl=0.0156496 u_clip=0.303279 delta_max=1.93938 eval_fourth=0.4375 deal_in=0.125
- cfg=12 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.1 rule_kl=0.001 pass=False non_top1=199 non_top1_pos=76 top1_changed=0.0131004 effective_margin=0.502839 scaled_prior_margin=0.750032 t_kl=0.00427219 t_clip=0.248908 u_kl=0.00794376 u_clip=0.406114 delta_max=0.697697 eval_fourth=0.375 deal_in=0.125
- cfg=13 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.1 rule_kl=0.005 pass=False non_top1=200 non_top1_pos=128 top1_changed=0 effective_margin=0.617455 scaled_prior_margin=0.750028 t_kl=0.00219441 t_clip=0.148305 u_kl=0.00387335 u_clip=0.271186 delta_max=0.548959 eval_fourth=0.375 deal_in=0.125
- cfg=14 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.001 pass=False non_top1=192 non_top1_pos=167 top1_changed=0.468468 effective_margin=0.275444 scaled_prior_margin=0.750026 t_kl=0.0161398 t_clip=0.292793 u_kl=0.0337985 u_clip=0.477477 delta_max=1.71932 eval_fourth=0.5625 deal_in=0.125
- cfg=15 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.005 pass=True non_top1=189 non_top1_pos=118 top1_changed=0.0357143 effective_margin=0.409365 scaled_prior_margin=0.750029 t_kl=0.008714 t_clip=0.147321 u_kl=0.0138469 u_clip=0.25 delta_max=1.19397 eval_fourth=0.375 deal_in=0.125

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
