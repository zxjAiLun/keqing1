# KeqingRL Tempered-Ratio PPO Diagnostic

source_type: `checkpoint`
ratio_mode: `tempered_current_logits`
candidate_summary: `reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/summary.csv`
source_config_ids: `93`
episodes: `2`
iterations: `1`
rule_score_scales: `0.25`
rule_score_scale_version: `keqingrl_rule_score_scale_v1`
temperatures: `1.25`
lrs: `0.003`
update_epochs_values: `1`
clip_eps_values: `0.2`
rule_kl_coef_values: `0.005`
delta_l2_coef_values: `0.001`
delta_clip_values: `0.0`
delta_clip_coef_values: `0.0`
low_rank_flip_topk_values: `3`
low_rank_flip_penalty_coef_values: `0.005`
weak_margin_threshold_values: `0.75`
weak_margin_flip_penalty_coef_values: `0.005`
entropy_coef: `0.005`
pass_criteria: `{'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'max_eval_fourth': 0.5, 'max_eval_deal_in': 0.25}`
adaptive_recovery: `{'enabled': True, 'role': 'adaptive_movement_diagnostic_not_candidate_selector', 'max_extra_epochs': 1, 'min_top1_changed': 0.02, 'max_top1_changed': 0.15, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'rollback_on_unstable_overmove_or_quality': True}`
movement_regularization: `{'low_rank_flip_topk_values': (3,), 'low_rank_flip_penalty_coef_values': (0.005,), 'weak_margin_threshold_values': (0.75,), 'weak_margin_flip_penalty_coef_values': (0.005,), 'weak_margin_threshold_units': 'unscaled_prior_logits'}`
movement_quality_gate: `{'enabled': True, 'role': 'candidate_selector', 'train_min_top1_changed': 0.02, 'train_max_top1_changed': 0.15, 'max_changed_prior_rank_mean': 3.0, 'max_rank_ge5_rate': 0.1, 'max_prior_margin_p50': 1.0}`
fresh_validation: `{'episodes': 1, 'seed_base': 202604460000, 'seed_stride': 1, 'seat_rotation': (0, 1), 'min_top1_changed': 0.01, 'max_top1_changed': 0.1, 'policy_mode': 'greedy'}`
eval_seed_registry_id: `base=202604450000:stride=1:count=2`
eval_seed_hash: `a7d21436cb51c25c`
eval_scope: `fixed-seed smoke; learner seats 0,1 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=0 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=1 clip=0.2 rule_kl=0.005 delta_l2=0.001 delta_clip=0/0 low_rank=3/0.005 weak_margin=0.75/0.005 pass=False non_top1=23 non_top1_pos=0 top1_changed=0 train_quality=False changed_rank=0 rank_ge5=0 margin_p50=0 low_rank_pen=3.999 weak_margin_pen=0.867861 fresh_top1=0 fresh_quality=False qualified_eval=False effective_margin=0.74255 scaled_prior_margin=0.750018 t_kl=1.05687e-05 t_clip=0 u_kl=0.00174557 u_clip=0 delta_max=0.0301919 recovery_extra=1 recovery_stop=budget_exhausted eval_fourth=nan deal_in=nan

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
