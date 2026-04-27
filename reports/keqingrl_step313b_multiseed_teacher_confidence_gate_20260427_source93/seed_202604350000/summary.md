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
rule_kl_coef_values: `0.001`
delta_l2_coef_values: `0.0`
delta_clip_values: `0.0`
delta_clip_coef_values: `0.0`
topk_ranking_aux: `{'modes': ('teacher-ce',), 'coef_values': (0.03,), 'topk': 3, 'teacher_sources': ('rule-component-v1',), 'teacher_temperature': 1.0, 'teacher_confidence_gate': True, 'teacher_entropy_max_values': (0.8, 1.0), 'teacher_margin_min_values': (0.2,), 'teacher_prior_agree_min_values': (0.0,), 'expanded_configs': ({'mode': 'teacher-ce', 'coef': 0.03, 'topk': 3, 'teacher_source': 'rule-component-v1', 'teacher_temperature': 1.0, 'teacher_confidence_gate': True, 'teacher_entropy_max': 0.8, 'teacher_margin_min': 0.2, 'teacher_prior_agree_min': 0.0, 'teacher_version': 'action_feature_component_reranker_v1', 'teacher_topk': 3, 'teacher_target_type': 'topk_distribution', 'teacher_contract_version': 'keqingrl_topk_teacher_v1'}, {'mode': 'teacher-ce', 'coef': 0.03, 'topk': 3, 'teacher_source': 'rule-component-v1', 'teacher_temperature': 1.0, 'teacher_confidence_gate': True, 'teacher_entropy_max': 1.0, 'teacher_margin_min': 0.2, 'teacher_prior_agree_min': 0.0, 'teacher_version': 'action_feature_component_reranker_v1', 'teacher_topk': 3, 'teacher_target_type': 'topk_distribution', 'teacher_contract_version': 'keqingrl_topk_teacher_v1'}), 'implemented_modes': ('none', 'teacher-ce'), 'teacher_contract_version': 'keqingrl_topk_teacher_v1', 'teacher_source_note': 'rule-prior-topk is the negative control; rule-components is a legacy alias; rule-component-v1 is an action-feature diagnostic reranker'}`
low_rank_flip_topk_values: `3`
low_rank_flip_penalty_coef_values: `0.0`
weak_margin_threshold_values: `0.75`
weak_margin_flip_penalty_coef_values: `0.0`
support_policy: `{'modes': ('support-only-topk',), 'expanded_configs': ({'support_policy_mode': 'support-only-topk', 'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'},), 'legacy_delta_support_args_used': False, 'scope': 'policy_forward_rollout_loss_fresh_eval'}`
delta_support_projection: `{'modes': ('all',), 'topk_values': (3,), 'margin_threshold_values': (0.75,), 'outside_support_delta_modes': ('zero',), 'expanded_configs': ({'support_policy_mode': 'unrestricted', 'mode': 'all', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'projected_policy_forward_and_loss', 'margin_threshold_units': 'unscaled_prior_logits'}`
actor_update_support: `{'modes': ('all',), 'topk_values': (3,), 'margin_threshold_values': (0.75,), 'expanded_configs': ({'mode': 'all', 'topk': 3, 'margin_threshold': 0.75, 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'actor_policy_loss_only', 'value_rank_losses_weighted': False, 'margin_threshold_units': 'unscaled_prior_logits'}`
entropy_coef: `0.005`
pass_criteria: `{'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'max_eval_fourth': 0.5, 'max_eval_deal_in': 0.25}`
adaptive_recovery: `{'enabled': False, 'role': 'adaptive_movement_diagnostic_not_candidate_selector', 'max_extra_epochs': 0, 'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'rollback_on_unstable_overmove_or_quality': True}`
movement_regularization: `{'low_rank_flip_topk_values': (3,), 'low_rank_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_values': (0.75,), 'weak_margin_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_units': 'unscaled_prior_logits'}`
movement_quality_gate: `{'enabled': True, 'role': 'candidate_selector', 'train_min_top1_changed': 0.02, 'train_max_top1_changed': 0.15, 'max_changed_prior_rank_mean': 3.0, 'max_rank_ge5_rate': 0.1, 'max_prior_margin_p50': 3.1}`
fresh_validation: `{'episodes': 16, 'seed_base': 202604370000, 'seed_stride': 1, 'seat_rotation': (0,), 'min_top1_changed': 0.01, 'max_top1_changed': 0.1, 'policy_mode': 'greedy'}`
eval_seed_registry_id: `base=202604360000:stride=1:count=16`
eval_seed_hash: `080515d48a60cb76`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=0 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 ranking_aux=teacher-ce/rule-component-v1/0.03/k3 teacher_conf=True/0.8/0.2 low_rank=3/0 weak_margin=0.75/0 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=all/3/0.75 pass=False non_top1=112 non_top1_pos=86 actor_kept=1 dropped_pos=0 kept_non_top1_pos=86 top1_changed=0.265116 train_quality=False changed_rank=2.35088 rank_ge5=0 margin_p50=3 low_rank_pen=0 weak_margin_pen=0.544005 rank_aux=0 teacher_kl=0 teacher_agree=0.55814 teacher_prior_agree=0.646512 teacher_rule_top1_rank=1.45581 teacher_margin=0.463289 teacher_entropy=1.02979 teacher_conf_keep=0 fresh_top1=0.232227 fresh_quality=False qualified_eval=False effective_margin=0.559263 scaled_prior_margin=0.750047 t_kl=0.0138358 t_clip=0.223256 u_kl=0.0252525 u_clip=0.353488 delta_max=2.71965 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=1 source=93 scale=0.25 temp=1.25 lr=0.003 epochs=8 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 ranking_aux=teacher-ce/rule-component-v1/0.03/k3 teacher_conf=True/1/0.2 low_rank=3/0 weak_margin=0.75/0 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=all/3/0.75 pass=False non_top1=130 non_top1_pos=106 actor_kept=1 dropped_pos=0 kept_non_top1_pos=106 top1_changed=0.135021 train_quality=True changed_rank=2.6875 rank_ge5=0 margin_p50=3.001 low_rank_pen=0 weak_margin_pen=0.548736 rank_aux=1.05615 teacher_kl=0.112002 teacher_agree=0.599156 teacher_prior_agree=0.687764 teacher_rule_top1_rank=1.40506 teacher_margin=0.496881 teacher_entropy=1.01841 teacher_conf_keep=0.299578 fresh_top1=0.116505 fresh_quality=False qualified_eval=False effective_margin=0.504069 scaled_prior_margin=0.750046 t_kl=0.0104915 t_clip=0.185654 u_kl=0.0161294 u_clip=0.223629 delta_max=2.11115 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
