# KeqingRL Tempered-Ratio PPO Diagnostic

source_type: `checkpoint`
ratio_mode: `tempered_current_logits`
candidate_summary: `reports/keqingrl_step39cd_penalty_full_seed_202604340000_source93/k5_coef001_diagnostic_no_quality_gate/summary.csv`
source_config_ids: `93`
episodes: `4`
iterations: `2`
rule_score_scales: `0.25`
rule_score_scale_version: `keqingrl_rule_score_scale_v1`
temperatures: `1.25`
lrs: `0.0025`
update_epochs_values: `5`
clip_eps_values: `0.2`
rule_kl_coef_values: `0.0`
delta_l2_coef_values: `0.0`
delta_clip_values: `0.0`
delta_clip_coef_values: `0.0`
topk_ranking_aux: `{'modes': ('teacher-ce',), 'coef_values': (5.0,), 'topk': 3, 'teacher_sources': ('mortal-discard-q',), 'teacher_temperature': 0.1, 'teacher_confidence_gate': False, 'teacher_entropy_max_values': (1000000000.0,), 'teacher_margin_min_values': (0.0,), 'teacher_prior_agree_min_values': (0.0,), 'expanded_configs': ({'mode': 'teacher-ce', 'coef': 5.0, 'topk': 3, 'teacher_source': 'mortal-discard-q', 'teacher_temperature': 0.1, 'teacher_confidence_gate': False, 'teacher_entropy_max': 1000000000.0, 'teacher_margin_min': 0.0, 'teacher_prior_agree_min': 0.0, 'teacher_version': 'mortal_discard_teacher_v1', 'teacher_topk': 3, 'teacher_target_type': 'topk_distribution', 'teacher_contract_version': 'keqingrl_topk_teacher_v1'},), 'implemented_modes': ('none', 'teacher-ce'), 'teacher_contract_version': 'keqingrl_topk_teacher_v1', 'teacher_source_note': 'mortal-discard-q is the only allowed strength teacher source; rule-prior-topk is a diagnostic negative control; rule-components is a legacy alias; rule-component-v1 is an action-feature diagnostic reranker; mortal-discard-q requires Mortal q/mask tensors in policy_input.obs.extras'}`
low_rank_flip_topk_values: `3`
low_rank_flip_penalty_coef_values: `0.0`
weak_margin_threshold_values: `0.75`
weak_margin_flip_penalty_coef_values: `0.01,0.05,0.1`
support_policy: `{'modes': ('support-only-topk',), 'expanded_configs': ({'support_policy_mode': 'support-only-topk', 'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'},), 'legacy_delta_support_args_used': False, 'scope': 'policy_forward_rollout_loss_fresh_eval'}`
delta_support_projection: `{'modes': ('topk',), 'topk_values': (3,), 'margin_threshold_values': (0.75,), 'outside_support_delta_modes': ('zero',), 'expanded_configs': ({'support_policy_mode': 'delta-topk-zero', 'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'projected_policy_forward_and_loss', 'margin_threshold_units': 'unscaled_prior_logits'}`
actor_update_support: `{'modes': ('topk',), 'topk_values': (3,), 'margin_threshold_values': (0.75,), 'expanded_configs': ({'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'actor_policy_loss_only', 'value_rank_losses_weighted': False, 'margin_threshold_units': 'unscaled_prior_logits'}`
entropy_coef: `0.0`
pass_criteria: `{'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'max_eval_fourth': 0.5, 'max_eval_deal_in': 0.25}`
adaptive_recovery: `{'enabled': False, 'role': 'adaptive_movement_diagnostic_not_candidate_selector', 'max_extra_epochs': 0, 'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'rollback_on_unstable_overmove_or_quality': True}`
movement_regularization: `{'low_rank_flip_topk_values': (3,), 'low_rank_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_values': (0.75,), 'weak_margin_flip_penalty_coef_values': (0.01, 0.05, 0.1), 'weak_margin_threshold_units': 'unscaled_prior_logits'}`
movement_quality_gate: `{'enabled': True, 'role': 'candidate_selector', 'train_min_top1_changed': 0.02, 'train_max_top1_changed': 0.2, 'max_changed_prior_rank_mean': 3.0, 'max_rank_ge5_rate': 0.1, 'max_prior_margin_p50': 1.0}`
fresh_validation: `{'episodes': 2, 'seed_base': 202604320000, 'seed_stride': 1, 'seat_rotation': (0,), 'min_top1_changed': 0.0, 'max_top1_changed': 0.2, 'policy_mode': 'greedy', 'per_iteration_early_stop_enabled': False, 'selection': 'best train+fresh gate pass; otherwise lowest fresh violation, then train/stability violation'}`
eval_seed_registry_id: `base=202604310000:stride=1:count=4`
eval_seed_hash: `b22f97dd19fd1449`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=0 source=93 scale=0.25 temp=1.25 lr=0.0025 epochs=5 clip=0.2 rule_kl=0 delta_l2=0 delta_clip=0/0 ranking_aux=teacher-ce/mortal-discard-q/5/k3 teacher_conf=False/1e+09/0 low_rank=3/0 weak_margin=0.75/0.01 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=topk/3/0.75 early_stop=False/iter1 pass=False non_top1=48 non_top1_pos=48 actor_kept=1 dropped_pos=0 kept_non_top1_pos=48 top1_changed=0 train_quality=False changed_rank=0 rank_ge5=0 margin_p50=0 low_rank_pen=0 weak_margin_pen=0.519347 rank_aux=0.929539 teacher_kl=0.902805 teacher_agree=0.625 teacher_prior_agree=0.625 teacher_rule_top1_rank=1.48611 teacher_margin=2.25407 teacher_entropy=0.026734 teacher_conf_keep=1 fresh_top1=0 fresh_quality=True qualified_eval=False effective_margin=0.706962 scaled_prior_margin=0.750028 t_kl=0.00102304 t_clip=0 u_kl=0.00521062 u_clip=0.0138889 delta_max=0.232699 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=1 source=93 scale=0.25 temp=1.25 lr=0.0025 epochs=5 clip=0.2 rule_kl=0 delta_l2=0 delta_clip=0/0 ranking_aux=teacher-ce/mortal-discard-q/5/k3 teacher_conf=False/1e+09/0 low_rank=3/0 weak_margin=0.75/0.05 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=topk/3/0.75 early_stop=False/iter1 pass=False non_top1=20 non_top1_pos=6 actor_kept=1 dropped_pos=0 kept_non_top1_pos=6 top1_changed=0.0952381 train_quality=False changed_rank=2.25 rank_ge5=0 margin_p50=3 low_rank_pen=0 weak_margin_pen=0.513691 rank_aux=0.784435 teacher_kl=0.74253 teacher_agree=0.690476 teacher_prior_agree=0.738095 teacher_rule_top1_rank=1.35714 teacher_margin=2.78442 teacher_entropy=0.0419048 teacher_conf_keep=1 fresh_top1=0.419355 fresh_quality=False qualified_eval=False effective_margin=0.623522 scaled_prior_margin=0.750048 t_kl=0.0293125 t_clip=0.357143 u_kl=0.049778 u_clip=0.47619 delta_max=1.08604 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan
- cfg=2 source=93 scale=0.25 temp=1.25 lr=0.0025 epochs=5 clip=0.2 rule_kl=0 delta_l2=0 delta_clip=0/0 ranking_aux=teacher-ce/mortal-discard-q/5/k3 teacher_conf=False/1e+09/0 low_rank=3/0 weak_margin=0.75/0.1 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=topk/3/0.75 early_stop=False/iter1 pass=False non_top1=34 non_top1_pos=12 actor_kept=1 dropped_pos=0 kept_non_top1_pos=12 top1_changed=0 train_quality=False changed_rank=0 rank_ge5=0 margin_p50=0 low_rank_pen=0 weak_margin_pen=0.499815 rank_aux=0.843814 teacher_kl=0.794555 teacher_agree=0.698413 teacher_prior_agree=0.698413 teacher_rule_top1_rank=1.36508 teacher_margin=2.53139 teacher_entropy=0.049259 teacher_conf_keep=1 fresh_top1=0 fresh_quality=True qualified_eval=False effective_margin=0.703386 scaled_prior_margin=0.75004 t_kl=0.0154299 t_clip=0.190476 u_kl=0.0340159 u_clip=0.365079 delta_max=0.926308 recovery_extra=0 recovery_stop=disabled eval_fourth=nan deal_in=nan

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
