# KeqingRL Tempered-Ratio PPO Diagnostic

source_type: `checkpoint`
ratio_mode: `tempered_current_logits`
candidate_summary: `reports/keqingrl_step39cd_penalty_full_seed_202604340000_source93/k5_coef001_diagnostic_no_quality_gate/summary.csv`
source_config_ids: `93`
episodes: `32`
iterations: `1`
rule_score_scales: `0.25`
rule_score_scale_version: `keqingrl_rule_score_scale_v1`
temperatures: `1.25`
lrs: `0.0001,0.0003`
update_epochs_values: `1`
clip_eps_values: `0.1,0.2`
rule_kl_coef_values: `0.001`
delta_l2_coef_values: `0.0`
delta_clip_values: `0.0`
delta_clip_coef_values: `0.0`
topk_ranking_aux: `{'modes': ('teacher-ce',), 'coef_values': (1.0,), 'topk': 3, 'teacher_sources': ('mortal-action-q',), 'teacher_temperature': 1.0, 'teacher_confidence_gate': False, 'teacher_entropy_max_values': (1000000000.0,), 'teacher_margin_min_values': (0.0,), 'teacher_prior_agree_min_values': (0.0,), 'expanded_configs': ({'mode': 'teacher-ce', 'coef': 1.0, 'topk': 3, 'teacher_source': 'mortal-action-q', 'teacher_temperature': 1.0, 'teacher_confidence_gate': False, 'teacher_entropy_max': 1000000000.0, 'teacher_margin_min': 0.0, 'teacher_prior_agree_min': 0.0, 'teacher_version': 'mortal_action_teacher_v1', 'teacher_topk': 3, 'teacher_target_type': 'topk_distribution', 'teacher_contract_version': 'keqingrl_topk_teacher_v1'},), 'implemented_modes': ('none', 'teacher-ce'), 'teacher_contract_version': 'keqingrl_topk_teacher_v1', 'teacher_source_note': 'mortal-discard-q and mortal-action-q are the only allowed Mortal teacher sources; rule-prior-topk is a diagnostic negative control; rule-components is a legacy alias; rule-component-v1 is an action-feature diagnostic reranker; Mortal teacher sources require Mortal q/mask tensors in policy_input.obs.extras'}`
action_scope: `{'self_turn_action_types': ('DISCARD', 'REACH_DISCARD', 'TSUMO', 'RYUKYOKU'), 'response_action_types': ('PASS', 'RON', 'PON', 'CHI'), 'forced_autopilot_action_types': ('TSUMO', 'RON', 'RYUKYOKU')}`
low_rank_flip_topk_values: `3`
low_rank_flip_penalty_coef_values: `0.0`
weak_margin_threshold_values: `0.75`
weak_margin_flip_penalty_coef_values: `0.0`
support_policy: `{'modes': ('support-only-topk',), 'expanded_configs': ({'support_policy_mode': 'support-only-topk', 'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'},), 'legacy_delta_support_args_used': False, 'scope': 'policy_forward_rollout_loss_fresh_eval'}`
delta_support_projection: `{'modes': ('topk',), 'topk_values': (3,), 'margin_threshold_values': (0.75,), 'outside_support_delta_modes': ('zero',), 'expanded_configs': ({'support_policy_mode': 'delta-topk-zero', 'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'outside_support_delta_mode': 'zero', 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'projected_policy_forward_and_loss', 'margin_threshold_units': 'unscaled_prior_logits'}`
actor_update_support: `{'modes': ('topk',), 'topk_values': (3,), 'margin_threshold_values': (0.75,), 'expanded_configs': ({'mode': 'topk', 'topk': 3, 'margin_threshold': 0.75, 'margin_threshold_units': 'unscaled_prior_logits'},), 'scope': 'actor_policy_loss_only', 'value_rank_losses_weighted': False, 'margin_threshold_units': 'unscaled_prior_logits'}`
entropy_coef: `0.005`
pass_criteria: `{'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'max_eval_fourth': 0.5, 'max_eval_deal_in': 0.25}`
adaptive_recovery: `{'enabled': False, 'role': 'adaptive_movement_diagnostic_not_candidate_selector', 'max_extra_epochs': 0, 'min_top1_changed': 0.02, 'max_top1_changed': 0.25, 'max_tempered_kl': 0.03, 'max_tempered_clip': 0.3, 'max_untempered_clip': 0.8, 'rollback_on_unstable_overmove_or_quality': True}`
movement_regularization: `{'low_rank_flip_topk_values': (3,), 'low_rank_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_values': (0.75,), 'weak_margin_flip_penalty_coef_values': (0.0,), 'weak_margin_threshold_units': 'unscaled_prior_logits'}`
movement_quality_gate: `{'enabled': False, 'role': 'candidate_selector', 'train_min_top1_changed': 0.02, 'train_max_top1_changed': 0.15, 'max_changed_prior_rank_mean': 3.0, 'max_rank_ge5_rate': 0.1, 'max_prior_margin_p50': 1.0}`
terminal_coverage_gate: `{'enabled': True, 'outcome_gate_enabled': False, 'role': 'qualification_gate_for_terminal_decision_opportunity_coverage', 'min_legal_terminal_rows': 1, 'min_score_changed_episode_rate': 0.0, 'min_legal_agari_rows': 1, 'min_prepared_legal_terminal_rows': 0, 'min_prepared_legal_agari_rows': 0, 'min_selected_agari_count': 0, 'min_selected_agari_episode_rate': 0.0, 'note': 'qualification is opportunity-based by default; score_changed/selected_agari thresholds are outcome diagnostics and only gate when outcome_gate_enabled=true'}`
fresh_validation: `{'episodes': 0, 'seed_base': 202604320000, 'seed_stride': 1, 'seat_rotation': (0,), 'min_top1_changed': 0.01, 'max_top1_changed': 0.1, 'policy_mode': 'greedy', 'per_iteration_early_stop_enabled': False, 'selection': 'best train+fresh gate pass; otherwise lowest fresh violation, then train/stability violation'}`
eval_seed_registry_id: `base=202604310000:stride=1:count=16`
eval_seed_hash: `0a023780fd4aa5d1`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results


## Contract Scoreboard

- diagnostic_only_outcome_note: `outcome counters do not gate unless --terminal-coverage-outcome-gate is enabled`
- cfg=0 teacher=mortal-action-q role=primary_action_q_teacher legal_owner_pass=True action_identity_pass=True mapping=0/1 opportunity_gate=False outcome_gate=False score_changed=0 selected_agari=0 qualified=False reason=blocked_by_mask_parity_gap notes=blocked by mask parity gap|unqualified opportunity coverage
- cfg=1 teacher=mortal-action-q role=primary_action_q_teacher legal_owner_pass=True action_identity_pass=True mapping=0/1 opportunity_gate=False outcome_gate=False score_changed=0 selected_agari=0 qualified=False reason=blocked_by_mask_parity_gap notes=blocked by mask parity gap|unqualified opportunity coverage
- cfg=2 teacher=mortal-action-q role=primary_action_q_teacher legal_owner_pass=True action_identity_pass=True mapping=0/1 opportunity_gate=False outcome_gate=False score_changed=0 selected_agari=0 qualified=False reason=blocked_by_mask_parity_gap notes=blocked by mask parity gap|unqualified opportunity coverage
- cfg=3 teacher=mortal-action-q role=primary_action_q_teacher legal_owner_pass=True action_identity_pass=True mapping=0/1 opportunity_gate=False outcome_gate=False score_changed=0 selected_agari=0 qualified=False reason=blocked_by_mask_parity_gap notes=blocked by mask parity gap|unqualified opportunity coverage

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
