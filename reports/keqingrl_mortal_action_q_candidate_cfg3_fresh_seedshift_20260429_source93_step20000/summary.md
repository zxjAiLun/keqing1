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
lrs: `0.0085`
update_epochs_values: `3`
clip_eps_values: `0.2`
rule_kl_coef_values: `0.001`
delta_l2_coef_values: `0.0`
delta_clip_values: `0.0`
delta_clip_coef_values: `0.0`
topk_ranking_aux: `{'modes': ('teacher-ce',), 'coef_values': (10.0,), 'topk': 3, 'teacher_sources': ('mortal-action-q',), 'teacher_temperature': 1.0, 'teacher_confidence_gate': False, 'teacher_entropy_max_values': (1000000000.0,), 'teacher_margin_min_values': (0.0,), 'teacher_prior_agree_min_values': (0.0,), 'expanded_configs': ({'mode': 'teacher-ce', 'coef': 10.0, 'topk': 3, 'teacher_source': 'mortal-action-q', 'teacher_temperature': 1.0, 'teacher_confidence_gate': False, 'teacher_entropy_max': 1000000000.0, 'teacher_margin_min': 0.0, 'teacher_prior_agree_min': 0.0, 'teacher_version': 'mortal_action_teacher_v1', 'teacher_topk': 3, 'teacher_target_type': 'topk_distribution', 'teacher_contract_version': 'keqingrl_topk_teacher_v1'},), 'implemented_modes': ('none', 'teacher-ce'), 'teacher_contract_version': 'keqingrl_topk_teacher_v1', 'teacher_source_note': 'mortal-discard-q and mortal-action-q are the only allowed Mortal teacher sources; rule-prior-topk is a diagnostic negative control; rule-components is a legacy alias; rule-component-v1 is an action-feature diagnostic reranker; Mortal teacher sources require Mortal q/mask tensors in policy_input.obs.extras'}`
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
movement_quality_gate: `{'enabled': True, 'role': 'candidate_selector', 'train_min_top1_changed': 0.02, 'train_max_top1_changed': 0.15, 'max_changed_prior_rank_mean': 3.0, 'max_rank_ge5_rate': 0.1, 'max_prior_margin_p50': 3.5}`
terminal_coverage_gate: `{'enabled': True, 'outcome_gate_enabled': False, 'role': 'qualification_gate_for_terminal_decision_opportunity_coverage', 'min_legal_terminal_rows': 1, 'min_score_changed_episode_rate': 0.0, 'min_legal_agari_rows': 1, 'min_prepared_legal_terminal_rows': 0, 'min_prepared_legal_agari_rows': 0, 'min_selected_agari_count': 0, 'min_selected_agari_episode_rate': 0.0, 'note': 'qualification is opportunity-based by default; score_changed/selected_agari thresholds are outcome diagnostics and only gate when outcome_gate_enabled=true'}`
fresh_validation: `{'episodes': 32, 'seed_base': 202604320000, 'seed_stride': 1, 'seat_rotation': (0,), 'min_top1_changed': 0.01, 'max_top1_changed': 0.1, 'policy_mode': 'greedy', 'per_iteration_early_stop_enabled': False, 'selection': 'best train+fresh gate pass; otherwise lowest fresh violation, then train/stability violation'}`
eval_seed_registry_id: `base=202604310000:stride=1:count=16`
eval_seed_hash: `0a023780fd4aa5d1`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Results

- cfg=0 source=93 scale=0.25 temp=1.25 lr=0.0085 epochs=3 clip=0.2 rule_kl=0.001 delta_l2=0 delta_clip=0/0 ranking_aux=teacher-ce/mortal-action-q/10/k3 teacher_conf=False/1e+09/0 low_rank=3/0 weak_margin=0.75/0 support_policy=support-only-topk delta_support=topk/3/0.75/zero actor_support=topk/3/0.75 early_stop=False/iter0 pass=True non_top1=252 non_top1_pos=226 actor_kept=0.998347 dropped_pos=0 kept_non_top1_pos=226 top1_changed=0.031405 train_quality=True changed_rank=2.05263 rank_ge5=0 margin_p50=3 low_rank_pen=0.00248746 weak_margin_pen=0.459402 rank_aux=0.908849 teacher_kl=0.566287 teacher_agree=0.635628 teacher_prior_agree=0.651822 teacher_rule_top1_rank=1.4332 teacher_margin=3.57042 teacher_entropy=0.342562 teacher_conf_keep=0.816529 reach_n=2 reach_policy=1 reach_teacher=1 non_discard_n=110 call_n=108 agari_n=0 score_changed_ep=29/32 legal_agari_rows=17 selected_agari=17 terminal_quality=True fresh_top1=0.0292969 fresh_quality=True qualified_eval=True effective_margin=0.792911 scaled_prior_margin=0.8294 t_kl=0.0134062 t_clip=0.204959 u_kl=0.0265724 u_clip=0.330579 delta_max=0.779643 recovery_extra=0 recovery_stop=disabled eval_fourth=0.125 deal_in=0

## Contract Scoreboard

- diagnostic_only_outcome_note: `outcome counters do not gate unless --terminal-coverage-outcome-gate is enabled`
- cfg=0 teacher=mortal-action-q role=primary_action_q_teacher legal_owner_pass=True action_identity_pass=True mapping=605/605 opportunity_gate=True outcome_gate=False score_changed=29 selected_agari=17 qualified=True reason=qualified_for_eval notes=qualified for paired eval

## Artifacts

- `tempered_ratio_pilot.json`
- `summary.csv`
- `iterations.csv`
- `batch_steps.csv`
- `advantage_audit.csv`
