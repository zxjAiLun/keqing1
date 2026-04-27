# KeqingRL Paired Candidate Evaluation

diagnostic_scope: `movement quality only; eval_episodes=1 is not strength evidence`
strength_note: `this report diagnoses why diagnostic gate candidates fail movement quality before any stronger eval`

mode: `summary`
source_type: `checkpoint`
candidate_summary: `reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/checkpoint_summary_merged.csv`
checkpoint_path: `None`
config_path: `None`
source_config_ids: `None`
rerun_config_ids: `None`
eval_seed_registry_id: `base=202604430000:stride=1:count=1`
eval_seed_hash: `6c443b27e66f58f9`
eval_seed_count: `1`
diagnostic_episode_count: `4`
seat_rotation: `0,1,2,3`
policy_mode: `greedy`
training_rollout_reuse: `false`
movement_quality_note: `neural_delta_abs_* is pure neural_delta; scaled_prior_delta_abs_* isolates scale contribution; final_minus_unscaled_prior_abs_* is compatibility/diagnostic only.`

## Required Fields

- checkpoint_path / checkpoint_sha256
- source_type / source_config_id / config_key
- eval_seed_registry_id / eval_seed_hash
- learner_deal_in_rate
- rulebase_fallback_count
- forced_terminal_preempt_count / autopilot_terminal_count
- illegal_action_rate_fail_closed / fallback_rate_fail_closed / forced_terminal_missed_fail_closed

## Candidate Results

- opponent=rule_prior_greedy candidate_id=source_93_rerun_202604370000 source_type=checkpoint source_id=93 delta_vs_zero=0.125 delta_vs_untrained=0.125 rank_pt=0.125 mean_rank=2.25 fourth=0.25 learner_deal_in=0 rule_agree=0.84127 top1_changed=0.15873 neural_delta_mean=0.29726 neural_delta_max=0.836407 changed_rank_mean=7.65 rank_ge5=0.875 changed_margin_p50=3.007 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604370000/checkpoint_config_000/policy_final.pt
- opponent=rule_prior_greedy candidate_id=source_93_rerun_202604360000 source_type=checkpoint source_id=93 delta_vs_zero=0 delta_vs_untrained=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0 rule_agree=0.94186 top1_changed=0.0581395 neural_delta_mean=0.176406 neural_delta_max=0.681503 changed_rank_mean=6.26667 rank_ge5=0.666667 changed_margin_p50=3.005 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604360000/checkpoint_config_000/policy_final.pt
- opponent=rule_prior_greedy candidate_id=source_93_rerun_202604340000 source_type=checkpoint source_id=93 delta_vs_zero=0 delta_vs_untrained=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0 rule_agree=0.726562 top1_changed=0.273438 neural_delta_mean=0.303992 neural_delta_max=0.944591 changed_rank_mean=6.25714 rank_ge5=0.585714 changed_margin_p50=3.004 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604340000/checkpoint_config_000/policy_final.pt
- opponent=rule_prior_greedy candidate_id=source_93_rerun_202604350000 source_type=checkpoint source_id=93 delta_vs_zero=0 delta_vs_untrained=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0 rule_agree=0.813688 top1_changed=0.186312 neural_delta_mean=0.287835 neural_delta_max=1.35186 changed_rank_mean=5.91837 rank_ge5=0.55102 changed_margin_p50=3.004 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604350000/checkpoint_config_000/policy_final.pt
- opponent=rulebase candidate_id=source_93_rerun_202604370000 source_type=checkpoint source_id=93 delta_vs_zero=0.125 delta_vs_untrained=0.125 rank_pt=0.125 mean_rank=2.25 fourth=0.25 learner_deal_in=0 rule_agree=0.84127 top1_changed=0.15873 neural_delta_mean=0.29726 neural_delta_max=0.836407 changed_rank_mean=7.65 rank_ge5=0.875 changed_margin_p50=3.007 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604370000/checkpoint_config_000/policy_final.pt
- opponent=rulebase candidate_id=source_93_rerun_202604360000 source_type=checkpoint source_id=93 delta_vs_zero=0 delta_vs_untrained=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0 rule_agree=0.94186 top1_changed=0.0581395 neural_delta_mean=0.176406 neural_delta_max=0.681503 changed_rank_mean=6.26667 rank_ge5=0.666667 changed_margin_p50=3.005 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604360000/checkpoint_config_000/policy_final.pt
- opponent=rulebase candidate_id=source_93_rerun_202604340000 source_type=checkpoint source_id=93 delta_vs_zero=0 delta_vs_untrained=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0 rule_agree=0.726562 top1_changed=0.273438 neural_delta_mean=0.303992 neural_delta_max=0.944591 changed_rank_mean=6.25714 rank_ge5=0.585714 changed_margin_p50=3.004 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604340000/checkpoint_config_000/policy_final.pt
- opponent=rulebase candidate_id=source_93_rerun_202604350000 source_type=checkpoint source_id=93 delta_vs_zero=0 delta_vs_untrained=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0 rule_agree=0.813688 top1_changed=0.186312 neural_delta_mean=0.287835 neural_delta_max=1.35186 changed_rank_mean=5.91837 rank_ge5=0.55102 changed_margin_p50=3.004 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604350000/checkpoint_config_000/policy_final.pt

## Decision

At least one checkpoint changes top-1 or has non-trivial delta; continue paired strength/stability evaluation before changing learning signal.
