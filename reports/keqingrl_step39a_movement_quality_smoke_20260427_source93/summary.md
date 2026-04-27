# KeqingRL Paired Candidate Evaluation

mode: `summary`
source_type: `checkpoint`
candidate_summary: `reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/checkpoint_summary_merged.csv`
checkpoint_path: `None`
config_path: `None`
source_config_ids: `[93]`
rerun_config_ids: `[202604360000]`
eval_seed_registry_id: `base=202604390000:stride=1:count=2`
eval_seed_hash: `6d400420ce99d4ed`
eval_seed_count: `2`
diagnostic_episode_count: `1`
seat_rotation: `0,1`
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

- opponent=rule_prior_greedy candidate_id=93 source_type=checkpoint source_id=93 delta_vs_zero=0 delta_vs_untrained=0 rank_pt=-0.125 mean_rank=2.75 fourth=0.25 learner_deal_in=0 rule_agree=0.928571 top1_changed=0.0714286 neural_delta_mean=0.168007 neural_delta_max=0.558692 changed_rank_mean=7 rank_ge5=1 changed_margin_p50=3.0055 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604360000/checkpoint_config_000/policy_final.pt
- opponent=rulebase candidate_id=93 source_type=checkpoint source_id=93 delta_vs_zero=0 delta_vs_untrained=0 rank_pt=-0.125 mean_rank=2.75 fourth=0.25 learner_deal_in=0 rule_agree=0.928571 top1_changed=0.0714286 neural_delta_mean=0.168007 neural_delta_max=0.558692 changed_rank_mean=7 rank_ge5=1 changed_margin_p50=3.0055 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604360000/checkpoint_config_000/policy_final.pt

## Decision

At least one checkpoint changes top-1 or has non-trivial delta; continue paired strength/stability evaluation before changing learning signal.
