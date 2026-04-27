# KeqingRL Paired Candidate Evaluation

mode: `summary`
source_type: `checkpoint`
candidate_summary: `reports/keqingrl_step313c_early_stop_saved_candidates_20260427_source93/checkpoint_summary.csv`
checkpoint_path: `None`
config_path: `None`
source_config_ids: `[93]`
rerun_config_ids: `None`
eval_seed_registry_id: `base=202604410000:stride=1:count=1`
eval_seed_hash: `d2887d7fd41e719e`
eval_seed_count: `1`
diagnostic_episode_count: `1`
seat_rotation: `0,1,2,3`
policy_mode: `greedy`
training_rollout_reuse: `false`
strength_proxy_note: `paired seat-rotation sanity/strength-proxy only; this is not proof of model strength.`
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

- opponent=rulebase candidate_id=source_93_seed_202604340000 source_type=checkpoint source_id=93 delta_vs_zero=0 delta_vs_untrained=0 delta_vs_source=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.25 rule_agree=0.960784 top1_changed=0.0392157 neural_delta_mean=0.0833257 neural_delta_max=0.801771 changed_rank_mean=2 rank_ge5=0 changed_margin_p50=3 checkpoint=reports/keqingrl_step313c_early_stop_saved_candidates_20260427_source93/seed_202604340000/checkpoint_config_000/policy_final.pt
- opponent=rulebase candidate_id=source_93_seed_202604370000 source_type=checkpoint source_id=93 delta_vs_zero=0 delta_vs_untrained=0 delta_vs_source=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.25 rule_agree=1 top1_changed=0 neural_delta_mean=0.0672229 neural_delta_max=0.821352 changed_rank_mean=0 rank_ge5=0 changed_margin_p50=0 checkpoint=reports/keqingrl_step313c_early_stop_saved_candidates_20260427_source93/seed_202604370000/checkpoint_config_000/policy_final.pt
- opponent=rulebase candidate_id=source_93_seed_202604360000 source_type=checkpoint source_id=93 delta_vs_zero=-0.375 delta_vs_untrained=-0.375 delta_vs_source=-0.375 rank_pt=-0.375 mean_rank=2.75 fourth=0.5 learner_deal_in=0.5 rule_agree=0.98 top1_changed=0.02 neural_delta_mean=0.050853 neural_delta_max=0.647915 changed_rank_mean=2 rank_ge5=0 changed_margin_p50=3 checkpoint=reports/keqingrl_step313c_early_stop_saved_candidates_20260427_source93/seed_202604360000/checkpoint_config_000/policy_final.pt

## Decision

At least one checkpoint changes top-1 or has non-trivial delta; continue paired strength/stability evaluation before changing learning signal.
