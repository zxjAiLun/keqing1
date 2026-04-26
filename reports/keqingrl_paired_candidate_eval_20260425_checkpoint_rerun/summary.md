# KeqingRL Paired Candidate Evaluation

candidate_summary: `reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/summary.csv`
eval_seed_registry_id: `base=202604250000:stride=1:count=64`
eval_seed_hash: `809ce4e45eb9ebe3`
eval_seed_count: `64`
diagnostic_episode_count: `4`
seat_rotation: `0,1,2,3`
policy_mode: `greedy`
training_rollout_reuse: `false`

## Required Fields

- checkpoint_path / checkpoint_sha256
- source_config_id / config_key
- eval_seed_registry_id / eval_seed_hash
- learner_deal_in_rate
- rulebase_fallback_count
- illegal_action_rate_fail_closed / fallback_rate_fail_closed / forced_terminal_missed_fail_closed

## Candidate Results

- opponent=rule_prior_greedy source_id=57 delta_vs_zero=0 delta_vs_untrained=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.117188 rule_agree=1 top1_changed=0 delta_mean=0.000149401 delta_max=0.000202656 checkpoint=reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/configs/rerun_001_source_57/policy_final.pt
- opponent=rule_prior_greedy source_id=93 delta_vs_zero=0 delta_vs_untrained=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.117188 rule_agree=1 top1_changed=0 delta_mean=0.000117531 delta_max=0.00023222 checkpoint=reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/configs/rerun_000_source_93/policy_final.pt
- opponent=rule_prior_greedy source_id=84 delta_vs_zero=0 delta_vs_untrained=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.117188 rule_agree=1 top1_changed=0 delta_mean=0.000119077 delta_max=0.000252008 checkpoint=reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/configs/rerun_002_source_84/policy_final.pt
- opponent=rule_prior_greedy source_id=8 delta_vs_zero=0 delta_vs_untrained=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.117188 rule_agree=1 top1_changed=0 delta_mean=0.000186681 delta_max=0.000825644 checkpoint=reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/configs/rerun_003_source_8/policy_final.pt
- opponent=rulebase source_id=57 delta_vs_zero=0 delta_vs_untrained=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.117188 rule_agree=1 top1_changed=0 delta_mean=0.000149401 delta_max=0.000202656 checkpoint=reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/configs/rerun_001_source_57/policy_final.pt
- opponent=rulebase source_id=93 delta_vs_zero=0 delta_vs_untrained=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.117188 rule_agree=1 top1_changed=0 delta_mean=0.000117531 delta_max=0.00023222 checkpoint=reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/configs/rerun_000_source_93/policy_final.pt
- opponent=rulebase source_id=84 delta_vs_zero=0 delta_vs_untrained=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.117188 rule_agree=1 top1_changed=0 delta_mean=0.000119077 delta_max=0.000252008 checkpoint=reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/configs/rerun_002_source_84/policy_final.pt
- opponent=rulebase source_id=8 delta_vs_zero=0 delta_vs_untrained=0 rank_pt=0 mean_rank=2.5 fourth=0.25 learner_deal_in=0.117188 rule_agree=1 top1_changed=0 delta_mean=0.000186681 delta_max=0.000825644 checkpoint=reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/configs/rerun_003_source_8/policy_final.pt

## Decision

Current checkpoints remain rule-prior equivalent on top-1 decisions; prioritize learning-signal research before more eval scaling.
