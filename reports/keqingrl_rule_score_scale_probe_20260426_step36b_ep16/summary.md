# KeqingRL Rule Score Scale Probe

source_type: `checkpoint`
training: `false`
diagnostic_only: `true`
candidate_summary: `reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/summary.csv`
episodes: `16`
eval_episodes: `16`
scales: `1.0,0.5,0.25,0.1`
seed_registry_id: `base=202604260000:stride=1:count=16`
seed_hash: `1a118258d3dc9335`

## Results

- source=93 rerun=0 opponent=rulebase scale=1 top1_changed=0 rule_agree=1 remaining_margin=3.00012 scaled_prior_margin=3.00012 delta_max=0.000237783 kl_vs_prior=9.31559e-07 clip_vs_prior=0 eval_rank=-0.15625 fourth=0.4375 deal_in=0.1875
- source=93 rerun=0 opponent=rulebase scale=0.5 top1_changed=0 rule_agree=1 remaining_margin=1.50007 scaled_prior_margin=1.50006 delta_max=0.000251455 kl_vs_prior=0.495238 clip_vs_prior=1 eval_rank=-0.15625 fourth=0.4375 deal_in=0.1875
- source=93 rerun=0 opponent=rulebase scale=0.25 top1_changed=0 rule_agree=1 remaining_margin=0.750033 scaled_prior_margin=0.750034 delta_max=0.000247392 kl_vs_prior=0.890497 clip_vs_prior=1 eval_rank=-0.15625 fourth=0.4375 deal_in=0.1875
- source=93 rerun=0 opponent=rulebase scale=0.1 top1_changed=0 rule_agree=1 remaining_margin=0.300015 scaled_prior_margin=0.300015 delta_max=0.000231558 kl_vs_prior=1.10467 clip_vs_prior=1 eval_rank=-0.15625 fourth=0.4375 deal_in=0.1875
- source=57 rerun=1 opponent=rule_prior_greedy scale=1 top1_changed=0 rule_agree=1 remaining_margin=3.00013 scaled_prior_margin=3.00013 delta_max=0.000219999 kl_vs_prior=-5.05412e-08 clip_vs_prior=0 eval_rank=-0.15625 fourth=0.4375 deal_in=0.1875
- source=57 rerun=1 opponent=rule_prior_greedy scale=0.5 top1_changed=0 rule_agree=1 remaining_margin=1.50007 scaled_prior_margin=1.50007 delta_max=0.000226188 kl_vs_prior=0.512479 clip_vs_prior=1 eval_rank=-0.15625 fourth=0.4375 deal_in=0.1875
- source=57 rerun=1 opponent=rule_prior_greedy scale=0.25 top1_changed=0 rule_agree=1 remaining_margin=0.750036 scaled_prior_margin=0.750033 delta_max=0.000218261 kl_vs_prior=0.877135 clip_vs_prior=1 eval_rank=-0.15625 fourth=0.4375 deal_in=0.1875
- source=57 rerun=1 opponent=rule_prior_greedy scale=0.1 top1_changed=0 rule_agree=1 remaining_margin=0.300017 scaled_prior_margin=0.300013 delta_max=0.000230763 kl_vs_prior=1.11857 clip_vs_prior=1 eval_rank=-0.15625 fourth=0.4375 deal_in=0.1875
- source=8 rerun=3 opponent=rule_prior_greedy scale=1 top1_changed=0 rule_agree=1 remaining_margin=3.00013 scaled_prior_margin=3.00014 delta_max=0.00110418 kl_vs_prior=1.14743e-05 clip_vs_prior=0 eval_rank=-0.15625 fourth=0.4375 deal_in=0.1875
- source=8 rerun=3 opponent=rule_prior_greedy scale=0.5 top1_changed=0 rule_agree=1 remaining_margin=1.50005 scaled_prior_margin=1.50006 delta_max=0.00102875 kl_vs_prior=0.526544 clip_vs_prior=1 eval_rank=-0.15625 fourth=0.4375 deal_in=0.1875
- source=8 rerun=3 opponent=rule_prior_greedy scale=0.25 top1_changed=0 rule_agree=1 remaining_margin=0.75003 scaled_prior_margin=0.750035 delta_max=0.00102929 kl_vs_prior=0.863708 clip_vs_prior=1 eval_rank=-0.15625 fourth=0.4375 deal_in=0.1875
- source=8 rerun=3 opponent=rule_prior_greedy scale=0.1 top1_changed=0 rule_agree=1 remaining_margin=0.299989 scaled_prior_margin=0.300012 delta_max=0.000993315 kl_vs_prior=1.13238 clip_vs_prior=1 eval_rank=-0.15625 fourth=0.4375 deal_in=0.1875

## Artifacts

- `rule_score_scale_probe.json`
- `summary.csv`
