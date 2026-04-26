# KeqingRL Controlled Discard-Only Candidate Rerun

seed base: `20260425`
configs: `8`
iterations/config: `5`
rollout_episodes/config/iter: `8`
eval_episodes/config: `16`

## Scope

- style_context: neutral
- action_scope: DISCARD only
- reward_spec: fixed default
- opponents: rule_prior_greedy, rulebase
- no teacher imitation, no full action RL, no style variants

## Stability Checks

- max rulebase_fallback_count: 0
- max rulebase_chosen_missing_count: 0
- max rulebase_chosen_not_found_count: 0
- max rulebase_batch_unsupported_count: 0
- fail-closed note: hard raise gates, not observed recoverable event rates
- trace counters pending: illegal_attempt_count, fallback_policy_count, forced_terminal_preempt_count, forced_terminal_missed_count, autopilot_terminal_count
- max illegal_action_rate_fail_closed: 0
- max fallback_rate_fail_closed: 0
- max forced_terminal_missed_fail_closed: 0
- caveat: 16 eval episodes is still small; use as candidate filter, not final strength evidence

## Ranked Candidates

- source_id=93 opponent=rulebase rule_kl=0.01 entropy=0.01 lr=3e-05 eval_rank_pt=0.375 mean_rank=1.875 fourth=0.1875 deal_in=0 win=0.1875 approx_kl=3.25545e-12 clip=0 rule_kl_obs=-1.62783e-08 rule_agree=1 delta_mean=9.3963e-05 delta_max=0.000214409 top1_changed=0
- source_id=57 opponent=rule_prior_greedy rule_kl=0.05 entropy=0.01 lr=3e-05 eval_rank_pt=0.34375 mean_rank=1.9375 fourth=0.1875 deal_in=0 win=0.125 approx_kl=1.95222e-12 clip=0 rule_kl_obs=1.61629e-09 rule_agree=1 delta_mean=9.80457e-05 delta_max=0.00030652 top1_changed=0
- source_id=8 opponent=rule_prior_greedy rule_kl=0 entropy=0.005 lr=0.0003 eval_rank_pt=0.34375 mean_rank=1.9375 fourth=0.1875 deal_in=0 win=0.1875 approx_kl=1.40321e-10 clip=0 rule_kl_obs=-8.59614e-09 rule_agree=1 delta_mean=0.000628929 delta_max=0.00165524 top1_changed=0
- source_id=84 opponent=rulebase rule_kl=0.01 entropy=0 lr=3e-05 eval_rank_pt=0.25 mean_rank=2 fourth=0.25 deal_in=0 win=0.375 approx_kl=3.42399e-12 clip=0 rule_kl_obs=-1.67513e-08 rule_agree=1 delta_mean=4.6311e-05 delta_max=0.00012886 top1_changed=0
- source_id=45 opponent=rule_prior_greedy rule_kl=0.02 entropy=0.01 lr=3e-05 eval_rank_pt=0.15625 mean_rank=2.0625 fourth=0.3125 deal_in=0 win=0.3125 approx_kl=5.58143e-12 clip=0 rule_kl_obs=-9.35228e-09 rule_agree=1 delta_mean=5.36666e-05 delta_max=0.000176421 top1_changed=0
- source_id=46 opponent=rule_prior_greedy rule_kl=0.02 entropy=0.01 lr=0.0001 eval_rank_pt=-0.03125 mean_rank=2.3125 fourth=0.375 deal_in=0 win=0.25 approx_kl=1.77069e-11 clip=0 rule_kl_obs=-2.01907e-08 rule_agree=1 delta_mean=0.000274374 delta_max=0.000539162 top1_changed=0
- source_id=105 opponent=rulebase rule_kl=0.02 entropy=0.01 lr=3e-05 eval_rank_pt=-0.21875 mean_rank=2.5625 fourth=0.4375 deal_in=0 win=0.0625 approx_kl=1.1505e-11 clip=0 rule_kl_obs=-1.41438e-08 rule_agree=1 delta_mean=0.000235138 delta_max=0.000412679 top1_changed=0
- source_id=64 opponent=rulebase rule_kl=0 entropy=0.001 lr=0.0001 eval_rank_pt=-0.21875 mean_rank=2.5625 fourth=0.4375 deal_in=0 win=0.3125 approx_kl=3.67075e-11 clip=0 rule_kl_obs=3.04694e-09 rule_agree=1 delta_mean=0.000455741 delta_max=0.00112392 top1_changed=0


## Rerun Config Mapping

- rerun_config_id=0 source_id=105 config_key=rulebase/rule_kl=0.02/entropy=0.01/lr=3e-05
- rerun_config_id=1 source_id=45 config_key=rule_prior_greedy/rule_kl=0.02/entropy=0.01/lr=3e-05
- rerun_config_id=2 source_id=46 config_key=rule_prior_greedy/rule_kl=0.02/entropy=0.01/lr=0.0001
- rerun_config_id=3 source_id=57 config_key=rule_prior_greedy/rule_kl=0.05/entropy=0.01/lr=3e-05
- rerun_config_id=4 source_id=64 config_key=rulebase/rule_kl=0/entropy=0.001/lr=0.0001
- rerun_config_id=5 source_id=84 config_key=rulebase/rule_kl=0.01/entropy=0/lr=3e-05
- rerun_config_id=6 source_id=8 config_key=rule_prior_greedy/rule_kl=0/entropy=0.005/lr=0.0003
- rerun_config_id=7 source_id=93 config_key=rulebase/rule_kl=0.01/entropy=0.01/lr=3e-05
## Artifacts

- `sweep.json`
- `summary.csv`
- `iterations.csv`
- `configs/*/` per-config raw reports
