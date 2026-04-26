# KeqingRL Controlled Discard-Only Fair-Seed Candidate Rerun

seed base: `20260425`
configs: `4`
iterations/config: `5`
rollout_episodes/config/iter: `16`
eval_episodes/config: `64`
eval_seed_registry_id: `base=202604250000:stride=1:count=64`
eval_seed_count: `64`
eval_seed_hash: `809ce4e45eb9ebe3`

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
- eval seeds are identical across all configs
- caveat: 64 eval episodes is a stronger filter, still not final strength evidence

## Ranked Candidates

- source_id=93 train_seed_base=29560425 opponent=rulebase rule_kl=0.01 entropy=0.01 lr=3e-05 eval_rank_pt=-0.140625 mean_rank=2.46875 fourth=0.40625 deal_in=0 win=0.234375 approx_kl=2.50069e-12 clip=0 rule_kl_obs=-1.35861e-08 rule_agree=1 delta_mean=3.88442e-05 delta_max=0.000106141 top1_changed=0
- source_id=84 train_seed_base=28660425 opponent=rulebase rule_kl=0.01 entropy=0 lr=3e-05 eval_rank_pt=-0.140625 mean_rank=2.46875 fourth=0.40625 deal_in=0 win=0.234375 approx_kl=3.26981e-12 clip=0 rule_kl_obs=-1.8335e-08 rule_agree=1 delta_mean=3.37575e-05 delta_max=0.000125993 top1_changed=0
- source_id=57 train_seed_base=25960425 opponent=rule_prior_greedy rule_kl=0.05 entropy=0.01 lr=3e-05 eval_rank_pt=-0.140625 mean_rank=2.46875 fourth=0.40625 deal_in=0 win=0.234375 approx_kl=5.37863e-12 clip=0 rule_kl_obs=-1.04534e-08 rule_agree=1 delta_mean=0.000142874 delta_max=0.000371547 top1_changed=0
- source_id=8 train_seed_base=21060425 opponent=rule_prior_greedy rule_kl=0 entropy=0.005 lr=0.0003 eval_rank_pt=-0.140625 mean_rank=2.46875 fourth=0.40625 deal_in=0 win=0.234375 approx_kl=1.16441e-10 clip=0 rule_kl_obs=-9.79571e-09 rule_agree=1 delta_mean=0.0003329 delta_max=0.00108699 top1_changed=0


## Rerun Config Mapping

- rerun_config_id=0 source_id=57 config_key=rule_prior_greedy/rule_kl=0.05/entropy=0.01/lr=3e-05
- rerun_config_id=1 source_id=84 config_key=rulebase/rule_kl=0.01/entropy=0/lr=3e-05
- rerun_config_id=2 source_id=8 config_key=rule_prior_greedy/rule_kl=0/entropy=0.005/lr=0.0003
- rerun_config_id=3 source_id=93 config_key=rulebase/rule_kl=0.01/entropy=0.01/lr=3e-05
## Artifacts

- `sweep.json`
- `summary.csv`
- `iterations.csv`
- `configs/*/` per-config raw reports
