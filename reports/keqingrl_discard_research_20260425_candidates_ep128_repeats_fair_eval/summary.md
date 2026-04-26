# KeqingRL Controlled Discard-Only 128-Episode Repeat Fair Eval

seed base: `20260425`
configs: `2`
repeat_count/config: `3`
iterations/config/repeat: `5`
rollout_episodes/config/iter: `16`
eval_episodes/config/repeat: `128`
eval_seed_registry_id: `base=202604250000:stride=1:count=128`
eval_seed_count: `128`
eval_seed_hash: `5e7ff5dac19cd9f8`

## Scope

- style_context: neutral
- action_scope: DISCARD only
- reward_spec: fixed default
- opponents: rulebase
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
- eval seeds are identical across all configs and repeats
- training seeds vary by repeat_id

## Aggregate

- source_id=84 opponent=rulebase rule_kl=0.01 entropy=0 lr=3e-05 rank_pt_mean=-0.0664062 rank_pt_stdev=0 fourth_mean=0.375 fourth_stdev=0 delta_max_mean=0.00010629 delta_max_max=0.000125993
- source_id=93 opponent=rulebase rule_kl=0.01 entropy=0.01 lr=3e-05 rank_pt_mean=-0.0664062 rank_pt_stdev=0 fourth_mean=0.375 fourth_stdev=0 delta_max_mean=0.000147973 delta_max_max=0.000196741

## Per Repeat

- repeat=0 source_id=84 train_seed_base=28660425 eval_rank_pt=-0.0664062 mean_rank=2.38281 fourth=0.375 win=0.203125 delta_max=0.000125993
- repeat=0 source_id=93 train_seed_base=29560425 eval_rank_pt=-0.0664062 mean_rank=2.38281 fourth=0.375 win=0.203125 delta_max=0.000106141
- repeat=1 source_id=84 train_seed_base=28670425 eval_rank_pt=-0.0664062 mean_rank=2.38281 fourth=0.375 win=0.203125 delta_max=9.33861e-05
- repeat=1 source_id=93 train_seed_base=29570425 eval_rank_pt=-0.0664062 mean_rank=2.38281 fourth=0.375 win=0.203125 delta_max=0.000196741
- repeat=2 source_id=84 train_seed_base=28680425 eval_rank_pt=-0.0664062 mean_rank=2.38281 fourth=0.375 win=0.203125 delta_max=9.94894e-05
- repeat=2 source_id=93 train_seed_base=29580425 eval_rank_pt=-0.0664062 mean_rank=2.38281 fourth=0.375 win=0.203125 delta_max=0.000141038


## Rerun Config Mapping

- rerun_config_id=0 source_id=84 config_key=rulebase/rule_kl=0.01/entropy=0/lr=3e-05
- rerun_config_id=1 source_id=93 config_key=rulebase/rule_kl=0.01/entropy=0.01/lr=3e-05
## Artifacts

- `sweep.json`
- `aggregate.csv`
- `summary.csv`
- `iterations.csv`
- `configs/*/` per-config raw reports
