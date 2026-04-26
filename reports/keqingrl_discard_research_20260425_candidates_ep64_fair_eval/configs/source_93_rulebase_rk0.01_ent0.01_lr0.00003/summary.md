# KeqingRL Controlled Discard-Only Sweep

seed: `29560425`
configs: `1`
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
- opponents: rulebase
- no teacher imitation, no full action RL, no style variants

## Stability Checks

- max illegal_action_rate: 0
- max fallback_rate: 0
- max forced_terminal_missed: 0
- caveat: this is a tiny smoke-scale matrix, not strength evidence

## Top Configs

- id=0 opponent=rulebase rule_kl=0.01 entropy=0.01 lr=3e-05 eval_rank_pt=-0.140625 fourth=0.40625 deal_in=0 approx_kl=2.50069e-12 clip=0 rule_kl_obs=-1.35861e-08 rule_agree=1 delta_mean=3.88442e-05 delta_max=0.000106141 top1_changed=0

## Artifacts

- `sweep.json`
- `summary.csv`
- `iterations.csv`
