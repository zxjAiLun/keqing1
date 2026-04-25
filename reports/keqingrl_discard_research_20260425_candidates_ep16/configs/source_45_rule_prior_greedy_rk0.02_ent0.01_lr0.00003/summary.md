# KeqingRL Controlled Discard-Only Sweep

seed: `24760425`
configs: `1`
iterations/config: `5`
rollout_episodes/config/iter: `8`
eval_episodes/config: `16`

## Scope

- style_context: neutral
- action_scope: DISCARD only
- reward_spec: fixed default
- opponents: rule_prior_greedy
- no teacher imitation, no full action RL, no style variants

## Stability Checks

- max illegal_action_rate: 0
- max fallback_rate: 0
- max forced_terminal_missed: 0
- caveat: this is a tiny smoke-scale matrix, not strength evidence

## Top Configs

- id=0 opponent=rule_prior_greedy rule_kl=0.02 entropy=0.01 lr=3e-05 eval_rank_pt=0.15625 fourth=0.3125 deal_in=0 approx_kl=5.58143e-12 clip=0 rule_kl_obs=-9.35228e-09 rule_agree=1 delta_mean=5.36666e-05 delta_max=0.000176421 top1_changed=0

## Artifacts

- `sweep.json`
- `summary.csv`
- `iterations.csv`
