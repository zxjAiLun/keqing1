# KeqingRL Controlled Discard-Only Sweep

seed: `28660425`
configs: `1`
iterations/config: `5`
rollout_episodes/config/iter: `16`
eval_episodes/config: `128`
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

- max illegal_action_rate: 0
- max fallback_rate: 0
- max forced_terminal_missed: 0
- caveat: this is a tiny smoke-scale matrix, not strength evidence

## Top Configs

- id=0 opponent=rulebase rule_kl=0.01 entropy=0 lr=3e-05 eval_rank_pt=-0.0664062 fourth=0.375 deal_in=0 approx_kl=1.61122e-12 clip=0 rule_kl_obs=1.46907e-09 rule_agree=1 delta_mean=2.79741e-05 delta_max=9.94894e-05 top1_changed=0

## Artifacts

- `sweep.json`
- `summary.csv`
- `iterations.csv`
