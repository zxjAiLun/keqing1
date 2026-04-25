# KeqingRL Controlled Discard-Only Sweep

seed: `28660425`
configs: `1`
iterations/config: `5`
rollout_episodes/config/iter: `8`
eval_episodes/config: `16`

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

- id=0 opponent=rulebase rule_kl=0.01 entropy=0 lr=3e-05 eval_rank_pt=0.25 fourth=0.25 deal_in=0 approx_kl=3.42399e-12 clip=0 rule_kl_obs=-1.67513e-08 rule_agree=1 delta_mean=4.6311e-05 delta_max=0.00012886 top1_changed=0

## Artifacts

- `sweep.json`
- `summary.csv`
- `iterations.csv`
