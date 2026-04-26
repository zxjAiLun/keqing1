# KeqingRL Step 3.8B Scale=0.25 Multi-Seed Validation

source_config_id: `93`
rule_score_scale: `0.25`
rule_score_scale_version: `keqingrl_rule_score_scale_v1`
seed_registry: `202604300000,202604310000,202604320000,202604330000`
episodes: `16`
iterations: `3`
behavior_temperature: `1.25`
eval_scope: `fixed-seed smoke; learner seat 0 only`
eval_strength_note: `sanity check only; not duplicate strength evidence`

## Config Summary

- cfg6 `cfg6_lr0.001_epochs8_clip0.2_rulekl0.001` pass=2/4 movement=2/4 stability=4/4 eval_sanity=4/4 top1_mean=0.0345528 top1_range=[0,0.0853659] t_kl_mean=0.00570843 t_clip_mean=0.0856733 u_clip_mean=0.189067 eval_fourth_mean=0.328125 deal_in_mean=0.078125
- cfg10 `cfg10_lr0.003_epochs4_clip0.2_rulekl0.001` pass=2/4 movement=3/4 stability=3/4 eval_sanity=4/4 top1_mean=0.0664606 top1_range=[0,0.168182] t_kl_mean=0.00691468 t_clip_mean=0.0968662 u_clip_mean=0.196392 eval_fourth_mean=0.328125 deal_in_mean=0.0625
- cfg15 `cfg15_lr0.003_epochs8_clip0.2_rulekl0.005` pass=2/4 movement=2/4 stability=4/4 eval_sanity=3/4 top1_mean=0.302538 top1_range=[0.141079,0.53527] t_kl_mean=0.013236 t_clip_mean=0.22921 u_clip_mean=0.325255 eval_fourth_mean=0.40625 deal_in_mean=0.125

## Seed Rows

- cfg10 seed=202604300000 pass=False top1=0 eff_margin=0.720402 non_top1_pos=77 t_kl=0.000896987 t_clip=0 u_clip=0.0152284 delta_mean=0.0812475 delta_max=0.14738 eval_fourth=0.375 deal_in=0.125
- cfg10 seed=202604310000 pass=True top1=0.168182 eff_margin=0.366977 non_top1_pos=111 t_kl=0.00524628 t_clip=0.05 u_clip=0.168182 delta_mean=0.287571 delta_max=0.93637 eval_fourth=0.375 deal_in=0.0625
- cfg10 seed=202604320000 pass=True top1=0.0244898 eff_margin=0.439234 non_top1_pos=123 t_kl=0.00307713 t_clip=0.0163265 u_clip=0.130612 delta_mean=0.193156 delta_max=0.839838 eval_fourth=0.25 deal_in=0.0625
- cfg10 seed=202604330000 pass=False top1=0.0731707 eff_margin=0.512163 non_top1_pos=132 t_kl=0.0184383 t_clip=0.321138 u_clip=0.471545 delta_mean=0.282619 delta_max=1.16259 eval_fourth=0.3125 deal_in=0
- cfg15 seed=202604300000 pass=False top1=0.288991 eff_margin=0.315598 non_top1_pos=151 t_kl=0.0111754 t_clip=0.215596 u_clip=0.37156 delta_mean=0.411418 delta_max=2.4045 eval_fourth=0.5 deal_in=0.3125
- cfg15 seed=202604310000 pass=True top1=0.141079 eff_margin=0.483732 non_top1_pos=180 t_kl=0.0123282 t_clip=0.219917 u_clip=0.307054 delta_mean=1.11262 delta_max=1.97481 eval_fourth=0.4375 deal_in=0.0625
- cfg15 seed=202604320000 pass=False top1=0.53527 eff_margin=0.187463 non_top1_pos=118 t_kl=0.0136259 t_clip=0.19917 u_clip=0.319502 delta_mean=0.433308 delta_max=1.88339 eval_fourth=0.3125 deal_in=0.0625
- cfg15 seed=202604330000 pass=True top1=0.244813 eff_margin=0.273666 non_top1_pos=124 t_kl=0.0158145 t_clip=0.282158 u_clip=0.302905 delta_mean=0.403209 delta_max=1.16726 eval_fourth=0.375 deal_in=0.0625
- cfg6 seed=202604300000 pass=False top1=0 eff_margin=0.67265 non_top1_pos=64 t_kl=0.00056687 t_clip=0 u_clip=0 delta_mean=0.13773 delta_max=0.3271 eval_fourth=0.375 deal_in=0.125
- cfg6 seed=202604310000 pass=True top1=0.0853659 eff_margin=0.399342 non_top1_pos=142 t_kl=0.00873421 t_clip=0.142276 u_clip=0.284553 delta_mean=0.359037 delta_max=1.02214 eval_fourth=0.375 deal_in=0.125
- cfg6 seed=202604320000 pass=False top1=0 eff_margin=0.539212 non_top1_pos=120 t_kl=0.00345338 t_clip=0.0378151 u_clip=0.130252 delta_mean=0.123388 delta_max=0.489102 eval_fourth=0.25 deal_in=0.0625
- cfg6 seed=202604330000 pass=True top1=0.0528455 eff_margin=0.499289 non_top1_pos=150 t_kl=0.0100793 t_clip=0.162602 u_clip=0.341463 delta_mean=0.254725 delta_max=0.992114 eval_fourth=0.3125 deal_in=0

## Interpretation

- scale=0.25 does produce top1 movement across multiple seed registries, but no tested 3-iteration config passed all four seeds.
- cfg10 is the best balance: 2/4 strict pass, 3/4 movement pass, and the failed moving seed only missed strict pass on tempered_clip=0.321138.
- cfg6 is safer but under-moves on two seed registries.
- cfg15 is too aggressive: it over-moves on two seed registries and worsened deal-in on one smoke eval.
- Do not proceed to paired seat-rotation strength eval yet; first run 3.8C-style regularization/recovery around cfg10 or increase iterations carefully with tighter recovery.

## Artifacts

- `step38b_multiseed_summary.json`
- `step38b_multiseed_summary.csv`
