# KeqingRL Step 3.12 Multiseed Teacher Compare

scope_note: `diagnostic gate candidate comparison only; not strength proof`
eval_scope: `fixed-seed smoke; learner seat 0 only when gate passes`
seed_registry: `202604340000,202604350000,202604360000,202604370000`

## Summary

- teacher=rule-prior-topk coef=0.01 qualified=1/4 train_gate=1/4 fresh_gate=1/4 half_hands=272 elapsed_seconds=280.44 top1_mean=0.228298 fresh_mean=0.198169 rank_mean=2.55327 rank_ge5_max=0
- teacher=rule-component-v1 coef=0.03 qualified=1/4 train_gate=1/4 fresh_gate=1/4 half_hands=272 elapsed_seconds=293.95 top1_mean=0.240878 fresh_mean=0.182242 rank_mean=2.564 rank_ge5_max=0

## Runs

- teacher=rule-prior-topk coef=0.01 seed=202604340000 half_hands=80 elapsed_seconds=80.45 qualified=True train_gate=True fresh_gate=True top1=0.0865801 fresh=0.0422535 rank=2.6 rank_ge5=0 teacher_prior_agree=1 t_kl=0.010963 t_clip=0.160173 u_clip=0.333333 eval_fourth=0.4375 deal_in=0.1875
- teacher=rule-prior-topk coef=0.01 seed=202604350000 half_hands=64 elapsed_seconds=66.04 qualified=False train_gate=False fresh_gate=False top1=0.409091 fresh=0.284444 rank=2.44444 rank_ge5=0 teacher_prior_agree=1 t_kl=0.0117387 t_clip=0.186364 u_clip=0.336364 eval_fourth=None deal_in=None
- teacher=rule-prior-topk coef=0.01 seed=202604360000 half_hands=64 elapsed_seconds=66.9 qualified=False train_gate=False fresh_gate=False top1=0.239382 fresh=0.33617 rank=2.53226 rank_ge5=0 teacher_prior_agree=1 t_kl=0.0121584 t_clip=0.204633 u_clip=0.409266 eval_fourth=None deal_in=None
- teacher=rule-prior-topk coef=0.01 seed=202604370000 half_hands=64 elapsed_seconds=67.05 qualified=False train_gate=False fresh_gate=False top1=0.178138 fresh=0.129808 rank=2.63636 rank_ge5=0 teacher_prior_agree=1 t_kl=0.0139933 t_clip=0.210526 u_clip=0.384615 eval_fourth=None deal_in=None
- teacher=rule-component-v1 coef=0.03 seed=202604340000 half_hands=80 elapsed_seconds=80.22 qualified=True train_gate=True fresh_gate=True top1=0.134199 fresh=0.0619048 rank=2.58065 rank_ge5=0 teacher_prior_agree=0.705628 t_kl=0.0143596 t_clip=0.233766 u_clip=0.445887 eval_fourth=0.4375 deal_in=0.125
- teacher=rule-component-v1 coef=0.03 seed=202604350000 half_hands=64 elapsed_seconds=69.99 qualified=False train_gate=False fresh_gate=False top1=0.396396 fresh=0.316742 rank=2.54545 rank_ge5=0 teacher_prior_agree=0.617117 t_kl=0.0130118 t_clip=0.216216 u_clip=0.351351 eval_fourth=None deal_in=None
- teacher=rule-component-v1 coef=0.03 seed=202604360000 half_hands=64 elapsed_seconds=71.35 qualified=False train_gate=False fresh_gate=False top1=0.252918 fresh=0.212389 rank=2.70769 rank_ge5=0 teacher_prior_agree=0.657588 t_kl=0.011785 t_clip=0.151751 u_clip=0.342412 eval_fourth=None deal_in=None
- teacher=rule-component-v1 coef=0.03 seed=202604370000 half_hands=64 elapsed_seconds=72.39 qualified=False train_gate=False fresh_gate=False top1=0.18 fresh=0.137931 rank=2.42222 rank_ge5=0 teacher_prior_agree=0.724 t_kl=0.013256 t_clip=0.236 u_clip=0.332 eval_fourth=None deal_in=None

## Artifacts

- `summary.csv`
- `summary.json`
