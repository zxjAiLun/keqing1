# KeqingRL Mortal Action-Q Imitation

candidate_summary: `reports/keqingrl_mortal_action_q_imitation_train_20260429_source93_step20000_iter16/checkpoint_summary.csv`
teacher_source: `mortal-action-q`
teacher_checkpoint: `artifacts/mortal_training/mortal.pth`
teacher_support: `topk`
teacher_topk: `3`
episodes: `2`
iterations: `1`

## Results

- cfg=0 source=93 mapping=45/45 fail_closed=0 teacher_ce=0.750243 teacher_kl=0.483819 teacher_agree=0.75 top1_parent=0 top1_source=0.2 changed_rank=2 rank_ge5=0 checkpoint=reports/keqingrl_mortal_action_q_imitation_dedup_post_rollout_smoke_20260430/checkpoint_config_000/policy_final.pt

## Checkpoints

- `reports/keqingrl_mortal_action_q_imitation_dedup_post_rollout_smoke_20260430/checkpoint_config_000/policy_final.pt` sha256=`e7d577ac92411229945469f08cd38dfd20ca8e5f93bb585b0131a9210503e567`
