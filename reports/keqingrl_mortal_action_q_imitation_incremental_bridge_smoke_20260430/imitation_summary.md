# KeqingRL Mortal Action-Q Imitation

candidate_summary: `reports/keqingrl_mortal_action_q_imitation_train_20260429_source93_step20000_iter16/checkpoint_summary.csv`
teacher_source: `mortal-action-q`
teacher_checkpoint: `artifacts/mortal_training/mortal.pth`
teacher_support: `topk`
teacher_topk: `3`
episodes: `2`
iterations: `1`

## Results

- cfg=0 source=93 mapping=45/45 fail_closed=0 teacher_ce=0.750243 teacher_kl=0.483819 teacher_agree=0.75 top1_parent=0 top1_source=0.2 changed_rank=2 rank_ge5=0 checkpoint=reports/keqingrl_mortal_action_q_imitation_incremental_bridge_smoke_20260430/checkpoint_config_000/policy_final.pt
- cfg=1 source=93 mapping=41/41 fail_closed=0 teacher_ce=0.94786 teacher_kl=0.516167 teacher_agree=0.5 top1_parent=0 top1_source=0.195122 changed_rank=2 rank_ge5=0 checkpoint=reports/keqingrl_mortal_action_q_imitation_incremental_bridge_smoke_20260430/checkpoint_config_001/policy_final.pt

## Checkpoints

- `reports/keqingrl_mortal_action_q_imitation_incremental_bridge_smoke_20260430/checkpoint_config_000/policy_final.pt` sha256=`68724d4eb5cd730dcf27791fcd4bbf29590fdb94415afa35ffbb50c916da60c9`
- `reports/keqingrl_mortal_action_q_imitation_incremental_bridge_smoke_20260430/checkpoint_config_001/policy_final.pt` sha256=`e4184ffdc10949db31f93c52bb1453f43aaf871847da6bcf8f2d83cfe7ba6481`
