# KeqingRL Mortal Action-Q Imitation

candidate_summary: `reports/keqingrl_mortal_action_q_imitation_train_20260429_source93_step20000_iter16/checkpoint_summary.csv`
teacher_source: `mortal-action-q`
teacher_checkpoint: `artifacts/mortal_training/mortal.pth`
teacher_support: `topk`
teacher_topk: `3`
episodes: `2`
iterations: `1`

## Results

- cfg=0 source=93 mapping=153/153 fail_closed=0 teacher_ce=0.851435 teacher_kl=0.414404 teacher_agree=0.736 top1_parent=0.0130719 top1_source=0.143791 changed_rank=2 rank_ge5=0 checkpoint=reports/keqingrl_mortal_action_q_imitation_all_seats_smoke_20260430/checkpoint_config_000/policy_final.pt

## Checkpoints

- `reports/keqingrl_mortal_action_q_imitation_all_seats_smoke_20260430/checkpoint_config_000/policy_final.pt` sha256=`0ea25432d26bd06de88f2cde8b1d220cbcaacd7d99449d08e92f4ad984e7955d`
