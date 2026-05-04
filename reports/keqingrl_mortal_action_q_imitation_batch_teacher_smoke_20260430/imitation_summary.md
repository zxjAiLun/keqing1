# KeqingRL Mortal Action-Q Imitation

candidate_summary: `reports/keqingrl_mortal_action_q_imitation_train_20260429_source93_step20000_iter16/checkpoint_summary.csv`
teacher_source: `mortal-action-q`
teacher_checkpoint: `artifacts/mortal_training/mortal.pth`
teacher_support: `topk`
teacher_topk: `3`
episodes: `2`
iterations: `1`

## Results

- cfg=0 source=93 mapping=45/45 fail_closed=0 teacher_ce=0.750243 teacher_kl=0.483819 teacher_agree=0.75 top1_parent=0 top1_source=0.2 changed_rank=2 rank_ge5=0 checkpoint=reports/keqingrl_mortal_action_q_imitation_batch_teacher_smoke_20260430/checkpoint_config_000/policy_final.pt
- cfg=1 source=93 mapping=41/41 fail_closed=0 teacher_ce=0.94786 teacher_kl=0.516167 teacher_agree=0.5 top1_parent=0 top1_source=0.195122 changed_rank=2 rank_ge5=0 checkpoint=reports/keqingrl_mortal_action_q_imitation_batch_teacher_smoke_20260430/checkpoint_config_001/policy_final.pt

## Checkpoints

- `reports/keqingrl_mortal_action_q_imitation_batch_teacher_smoke_20260430/checkpoint_config_000/policy_final.pt` sha256=`5c65765e491d1c6cfa56f2672fcfec5959b64c977132986d3ab558af0c81b56d`
- `reports/keqingrl_mortal_action_q_imitation_batch_teacher_smoke_20260430/checkpoint_config_001/policy_final.pt` sha256=`d8d1439481384da37d9236c64a29360493a388ae26b0ec114dc2922570edda3e`
