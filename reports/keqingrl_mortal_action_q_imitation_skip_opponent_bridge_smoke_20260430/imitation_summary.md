# KeqingRL Mortal Action-Q Imitation

candidate_summary: `reports/keqingrl_mortal_action_q_imitation_train_20260429_source93_step20000_iter16/checkpoint_summary.csv`
teacher_source: `mortal-action-q`
teacher_checkpoint: `artifacts/mortal_training/mortal.pth`
teacher_support: `topk`
teacher_topk: `3`
episodes: `2`
iterations: `1`

## Results

- cfg=0 source=93 mapping=45/45 fail_closed=0 teacher_ce=0.750243 teacher_kl=0.483819 teacher_agree=0.75 top1_parent=0 top1_source=0.2 changed_rank=2 rank_ge5=0 checkpoint=reports/keqingrl_mortal_action_q_imitation_skip_opponent_bridge_smoke_20260430/checkpoint_config_000/policy_final.pt
- cfg=1 source=93 mapping=41/41 fail_closed=0 teacher_ce=0.94786 teacher_kl=0.516167 teacher_agree=0.5 top1_parent=0 top1_source=0.195122 changed_rank=2 rank_ge5=0 checkpoint=reports/keqingrl_mortal_action_q_imitation_skip_opponent_bridge_smoke_20260430/checkpoint_config_001/policy_final.pt

## Checkpoints

- `reports/keqingrl_mortal_action_q_imitation_skip_opponent_bridge_smoke_20260430/checkpoint_config_000/policy_final.pt` sha256=`49252f4c491d58e33588ef089b736a7767e133e0af37467da88795045930ad1f`
- `reports/keqingrl_mortal_action_q_imitation_skip_opponent_bridge_smoke_20260430/checkpoint_config_001/policy_final.pt` sha256=`cf3e76a4e6d627f1d7221f2ec55dc661449a69fe91726c7490703a88e209b48d`
