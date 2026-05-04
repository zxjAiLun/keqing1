# KeqingRL Mortal Action-Q Imitation

candidate_summary: `reports/keqingrl_mortal_action_q_imitation_train_20260429_source93_step20000_cfg3_iter8/checkpoint_summary.csv`
teacher_source: `mortal-action-q`
teacher_checkpoint: `artifacts/mortal_training/mortal.pth`
teacher_support: `topk`
teacher_topk: `3`
episodes: `2`
iterations: `1`

## Results

- cfg=0 source=93 mapping=39/39 fail_closed=0 teacher_ce=0.796614 teacher_kl=0.585223 teacher_agree=0.75 top1_parent=0 top1_source=0.153846 changed_rank=0 rank_ge5=0 checkpoint=reports/keqingrl_mortal_action_q_imitation_script_smoke_20260429/checkpoint_config_000/policy_final.pt

## Checkpoints

- `reports/keqingrl_mortal_action_q_imitation_script_smoke_20260429/checkpoint_config_000/policy_final.pt` sha256=`0f529279c906704b72dc9201559de127629dc109d229ae499ad051f35b0b40a8`
