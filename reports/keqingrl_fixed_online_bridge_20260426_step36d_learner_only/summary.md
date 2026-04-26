# KeqingRL Fixed-Batch To Online Bridge

source_type: `fixed_batch_overfit_bridge`
candidate_summary: `reports/keqingrl_discard_research_20260425_candidates_checkpoint_rerun/summary.csv`
fixed_episodes: `16`
online_episodes: `16`
fixed_seed_registry_id: `base=202604260000:stride=1:count=16`
online_seed_registry_id: `base=202604270000:stride=1:count=16`

## Results

- source=93 rerun=0 status=FIXED_ONLY_WEAK_ONLINE_TRANSFER overfit_pass=True overfit_epochs=43 fixed_before=0 fixed_after=0.0201613 online_pre=0 online_post=0 fixed_post_online=0.0483871 online_kl=0.00521607 online_clip=0.165323 online_policy_loss=-0.0246842 online_rule_agreement=1 online_delta_mean=1.24664 online_delta_max=2.48474 stable_update=True kl_stable=True checkpoint=reports/keqingrl_fixed_online_bridge_20260426_step36d_learner_only/checkpoints/source_93_rerun_0_fixed_batch_overfit.pt post_online_checkpoint=reports/keqingrl_fixed_online_bridge_20260426_step36d_learner_only/checkpoints/source_93_rerun_0_post_online.pt

## Artifacts

- `fixed_online_bridge.json`
- `summary.csv`
- `overfit_curve.csv`
- `checkpoints/*.pt`
