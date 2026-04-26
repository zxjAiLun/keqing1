# KeqingRL Step 3.8C Gate Registry2 Paired Seat-Rotation Strength-Proxy Eval

scope_note: `This is a stronger paired seat-rotation sanity/strength-proxy eval for diagnostic gate candidates; it is not proof of strength.`
source_candidate: `adaptive recovery gate registry2 final checkpoints`
rule_score_scale: `0.25`
rule_score_scale_version: `keqingrl_rule_score_scale_v1`
eval_seed_registry_id: `base=202604380000:stride=1:count=64`
seat_rotation: `0,1,2,3`
opponents: `rule_prior_greedy,rulebase`
run_count: `4 candidates + 2 baselines, 2 opponents, 64 eval games each = 768 eval games; diagnostics = 192 episodes; total script-counted games/episodes = 960; max_kyokus=1`
elapsed: `2969.85s`
checkpoint_generation_count: `4 checkpoint reruns * (3*16 rollout + 16 eval) = 256 script-counted games; successful parallel wall time = 168.65s`

## Strength-Proxy Readout

- Best candidate was `source_93_rerun_202604360000`: `delta_vs_zero=-0.0214844`, `rank_pt=-0.0214844`, `fourth=0.257812`, `deal_in=0.109375`.
- All 4 adaptive-gate candidates were negative vs `zero_delta_rule_prior` on paired rank_pt for both opponents.
- The paired proxy does not support claiming strength improvement from the diagnostic gate candidate.
- The diagnostic gate remains useful for controlled top1 movement/recovery, but this eval says that movement is not yet translating into positive paired seat-rotation proxy performance.

---

# KeqingRL Paired Candidate Evaluation

mode: `summary`
source_type: `checkpoint`
candidate_summary: `reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/checkpoint_summary_merged.csv`
checkpoint_path: `None`
config_path: `None`
source_config_ids: `None`
rerun_config_ids: `None`
eval_seed_registry_id: `base=202604380000:stride=1:count=64`
eval_seed_hash: `3d01161d5213f899`
eval_seed_count: `64`
diagnostic_episode_count: `4`
seat_rotation: `0,1,2,3`
policy_mode: `greedy`
training_rollout_reuse: `false`

## Required Fields

- checkpoint_path / checkpoint_sha256
- source_type / source_config_id / config_key
- eval_seed_registry_id / eval_seed_hash
- learner_deal_in_rate
- rulebase_fallback_count
- forced_terminal_preempt_count / autopilot_terminal_count
- illegal_action_rate_fail_closed / fallback_rate_fail_closed / forced_terminal_missed_fail_closed

## Candidate Results

- opponent=rule_prior_greedy candidate_id=source_93_rerun_202604360000 source_type=checkpoint source_id=93 delta_vs_zero=-0.0214844 delta_vs_untrained=-0.0214844 rank_pt=-0.0214844 mean_rank=2.52734 fourth=0.257812 learner_deal_in=0.109375 rule_agree=0.977654 top1_changed=0.0223464 delta_mean=2.14709 delta_max=2.73094 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604360000/checkpoint_config_000/policy_final.pt
- opponent=rule_prior_greedy candidate_id=source_93_rerun_202604340000 source_type=checkpoint source_id=93 delta_vs_zero=-0.119141 delta_vs_untrained=-0.119141 rank_pt=-0.119141 mean_rank=2.69922 fourth=0.269531 learner_deal_in=0.0820312 rule_agree=0.757895 top1_changed=0.242105 delta_mean=2.0015 delta_max=3.03237 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604340000/checkpoint_config_000/policy_final.pt
- opponent=rule_prior_greedy candidate_id=source_93_rerun_202604370000 source_type=checkpoint source_id=93 delta_vs_zero=-0.121094 delta_vs_untrained=-0.121094 rank_pt=-0.121094 mean_rank=2.67188 fourth=0.285156 learner_deal_in=0.121094 rule_agree=0.723618 top1_changed=0.276382 delta_mean=2.02167 delta_max=2.91596 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604370000/checkpoint_config_000/policy_final.pt
- opponent=rule_prior_greedy candidate_id=source_93_rerun_202604350000 source_type=checkpoint source_id=93 delta_vs_zero=-0.136719 delta_vs_untrained=-0.136719 rank_pt=-0.136719 mean_rank=2.70312 fourth=0.285156 learner_deal_in=0.101562 rule_agree=0.848168 top1_changed=0.151832 delta_mean=2.14358 delta_max=3.13999 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604350000/checkpoint_config_000/policy_final.pt
- opponent=rulebase candidate_id=source_93_rerun_202604360000 source_type=checkpoint source_id=93 delta_vs_zero=-0.0214844 delta_vs_untrained=-0.0214844 rank_pt=-0.0214844 mean_rank=2.52734 fourth=0.257812 learner_deal_in=0.109375 rule_agree=0.977654 top1_changed=0.0223464 delta_mean=2.14709 delta_max=2.73094 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604360000/checkpoint_config_000/policy_final.pt
- opponent=rulebase candidate_id=source_93_rerun_202604340000 source_type=checkpoint source_id=93 delta_vs_zero=-0.119141 delta_vs_untrained=-0.119141 rank_pt=-0.119141 mean_rank=2.69922 fourth=0.269531 learner_deal_in=0.0820312 rule_agree=0.757895 top1_changed=0.242105 delta_mean=2.0015 delta_max=3.03237 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604340000/checkpoint_config_000/policy_final.pt
- opponent=rulebase candidate_id=source_93_rerun_202604370000 source_type=checkpoint source_id=93 delta_vs_zero=-0.121094 delta_vs_untrained=-0.121094 rank_pt=-0.121094 mean_rank=2.67188 fourth=0.285156 learner_deal_in=0.121094 rule_agree=0.723618 top1_changed=0.276382 delta_mean=2.02167 delta_max=2.91596 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604370000/checkpoint_config_000/policy_final.pt
- opponent=rulebase candidate_id=source_93_rerun_202604350000 source_type=checkpoint source_id=93 delta_vs_zero=-0.136719 delta_vs_untrained=-0.136719 rank_pt=-0.136719 mean_rank=2.70312 fourth=0.285156 learner_deal_in=0.101562 rule_agree=0.848168 top1_changed=0.151832 delta_mean=2.14358 delta_max=3.13999 checkpoint=reports/keqingrl_step38c_gate_registry2_checkpoints_20260426_source93/gate_seed_202604350000/checkpoint_config_000/policy_final.pt

## Decision

At least one checkpoint changes top-1 or has non-trivial delta; continue paired strength/stability evaluation before changing learning signal.
