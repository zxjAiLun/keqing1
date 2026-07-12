# Keqing1 Current Scope

Keqing1 is a local riichi mahjong research workspace focused on native Mortal/libriichi selfplay, offline DQN training, and local replay review.

Active local models:

- `artifacts/model_v4_20240308_best_min.pth`: external v4 reference and synthetic-data actor.
- `artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth`: 70k anchor and V2 parent.
- V2 checkpoint: created under `artifacts/experiments/model_pool_2026_07/V2_population_mixed_v4_warmstart_2026_07/checkpoints/` after training.

The replay GUI supports local v4, 70k, and the optional V2 candidate. External reviewer, NAGA, teacher-CE, and manual-audit workflows are intentionally out of scope.
