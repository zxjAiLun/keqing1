# Mortal Action Contract

Mortal DQN uses 46 action ids. Native selfplay and local replay review obtain observations, legal masks, and action semantics from Mortal/libriichi.

Local review stores legal-action Q values compactly:

```text
mask_bits: integer bitset over action ids 0..45
q_values: compact list aligned to set bits in mask_bits
```

The active boundary is deliberately local-only: `training/mortal/four_player_native.py`, `training/mortal/selfplay_native.py`, `training/run_mortal_dqn_offline.py`, and the replay GUI. External reviewer formats and teacher-action overlays are not part of the training contract.
