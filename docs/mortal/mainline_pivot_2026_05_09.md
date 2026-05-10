# Mortal Encoding Mainline Pivot

Updated: 2026-05-10

## Decision

The active strength route is Mortal-native development:

```text
libriichi state / Mortal obs_repr
-> Mortal Brain encoder
-> Mortal Dueling DQN action-value head
-> optional future policy / value / rank heads on the same backbone
```

Do not start a new self-authored Mahjong observation encoding for strong-model
work. Any future fine-tuning, continued training, selfplay training, reward
changes, rank/point/GRP changes, or policy/value-head experiments should begin
from Mortal/libriichi encoding and remain compatible with Mortal checkpoints
unless a later design document explicitly changes this boundary.

The detailed sync from the two Mortal deep research reports is:

```text
docs/mortal/deep_research_sync_2026_05_10.md
```

## Rationale

The useful project-level conclusion from the KeqingRL and keqingv experiments is
that Mahjong model quality was gated less by optimizer or head choice and more
by observation quality. Repeatedly rebuilding local features, preprocessing,
cache schemas, action contracts, and materialized datasets consumed effort
without matching Mortal's already available representation of:

- visible and invisible state needed by Mortal
- seen tiles and river state
- calls, riichi, dora, red fives, and tsumogiri/tedashi context
- legal action masks
- EV/GRP-related training support
- Brain encoder and 46-action DQN value layout

The previous independent routes are therefore archived as learning assets and
tooling references, not as active strength baselines.

## Keep Active

- Mortal checkpoint continued training and fine-tuning
- Mortal reward, rank, point, and GRP experiments
- riichienv 4-Mortal selfplay replay generation
- checkpoint-vs-checkpoint evaluation and comparison
- Mortal Q/mask sidecars and replay review GUI
- action/mask audits where they protect Mortal tooling
- future policy/value heads attached to Mortal-compatible encoding

## Archive By Default

- xmodel supervised retraining
- keqingv3 / keqingv3.1 feature expansion
- KeqingRL independent policy or Action-Q imitation as the strength route
- materialize-heavy policy distillation as the main path
- legacy BattleManager selfplay
- large custom feature preprocessing pipelines

## Immediate Next Direction

The next high-leverage step is not another local observation schema. It is to
make the Mortal-native loop easier to run and evaluate:

1. Verify the local Mortal/libriichi build and checkpoint loading path.
2. Run a small Mortal selfplay replay generation smoke.
3. Prepare or refresh Mortal-format gzipped MJAI training data.
4. Run a short Mortal DQN/GRP continued-training smoke from an existing Mortal
   checkpoint.
5. Compare checkpoints through the same riichienv evaluation surface and review
   representative Q/mask sidecars in the GUI.
