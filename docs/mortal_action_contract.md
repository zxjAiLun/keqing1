# Mortal Action Contract

Updated: 2026-05-08

This document describes the active Mortal runtime/tooling boundary after the
project contraction.

## Active Boundary

Mortal is the default model/runtime owner for current work:

- observations and legal actions come from Mortal/libriichi/riichienv
- checkpoint inference uses `src/inference/mortal_bot.py`
- selfplay replay generation uses `scripts/mortal/generate_riichienv_selfplay_replays.py`
- replay Q/mask sidecars use `scripts/mortal/materialize_replay_sidecars.py`

The active workflow should not route through deleted or archived `keqingrl`,
`xmodel`, or `keqingv` model stacks.

## Action Space

Mortal DQN uses 46 action ids. Local sidecar tools store compact Q values plus a
bit mask:

```text
mask_bits: integer bitset over action ids 0..45
q_values: compact list aligned to set bits in mask_bits
```

`scripts/mortal/generate_riichienv_selfplay_replays.py` can optionally expand
that compact representation into:

```text
expanded_q_values[46]
action_mask[46]
```

## Active Outputs

Selfplay generator output:

- `replays/*.mjson`
- `replays/*.decisions.json`
- `replays/*.json`
- `replays/manifest.json`
- `summary.json`

Sidecar materializer output:

- `<replay>.decisions.json`
- `sidecar_materialize_summary.json`

The sidecar schema is:

```json
{
  "by_actor": {
    "0": [
      {
        "event_index": 0,
        "actor": 0,
        "event_type": "tsumo",
        "chosen_action": {},
        "mortal_meta": {
          "mask_bits": 0,
          "q_values": [],
          "eval_time_ns": 0
        }
      }
    ]
  }
}
```

## Archived Context

Older versions of this document described a KeqingRL Action-Q imitation adapter.
That route is archived and its implementation may be removed from the working
tree. Treat those notes only as historical audit context under the frozen
`archive-keqingrl-mortal-imitation-202605` tag.
