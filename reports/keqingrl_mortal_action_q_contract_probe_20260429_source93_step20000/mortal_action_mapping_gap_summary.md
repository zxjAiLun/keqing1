# Mortal Action-Q Mapping Gap Summary

## Contract Result

- Branch: `B / mask parity blocked`
- All configs stopped before training/eval because Mortal exposed an out-of-scope action id.
- This is not teacher strength evidence and not an opportunity/outcome conclusion.

## Aggregates

### Mismatch kind
- `extra_mortal`: 4

### Legal action type
- `PASS`: 4
- `PON`: 4

### Missing legal type
- none

### Extra Mortal id
- `42`: 4

## Interpretation

- KeqingRL controlled legal rows were `PON + PASS`.
- Mortal mask exposed `[41, 42, 45]`: `PON`, `KAN`, `PASS`.
- The only gap is extra Mortal id `42` (`KAN`) while the planned response scope excludes `DAIMINKAN`.
- No missing `PON` or `PASS` mapping was observed in this probe.

## Replay Examples

### Example 0 cfg=0

- actor: `1`
- legal_action_types_json: `["PON","PASS"]`
- extra_mortal_action_ids_json: `[42]`
- hand_json: `["1m","2s","3s","4p","6p","6p","7p","7s","8p","8s","8s","8s","9p"]`
- last_discard_json: `{"actor":2,"pai":"8s","pai_raw":"8s"}`
- readable: `replays/cfg0/episode_000_seed_202604300000.readable.md`
- decisions: `replays/cfg0/episode_000_seed_202604300000.decisions.csv`

### Example 1 cfg=1

- actor: `0`
- legal_action_types_json: `["PON","PASS"]`
- extra_mortal_action_ids_json: `[42]`
- hand_json: `["3s","6m","6p","7m","7p","7s","8p","8s","8s","8s","9s","F","F"]`
- last_discard_json: `{"actor":1,"pai":"8s","pai_raw":"8s"}`
- readable: `replays/cfg1/episode_000_seed_202604400000.readable.md`
- decisions: `replays/cfg1/episode_000_seed_202604400000.decisions.csv`

### Example 2 cfg=2

- actor: `3`
- legal_action_types_json: `["PON","PASS"]`
- extra_mortal_action_ids_json: `[42]`
- hand_json: `["1p","1p","3p","4p","7p","9p","E","E","E","P"]`
- last_discard_json: `{"actor":1,"pai":"E","pai_raw":"E"}`
- readable: `replays/cfg2/episode_000_seed_202604500000.readable.md`
- decisions: `replays/cfg2/episode_000_seed_202604500000.decisions.csv`

## Next Engineering Choice

Use one of these explicit paths before rerunning full contract probe:

1. Add `DAIMINKAN` to response scope for strict Mortal full-mask parity, then rerun contract probe.
2. Keep KAN out of learner scope and make the contract explicitly scope-aware for extra Mortal ids, while still failing on missing in-scope legal actions.

Path 1 is stricter and fastest to validate; Path 2 matches the staged plan that keeps KAN family later, but needs a code change to distinguish out-of-scope extra ids from true mismatch.
