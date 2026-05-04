# Mortal Action Contract

Updated: 2026-05-04

This document follows the local `third_party/Mortal` repository. Mortal may be
used as a teacher only through trained Mortal checkpoints. Do not use `xmodel`,
`xmodel1`, or `keqingv4` checkpoints, logits, scores, or rollouts as teacher
sources.

## 2026-04-28 Correction: Coverage Is Not Outcome

The latest Mortal teacher probes changed the interpretation of earlier
`topK` / gate results. Earlier notes that read as "Mortal teacher did not move
topK" or "discard-only teacher did not pass strength gate" are now considered
**unqualified diagnostics** unless the run also proves terminal/action-opportunity
coverage.

Deprecated interpretations:

```text
WRONG: no topK movement means Mortal teacher is weak.
RIGHT: no topK movement on a rollout without terminal/agari opportunity coverage
       is not a valid teacher-strength conclusion.

WRONG: score_changed_episode_count is a proxy for agari coverage.
RIGHT: score changes can come from riichi sticks or ryukyoku tenpai payments.
       Gate on legal terminal/agari opportunities, not score changes alone.

WRONG: gate should require actual hora / selected agari count.
RIGHT: actual hora is an outcome variable affected by wall luck and short-sample
       variance. Default qualification gates must be opportunity-based.

WRONG: discard-only teacher gate can judge Mahjong strength.
RIGHT: discard-only is an infrastructure/contract diagnostic. Strength-relevant
       teacher use must cover reach, terminal, pass/call, and later kan decisions.
```

Current gate contract:

```text
--terminal-coverage-gate
  qualifies a run by opportunity coverage only:
    terminal_coverage_legal_terminal_row_count
    terminal_coverage_legal_agari_row_count
    terminal_coverage_prepared_legal_terminal_row_count
    terminal_coverage_prepared_legal_agari_row_count

--terminal-coverage-outcome-gate
  optional, off by default. Only when enabled may score_changed / selected_agari
  thresholds affect qualification.
```

Outcome counters remain useful diagnostics:

```text
terminal_coverage_score_changed_episode_count
terminal_coverage_score_changed_without_selected_agari_episode_count
terminal_coverage_selected_agari_count
```

but they must not be used by default to accept/reject a teacher because they
measure what happened after stochastic wall progression, not whether the batch
contained the decision opportunities needed to test the contract.

When a result looks surprising, export and inspect the MJAI replay before
updating conclusions:

```text
scripts/export_keqingrl_mjai_replay.py
*.mjai.jsonl
*.readable.md
*.decisions.csv
```

## Source Files

The action contract is defined by:

```text
third_party/Mortal/libriichi/src/consts.rs
third_party/Mortal/libriichi/src/state/obs_repr.rs
third_party/Mortal/libriichi/src/agent/mortal.rs
third_party/Mortal/libriichi/src/dataset/gameplay.rs
```

`consts.rs` sets `ACTION_SPACE = 46`. `PlayerState.encode_obs(version=4,
at_kan_select=false)` returns:

```text
obs  shape = (1012, 34)
mask shape = (46,)
```

Mortal's DQN outputs one Q value per action id and masks illegal actions to
`-inf`.

## Action Ids

| Mortal id | Meaning | KeqingRL mapping |
| --- | --- | --- |
| `0..33` | discard a normal 34-tile id | `ActionType.DISCARD(tile=...)` |
| `34` | discard red `5m` | `DISCARD(tile=5m)`, red collapsed |
| `35` | discard red `5p` | `DISCARD(tile=5p)`, red collapsed |
| `36` | discard red `5s` | `DISCARD(tile=5s)`, red collapsed |
| `37` | riichi declaration | `ActionType.REACH_DISCARD`, combined with discard tile Q |
| `38` | chi low | `ActionType.CHI` where called tile is sequence low end |
| `39` | chi mid | `ActionType.CHI` where called tile is sequence middle |
| `40` | chi high | `ActionType.CHI` where called tile is sequence high end |
| `41` | pon | `ActionType.PON` |
| `42` | kan decision | `DAIMINKAN` / `ANKAN` / `KAKAN`, coarse score |
| `43` | agari | `TSUMO` or `RON`, context decides |
| `44` | ryukyoku | `ActionType.RYUKYOKU` |
| `45` | pass / none | `ActionType.PASS` |

Ids `0..36` also serve as tile-choice ids in Mortal's separate
`at_kan_select=true` observation. KeqingRL now consumes that second observation
when KAN actions are in the controlled row, so multiple legal kan choices can
be scored by concrete tile.

## Mapping Rules

KeqingRL/Rust remains the legal action owner:

```text
Rust legal enumeration -> KeqingRL ActionSpec canonical_key -> adapter lookup
```

Mortal only supplies scores:

```text
Mortal q_values[46] + mortal_mask[46] + KeqingRL legal_actions
-> teacher_scores over KeqingRL legal_actions
```

The adapter must not let Mortal generate actions or override the KeqingRL legal
set.

Discard mapping collapses red fives:

```text
5m -> Mortal ids 4 and 34
5p -> Mortal ids 13 and 35
5s -> Mortal ids 22 and 36
```

If multiple red/normal ids are masked for one KeqingRL tile, the adapter uses
the maximum masked Q value.

Reach mapping is a composite score. Mortal id `37` emits only `reach`.
KeqingRL `REACH_DISCARD` expands to:

```text
{"type": "reach", "actor": actor}
{"type": "dahai", "actor": actor, "pai": ..., "tsumogiri": ...}
```

The adapter scores each concrete KeqingRL reach-discard candidate as:

```text
REACH_DISCARD(tile) = q[37] + max(q[discard_ids_for_tile])
```

The row is available only when both Mortal's reach id and the concrete discard
tile id are legal in the Mortal mask. Red-five discard ids use the same
red/normal max rule as normal discard scoring.

For topK teacher training, every `REACH_DISCARD(tile)` candidate now has a
source key that includes id `37` and that tile's discard ids. Multiple reach
discard candidates therefore remain distinct in topK while still sharing the
same reach declaration component. This supports the intended comparison:

```text
reach declaration q37 + concrete discard q vs non-reach discard q
```

The pilot's support-only topK projection uses the same source-deduped support
when `legal_actions` are available. That keeps teacher-selected distinct-source
candidates inside the policy support gate.

Agari mapping is also contextual. Mortal id `43` represents `hora`; the current
state determines whether that means tsumo or ron.

Chi mapping is derived from `(pai, consumed)`:

```text
chi low  -> consumed = pai+1, pai+2
chi mid  -> consumed = pai-1, pai+1
chi high -> consumed = pai-2, pai-1
```

Honors, malformed consumed sets, missing `from_who`, or ambiguous call payloads
must fail closed.

Kan mapping uses id `42` for the first-stage kan decision. If multiple concrete
KeqingRL kan choices are present, the adapter also requires an
`at_kan_select=true` Q/mask pass and adds the tile-choice Q to id42. Without
that second pass, multiple kan choices still fail closed as `ambiguous_kan`.

## Mask Contract

For strict scoring, every KeqingRL legal action must map to at least one masked
Mortal action id, and every masked Mortal action id must be represented in the
KeqingRL legal set being scored.

Failures are reported instead of silently falling back:

```text
missing_legal_keys
extra_mortal_action_ids
```

This strict check is appropriate when the input legal set is the current full
KeqingRL legal set for the probe scope. If a future experiment intentionally
passes only a subset, it must explicitly opt out of extra-id strictness and
record that in the report.

Stage-1 reach probes use a scoped legal set (`DISCARD + REACH_DISCARD`) and set
`--no-mortal-teacher-strict-extra-mask`. This still fails if any controlled
KeqingRL action is missing from Mortal's mask, but it does not fail merely
because Mortal also marks an action id outside the current scoped learner slice.

Reach candidates are filtered by basic declaration preconditions before entering
the learner slice. In particular, candidates with fewer than four wall tiles
left are removed, matching Mortal's `tiles_left >= 4` riichi mask rule.

Response pass is only a Mortal action when there is a real response
opportunity. KeqingRL may record actor-scoped internal pass events such as
`{"type":"none","actor":2}` so `keqing_core` can track same-cycle furiten during
sequential response handling; the Mortal observation bridge drops those events
and keeps only the actorless all-pass `{"type":"none"}` event expected by
Mortal.

For sequential responses after a riichi discard, the bridge also trims trailing
actor-scoped `none` events before deciding whether to hide `reach_accepted`.
Mortal must see the riichi declaration discard as still open for the remaining
responders; otherwise it returns an empty response mask while KeqingCore still
offers `PON/PASS`.

Calls are not legal on the final discard when `remaining_wall == 0`. The Rust
legal owner filters `CHI` / `PON` / `DAIMINKAN` in that houtei response window
to match Mortal's mask; ron remains governed by the hora truth check and
furiten state.

## Current Implementation

The adapter lives in:

```text
src/keqingrl/mortal_teacher.py
```

Key APIs:

```text
mortal_discard_scores_for_legal_actions(...)
mortal_scores_for_legal_actions(...)
mortal_action_topk_teacher_context(...)
assert_mortal_action_mask_compatible(...)
```

`DiscardOnlyMahjongEnv` now attaches Mortal Q/mask extras for non-discard
controlled turns when the full-action mask is compatible with the current
KeqingRL legal `ActionSpec` row.

The tempered-ratio pilot exposes this as a contract/probe path:

```text
teacher_source=mortal-action-q
--self-turn-action-types DISCARD REACH_DISCARD TSUMO ANKAN KAKAN RYUKYOKU
--response-action-types PASS RON PON CHI DAIMINKAN
--no-mortal-teacher-strict-extra-mask
```

The current model-growth path is the dedicated imitation script:

```text
scripts/run_keqingrl_mortal_imitation.py
```

It trains KeqingRL to imitate Mortal action-Q over the same KeqingCore-owned
legal rows. The current best student checkpoint is:

```text
reports/keqingrl_mortal_action_q_imitation_train_20260430_source93_step20000_allseats_lr003_cont1/checkpoint_config_000/policy_iter_0004.pt
teacher_kl=0.401925
teacher_agreement=0.686201
mapping=5422/5422
fail_closed=0
```

The imitation path includes deferred Mortal runtime evaluation, batched teacher
Q evaluation, incremental MortalObservationBridge caching, optional post-rollout
bridge materialization, all-seat learner collection, and iteration-specific
checkpoints. These are performance/operational changes only; they do not
change Mortal's legal ownership role. KeqingCore still owns legal actions.

`teacher_source=mortal-discard-q` remains available for historical discard-only
diagnostics.

For the current default response-stage probe, KAN family actions stay out of
learner scope. If Mortal marks id `42` legal while KeqingRL is controlling only
`PASS/RON/PON/CHI`, the pilot records that as an out-of-scope extra Mortal id in
`mortal_action_mapping_audit.*`. It does not block this scoped contract when
`--no-mortal-teacher-strict-extra-mask` is set. Missing controlled legal actions
and unsupported mappings still fail closed.

As of 2026-05-04, KeqingRL has the first KAN-training contract wired:

```text
normal Mortal obs/mask/q      -> score id42
at_kan_select=true obs/mask/q -> score concrete kan tile id
concrete KeqingRL KAN score   -> q42 + q_kan_select[tile]
```

This lets multiple legal `DAIMINKAN` / `ANKAN` / `KAKAN` rows be distinguished
without letting Mortal own legality. If a row contains multiple KAN candidates
but the kan-select Q/mask extras are absent, mapping still fails closed as
`ambiguous_kan`.

The pilot writes a main contract scoreboard:

```text
contract_scoreboard.csv
contract_scoreboard.md
```

It also writes response-window mask diagnostics:

```text
mortal_action_mapping_audit.csv
mortal_action_mapping_examples.jsonl
```

The audit rows are not fallback teacher data. Training and teacher CE still fail
closed on mask/key mismatch; the audit only makes the mismatch countable and
replayable.

## Known Limits

The current full-action scorer is suitable for Mortal imitation over the
default controlled scope. It is not yet a full Mortal policy bridge.

Known limits:

```text
REACH_DISCARD uses q37 + concrete discard tile Q and is in the default scope.
ANKAN, DAIMINKAN, and KAKAN use first-stage id42 + at_kan_select tile-Q scoring
  and are in the imitation defaults after the 2026-05-04 scoped probes.
Call ids do not encode red-five selection; Mortal reconstructs consumed red use from state.
from_who is validated for presence but not independently inferred from Mortal mask.
Response-window PASS/RON/PON/CHI is in the current imitation scope, but still
  fails closed on missing controlled legal mappings.
```

Those limits are acceptable for the next probe order:

```text
1. DISCARD + REACH_DISCARD
2. PASS / RON / PON / CHI
3. KAN family with at_kan_select support
```
