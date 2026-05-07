# KeqingRL Mortal Action-Q Handoff

Updated: 2026-05-04

> 2026-05-06 update: this document is no longer the freshest operational
> handoff. Read
> `docs/keqingrl/keqingrl_riichienv_mortal_imitation_2026_05_06.md` first for
> the current RiichiEnv-native Mortal replay route, rule-free defaults,
> all-seat correction, GUI review workflow, and CHI kuikae legality fix.
>
> 2026-05-07 update: the active large-data route is now Mortal reviewed replay
> sidecars -> KeqingRL tensor shards -> fast shard train/eval. The current
> representation hypothesis is under-call caused by teacher/student information
> mismatch, not a value/rank-head problem. `keqingrl_action_feature_v2` keeps the
> old 8 action-feature columns as a prefix and appends response/call context
> such as last discard, call availability, call/pass family flags, consumed
> tiles, CHI shape, dora/aka target flags, and relative `from_who`. Old v1
> checkpoints can warm-start v2 shards by expanding the first action projection
> layer with zero-initialized new columns.

This is the current handoff for the KeqingRL model line. It records the route
from early KeqingRL development to the current Mortal Action-Q imitation
mainline, including wrong turns, fixes, unresolved limits, and the artifacts a
new agent should read first.

## Current One-Line Status

KeqingRL is now trained as a Mortal Action-Q imitation student:

```text
KeqingCore enumerates legal ActionSpec rows
-> Mortal checkpoint scores those legal rows with action-Q
-> KeqingRL policy learns Mortal's distribution over those rows
```

The active teacher is only:

```text
teacher_source=mortal-action-q
checkpoint=artifacts/mortal_training/mortal.pth
```

Do not use `xmodel`, `xmodel1`, or `keqingv4` as teacher sources. They are
frozen references/baselines only.

## 2026-05-06 RiichiEnv Native Update

The current operational route has moved from the legacy BattleManager bridge
rollout to RiichiEnv-native Mortal replay generation:

```text
RiichiEnv 4-Mortal selfplay
-> game_*.mjson + game_*.decisions.json compact sidecar
-> replay-to-KeqingRL ActionSpec batch
-> full-legal Mortal Q imitation
```

Mortal plays first and writes replay/sidecar artifacts; KeqingRL learns from the
materialized replay batch afterward. This is not online RL.

Current training defaults:

```text
rollout_source=riichienv-mortal-selfplay-replay
rollout_behavior=mortal-teacher
teacher_support=full-legal
rule_score_scale=0.0
support_policy_mode=unrestricted
delta_support_mode=all
student_logit_source=neural_delta_only
```

Rulebase is diagnostic only. Do not use Rule Rank as a meaningful target or
review signal.

Because all four Mortal seats use the same checkpoint, collecting all seats is
not strategically independent. The next short probes should prefer:

```text
learner_seats=0
episodes=8 or 16
multiple independent seeds
```

The latest RiichiEnv-native continuation checkpoint is:

```text
reports/keqingrl_mortal_action_q_imitation_riichienv_train_20260506_seedE_ep8_iter4_shared_cont_from_iter2/checkpoint_config_000/policy_iter_0002.pt
sha256=8010cc8c218b8aad6ca36ef5ac48caa4d68b83430e430d4f165c5d701235a8d6
rows=5108
mapping=5108/5108
fail_closed=0
teacher_ce=1.53985
teacher_kl=0.74140
teacher_agree=0.53681
```

This is a functional native-route candidate, not a promoted best.

The latest strict native seat-0 probe after replay conversion fixes is:

```text
reports/keqingrl_mortal_action_q_imitation_riichienv_seedF_ep8_seat0_20260506_fix_reach_kakan/checkpoint_config_000/policy_iter_0002.pt
sha256=2c38988038f290115e67790b3a53c6ceb6ca8fb6ebec7577bc4f6b15ad4421c9
episodes=8
iterations=2
learner_seats=0
mapping=1345/1345
teacher_valid=1345/1345
fail_closed=0
teacher_ce=1.46836
teacher_kl=0.904697
teacher_agree=0.552416
rank_ge5=0.730697
max_rss~=1.48 GiB
```

This proves the current RiichiEnv native path can close the strict mapping
contract for that seed, but it is not promoted because full-legal/rule-free
`rank_ge5` remains too high.

Important correction: a reviewed `RON / CHI / PASS` row after a riichi
declaration discard was initially treated as a Mortal native CHI-Q limitation.
That was wrong. The actual issue was CHI legality: after eating `5p` with
`3p 4p` from hand `2p 3p 4p 5p`, the remaining `2p 5p` are both kuikae-forbidden
discard candidates, so CHI leaves no legal post-call discard and must not enter
KeqingRL controlled legal actions. `src/keqingrl/env.py` now filters CHI rows
with no legal post-call discard before teacher scoring, matching Mortal/libriichi
`forbidden_tiles` behavior.

Two more replay conversion corrections are now part of the contract:

```text
1. Merge replay sample legal actions with runtime KeqingCore legal enumeration
   so KAN options are not hidden by incomplete replay legal rows.
2. Treat reach declaration + discard as one `REACH_DISCARD(tile)` row and skip
   the post-reach discard duplicate sample.
```

Current detailed handoff:

```text
docs/keqingrl/keqingrl_riichienv_mortal_imitation_2026_05_06.md
```

## Current Best Student Checkpoint

The best known KeqingRL imitation checkpoint is:

```text
reports/keqingrl_mortal_action_q_imitation_train_20260430_source93_step20000_allseats_lr003_cont1/checkpoint_config_000/policy_iter_0004.pt
sha256: 0e2acefc502472e804afebd272a8fb0e30093434a69125b9ec09c19ac17ad1a2
```

Its final reported metrics:

```text
teacher_kl=0.401925
teacher_ce=0.809471
teacher_agreement=0.686201
mapping=5422/5422
fail_closed=0
rank_ge5=0
```

Later `lr=0.003` continuation and `lr=0.0025` probe did not beat this
checkpoint. Do not blindly continue the same learning-rate chain without a new
hypothesis.

## Read First

1. `docs/project_progress.md`
2. `docs/agent_sync.md`
3. `docs/project_overview_current.md`
4. `docs/keqingrl/keqingrl_riichienv_mortal_imitation_2026_05_06.md`
5. `docs/keqingrl/keqingrl_mortal_action_q_handoff_2026_05_04.md`
6. `docs/mortal_action_contract.md`
7. `docs/keqingrl/mortal_training_workflow.md`
8. `scripts/run_keqingrl_mortal_imitation.py`
9. `tests/test_keqingrl_mortal_imitation.py`

## Development History

### Phase 0: Frozen Supervised Assets

Earlier work revolved around supervised/offline model assets such as
`xmodel1` and `keqingv4`. Those assets remain useful as references for runtime
adapters, Rust ownership, feature ideas, review tooling, and baseline behavior.

They are no longer valid teacher sources for KeqingRL. The project needs a
teacher whose own result has been trained and audited in this repo. The active
teacher source is therefore trained Mortal, not unvalidated `xmodel` or
`keqingv4` outputs.

### Phase 1: KeqingRL Contract Hardening

Before Mortal was useful, the KeqingRL/Rust contract had to be made stable.
The critical work was:

- legal action order and `ActionSpec.canonical_key`
- KeqingCore/Rust legal enumeration as the single legal owner
- terminal/autopilot handling for forced actions
- learner/opponent PPO batch contamination fix
- `rule_score_scale` metadata and rollout contract versioning
- support-only topK wrapper
- fresh validation and movement diagnostics

This was necessary because an earlier apparent improvement was caused by
opponent-step leakage, not real policy learning. The lesson is still active:
never reinterpret an old action index by rebuilding legal actions later.

### Phase 2: PPO/Rule-Prior Diagnostics

The initial KeqingRL learner used:

```text
rulebase prior + neural delta + actor-critic PPO
```

Diagnostics showed:

- PPO was not completely dead.
- `rule_score_scale=1.0` made the rule prior margin too strong.
- `rule_score_scale=0.25` allowed movement.
- sample-level support masks and soft penalties did not solve the core issue.
- support-only topK was structurally useful but did not by itself produce a
  reliable strength path.

The conclusion shifted from "PPO cannot learn" to "the internal ordering signal
inside rule-prior topK is not strong enough."

### Phase 3: Mortal Integration

Mortal became the right teacher candidate only after the KeqingRL contract was
stable enough to expose mismatches. The source of truth is the local
`third_party/Mortal` clone aligned with upstream `Equim-chan/Mortal`.

Implemented contract:

```text
Mortal q_values[46] + Mortal mask[46] + KeqingRL legal ActionSpec rows
-> teacher scores over KeqingRL legal rows
```

KeqingCore still owns legality. Mortal only scores legal actions supplied by
KeqingRL.

Important mapping facts:

- `0..36`: discard ids, red fives collapsed by the adapter.
- `37`: reach declaration. `REACH_DISCARD(tile)` is scored as q37 plus that
  tile's Mortal discard Q, so reach candidates are tile-level distinct.
- `38..40`: chi low/mid/high.
- `41`: pon.
- `42`: kan family, currently out of training scope.
- `43`: agari, context decides `TSUMO` vs `RON`.
- `44`: ryukyoku.
- `45`: pass.

`contract_scoreboard.*` and `mortal_action_mapping_audit.*` were added so
response mask gaps are counted and replayable instead of silently becoming
"teacher weakness."

### Phase 4: Interpretation Corrections

Several earlier interpretations are now explicitly deprecated:

```text
WRONG: discard-only no-pass proves Mortal teacher is weak.
RIGHT: discard-only is only infrastructure / mask / teacher-consumption
       diagnostic.

WRONG: no ron in 32 games means the teacher has no value.
RIGHT: short outcome samples are luck-heavy. If the batch has no qualified
       opportunity coverage, it cannot judge the teacher.

WRONG: score_changed is an agari proxy.
RIGHT: score can change through riichi sticks and ryukyoku tenpai payments.

WRONG: actual hora should be the default gate.
RIGHT: actual hora is stochastic outcome. Default gates are opportunity and
       contract gates.

WRONG: paired eval must block all training.
RIGHT: once the user chose imitation, paired eval became diagnostic rather than
       the main training gate.
```

The current training objective is imitation fidelity to Mortal Action-Q, not
short-run paired strength.

### Phase 5: Dedicated Mortal Imitation

The dedicated script is:

```text
scripts/run_keqingrl_mortal_imitation.py
```

It avoids rank reward and PPO advantage as the main objective. The core loss is:

```text
teacher_scores = Mortal scores over KeqingRL legal actions
teacher_probs = softmax(teacher_scores / teacher_temperature)
policy_log_probs = log_softmax(policy logits over the same legal actions)
loss = CE(teacher_probs, policy_log_probs)
```

Default support remains `topk=3` because that path was already stable. The
script writes:

```text
imitation_summary.csv
imitation_summary.md
imitation_iterations.csv
mortal_action_mapping_audit.csv
mortal_action_mapping_examples.jsonl
checkpoint_summary.csv
checkpoint_iterations.csv
changed_decisions.csv
changed_decisions.readable.md
```

The first direct imitation run was:

```text
reports/keqingrl_mortal_action_q_imitation_train_20260429_source93_step20000_cfg3_iter8
```

Then all-seat training and learning-rate probes produced the current best
checkpoint listed above.

## Performance Work

The training bottleneck was not optimizer time. It was rollout/selfplay/env plus
Mortal observation construction from event history.

Implemented optimizations:

- Deferred Mortal runtime: rollout stores encoded Mortal obs/mask, and Mortal Q
  is evaluated later in batches.
- `--mortal-teacher-eval-batch-size`, default `256`.
- Incremental `MortalObservationBridge` cache per actor, cleared on `env.reset()`.
- Skip Mortal extras for opponent/non-learner observations.
- Post-rollout bridge materialization: rollout can store
  `mortal_teacher_events`, then batch code builds Mortal obs later.
- Candidate summary dedupe for stale duplicate checkpoint rows.
- Iteration-specific checkpoints (`policy_iter_000N.pt`) instead of overwriting
  only `policy_final.pt`.
- All-seat learner collection (`--learner-seats 0 1 2 3`) improved row
  throughput versus seat-0 only.

Observed smoke comparison:

```text
seat0-only: ~45 rows, ~2.15 sec, ~20.9 rows/sec
all seats:  ~153 rows, ~2.94 sec, ~52.1 rows/sec
```

Remaining performance limit:

```text
selfplay/env remains serial and event-heavy.
parallel episode collection is not implemented.
```

## Live Gateway Fix

A separate live-play bug was fixed in `src/gateway/riichi_dev_client.py`.

Root cause:

```text
wire dahai payload omitted tsumogiri:false
```

Some server windows interpreted the discard as current-draw tsumogiri, causing
echo mismatch such as:

```text
sent:     {"actor":0,"pai":"8m","type":"dahai"}
observed: {"actor":0,"pai":"7p","tsumogiri":true,"type":"dahai"}
```

Fix:

```text
all dahai wire payloads include tsumogiri
tile == latest self tsumo -> true
otherwise -> false
```

This is independent from the imitation training path, but it matters for live
riichi.dev use.

## Current Training Scope

Controlled self-turn actions:

```text
DISCARD
REACH_DISCARD
TSUMO
ANKAN
KAKAN
RYUKYOKU
```

Controlled response actions:

```text
PASS
RON
PON
CHI
DAIMINKAN
```

Forced/autopilot actions:

```text
TSUMO
RON
RYUKYOKU
```

Mortal id `42` is the coarse KAN family id. Missing controlled legal actions
still fail closed.

2026-05-04 implementation update: the first KAN mapping contract is now wired.
For concrete KAN rows, KeqingRL can evaluate the normal Mortal decision Q for
id42 and a second `at_kan_select=true` Mortal pass for the concrete kan tile:

```text
KAN ActionSpec score = q[id42] + q_kan_select[tile]
```

This resolves the pure id42 ambiguity in the mapping layer. A small 2026-05-04
probe passed for `ANKAN + DAIMINKAN`:

```text
reports/keqingrl_mortal_action_q_imitation_kan_scope_probe_20260504_ankan_daiminkan_from_best_ep8
mapping=666/666
fail_closed=0
```

`ANKAN` and `DAIMINKAN` entered the imitation script defaults first. A follow-up
KAKAN scope probe passed:

```text
reports/keqingrl_mortal_action_q_imitation_kakan_scope_probe_20260504_from_best_ep32
mapping=584/584
fail_closed=0
```

`KAKAN` is therefore also in the imitation script defaults.

## Unresolved Issues

- `REACH_DISCARD` now uses Mortal id `37` plus concrete discard tile Q, so
  reach-vs-discard and reach-tile choice are both represented in teacher scores.
- Call ids do not encode every red-five detail; current adapter relies on
  state/context and fails closed on malformed payloads.
- `changed_decisions.*` is useful but not yet a full hand/state/event-rich
  reviewer for every changed row. The readable replay exporter is still the
  better manual inspection path.
- All-seat imitation improves data throughput but is not proof of Mahjong
  strength.
- The current best was found with topK=3. A small `teacher_topk=4` probe is a
  reasonable next hypothesis.
- Selfplay/env is still serial. Further speed work likely needs parallel
  episode workers or deeper state/observation caching.

## Suggested Next Step

Use the current best checkpoint as the new parent and test one bounded
hypothesis, not another blind learning-rate continuation.

Recommended next probe:

```text
teacher_support=topk
teacher_topk=4
lr around 0.003
all learner seats
Mortal checkpoint artifacts/mortal_training/mortal.pth
parent checkpoint current best policy_iter_0004.pt
```

Keep the stopping rules:

```text
fail_closed > 0 -> stop and debug mapping
mapping_available_rate < 1.0 -> stop and debug contract
rank_ge5_rate > 0.05 -> stop or reduce aggressiveness
teacher_kl not improving for two stages -> stop this branch
top1_changed explodes > 0.25 -> stop this branch
```

## Verification Already Run

The implementation work for Mortal imitation and bridge optimization was
previously verified with:

```bash
uv run pytest \
  tests/test_mortal_action_mapping.py \
  tests/test_keqingrl_tempered_ratio_mortal_teacher.py \
  tests/test_mortal_observation_bridge.py \
  tests/test_keqingrl_env_contract.py \
  tests/test_keqingrl_mortal_imitation.py -q

uv run ruff check \
  scripts/run_keqingrl_mortal_imitation.py \
  src/keqingrl/mortal_teacher.py \
  src/keqingrl/mortal_observation.py \
  src/keqingrl/env.py \
  src/keqingrl/selfplay.py \
  src/keqingrl/rollout.py \
  tests/test_keqingrl_mortal_imitation.py \
  tests/test_mortal_observation_bridge.py
```

This handoff update is documentation-only; rerun the test set after any code
change.
