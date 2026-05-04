# Mortal Training Workflow

Updated: 2026-05-04

This document records the local Mortal training workflow used in this repo.
It is the operational reference for producing a Mortal checkpoint for
inference, bot play, and replay review.

## Current Status

This file is the Mortal checkpoint/data workflow reference. For the current
KeqingRL student handoff, read:

```text
docs/keqingrl/keqingrl_mortal_action_q_handoff_2026_05_04.md
```

The current training checkpoint is:

```text
artifacts/mortal_training/mortal.pth
steps: 24800
```

The runtime/playable snapshot is copied separately to:

```text
artifacts/mortal_serving/mortal.pth
```

Fixed snapshots also exist:

```text
artifacts/mortal_training/mortal_step20000.pth
artifacts/mortal_training/mortal_step24800.pth
```

The current mainline dataset is the no-ds3 packaged dataset:

```text
artifacts/mortal_mjai_gz/train/**/*.json.gz
```

The first packaged dataset included `ds3`. Later inspection showed that
`artifacts/converted_mjai/ds3` is the union of these five source directories:

```text
closehand&atk
closehand&def
superaggresive
superavoid4
superopenhand
```

So the first dataset over-weighted that group. The generated mainline Mortal
dataset now excludes `ds3`; the raw source directory is still kept.

## Source Data

Raw converted mjai source logs live under:

```text
artifacts/converted_mjai/**/*.mjson
```

Do not delete `artifacts/converted_mjai/ds3`. It is kept as source data.
When preparing Mortal training data, exclude `ds3` during packaging.

The corrected no-ds3 packaged dataset is:

```text
artifacts/mortal_mjai_gz
```

The corrected no-ds3 training directory is:

```text
artifacts/mortal_training
```

Verified counts:

```text
source_count: 72738
train_count: 69101
val_count: 3637
ds3 entries in new manifest: 0
converted_mjai/ds3 source files still present: 33027
```

After excluding `ds3`, the remaining train set has only minor overlap:

```text
train file entries: 69101
train unique log ids: 68700
duplicate file entries: 401
```

## Prepare Dataset

Use the project-side packaging script. It gzips `.mjson` logs into Mortal's
expected `.json.gz` format and writes a matching config.

```bash
uv run python scripts/prepare_mortal_training.py \
  --source-dir artifacts/converted_mjai \
  --output-dir artifacts/mortal_mjai_gz \
  --training-dir artifacts/mortal_training \
  --exclude-dir ds3 \
  --seed 20260428 \
  --val-ratio 0.05 \
  --device cuda:0
```

The important flag is:

```text
--exclude-dir ds3
```

This excludes `ds3` from the packaged training data without touching the
source directory.

If reusing an existing training directory after changing data globs, remove or
rebuild cached indexes before training:

```bash
rm -f artifacts/mortal_training/file_index.pth
rm -f artifacts/mortal_training/grp_file_index.pth
```

The DQN finite runner will rebuild `file_index.pth` automatically.

## Current Mainline Checkpoints

The current no-ds3 checkpoint is in the mainline training directory:

```text
artifacts/mortal_training/mortal.pth
artifacts/mortal_training/mortal_step20000.pth
artifacts/mortal_training/mortal_step24800.pth
```

Mortal DQN training also needs `grp.pth` at runtime. `grp.pth` is not used for
inference, but the Mortal dataloader loads it to build reward targets while
training:

```text
artifacts/mortal_training/grp.pth
```

The no-ds3 GRP checkpoint should be trained on the no-ds3 gz dataset, not copied
from the old mixed-data run. Keep the copied fallback only as a backup:

```text
artifacts/mortal_training/grp_legacy_copied_from_mortal_training_step2000.pth
```

Train the no-ds3 GRP checkpoint with the finite project runner:

```bash
rm -f artifacts/mortal_training/grp.pth

uv run python scripts/run_mortal_grp_offline.py \
  --config artifacts/mortal_training/config.toml \
  --target-steps 10000 \
  --num-workers 0 \
  --val-steps 100 \
  --log-every 50
```

After training, keep a fixed snapshot:

```bash
cp -a artifacts/mortal_training/grp.pth \
  artifacts/mortal_training/grp_no_ds3_step10000.pth
```

The current no-ds3 GRP checkpoint is `steps=10000` and has SHA-256 prefix
`ded2ba91c2ee2da7`. The previous copied checkpoint had SHA-256 prefix
`eba52019cb7356af`.

Verify:

```bash
uv run python - <<'PY'
import torch
from pathlib import Path

for name in ["mortal.pth", "mortal_step20000.pth", "grp.pth"]:
    state = torch.load(Path("artifacts/mortal_training") / name, map_location="cpu", weights_only=True)
    print(name, "steps=", state.get("steps"))
PY
```

Expected after the current promotion:

```text
mortal.pth steps= 24800
mortal_step20000.pth steps= 20000
grp.pth steps= 10000
```

## Train Mortal Brain + DQN

Use the finite project runner, not upstream `train.py`, for controlled chunks:

```bash
uv run python scripts/run_mortal_dqn_offline.py \
  --config artifacts/mortal_training/config.toml \
  --target-steps 100000 \
  --num-workers 0 \
  --log-every 50
```

Console logs include windowed metrics every `--log-every` trained steps.
For DQN this includes `loss_total`, `dqn_loss`, `cql_loss`,
`next_rank_loss`, `next_rank_acc`, `q_mean`, `target_mean`, `q_abs_err`,
and `lr`. For GRP this includes train `loss`, train `acc`, `lr`, and a
validation `loss/acc` summary after validation.

The runner resumes from `state_file` in the config:

```text
artifacts/mortal_training/mortal.pth
```

It saves the same Mortal-native checkpoint keys needed for inference:

```text
mortal
current_dqn
aux_net
optimizer
scheduler
scaler
steps
config
```

The runner saves periodically using `control.save_every` from the config.
Current value:

```text
save_every = 400
```

After a target finishes, preserve a fixed snapshot:

```bash
cp artifacts/mortal_training/mortal.pth \
   artifacts/mortal_training/mortal_step100000.pth
```

## Checkpoint Verification

Use this after every training chunk:

```bash
uv run python - <<'PY'
import torch
from pathlib import Path

for name in ["mortal.pth", "mortal_step100000.pth"]:
    path = Path("artifacts/mortal_training") / name
    state = torch.load(path, map_location="cpu", weights_only=True)
    keys_ok = all(key in state for key in ("mortal", "current_dqn", "aux_net", "config"))
    print(name, "steps=", state.get("steps"), "keys_ok=", keys_ok)
PY
```

## Runtime Use

Training writes the mutable checkpoint here:

```text
artifacts/mortal_training/mortal.pth
```

Runtime/review/riichi.dev/selfplay defaults load a separate serving snapshot:

```text
artifacts/mortal_serving/mortal.pth
```

Promote a trained checkpoint to runtime only when you want the bot to use that
version:

```bash
mkdir -p artifacts/mortal_serving
cp -a artifacts/mortal_training/mortal.pth \
  artifacts/mortal_serving/mortal.pth
```

This copy step intentionally decouples live games from ongoing training. The
training loop may continue to update `artifacts/mortal_training/mortal.pth`
without changing the model already loaded by runtime entrypoints. If the local
service is already running, restart it after replacing the serving checkpoint so
the model is reloaded:

```bash
uv run python src/main.py --port 18080 local
```

## RiichiEnv Local Test

The riichi.dev gateway can run the same Agent-style path locally with
`riichienv.RiichiEnv`. This does not use WebSocket or the online server:

```bash
uv run python src/gateway/riichi_dev_client.py \
  --mode local \
  --bot-name mortal \
  --project-root . \
  --device cuda \
  --game-mode 2 \
  --seed 42
```

For Mortal local simulation, the runner keeps one Mortal-backed agent per seat
so each seat has its own `obs.new_events()` stream and native mjai bot state.
The online riichi.dev mode only sends responses to `request_action`; other
protocol events are informational because the serialized Observation already
contains the per-seat unseen event stream.

For online play, Mortal's raw MJAI response is used only to select a legal
`riichienv.Action` through `obs.select_action_from_mjai()`. The WebSocket
response is then normalized as a minimal MJAI wire event. In particular,
call-pass responses are sent as actorless `{"type":"none"}`. Discards always
include an explicit `tsumogiri` boolean. This is intentionally more explicit
than some RiichiLab request-side `possible_actions`, because omitting
`tsumogiri:false` can make a hand discard ambiguous on the WebSocket wire.
Call actions include an explicit `target`; when `riichienv.Action.to_mjai()`
omits it, the client fills it from the latest `dahai` in the current
`obs.new_events()` batch and fails closed to `none` if that target cannot be
derived.
RiichiEnv legal actions and RiichiLab `possible_actions` may have slightly
different request-side shapes, so the audit log records both the parsed
`response` and the exact `wire_payload` string that was sent.

RiichiLab observations can occasionally return an `obs.new_events()` batch that
crosses a hand boundary, for example a previous-hand tail ending with
`ryukyoku/end_kyoku` followed by the current hand's `start_kyoku`. The gateway
resets native Mortal at hand boundaries and, before feeding events into Mortal,
drops stale events before the last `start_kyoku` while preserving an initial
`start_game -> start_kyoku` pair. This prevents previous-hand discards from
being applied to a freshly reset native Mortal state.

Use `--game-mode 1` for a faster East-only smoke. If Mortal emits an illegal
MJAI response, the legality guard logs `mortal legality guard fallback` and
falls back to a legal action instead of crashing the game. If native Mortal
raises a runtime state-sync error while consuming observation events, the
gateway logs `mortal observation sync failed` and also falls back to a legal
action so the WebSocket client does not disconnect.

## Review And Bot Use

`mortal.pth` is directly usable for inference. There is no separate q-value
checkpoint to train. Runtime q-values are produced by:

```text
Brain + current_dqn
```

The main service replay path accepts:

```text
bot_type=mortal
```

Smoke-test the service path after promotion:

```bash
uv run python - <<'PY'
import json
from urllib import parse, request

events = [
    {"type": "start_game", "names": ["P0", "P1", "P2", "P3"], "kyoku_first": 0, "aka_flag": True},
    {
        "type": "start_kyoku",
        "bakaze": "E",
        "dora_marker": "1m",
        "kyoku": 1,
        "honba": 0,
        "kyotaku": 0,
        "oya": 0,
        "scores": [25000, 25000, 25000, 25000],
        "tehais": [
            ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "E", "S", "W", "N"],
            ["1p", "1p", "2p", "2p", "3p", "3p", "4p", "4p", "5p", "5p", "6p", "6p", "7p"],
            ["1s", "1s", "2s", "2s", "3s", "3s", "4s", "4s", "5s", "5s", "6s", "6s", "7s"],
            ["8m", "8m", "9m", "9m", "8p", "8p", "9p", "9p", "8s", "8s", "9s", "9s", "P"],
        ],
    },
    {"type": "tsumo", "actor": 0, "pai": "5m"},
]

form = parse.urlencode({
    "json_text": json.dumps(events, ensure_ascii=False),
    "input_type": "mjai",
    "player_id": "0",
    "bot_type": "mortal",
}).encode()

req = request.Request(
    "http://127.0.0.1:18080/api/replay",
    data=form,
    headers={"Content-Type": "application/x-www-form-urlencoded"},
)
with request.urlopen(req, timeout=30) as resp:
    body = json.loads(resp.read().decode())

print("status", resp.status)
print("bot_type", body.get("bot_type"))
print("log_len", len(body.get("log", [])))
print("mortal_meta_present", bool(body.get("log", [{}])[0].get("mortal_meta")))
PY
```

Expected:

```text
status 200
bot_type mortal
log_len 1
mortal_meta_present True
```

## Notes

- `grp.pth` is not needed for inference.
- The finite DQN runner used here does not need a separate q-value artifact.
- The no-ds3 package was created after the 20k checkpoint; the 20k checkpoint is
  not a pure no-ds3-from-zero model.
- Continue training from 20k on the no-ds3 package if the goal is a cleaner
  playable model.

## KeqingRL Teacher Probe Notes

Mortal checkpoints may be used as KeqingRL teachers only through Mortal-native
`Brain + current_dqn` runtime Q values. Do not use `xmodel`, `xmodel1`, or
`keqingv4` checkpoints/logits as teacher sources.

Current KeqingRL growth no longer treats paired eval as the default blocker.
The active path is direct Mortal Action-Q imitation:

```text
scripts/run_keqingrl_mortal_imitation.py
```

Current best student checkpoint:

```text
reports/keqingrl_mortal_action_q_imitation_train_20260430_source93_step20000_allseats_lr003_cont1/checkpoint_config_000/policy_iter_0004.pt
sha256: 0e2acefc502472e804afebd272a8fb0e30093434a69125b9ec09c19ac17ad1a2
teacher_kl=0.401925
teacher_ce=0.809471
teacher_agreement=0.686201
mapping=5422/5422
fail_closed=0
rank_ge5=0
```

Later `lr=0.003` continuation and `lr=0.0025` probe did not beat this
checkpoint. Treat older paired/movement notes below as historical diagnostics;
they explain the path into imitation but do not override the current best
checkpoint above.

Current KeqingRL teacher modes:

```text
mortal-discard-q  historical discard-only contract diagnostic
mortal-action-q   46-id Mortal scorer over KeqingRL legal ActionSpec rows
```

Discard-only probe results are not strength evidence. They do not test reach,
terminal, pass/call, kan, or push/fold decisions.

Latest correction:

```text
WRONG: no topK movement on a discard-only or terminal-poor rollout proves the teacher is weak.
RIGHT: teacher/topK conclusions are unqualified unless the batch covers the action opportunities being tested.

WRONG: score_changed means agari coverage.
RIGHT: score changes can come from riichi sticks or ryukyoku tenpai payments.

WRONG: actual hora count should gate teacher acceptance.
RIGHT: actual hora is outcome luck; default coverage gate must use legal-opportunity rows.
```

Use `--terminal-coverage-gate` to qualify by opportunity coverage:

```text
terminal_coverage_legal_terminal_row_count
terminal_coverage_legal_agari_row_count
terminal_coverage_prepared_legal_terminal_row_count
terminal_coverage_prepared_legal_agari_row_count
```

`score_changed` and `selected_agari` stay diagnostic-only unless
`--terminal-coverage-outcome-gate` is explicitly enabled.

The Mortal action-Q pilot writes a separate contract scoreboard:

```text
contract_scoreboard.csv
contract_scoreboard.md
```

The scoreboard is the primary interpretation surface. It distinguishes:

```text
deprecated_discard_diagnostic
primary_action_q_teacher
diagnostic_control
no_teacher_control
```

It also records `mortal_action_q_mapping_available_rate`,
`mortal_action_q_fail_closed_count`, opportunity gate status, and diagnostic
outcome counters. Outcome counters do not gate by default.

Current response-stage probes control `PASS/RON/PON/CHI` and keep KAN family
out of learner scope. Use `--no-mortal-teacher-strict-extra-mask` for this
stage so extra Mortal id `42` is recorded as diagnostic-only. Missing controlled
legal actions, unsupported mappings, and ambiguous KAN mappings still fail
closed.

Latest 2026-04-29 result:

```text
contract probe: reports/keqingrl_mortal_action_q_contract_probe_20260429_source93_step20000_bridge_fix_ckpt
paired eval:    reports/keqingrl_mortal_action_q_paired_eval_20260429_source93_step20000_bridge_fix
contract:       mapping_available_rate=1.0, fail_closed=0, opportunity_gate=true
paired proxy:   delta_vs_zero=0, top1_changed=0
```

This validates the Mortal action-Q contract path for the staged response scope.
It does not validate strength improvement.

Latest movement-bearing 2026-04-29 candidate:

```text
calibration: reports/keqingrl_mortal_action_q_clip_margin_calibration_20260429_source93_step20000
candidate:   reports/keqingrl_mortal_action_q_candidate_cfg3_fresh_seedshift_20260429_source93_step20000
paired diag: reports/keqingrl_mortal_action_q_paired_eval_20260429_source93_step20000_cfg3_seedshift_diag32
checkpoint: reports/keqingrl_mortal_action_q_candidate_cfg3_fresh_seedshift_20260429_source93_step20000/checkpoint_config_000/policy_final.pt
```

Selected config:

```text
rule_score_scale=0.25
lr=0.0085
update_epochs=3
clip_eps=0.2
teacher_source=mortal-action-q
teacher_ce_coef=10
support_policy=support-only-topk
quality_max_prior_margin_p50=3.5
```

Observed gates:

```text
train top1_changed=0.0314
fresh top1_changed=0.0293
approx_kl=0.0134
clip_fraction=0.205
mortal_action_q_fail_closed_count=0
qualified_for_eval=True
```

Paired diag32 showed a small positive proxy:

```text
delta_vs_zero=+0.046875
eval diagnostic top1_changed=0.0184
changed_rank_mean=2.28
rank_ge5=0
changed_margin_p50=3.0
```

This is the first Mortal action-Q checkpoint with measurable top1 movement and
fresh validation. It is still only a strength proxy; promote it only after a
larger multi-seed paired/seat-rotation validation. The relaxed
`quality_max_prior_margin_p50=3.5` is a movement-calibration choice for raw
unscaled rule-prior gaps. It does not relax Mortal/KeqingCore mask parity.

Follow-up direct imitation-heavy training:

```text
run:        reports/keqingrl_mortal_action_q_imitation_train_20260429_source93_step20000_cfg3_iter8
checkpoint: reports/keqingrl_mortal_action_q_imitation_train_20260429_source93_step20000_cfg3_iter8/checkpoint_config_000/policy_final.pt
sha256:     645e41e5730c439e2355deb934bb2a0a388a84ea776073c8415e6c40daeb3b7e
```

This run follows the user direction to learn Mortal decisions directly instead
of treating paired validation as the blocker. It continued from the calibrated
cfg3 checkpoint with:

```text
episodes=64
iterations=8
lr=0.006
update_epochs=3
clip_eps=0.2
teacher_source=mortal-action-q
teacher_ce_coef=10
rule_kl=0
entropy=0
```

Final train summary:

```text
mapping_available=1204/1204
mortal_action_q_fail_closed_count=0
top1_changed=0.0407
changed_rank_mean=2.10
rank_ge5=0
approx_kl=0.0125
clip_fraction=0.1395
```

The only mapping audit rows remained `extra_mortal` id `42`, which is expected
while KAN is out of learner scope and `--no-mortal-teacher-strict-extra-mask`
is used. Missing controlled legal actions still fail closed.

Response mask parity failures are written to:

```text
mortal_action_mapping_audit.csv
mortal_action_mapping_examples.jsonl
```

These artifacts are for debugging only. They do not relax fail-closed training
or silently fall back to another teacher.

Before reinterpreting a surprising probe, export the exact seed with:

```bash
uv run python scripts/export_keqingrl_mjai_replay.py \
  --candidate-summary artifacts/.../summary.csv \
  --source-config-ids 93 \
  --output-dir reports/.../replays \
  --episode-index 0 \
  --seed-base 202604300000 \
  --torch-seed 202604300000
```

Inspect `episode_*.readable.md` and `episode_*.decisions.csv` before claiming a
run was "all ryukyoku", "no ron", or "teacher failed".
