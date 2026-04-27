# Mortal Training Runbook

Updated: 2026-04-28

This runbook follows the local `third_party/Mortal` Git repository. Do not use
`xmodel`, `xmodel1`, or `keqingv4` checkpoints, logits, scores, or rollout
outputs as teacher sources.

## Repository Contract

Local Mortal is aligned to upstream `Equim-chan/Mortal` main at:

```text
0cff2b52982be5b1163aa9a62fb01f03ce91e0d2
```

The Python extension contract is:

```text
cargo build -p libriichi --lib --release
cp target/release/libriichi.so mortal/libriichi.so
python -c "import libriichi"
```

The rebuilt local module imports as `libriichi`; `third_party/Mortal` is clean.

## Local Environment Status

Current project venv has Torch, but is missing Mortal's Python-side utilities:

```text
torch: ok
toml: missing
tqdm: missing
tensorboard: missing
```

Before running Mortal training in this venv, install at least:

```bash
uv pip install toml tqdm tensorboard
```

Mortal's docs recommend using its own conda env plus a separately installed
PyTorch. Either path is fine as long as `mortal/libriichi.so` is importable from
`third_party/Mortal/mortal`.

## Data Contract

Mortal training scripts expect gzip-compressed mjai NDJSON files:

```text
*.json.gz
```

The local converted data currently is uncompressed:

```text
artifacts/converted_mjai/**/*.mjson
count: 105765
size: 3.9G
```

Passing `.mjson` directly to Mortal fails with `invalid gzip header`. A gzipped
sample was verified through Mortal's loaders:

```text
GameplayLoader(version=4): games=4, obs=(1012, 34), mask_len=46
Grp.load_gz_log_files: grp_feature_shape=(9, 7)
```

Prepare a Mortal dataset directory by gzipping without changing the source data:

```bash
uv run python scripts/prepare_mortal_training.py \
  --source-dir artifacts/converted_mjai \
  --output-dir artifacts/mortal_mjai_gz \
  --training-dir artifacts/mortal_training \
  --val-ratio 0.05 \
  --device cuda:0
```

The script writes:

```text
artifacts/mortal_mjai_gz/train/**/*.json.gz
artifacts/mortal_mjai_gz/val/**/*.json.gz
artifacts/mortal_training/config.toml
artifacts/mortal_training/manifest.json
```

## Training Order

Mortal has two offline stages.

### 1. Train GRP

`mortal/train_grp.py` trains `GRP`, which is later used by
`RewardCalculator` to build DQN rewards.

Relevant config sections:

```toml
[grp]
state_file = '/abs/path/artifacts/mortal_training/grp.pth'

[grp.control]
device = 'cuda:0'
tensorboard_dir = '/abs/path/artifacts/mortal_training/tb_grp'
batch_size = 512
save_every = 2000
val_steps = 400

[grp.dataset]
train_globs = ['/abs/path/artifacts/mortal_mjai_gz/train/**/*.json.gz']
val_globs = ['/abs/path/artifacts/mortal_mjai_gz/val/**/*.json.gz']
file_index = '/abs/path/artifacts/mortal_training/grp_file_index.pth'
file_batch_size = 50
```

Command:

```bash
cd third_party/Mortal/mortal
MORTAL_CFG=/abs/path/artifacts/mortal_training/config.toml python train_grp.py
```

The upstream script loops forever. The project-side finite runner is preferred
for reproducible chunks:

```bash
uv run python scripts/run_mortal_grp_offline.py \
  --config artifacts/mortal_training/config.toml \
  --target-steps 2000 \
  --val-steps 50 \
  --num-workers 0
```

### 2. Train Mortal Brain + DQN

`mortal/train.py` trains:

```text
Brain(version=4)
DQN(version=4)
AuxNet
```

Relevant config sections:

```toml
[control]
version = 4
online = false
state_file = '/abs/path/artifacts/mortal_training/mortal.pth'
best_state_file = '/abs/path/artifacts/mortal_training/mortal_best.pth'
tensorboard_dir = '/abs/path/artifacts/mortal_training/tb_mortal'
device = 'cuda:0'
batch_size = 512
save_every = 400
test_every = 20000

[dataset]
globs = ['/abs/path/artifacts/mortal_mjai_gz/train/**/*.json.gz']
file_index = '/abs/path/artifacts/mortal_training/file_index.pth'
file_batch_size = 15
num_workers = 1
player_names_files = []
num_epochs = 1
enable_augmentation = false

[grp]
state_file = '/abs/path/artifacts/mortal_training/grp.pth'
```

Command:

```bash
cd third_party/Mortal/mortal
MORTAL_CFG=/abs/path/artifacts/mortal_training/config.toml python train.py
```

For finite bootstrap chunks, use the project-side runner:

```bash
uv run python scripts/run_mortal_dqn_offline.py \
  --config artifacts/mortal_training/config.toml \
  --target-steps 400 \
  --num-workers 0
```

It saves the same checkpoint keys needed by the KeqingRL teacher runtime:

```text
state['mortal']
state['current_dqn']
state['aux_net']
state['config']
```

## Current Blocking Point

Upstream `train.py` constructs `TestPlayer()` immediately, before the first
offline batch. `TestPlayer` loads:

```toml
[baseline.test]
state_file = '/path/to/baseline.pth'
```

So first-from-scratch offline training currently needs either:

- an existing trained Mortal-format baseline checkpoint for `baseline.test`, or
- a small local patch that instantiates `TestPlayer` lazily only when evaluation
  actually runs.

For this project, the baseline checkpoint must also be Mortal-format. Do not use
`xmodel`, `xmodel1`, or `keqingv4` as baseline or teacher artifacts. A lazy-eval
patch is the clean next step if no Mortal baseline checkpoint is available.

The project-side wrapper avoids editing `third_party/Mortal` by monkeypatching
`player.TestPlayer` into a lazy proxy before importing `mortal/train.py`:

```bash
uv run python scripts/run_mortal_train_offline.py \
  --config artifacts/mortal_training/config.toml
```

This lets offline training start without loading `baseline.test.state_file`
until the first configured `test_every` evaluation. Evaluation still needs a
Mortal-format baseline checkpoint once it actually runs.

## Output Needed By KeqingRL

After training, the checkpoint should contain:

```text
state['mortal']       -> Brain state_dict
state['current_dqn']  -> DQN state_dict
state['config']       -> Mortal config
```

KeqingRL should load those into Mortal `Brain + DQN`, call
`MortalEngine.react_batch(obs, masks, invisible_obs)`, and attach the returned
tensors to `PolicyInput.obs.extras`:

```text
mortal_q_values     shape [B, 46]
mortal_action_mask  shape [B, 46]
```

The current KeqingRL integration consumes only discard actions from these
tensors through `teacher_source=mortal-discard-q`.

The checkpoint runtime wrapper now lives at:

```text
src/keqingrl/mortal_runtime.py
```

It exposes `load_mortal_teacher_runtime(checkpoint_path, mortal_root=...)` and
returns a runtime whose `evaluate(obs, masks)` method produces the standard
extras keys above.

The live observation bridge now lives at:

```text
src/keqingrl/mortal_observation.py
```

It replays KeqingRL mjai events into Mortal `PlayerState.encode_obs(version=4)`
and returns Mortal-format `obs` plus the 46-wide action mask. `DiscardOnlyMahjongEnv`
accepts `mortal_teacher_runtime` and `mortal_observation_bridge`; on discard-only
learner turns it attaches `mortal_q_values` and `mortal_action_mask` to
`PolicyInput.obs.extras`. The current integration has tests for direct
observation parity and for extras surviving selfplay collection plus PPO batch
collation.

## KeqingRL Pilot Probe

Once a trained Mortal checkpoint exists, run the existing tempered-ratio pilot
with Mortal as the only strength teacher:

```bash
uv run python scripts/run_keqingrl_tempered_ratio_pilot.py \
  --candidate-summary artifacts/.../summary.csv \
  --source-config-ids 93 \
  --output-dir artifacts/.../mortal_discard_q_probe \
  --topk-ranking-aux-mode teacher-ce \
  --topk-ranking-aux-coef 0.05 \
  --topk-ranking-k 3 \
  --teacher-source mortal-discard-q \
  --teacher-temperature 1.0 \
  --mortal-teacher-checkpoint artifacts/mortal_training/mortal.pth \
  --mortal-root third_party/Mortal \
  --support-policy-mode support-only-topk \
  --delta-support-topk 3 \
  --rule-score-scales 0.25
```

The pilot fails closed if `teacher_source=mortal-discard-q` is selected without
`--mortal-teacher-checkpoint`. Non-Mortal diagnostic controls do not load a
Mortal runtime.

## Local Artifact Status

As of 2026-04-28, the local workspace has completed the first bootstrap pass:

```text
artifacts/mortal_mjai_gz/**/*.json.gz      105765 files, 745M
artifacts/mortal_training/grp.pth          steps=2000
artifacts/mortal_training/mortal_step400.pth steps=400
artifacts/mortal_training/mortal.pth       steps=2000
```

The current `mortal.pth` is a trained Mortal-format teacher artifact and is
useful for integration probes. It is not yet validated as a strength teacher;
only report it as a candidate after the KeqingRL train/fresh/movement/paired
gates pass. The previous step-400 checkpoint is preserved as
`mortal_step400.pth` for reproducibility.

Runtime smoke already passed:

```text
load_mortal_teacher_runtime(artifacts/mortal_training/mortal.pth)
env.observe(actor).obs.extras['mortal_q_values']     -> shape [1, 46]
env.observe(actor).obs.extras['mortal_action_mask']  -> shape [1, 46]
mortal-discard-q topK teacher context                -> OK
```

Mortal-only KeqingRL probes have been run under:

```text
reports/keqingrl_mortal_discard_q_smoke_20260428_source93_step400
reports/keqingrl_mortal_discard_q_screen_20260428_source93_step400
reports/keqingrl_mortal_discard_q_aggressive_movement_20260428_source93_step400
reports/keqingrl_mortal_discard_q_moderate_grid_20260428_source93_step400
reports/keqingrl_mortal_discard_q_narrow_grid_20260428_source93_step400
reports/keqingrl_mortal_discard_q_narrow_grid_20260428_source93_step2000
reports/keqingrl_mortal_discard_q_penalty_grid_20260428_source93_step2000
```

Observed status:

```text
integration smoke: passed
Mortal q/mask extras: passed
mortal-discard-q teacher CE: passed
moderate movement: possible
train/fresh/movement gate: not passed yet
```

The useful diagnostic facts are:

- Conservative settings (`coef<=1`, `teacher_temperature=1.0`) preserve low
  KL/clip but do not move top1.
- Aggressive settings (`coef=10`, `teacher_temperature=0.1`, high lr/epochs)
  prove the teacher can move topK ordering, but they overmove.
- The best narrow step-400 probe found `top1_changed=0.0556`, `t_kl=0.0123`,
  `t_clip=0.125`, but movement quality failed because changes were concentrated
  on high prior-margin decisions (`changed_prior_margin_p50 ~= 3.0`).
- Step-2000 did not remove this gate issue.
- Weak-margin flip penalty suppresses movement when strong enough and does not
  yet produce a clean pass.

Do not treat the current Mortal checkpoint as validated strength. The next
productive branch is either longer/better Mortal training or a teacher-consumer
gate that only applies Mortal CE on rows where Mortal disagreement is plausible
without flipping strong rule-prior margins.
