# Keqing1

Keqing1 is now organized around `KeqingRL-Lite` as the single model-growth
mainline.

- `keqingrl`: active growth mainline, rule-prior + neural-delta actor-critic
- `xmodel1`: frozen SL asset line and evaluation baseline
- `keqingv4`: frozen backup/runtime/Rust asset line
- `rulebase`: rule-prior baseline and early self-play/autopilot policy

The project no longer treats large supervised cache rebuilds, new aux heads, or
full `xmodel1` / `keqingv4` retraining as the default path to model strength.

## Environment

Install Python dependencies with:

```bash
uv sync
```

Install replay UI dependencies with:

```bash
cd src/replay_ui
npm install
```

Local note for this workstation: the current `.venv` is broken. Recreate it
before running Python tests:

```bash
rm -rf .venv
uv sync
```

## Current Mainline Flow

```text
public observation + ordered legal actions
-> Rust rulebase scoring
-> raw_rule_scores
-> centered/clipped prior_logits
-> RulePriorDeltaPolicy
-> rollout records with old_log_prob/value/rule priors
-> PPO update
-> JSONL review + fixed-seed eval
```

The policy contract is variable legal-action scoring, not fixed global 45-way
logits.

## Entry Points

Run local replay/review service:

```bash
uv run python src/main.py local --port 8000
```

Run gateway only:

```bash
uv run python src/main.py --gateway-port 11600 tenhou
```

Launch live compatibility bots:

```bash
uv run python scripts/launch_tenhou_bots.py --room L2147 --count 1 --bot rulebase --start-gateway
```

Supported bot names on active runtime surfaces are still:

- `rulebase`
- `xmodel1`
- `keqingv4`

`xmodel1` and `keqingv4` remain useful for comparison and asset extraction, but
they are not the growth mainline.

## Key Directories

- `src/keqingrl/`: active interactive policy, rollout, PPO, review contracts
- `rust/keqing_core/`: Rust semantic core and rulebase scorer
- `src/mahjong_env/`: shared public Mahjong semantics
- `src/inference/`: compatibility bot/runtime surfaces
- `src/xmodel1/`: frozen SL asset line
- `src/keqingv4/`: frozen backup/runtime asset line
- `docs/`: design docs, status boards, execution notes
- `plans/`: multi-session implementation plans

## Verification

Focused checks:

```bash
uv run pytest tests/test_keqingrl_actions.py tests/test_keqingrl_distribution.py -q
uv run pytest tests/test_keqingrl_rule_score.py tests/test_keqingrl_policy_lite.py -q
uv run pytest tests/test_keqingrl_env_contract.py tests/test_keqingrl_selfplay.py -q
uv run pytest tests/test_keqingrl_ppo_toy.py tests/test_keqingrl_training.py tests/test_keqingrl_review.py -q
cargo test --manifest-path rust/keqing_core/Cargo.toml rulebase
```
