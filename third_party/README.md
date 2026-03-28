# Third-party dependencies

- Target dependency: `Mortal/libriichi`
- Purpose: convert raw tenhou/majsoul records to mjai jsonl.
- Current environment cannot reach GitHub, so the bridge currently falls back to built-in converter.

## Expected setup when network is available

1. Clone Mortal into `third_party/Mortal`.
2. Build libriichi CLI or locate executable path.
3. Pass binary path to tools via `--libriichi-bin`.

The bridge in `src/convert/libriichi_bridge.py` automatically prefers libriichi if the binary is callable.

