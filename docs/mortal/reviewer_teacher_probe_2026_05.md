# Mortal Reviewer Teacher Probe

## Purpose

Phase R introduces official Mortal reviewer output as a sparse black-box teacher signal.

This is not weight distillation. The reviewer only gives preferences on states that appear in a full game log, and its detail table should be treated as action preference metadata rather than a precise value oracle.

## External Constraints

The reviewer Custom log input expects `tenhou.net/6` JSON. For custom logs, target player must be specified explicitly. Mortal reviewer supports four-player standard games, and only hanchan games are supported for the Mortal engine.

The useful reviewer networks for the first probe are:

- `3.0`: closer to the local `model_v4` reference level and more human-like.
- `4.1b`: balanced stronger public reviewer teacher.

Reviewer pages are retained for 15 days, so downloaded report JSON must be archived immediately.

## R0: Reviewer Input Smoke

R0 validates the local input side:

```text
Mortal arena mjai JSONL
  -> tenhou.net/6 JSON
  -> mjai-reviewer convlog round-trip
  -> manifest rows for manual/controlled reviewer upload
```

Generated local artifact:

`artifacts/experiments/reviewer_teacher_probe_2026_05/R0_reviewer_input_smoke`

Command:

```bash
PYTHONPATH=src uv run python scripts/mortal/prepare_reviewer_teacher_probe.py \
  --logs 'artifacts/eval/gate_10000h/Gate_v4_vs_70k_2500/logs/*.json.gz' \
  --output-root artifacts/experiments/reviewer_teacher_probe_2026_05 \
  --experiment-id R0_reviewer_input_smoke \
  --limit 5 \
  --target-player-name challenger \
  --networks 3.0,4.1b \
  --validate-convlog
```

R0 output:

- `input/*.tenhou6.json`: Custom log payloads for reviewer upload.
- `roundtrip_mjai/*.mjson`: convlog validation output.
- `manifest.jsonl`: source log, target player seat, network list, validation status.
- `summary.json`: run-level metadata.

All 5 generated Tenhou6 inputs passed local convlog validation.

## Target Player Handling

The 1v3 arena logs rotate challenger seat across `_a/_b/_c/_d`.

R0 uses:

```bash
--target-player-name challenger
```

so each manifest row records the correct target player seat:

- `_a`: challenger seat `0`
- `_b`: challenger seat `1`
- `_c`: challenger seat `2`
- `_d`: challenger seat `3`

This matters because reviewer custom logs do not auto-detect target player from URL metadata.

## R1 Plan

After a small manual/controlled upload confirms report JSON shape, R1 should parse archived report JSON and compute:

- teacher final-action agreement rate
- high-confidence disagreement rate
- disagreement composition by action family
- teacher top-1 probability and top-1/top-2 margin
- local actual action probability under teacher distribution

Use Q values only as ranking/margin metadata. The first training signal should be final action / soft preference, not raw-Q regression.

## R0 External Smoke

The reviewer site requires a valid Turnstile captcha response for `/review` submissions. A direct command-line POST is rejected by the server with `invalid captcha response`, so the upload step must go through the official browser page.

Use the first R0-local input for the initial smoke:

- source Tenhou6: `artifacts/experiments/reviewer_teacher_probe_2026_05/R0_reviewer_input_smoke/input/0001_320000_8192_a.tenhou6.json`
- target player: `0`
- networks: `3.0`, `4.1b`

Browser upload settings:

```text
Game log input: Custom log (tenhou.net/6 JSON)
Target player: 0
Engine: Mortal
Mortal network: 3.0 or 4.1b
UI: KillerDucky
Language: English
```

After each report page is generated, archive the JSON immediately:

```bash
PYTHONPATH=src uv run python scripts/mortal/archive_reviewer_reports.py \
  --source-manifest artifacts/experiments/reviewer_teacher_probe_2026_05/R0_reviewer_input_smoke/manifest.jsonl \
  --source-index 0 \
  --target-player 0 \
  --network 3.0 \
  --report https://mjai.ekyu.moe/report/<REPORT_ID>

PYTHONPATH=src uv run python scripts/mortal/archive_reviewer_reports.py \
  --source-manifest artifacts/experiments/reviewer_teacher_probe_2026_05/R0_reviewer_input_smoke/manifest.jsonl \
  --source-index 0 \
  --target-player 0 \
  --network 4.1b \
  --report https://mjai.ekyu.moe/report/<REPORT_ID>
```

Archive output:

```text
artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_smoke/
  reports/
    0001_320000_8192_a__3.0__p0.json
    0001_320000_8192_a__4.1b__p0.json
  report_manifest.jsonl
```

`report_manifest.jsonl` records the source Tenhou6 path, target player, reviewer network, report id, page URL, JSON URL, local JSON path, download time, and a shallow schema summary showing whether detail/action/Q-or-score-like fields were detected.
