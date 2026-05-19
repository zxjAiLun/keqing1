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

R0 external smoke completed on the first generated input:

| Network | Report id | Local JSON | total reviewed | matches |
| --- | --- | --- | ---: | ---: |
| `3.0` | `f531a830678058bd` | `artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_smoke/reports/0001_320000_8192_a__3.0__p0.json` | 119 | 113 |
| `4.1b` | `44cbfc905f0667fd` | `artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_smoke/reports/0001_320000_8192_a__4.1b__p0.json` | 119 | 96 |

Both reports use the same target player (`player_id=0`), `engine=Mortal`, `game_length=Hanchan`, `version=1.5.10`, and `temperature=0.1`. The useful decision schema is:

```text
review.kyokus[].entries[].actual
review.kyokus[].entries[].expected
review.kyokus[].entries[].details[].action
review.kyokus[].entries[].details[].q_value
review.kyokus[].entries[].details[].prob
```

This is sufficient to start R1 parser work for final-action agreement, high-confidence disagreement, action-family composition, and teacher probability/margin extraction.

## R1 Parser Smoke

R1 parses archived reviewer reports into decision-level and aligned teacher tables:

```bash
PYTHONPATH=src uv run python scripts/mortal/parse_reviewer_teacher_reports.py \
  --report-manifest artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_smoke/report_manifest.jsonl \
  --output-dir artifacts/experiments/reviewer_teacher_probe_2026_05/R1_parser_smoke \
  --top-k 20
```

Outputs:

```text
artifacts/experiments/reviewer_teacher_probe_2026_05/R1_parser_smoke/
  decision_table.jsonl
  decision_table.csv
  aligned_decisions.jsonl
  aligned_decisions.csv
  top_disagreements.jsonl
  summary.json
```

Smoke result on the first archived input:

| Metric | `3.0` | `4.1b` |
| --- | ---: | ---: |
| decisions | 119 | 119 |
| matches | 113 | 96 |
| match rate | 94.96% | 80.67% |
| high-confidence disagreements | 6 | 12 |
| mean actual-action teacher prob | 0.9448 | 0.8019 |

Aligned teacher summary:

| Pattern | Count |
| --- | ---: |
| actual matches both teachers | 94 |
| actual matches `3.0` only | 19 |
| actual matches `4.1b` only | 2 |
| actual matches neither | 4 |
| teacher agreement | 97 / 119 = 81.51% |

The first-report weak signal is consistent with `model_v4` looking closer to reviewer `3.0` than reviewer `4.1b`, but this remains a one-hanchan smoke result. R1 should next expand to the remaining prepared R0 inputs before drawing stable conclusions.
