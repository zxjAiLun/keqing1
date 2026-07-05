# Mortal Reviewer Teacher Probe

## Purpose

Phase R introduces official Mortal reviewer output as a sparse black-box teacher signal.

This is not weight distillation. The reviewer only gives preferences on states that appear in a full game log, and its detail table should be treated as action preference metadata rather than a precise value oracle.

## Current R-Series Conclusions

Current read as of 2026-06-09:

- R0 validated the local conversion path from arena mjai JSONL to `tenhou.net/6` JSON and back through mjai-reviewer convlog. The prepared five-input smoke set is usable, and target-player handling is correct for rotated challenger seats.
- R0 external smoke confirmed that official Mortal reviewer JSON is archiveable and parseable. The useful fields are `review.kyokus[].entries[].actual`, `expected`, and `details[].{action,q_value,prob}`.
- R1 parser smoke is sufficient for sparse preference labeling: it extracts decision tables, high-confidence disagreements, teacher top-1/top-2 margins, and aligned `3.0`/`4.1b` comparisons.
- On the first smoke hanchan, local `model_v4`-generated action choices look much closer to reviewer `3.0` than to `4.1b`: `3.0` matched 113/119 decisions, while `4.1b` matched 96/119.
- The first browser-assisted `4.1b` expansion on source index 1 parsed 118 decisions, with 102 matches, 16 mismatches, and 13 high-confidence disagreements. This is consistent with `4.1b` being a stronger correction source rather than a drop-in imitation target.
- Most high-confidence `4.1b` disagreements in the current two-report sample are discard-choice corrections, with only one call-family correction per parsed report. That makes the first reviewer correction set more about discard preference than fuuro policy.
- Direct cURL replay is not a reliable submission path because Turnstile responses expire or are single-use. Browser-assisted form fill plus immediate archive is the current working R1.5 route.
- Reviewer `4.1b` should remain a sparse black-box labeler for high-confidence disagreements. It should not be treated as a bulk selfplay generator or a raw-Q regression teacher.

Practical implication: use R-series output to build a small `4.1b` high-confidence disagreement correction/evaluation set, especially for GUI side-by-side review against local weights. Do not start a large reviewer-driven training run until more browser-assisted reports are archived and parsed across multiple source logs/seats.

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

## T-Series Teacher Transfer Direction

O-series online continuation has been closed as a local recipe search. T-series is now the active research route for training-method development, while `model_v4` remains the practical strongest local model.

The distinction matters:

- Practical model: use `model_v4` directly for play/review/strong baseline/teacher replay generation.
- Research checkpoint: use `T1@71000` to study how teacher action preference transfers strength into a 70k-derived student.

There are two different teacher sources:

- Public reviewer networks (`4.1a/b/c`, `3.0`, `4.0`) are black-box reviewers. Without their weights, they cannot be used as local selfplay generators. Use them to label preferences on existing Tenhou6 logs.
- Local checkpoints such as `model_v4` can be used as local data generators. They can produce replay logs at scale through local arenas.

### T1: model_v4 Teacher CE Transfer

T1 is the first positive teacher-guided student result:

| Item | Setting |
| --- | --- |
| Parent | `artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth` |
| Teacher source | local `artifacts/model_v4_20240308_best_min.pth` |
| Training logs | `model_v4 vs 3x70k`, filtered to `challenger` samples |
| Loss | offline DQN/CQL/Aux + teacher action CE |
| `teacher_ce_weight` | `0.1` |
| Student checkpoint | `artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth` |
| Step | `71000` |

T1 is not a replacement for `model_v4`. It proves that `model_v4` action preference can be converted into real student gate improvement.

Current T1 gate summary:

| Gate | Result |
| --- | ---: |
| `T1@71000 vs 3x70k`, final 5000h | +0.738 |
| `70k vs 3xT1@71000`, final 5000h | -1.377 |
| `T1@71000 vs 3x80k_game`, 1000h screen | -0.090 |
| `80k_game vs 3xT1@71000`, 1000h screen | +0.045 |

Interpretation:

- T1 is the current best trained student / proof-of-mechanism checkpoint.
- T1 is close to `80k_game` in the existing screen, but not clearly above it.
- `model_v4` remains the strongest available local model and should be used directly for practical play/review.

Near-term T-series priorities:

1. Freeze T1@71000 as the positive reference student.
2. Compare T1 against `80k_game` as needed; current 1000h evidence is near parity.
3. Run behavior readout for `70k`, `80k_game`, `T1@71000`, and `model_v4`. Done at `artifacts/experiments/teacher_transfer_2026_05/behavior_readout_four_model_100h/readout/`.
4. Try `T1b_teacher_ce_005` before higher teacher CE weights. Done; `T1b@71000` is at `artifacts/experiments/teacher_transfer_2026_05/T1b_teacher_ce_005/mortal.pth`.
5. Keep `T1c_teacher_ce_02` paused unless T1b/readout evidence suggests stronger imitation is needed.

Concrete variant preparation commands:

```bash
uv run python scripts/mortal/prepare_teacher_replay_transfer.py \
  --experiment-id T1b_teacher_ce_005 \
  --teacher-ce-weight 0.05 \
  --copy-parent-checkpoint

uv run python scripts/mortal/prepare_teacher_replay_transfer.py \
  --experiment-id T1c_teacher_ce_02 \
  --teacher-ce-weight 0.2 \
  --copy-parent-checkpoint
```

Run `T1b_teacher_ce_005` first. Do not start `T1c_teacher_ce_02` until T1b and the behavior readout indicate that stronger imitation is still plausible.

### T1/T1b Behavior Readout

The first four-model behavior readout used a same-table 100h arena with `70k`, `80k_game`, `T1@71000`, and `model_v4`.

Artifact: `artifacts/experiments/teacher_transfer_2026_05/behavior_readout_four_model_100h/readout/behavior_readout.md`

| Metric | `70k` | `80k_game` | `T1@71000` | `model_v4` |
| --- | ---: | ---: | ---: | ---: |
| Agari | 21.81% | 21.62% | 23.86% | 20.16% |
| Dealin | 15.77% | 13.44% | 11.78% | 12.17% |
| Fuuro rate | 31.65% | 29.89% | 27.56% | 29.70% |
| Riichi rate | 19.28% | 21.32% | 19.08% | 19.77% |
| After-fuuro agari | 32.31% | 29.32% | 37.81% | 31.15% |
| After-fuuro dealin | 17.23% | 16.61% | 13.78% | 16.07% |
| After-riichi agari | 48.48% | 51.60% | 54.08% | 45.81% |
| After-riichi dealin | 20.71% | 15.53% | 10.20% | 13.79% |
| Avg winning delta score | 6634.4 | 6909.0 | 6953.1 | 6492.8 |
| Avg open winning delta score | 4831.4 | 4907.8 | 4769.2 | 4296.8 |
| Avg call delta score | 113.8 | 126.7 | 592.9 | 193.4 |

T1 does not look like a simple aggressive imitation of `80k_game`: it calls less than all three references in this readout, has the lowest deal-in rate, and has the strongest after-fuuro / after-riichi outcome balance. Its mean absolute rate gap is almost identical to `80k_game` and `model_v4` in this 100h readout (`3.45pp` vs `3.46pp`), so the readout does not prove that T1 is stylistically closer to either one.

T1b used `teacher_ce_weight = 0.05`, trained from the same 70k parent to step 71000, then received a 100h behavior readout.

Artifacts:

- `artifacts/experiments/teacher_transfer_2026_05/T1b_teacher_ce_005/manifest.json`
- `artifacts/experiments/teacher_transfer_2026_05/behavior_readout_four_model_T1b_100h/readout/behavior_readout.md`
- `artifacts/experiments/teacher_transfer_2026_05/T1b_teacher_ce_005/gate_1000h_chunked/Gate_T1b_vs_70k/aggregated_metrics.json`

| Metric | `70k` | `80k_game` | `T1b@71000` | `model_v4` |
| --- | ---: | ---: | ---: | ---: |
| Agari | 22.67% | 19.72% | 22.86% | 23.23% |
| Dealin | 12.72% | 15.67% | 12.72% | 11.06% |
| Fuuro rate | 31.06% | 33.92% | 25.90% | 29.12% |
| Riichi rate | 18.06% | 19.72% | 22.30% | 19.08% |
| After-fuuro agari | 34.42% | 27.72% | 37.37% | 33.23% |
| After-fuuro dealin | 14.54% | 16.85% | 13.17% | 10.13% |
| After-riichi agari | 49.49% | 40.65% | 49.17% | 53.62% |
| After-riichi dealin | 15.82% | 16.36% | 16.94% | 15.46% |
| Avg winning delta score | 6424.0 | 6211.7 | 6309.7 | 6635.7 |
| Avg open winning delta score | 4325.9 | 4702.0 | 4416.2 | 4495.2 |
| Avg call delta score | 393.2 | 74.2 | 582.6 | 446.5 |

T1b did not show an obvious behavior crash in the 100h readout: call frequency stayed lower than the references, deal-in matched 70k and stayed below 80k, and after-fuuro outcomes remained good. The 1000h challenger gate against 70k was only mildly positive:

| Gate | Games | Rank counts | Tenhou avg pt |
| --- | ---: | --- | ---: |
| `T1b@71000 vs 3x70k`, chunked | 1000 | `[278, 229, 238, 255]` | +0.900 |
| `T1@71000 vs 3x70k`, earlier 1000h screen | 1000 | `[262, 246, 250, 242]` | +1.980 |
| `T1@71000 vs 3x70k`, final | 5000 | `[1331, 1239, 1157, 1273]` | +0.738 |

Interpretation: `teacher_ce_weight = 0.05` preserves a weak positive transfer signal, but this screen does not show it is better than the original `0.1`. Do not start `T1c_teacher_ce_02` from this evidence alone; if expanding T1b, treat it as a confirmation run, not as a promotion path.

Useful follow-up data mixtures after the first smoke:

```text
model_v4 vs 3x70k
model_v4 vs 3xmodel_v4
70k vs 3xmodel_v4
mixed model_v4 + 70k tables
```

Do not start with `70k trainee vs 3xmodel_v4` online rollout as the next structural experiment; it risks generating too many dominated states and noisy updates.

### Later: Stronger Teacher Correction

The next correction route should not be named as if T1 lacked teacher CE. T1 already uses teacher action CE. Later variants should add stronger or more selective preference signals:

- teacher CE weight sweep (`0.05` first, then `0.2` if warranted)
- model_v4 data distribution expansion (`model_v4 vs 3xmodel_v4`, `70k vs 3xmodel_v4`, mixed tables)
- reviewer `4.1b` high-confidence disagreement correction set

Reviewer `4.1b` should remain a sparse black-box preference labeler, not a bulk data generator. High-value labels are states where the actual 70k/T1/model_v4 action differs from the reviewer top action with a large reviewer margin.

## R1.5 Submission Automation

Manual reviewer upload is now the bottleneck. The batch submitter is:

```text
scripts/mortal/submit_reviewer_teacher_probe.py
```

It reads the R0 input manifest, submits each Tenhou6 custom log for each requested network, extracts the generated report id, archives `/report/<id>.json` with the existing report archiver, and writes:

```text
artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_batch/
  submit_manifest.jsonl
  submit_summary.json
  report_manifest.jsonl
  reports/*.json
```

The script does not solve or bypass Turnstile. It replays a browser-captured successful submit request and replaces only the safe form fields:

```text
input-method=tenhou6
tenhou6=<current JSON>
player-id=<manifest target seat>
engine=mortal
mortal-model-tag=<network>
ui=killerducky
lang=en
```

To prepare it, manually submit one Custom log in the browser, then use DevTools Network -> the `/review` request -> Copy as cURL and save it locally, for example:

```text
artifacts/experiments/reviewer_teacher_probe_2026_05/reviewer_submit.curl
```

Dry-run the batch plan first:

```bash
PYTHONPATH=src uv run python scripts/mortal/submit_reviewer_teacher_probe.py \
  --input-manifest artifacts/experiments/reviewer_teacher_probe_2026_05/R0_reviewer_input_smoke/manifest.jsonl \
  --submit-curl-file artifacts/experiments/reviewer_teacher_probe_2026_05/reviewer_submit.curl \
  --output-dir artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_batch \
  --networks 3.0,4.1b \
  --dry-run
```

For a cleaner real-submit smoke on a not-yet-reviewed row, use `--source-index` or `--start-index`:

```bash
PYTHONPATH=src uv run python scripts/mortal/submit_reviewer_teacher_probe.py \
  --input-manifest artifacts/experiments/reviewer_teacher_probe_2026_05/R0_reviewer_input_smoke/manifest.jsonl \
  --submit-curl-file artifacts/experiments/reviewer_teacher_probe_2026_05/reviewer_submit.curl \
  --output-dir artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_batch \
  --source-index 1 \
  --networks 3.0 \
  --dry-run
```

Then run without `--dry-run`. If the captured Turnstile response is expired or single-use, the script will record a clear `reviewer submit failed captcha validation` failure in `submit_manifest.jsonl`; in that case, switch to browser-driven submission rather than trying to reuse stale tokens.

Chrome on Windows may copy cURL in `cmd.exe` form with `^"` and line-continuation `^` characters. The submitter normalizes that form before parsing, so both bash-style and Windows cmd-style Copy-as-cURL are acceptable for dry-run validation.

Current R1.5 status as of 2026-06-09:

- A browser-captured `/review` POST cURL was saved locally as `artifacts/experiments/reviewer_teacher_probe_2026_05/reviewer_submit.curl`.
- Dry-run validation passed for `--source-index 0 --limit 1 --networks 4.1b`: the submitter parsed 14 form fields and found a captcha response.
- Real-submit smoke attempts for `--source-index 1 --limit 1 --networks 4.1b` failed with `reviewer submit failed captcha validation; capture a fresh browser submit cURL`. This means the captured Turnstile token is expired or single-use, not that the submitter failed to parse the request.
- The new preferred route is browser-assisted submission, not cURL replay.
- Existing smoke reports remain parseable. A current parser resmoke against `R0_external_smoke/report_manifest.jsonl` wrote `artifacts/experiments/reviewer_teacher_probe_2026_05/R1_parser_resmoke_20260604/`.
- The parser resmoke produced 238 decisions total and 119 aligned entries. For `4.1b`, it parsed 119 decisions, 96 matches, 23 mismatches, and 12 high-confidence disagreements.

### R1.5 Browser-Assisted Submission

Prepare browser tasks:

```bash
uv run python scripts/mortal/prepare_reviewer_browser_batch.py \
  --input-manifest artifacts/experiments/reviewer_teacher_probe_2026_05/R0_reviewer_input_smoke/manifest.jsonl \
  --output-dir artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_browser_batch \
  --source-index 1 \
  --limit 1 \
  --networks 4.1b
```

This writes:

```text
artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_browser_batch/browser_tasks/browser_batch.html
artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_browser_batch/browser_tasks/browser_tasks.jsonl
artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_browser_batch/browser_tasks/fill_scripts/*.js
```

Manual browser flow:

1. Open `browser_batch.html`.
2. Open `https://mjai.ekyu.moe/`.
3. Click `Copy fill JS` for one pending task.
4. Paste the JS into DevTools Console on the mjai page. It fills the real review form with the task's Tenhou6 JSON, target player, and Mortal network.
5. Complete Turnstile if shown and submit in the browser.
6. Copy the resulting report URL.
7. Run the task's archive command, replacing `REPORT_URL_OR_ID` with the report URL.

Example archive command:

```bash
uv run python scripts/mortal/archive_reviewer_browser_batch.py \
  --tasks artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_browser_batch/browser_tasks/browser_tasks.jsonl \
  --task-index 0 \
  --output-dir artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_browser_batch \
  --report REPORT_URL_OR_ID
```

Then parse the browser-batch report manifest:

```bash
uv run python scripts/mortal/parse_reviewer_teacher_reports.py \
  --report-manifest artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_browser_batch/report_manifest.jsonl \
  --output-dir artifacts/experiments/reviewer_teacher_probe_2026_05/R1_browser_batch_parse
```

Current browser-assisted result:

| Source | Network | Report id | Target player | Decisions | Matches | Mismatches | High-conf disagreements | Mean actual prob |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0002_320000_8192_b` | `4.1b` | `8b2e56f7d0ca12d8` | 1 | 118 | 102 | 16 | 13 | 0.8244 |

Artifacts:

- Report manifest: `artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_browser_batch/report_manifest.jsonl`
- Archived report: `artifacts/experiments/reviewer_teacher_probe_2026_05/R0_external_browser_batch/reports/0002_320000_8192_b__4.1b__p1.json`
- Parser summary: `artifacts/experiments/reviewer_teacher_probe_2026_05/R1_browser_batch_parse/summary.json`
- High-confidence disagreement list: `artifacts/experiments/reviewer_teacher_probe_2026_05/R1_browser_batch_parse/top_disagreements.jsonl`

Read: browser-assisted submission/archive works. The first expanded `4.1b` report has a similar disagreement profile to the initial `4.1b` smoke: most high-confidence corrections are discard choices, with one call-family correction. This supports using `4.1b` first as a reviewer overlay and curated correction set, not as a broad raw-Q target.

## T2a Risk-Gated Teacher CE

T2a tested a selective teacher CE route from `T1@71000`: keep the normal teacher CE weight at `0.05`, but set teacher CE weight to `0.0` on broad high-risk discard samples. The gate covered self-after-fuuro discards, opponent-riichi discards, after-fuuro-vs-riichi discards, dealer or big-lead states, and start-rank-1 states. DQN, CQL, GRP reward, and next-rank targets were unchanged.

Artifacts:

- Experiment: `artifacts/experiments/teacher_transfer_2026_05/T2a_risk_gated_teacher_ce_005/`
- Preflight: `preflight/risk_weight_alignment.json`
- Checkpoints: `checkpoints/mortal_t2a_71400.pth`, `checkpoints/mortal_t2a_71800.pth`
- Primary eval: `four_player_native_1000h/`
- Decision report: `T2a_decision_report.md`

Preflight passed over the full teacher replay pool: `10000` files, `1581313` samples, `1214264` matched raw explicit decisions, `mismatch_count=0`. CE active rate was `56.22%`, disabled rate was `43.78%`, and disabled discard rate was `60.37%`; this is inside the planned active-rate bounds.

Native random-seat 1000h result with `70k / 80k_game / T1_71000 / T2a_71800`:

| Model | Rank counts `[1,2,3,4]` | Avg rank | Avg pt |
| --- | ---: | ---: | ---: |
| `70k` | `[256,220,258,266]` | 2.534 | -2.97 |
| `80k_game` | `[264,223,256,257]` | 2.506 | -0.90 |
| `T1_71000` | `[240,282,246,232]` | 2.470 | +2.97 |
| `T2a_71800` | `[240,275,240,245]` | 2.490 | +0.90 |

Behavior readout:

| Metric | `T1_71000` | `T2a_71800` |
| --- | ---: | ---: |
| Agari | 21.85% | 21.36% |
| Houjuu | 13.68% | 12.48% |
| Fuuro | 26.80% | 25.82% |
| Riichi | 20.38% | 16.87% |
| After-fuuro agari | 34.32% | 35.80% |
| After-fuuro houjuu | 13.93% | 13.61% |
| After-riichi agari | 48.13% | 48.47% |
| After-riichi houjuu | 16.35% | 15.12% |

Read: T2a lowered houjuu and improved some after-fuuro/after-riichi outcomes, but lost enough agari and riichi pressure that it finished `2.07` avg pt behind T1. This fails the planned continuation rule (`T2a` worse than T1 by more than `1.0` avg pt). Do not promote T2a over T1 and do not expand this exact binary risk gate.

Next T-series design should not use broad CE-off gating. If continuing the idea, use softer/narrower weighting, such as nonzero CE on `vs_riichi` and late discard samples, and validate on a smaller staged smoke before another full 1000h native run.

## T2b Soft Risk-Gated Teacher CE

T2b tested the softer version of T2a: keep normal teacher CE at `0.05`, but use `0.02` instead of `0.0` on the same high-risk discard samples. The goal was to preserve T1's pressure while retaining T2a's lower-houjuu signal.

Artifacts:

- Experiment: `artifacts/experiments/teacher_transfer_2026_05/T2b_soft_risk_teacher_ce_005_002/`
- Preflight: `preflight/risk_weight_alignment.json`
- Checkpoint: `checkpoints/mortal_t2b_71400.pth`
- Stage-1 eval: `four_player_native_250h_71400/`
- Decision report: `T2b_decision_report.md`

Preflight passed over the full teacher replay pool: `10000` files, `1581313` samples, `1214264` matched raw explicit decisions, `mismatch_count=0`. CE weight mean was `0.03687`, gated sample rate was `43.78%`, and gated discard rate was `60.37%`.

Native random-seat 250h result with `70k / 80k_game / T1_71000 / T2b_71400`:

| Model | Rank counts `[1,2,3,4]` | Avg rank | Avg pt |
| --- | ---: | ---: | ---: |
| `70k` | `[66,61,58,65]` | 2.488 | -0.36 |
| `80k_game` | `[60,68,59,63]` | 2.500 | -0.18 |
| `T1_71000` | `[65,58,66,61]` | 2.492 | +0.90 |
| `T2b_71400` | `[59,63,67,61]` | 2.520 | -0.36 |

Behavior readout:

| Metric | `T1_71000` | `T2b_71400` |
| --- | ---: | ---: |
| Agari | 21.97% | 21.27% |
| Houjuu | 13.82% | 11.60% |
| Fuuro | 26.23% | 24.42% |
| Riichi | 21.04% | 17.82% |
| After-fuuro agari | 35.59% | 37.33% |
| After-fuuro houjuu | 14.69% | 12.75% |
| After-riichi agari | 48.94% | 47.61% |
| After-riichi houjuu | 15.14% | 16.22% |

Read: T2b preserved the intended lower-houjuu signal, but still lost enough riichi/agari pressure to fail the stage-1 continuation gate. T2b was `1.26` avg pt behind T1 in the 250h screen, and riichi rate fell by `3.22pp`. Do not train `T2b@71800`.

This stops the broad T-series risk-gated CE family for now. The next planning cycle should pivot to reviewer/NAGA/4.1b high-confidence small-sample diagnosis focused on late/vs-riichi and after-riichi contexts, without treating the reviewer as a hard oracle.

## R2 Outcome-Anchored Reviewer Casebook

R2 is a diagnostic-only casebook generated from existing T2a/T2b native logs. It does not start T3 training and does not convert NAGA/4.1b into hard labels. The goal is to determine whether the repeated T2 failure mode is concentrated in late/vs-riichi and after-riichi decisions, or whether the apparent disagreements are mostly style/outcome noise.

Artifacts:

- Experiment: `artifacts/experiments/reviewer_teacher_probe_2026_05/R2_late_vs_riichi_casebook_2026_06/`
- Generator: `scripts/mortal/build_r2_outcome_anchored_casebook.py`
- Casebook: `review_cases.html`
- Manifest: `case_manifest.jsonl`, `case_manifest.json`
- NAGA manual input: `naga_kyoku_blocks.txt`, `naga_kyoku_urls.txt`, `naga_kyoku_tenhou6_cases.zip`
- Full hanchan Tenhou6 archive: `hanchan_tenhou6_cases.zip`
- External review import template: `external_reviews_pending/import_template.jsonl`

Case selection:

| Bucket | Count |
| --- | ---: |
| `late_vs_riichi_negative` | 20 |
| `after_riichi_negative` | 20 |
| `after_fuuro_positive` | 10 |
| `neutral_control` | 10 |

Each case records source log, kyoku index, target seat with wind label, target model, focus slice, outcome, local source-action q/prob from the arena log, and local replay reviews from `70k`, `80k_game`, `T1_71000`, and the target model (`T2a_71800` or `T2b_71400`). The generated replay URL opens the project-native `/game-replay` page at the focus decision, so Tenhou6/NAGA links are secondary reviewer-input artifacts rather than the main casebook UI.

Validation:

- `case_manifest.jsonl` rows: `60`
- Bucket counts: `20 / 20 / 10 / 10`
- `naga_kyoku_blocks.txt` separators: `60`
- Cases with missing replay URL: `0`
- Cases with four local model reviews: `60`
- External reviewer status: pending for all `60`

Decision status: no training decision yet. Import NAGA/4.1b reports and human annotations first. If high-confidence reviewer disagreements concentrate in late/vs-riichi and look locally reasonable, the next design can be a narrow T3 targeted correction. If disagreements are scattered or style-only, stop reviewer-training and return to pressure/readout analysis around T1.
