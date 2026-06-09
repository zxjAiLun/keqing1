#!/usr/bin/env python3
"""Prepare browser-assisted reviewer submission tasks.

This does not bypass Turnstile. It prepares per-task JavaScript snippets that
fill the real mjai.ekyu.moe review form in a browser session. The user still
submits in the browser and completes any challenge there.
"""
from __future__ import annotations

import argparse
from collections.abc import Sequence
from datetime import datetime
from datetime import timezone
import html
import json
from pathlib import Path
import sys
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.mortal.submit_reviewer_teacher_probe import DEFAULT_INPUT_MANIFEST
from scripts.mortal.submit_reviewer_teacher_probe import DEFAULT_OUTPUT_DIR
from scripts.mortal.submit_reviewer_teacher_probe import load_existing_keys
from scripts.mortal.submit_reviewer_teacher_probe import load_jsonl
from scripts.mortal.submit_reviewer_teacher_probe import output_report_manifest_path
from scripts.mortal.submit_reviewer_teacher_probe import parse_csv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-manifest", type=Path, default=DEFAULT_INPUT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--networks", default=None, help="Override comma-separated networks; default uses each manifest row")
    parser.add_argument("--source-index", type=int, default=None, help="Prepare exactly one 0-based source manifest row")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-existing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def select_manifest_rows(
    *,
    input_manifest: Path,
    source_index: int | None,
    start_index: int,
    limit: int | None,
) -> list[tuple[int, dict[str, Any]]]:
    all_rows = list(enumerate(load_jsonl(input_manifest)))
    if source_index is not None:
        if source_index < 0 or source_index >= len(all_rows):
            raise IndexError(f"source-index {source_index} out of range for {input_manifest} with {len(all_rows)} rows")
        selected = [all_rows[source_index]]
    else:
        if start_index < 0:
            raise ValueError(f"start-index must be non-negative, got {start_index}")
        selected = all_rows[start_index:]
    if limit is not None:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        selected = selected[:limit]
    return selected


def build_fill_script(*, tenhou6_text: str, target_player: int, network: str) -> str:
    payload = {
        "tenhou6": tenhou6_text,
        "targetPlayer": str(target_player),
        "network": str(network),
    }
    payload_json = json.dumps(payload, ensure_ascii=False)
    return f"""(() => {{
  const task = {payload_json};
  const form = document.forms.reviewForm || document.querySelector('form[name="reviewForm"]');
  if (!form) {{
    throw new Error('reviewForm not found. Open https://mjai.ekyu.moe/ first.');
  }}

  const setValue = (selector, value) => {{
    const el = form.querySelector(selector) || document.querySelector(selector);
    if (!el) throw new Error(`missing field: ${{selector}}`);
    el.value = value;
    el.dispatchEvent(new Event('input', {{ bubbles: true }}));
    el.dispatchEvent(new Event('change', {{ bubbles: true }}));
    return el;
  }};

  const inputMethod = form.querySelector('input[name="input-method"][value="tenhou6"]');
  if (!inputMethod) throw new Error('tenhou6 input-method radio not found');
  inputMethod.checked = true;
  inputMethod.dispatchEvent(new Event('change', {{ bubbles: true }}));
  if (typeof window.changeInputMethod === 'function') window.changeInputMethod('tenhou6');

  setValue('textarea[name="tenhou6"]', task.tenhou6);
  setValue('select[name="player-id"]', task.targetPlayer);
  setValue('select[name="engine"]', 'mortal');
  if (typeof window.changeEngine === 'function') window.changeEngine('mortal');
  setValue('select[name="mortal-model-tag"]', task.network);
  setValue('select[name="ui"]', 'killerducky');
  setValue('select[name="lang"]', 'en');
  setValue('input[name="temperature"]', '');
  setValue('input[name="kyokus"]', '');

  const submit = form.querySelector('button[type="submit"], input[type="submit"], button:not([type])');
  if (submit) submit.scrollIntoView({{ block: 'center' }});
  console.log(`Filled reviewer task: player=${{task.targetPlayer}}, network=${{task.network}}`);
  console.log('Complete Turnstile if needed, submit the form, then archive the resulting report URL.');
}})();
"""


def task_output_name(task_index: int, source_index: int, target_player: int, network: str) -> str:
    safe_network = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in network)
    return f"task_{task_index:04d}_src{source_index:04d}_p{target_player}_{safe_network}"


def prepare_browser_batch(
    *,
    input_manifest: Path,
    output_dir: Path,
    networks_override: list[str] | None,
    source_index: int | None,
    start_index: int,
    limit: int | None,
    skip_existing: bool,
) -> dict[str, Any]:
    rows = select_manifest_rows(
        input_manifest=input_manifest,
        source_index=source_index,
        start_index=start_index,
        limit=limit,
    )
    existing = load_existing_keys(output_report_manifest_path(output_dir)) if skip_existing else set()
    task_dir = output_dir / "browser_tasks"
    fill_dir = task_dir / "fill_scripts"
    task_manifest_path = task_dir / "browser_tasks.jsonl"

    tasks: list[dict[str, Any]] = []
    task_index = 0
    for row_index, row in rows:
        tenhou6_path = Path(str(row["tenhou6_path"]))
        tenhou6_text = tenhou6_path.read_text(encoding="utf-8")
        networks = networks_override or list(row["networks"])
        for target_player in row["target_players"]:
            for network in networks:
                key = (str(tenhou6_path), int(target_player), str(network))
                status = "skipped_existing" if key in existing else "pending_browser_submit"
                name = task_output_name(task_index, row_index, int(target_player), str(network))
                fill_script_path = fill_dir / f"{name}.js"
                archive_command = (
                    "uv run python scripts/mortal/archive_reviewer_browser_batch.py "
                    f"--tasks {task_manifest_path} --task-index {task_index} "
                    f"--output-dir {output_dir} --report REPORT_URL_OR_ID"
                )
                task = {
                    "schema": "keqing.mortal.reviewer_browser_task.v1",
                    "task_index": task_index,
                    "status": status,
                    "source_manifest": str(input_manifest),
                    "source_index": row_index,
                    "source_log": row.get("source_log"),
                    "source_tenhou6_path": str(tenhou6_path),
                    "target_player": int(target_player),
                    "network": str(network),
                    "fill_script_path": str(fill_script_path),
                    "archive_command": archive_command,
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }
                if status != "skipped_existing":
                    fill_dir.mkdir(parents=True, exist_ok=True)
                    fill_script_path.write_text(
                        build_fill_script(
                            tenhou6_text=tenhou6_text,
                            target_player=int(target_player),
                            network=str(network),
                        ),
                        encoding="utf-8",
                    )
                tasks.append(task)
                task_index += 1

    task_dir.mkdir(parents=True, exist_ok=True)
    with task_manifest_path.open("w", encoding="utf-8") as handle:
        for task in tasks:
            handle.write(json.dumps(task, ensure_ascii=False, sort_keys=True) + "\n")
    html_path = task_dir / "browser_batch.html"
    html_path.write_text(render_html(tasks, task_manifest_path=task_manifest_path), encoding="utf-8")
    summary = {
        "schema": "keqing.mortal.reviewer_browser_batch.v1",
        "input_manifest": str(input_manifest),
        "output_dir": str(output_dir),
        "task_manifest": str(task_manifest_path),
        "html": str(html_path),
        "task_count": len(tasks),
        "status_counts": count_statuses(tasks),
    }
    (task_dir / "browser_batch_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def count_statuses(tasks: Sequence[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for task in tasks:
        key = str(task["status"])
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def render_html(tasks: Sequence[dict[str, Any]], *, task_manifest_path: Path) -> str:
    rows: list[str] = []
    for task in tasks:
        fill_path = Path(str(task["fill_script_path"]))
        script_text = fill_path.read_text(encoding="utf-8") if fill_path.exists() else ""
        rows.append(
            "<tr>"
            f"<td>{task['task_index']}</td>"
            f"<td>{task['status']}</td>"
            f"<td>{task['source_index']}</td>"
            f"<td>{task['target_player']}</td>"
            f"<td>{html.escape(str(task['network']))}</td>"
            f"<td><button data-script=\"{html.escape(script_text, quote=True)}\">Copy fill JS</button></td>"
            f"<td><code>{html.escape(str(task['archive_command']))}</code></td>"
            "</tr>"
        )
    return f"""<!doctype html>
<meta charset="utf-8">
<title>Reviewer Browser Batch</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 6px 8px; vertical-align: top; }}
code {{ white-space: pre-wrap; }}
button {{ cursor: pointer; }}
</style>
<h1>Reviewer Browser Batch</h1>
<p>Task manifest: <code>{html.escape(str(task_manifest_path))}</code></p>
<ol>
  <li>Open <a href="https://mjai.ekyu.moe/" target="_blank" rel="noreferrer noopener">https://mjai.ekyu.moe/</a>.</li>
  <li>For one pending task, click <b>Copy fill JS</b>.</li>
  <li>Paste it in the browser DevTools Console on the mjai page. It fills the real review form.</li>
  <li>Complete Turnstile if shown, submit in the browser, then copy the report URL.</li>
  <li>Run the shown archive command with <code>REPORT_URL_OR_ID</code> replaced by the report URL.</li>
</ol>
<table>
<thead><tr><th>Task</th><th>Status</th><th>Source</th><th>Player</th><th>Network</th><th>Fill</th><th>Archive command</th></tr></thead>
<tbody>
{''.join(rows)}
</tbody>
</table>
<script>
for (const button of document.querySelectorAll('button[data-script]')) {{
  button.addEventListener('click', async () => {{
    await navigator.clipboard.writeText(button.dataset.script);
    button.textContent = 'Copied';
    setTimeout(() => button.textContent = 'Copy fill JS', 1500);
  }});
}}
</script>
"""


def main() -> None:
    args = parse_args()
    summary = prepare_browser_batch(
        input_manifest=args.input_manifest,
        output_dir=args.output_dir,
        networks_override=parse_csv(args.networks),
        source_index=args.source_index,
        start_index=int(args.start_index),
        limit=args.limit,
        skip_existing=bool(args.skip_existing),
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
