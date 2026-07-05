#!/usr/bin/env python3
"""Import outcome-risk cases into the existing Replay UI storage.

This does not modify the main review UI. It materializes each selected case as a
normal replay and writes a compact HTML index that links to /game-replay.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from replay.api import run_replay_single_raw
from replay.bot import render_replay_json
from replay.normalize import normalize_replay_decisions
from replay.storage import get_storage


DEFAULT_T1_CHECKPOINT = (
    _REPO_ROOT
    / "artifacts"
    / "experiments"
    / "teacher_transfer_2026_05"
    / "T1_teacher_ce_01"
    / "mortal.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_T1_CHECKPOINT)
    parser.add_argument("--bot-type", default="t1_71000")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Re-import even when an import manifest exists")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_events(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def resolve_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = _REPO_ROOT / path
    return path


def target_focus_event_index(events: list[dict[str, Any]], *, kyoku_index: int, target_seat: int) -> int | None:
    current_kyoku = -1
    in_target_kyoku = False
    last_target_dahai: int | None = None
    first_terminal: int | None = None
    for idx, event in enumerate(events):
        event_type = str(event.get("type", ""))
        if event_type == "start_kyoku":
            current_kyoku += 1
            in_target_kyoku = current_kyoku == kyoku_index
            last_target_dahai = None
            first_terminal = None
            continue
        if not in_target_kyoku:
            continue
        if event_type == "dahai" and int(event.get("actor", -1)) == target_seat:
            last_target_dahai = idx
        if event_type in {"hora", "ryukyoku"} and first_terminal is None:
            first_terminal = idx
            if event_type == "hora" and int(event.get("target", -1)) == target_seat:
                return last_target_dahai or idx
        if event_type == "end_kyoku":
            return last_target_dahai or first_terminal or idx
    return last_target_dahai


def resolve_focus_step(decisions: dict[str, Any], focus_event_index: int | None) -> tuple[int | None, str]:
    if focus_event_index is None:
        return None, "missing"
    log = decisions.get("log", [])
    exact = [
        idx
        for idx, entry in enumerate(log)
        if isinstance(entry, dict) and entry.get("source_event_index") == focus_event_index
    ]
    if exact:
        return int(exact[0]), "exact"
    candidates = [
        (abs(int(entry["source_event_index"]) - int(focus_event_index)), idx)
        for idx, entry in enumerate(log)
        if isinstance(entry, dict) and isinstance(entry.get("source_event_index"), int)
    ]
    if not candidates:
        return None, "missing"
    _distance, idx = min(candidates, key=lambda item: (item[0], item[1]))
    return int(idx), "nearest"


def case_id(row: dict[str, Any]) -> str:
    return f"t1_risk_case_{int(row['case_index']):03d}"


def import_case(row: dict[str, Any], *, checkpoint: Path, bot_type: str) -> dict[str, Any]:
    events = load_events(resolve_path(str(row["source_log"])))
    target_seat = int(row["target_seat"])
    bot = run_replay_single_raw(
        events,
        player_id=target_seat,
        checkpoint=str(checkpoint),
        input_type="mjai",
        bot_type=bot_type,
    )
    decisions = normalize_replay_decisions(render_replay_json(bot))
    focus_event_index = target_focus_event_index(
        events,
        kyoku_index=int(row["kyoku_index"]),
        target_seat=target_seat,
    )
    focus_step, focus_resolution = resolve_focus_step(decisions, focus_event_index)
    decisions["outcome_risk_case"] = {
        "case_id": case_id(row),
        "case_index": int(row["case_index"]),
        "target_model": row.get("target_model"),
        "target_seat": target_seat,
        "target_seat_name": ["东家", "南家", "西家", "北家"][target_seat],
        "kyoku_index": int(row["kyoku_index"]),
        "tags": row.get("tags", []),
        "source_log": row.get("source_log"),
        "focus_event_index": focus_event_index,
        "focus_step": focus_step,
        "focus_resolution": focus_resolution,
    }
    storage = get_storage()
    replay_id = storage.save(
        events=events,
        decisions=decisions,
        bot_type=bot_type,
        player_names=decisions.get("player_names") or row.get("player_names"),
        checkpoint=str(checkpoint),
    )
    return {
        **decisions["outcome_risk_case"],
        "replay_id": replay_id,
        "player_names": row.get("player_names", []),
        "start_scores": row.get("start_scores", []),
        "dealer": row.get("dealer"),
        "turns": row.get("turns"),
        "agari": row.get("agari"),
        "houjuu": row.get("houjuu"),
        "delta_score": row.get("delta_score"),
    }


def build_index_html(rows: list[dict[str, Any]], *, base_url: str) -> str:
    data = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>T1 Risk Cases - Replay UI Index</title>
<style>
body {{ margin:0; font-family:Arial,"Microsoft YaHei",sans-serif; background:#f6f7f9; color:#15171a; }}
header {{ padding:14px 18px; background:#fff; border-bottom:1px solid #d8dde5; position:sticky; top:0; }}
h1 {{ margin:0 0 8px; font-size:18px; }}
.grid {{ display:grid; grid-template-columns: 360px minmax(0,1fr); min-height:calc(100vh - 72px); }}
.list {{ background:#fff; border-right:1px solid #d8dde5; overflow:auto; }}
.case {{ display:block; width:100%; text-align:left; border:0; border-bottom:1px solid #eef1f5; background:#fff; padding:10px; cursor:pointer; }}
.case.active {{ background:#eaf2ff; }}
.title {{ font-weight:700; }}
.meta {{ margin-top:4px; color:#59636f; font-size:12px; line-height:1.35; }}
main {{ padding:16px; }}
.panel {{ background:#fff; border:1px solid #d8dde5; border-radius:6px; padding:12px; margin-bottom:12px; }}
.row {{ display:grid; grid-template-columns: 140px 1fr; gap:8px; padding:4px 0; }}
.label-row {{ display:flex; flex-wrap:wrap; gap:8px; margin:10px 0; }}
button, a.button {{ height:34px; border:1px solid #b8c0cc; background:#fff; border-radius:6px; padding:0 10px; font-size:13px; display:inline-flex; align-items:center; text-decoration:none; color:#15171a; }}
button.primary, a.primary {{ background:#1f6feb; color:#fff; border-color:#1f6feb; }}
.label-row button.active {{ background:#1f6feb; color:#fff; border-color:#1f6feb; }}
textarea {{ width:100%; min-height:96px; border:1px solid #d8dde5; border-radius:6px; padding:8px; font-family:Arial,"Microsoft YaHei",sans-serif; }}
.actions {{ display:flex; flex-wrap:wrap; gap:8px; }}
</style>
</head>
<body>
<header>
  <h1>T1 风险 case 索引 - 使用项目原生 /game-replay</h1>
  <div class="actions">
    <button id="exportJson">导出标注 JSON</button>
    <button id="exportCsv">导出标注 CSV</button>
  </div>
</header>
<div class="grid">
  <aside class="list" id="list"></aside>
  <main>
    <div class="panel" id="detail"></div>
    <div class="panel">
      <div><strong>人工标注</strong> A=明显坏模式，B=有争议，C=T1没问题，D=丢弃</div>
      <div class="label-row" id="labels">
        <button data-label="A">A 坏模式</button>
        <button data-label="B">B 有争议</button>
        <button data-label="C">C 没问题</button>
        <button data-label="D">D 丢弃</button>
      </div>
      <textarea id="note" placeholder="备注"></textarea>
    </div>
  </main>
</div>
<script id="case-data" type="application/json">{data}</script>
<script>
const baseUrl = {json.dumps(base_url.rstrip('/'), ensure_ascii=False)};
const cases = JSON.parse(document.getElementById('case-data').textContent);
const key = 't1_risk_replay_ui_annotations_v1';
let current = 0;
let annotations = JSON.parse(localStorage.getItem(key) || '{{}}');
const list = document.getElementById('list');
const detail = document.getElementById('detail');
const labels = document.getElementById('labels');
const note = document.getElementById('note');
function id(c) {{ return String(c.case_index).padStart(3, '0'); }}
function replayUrl(c) {{
  const p = new URLSearchParams();
  p.set('id', c.replay_id);
  p.set('player_id', String(c.target_seat));
  if (c.focus_step !== null && c.focus_step !== undefined) {{
    p.set('focus_step', String(c.focus_step));
    p.set('step', String(c.focus_step));
    p.set('phase', 'pre');
  }}
  return baseUrl + '/game-replay?' + p.toString();
}}
function save() {{ localStorage.setItem(key, JSON.stringify(annotations)); }}
function ann(c) {{ const k=id(c); annotations[k] ||= {{label:'',note:''}}; return annotations[k]; }}
function saveNote() {{ ann(cases[current]).note = note.value; save(); }}
function renderList() {{
  list.innerHTML = cases.map((c,i) => `<button class="case ${{i===current?'active':''}}" data-i="${{i}}">
    <div class="title">CASE ${{id(c)}} · ${{c.target_seat_name}} · kyoku ${{c.kyoku_index}}</div>
    <div class="meta">${{c.tags.join(', ')}}<br>${{c.replay_id}}</div>
  </button>`).join('');
}}
function render() {{
  const c = cases[current];
  const a = ann(c);
  renderList();
  detail.innerHTML = `
    <div class="row"><strong>Case</strong><span>${{id(c)}}</span></div>
    <div class="row"><strong>主视角</strong><span>${{c.target_model}} / ${{c.target_seat_name}} / seat ${{c.target_seat}}</span></div>
    <div class="row"><strong>玩家</strong><span>${{c.player_names.join(' / ')}}</span></div>
    <div class="row"><strong>标签</strong><span>${{c.tags.join(', ')}}</span></div>
    <div class="row"><strong>结果</strong><span>houjuu=${{c.houjuu}} delta=${{c.delta_score}}</span></div>
    <div class="row"><strong>Replay</strong><span>${{c.replay_id}}</span></div>
    <div class="actions" style="margin-top:12px">
      <a class="button primary" href="${{replayUrl(c)}}" target="_blank" rel="noopener">打开项目原生 Review</a>
      <button onclick="navigator.clipboard.writeText('${{replayUrl(c)}}')">复制 Review URL</button>
    </div>`;
  [...labels.querySelectorAll('button')].forEach(b => b.classList.toggle('active', b.dataset.label === a.label));
  note.value = a.note || '';
}}
list.onclick = e => {{ const b=e.target.closest('button[data-i]'); if(!b) return; saveNote(); current=Number(b.dataset.i); render(); }};
labels.onclick = e => {{ const b=e.target.closest('button[data-label]'); if(!b) return; ann(cases[current]).label=b.dataset.label; save(); render(); }};
note.oninput = saveNote;
function rows() {{ saveNote(); return cases.map(c => ({{case:id(c), label:(annotations[id(c)]||{{}}).label||'', note:(annotations[id(c)]||{{}}).note||'', replay_url:replayUrl(c), ...c}})); }}
function download(name, text, type) {{ const blob=new Blob([text],{{type}}); const url=URL.createObjectURL(blob); const a=document.createElement('a'); a.href=url; a.download=name; a.click(); URL.revokeObjectURL(url); }}
document.getElementById('exportJson').onclick = () => download('t1_risk_replay_ui_annotations.json', JSON.stringify(rows(), null, 2), 'application/json');
document.getElementById('exportCsv').onclick = () => {{
  const r=rows(); const h=['case','label','note','replay_url','target_seat_name','tags','replay_id'];
  const esc=v => '"' + String(v ?? '').replaceAll('"','""') + '"';
  download('t1_risk_replay_ui_annotations.csv', [h.join(','), ...r.map(x => h.map(k => esc(k==='tags' ? x[k].join(';') : x[k])).join(','))].join('\\n')+'\\n', 'text/csv');
}};
render();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    import_manifest = args.output_dir / "replay_ui_import_manifest.jsonl"
    if import_manifest.exists() and not args.force:
        rows = load_jsonl(import_manifest)
    else:
        source_rows = load_jsonl(args.case_dir / "case_manifest.jsonl")
        if args.limit and args.limit > 0:
            source_rows = source_rows[: args.limit]
        rows = []
        for idx, row in enumerate(source_rows, start=1):
            print(f"importing {idx}/{len(source_rows)} case={row['case_index']} seat={row['target_seat']}", flush=True)
            rows.append(import_case(row, checkpoint=args.checkpoint, bot_type=str(args.bot_type)))
        with import_manifest.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
    index_html = args.output_dir / "replay_ui_case_index.html"
    index_html.write_text(build_index_html(rows, base_url=str(args.base_url)), encoding="utf-8")
    print(f"cases: {len(rows)}")
    print(f"manifest: {import_manifest}")
    print(f"index: {index_html}")


if __name__ == "__main__":
    main()
