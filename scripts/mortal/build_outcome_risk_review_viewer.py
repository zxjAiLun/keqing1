#!/usr/bin/env python3
"""Build a compact single-page viewer for outcome-risk review cases."""

from __future__ import annotations

import argparse
import gzip
import html
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def read_json(path: Path) -> str:
    return json.dumps(json.loads(path.read_text(encoding="utf-8")), ensure_ascii=False, separators=(",", ":"))


def read_jsonl_events(path: Path) -> list[dict[str, Any]]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def extract_kyoku_events(path: Path, kyoku_index: int) -> list[dict[str, Any]]:
    events = read_jsonl_events(path)
    selected: list[dict[str, Any]] = []
    current_index = -1
    in_target = False
    for event in events:
        event_type = str(event.get("type", ""))
        if event_type == "start_game":
            selected = [event]
            continue
        if event_type == "start_kyoku":
            current_index += 1
            in_target = current_index == kyoku_index
            if in_target:
                selected.append(event)
            continue
        if in_target:
            selected.append(event)
            if event_type == "end_kyoku":
                break
    return selected


def tenhou6_url(payload_json: str) -> str:
    return "https://tenhou.net/6/#json=" + payload_json


def tenhou5_viewer_url(payload_json: str) -> str:
    return "https://tenhou.net/5/#json=" + payload_json


def build_cases(case_dir: Path) -> list[dict[str, Any]]:
    rows = load_jsonl(case_dir / "case_manifest.jsonl")
    cases: list[dict[str, Any]] = []
    for row in rows:
        hanchan_path = Path(row["hanchan_tenhou6_path"])
        naga_path = Path(row["naga_kyoku_tenhou6_path"])
        if not hanchan_path.is_absolute():
            hanchan_path = Path.cwd() / hanchan_path
        if not naga_path.is_absolute():
            naga_path = Path.cwd() / naga_path
        source_log = Path(row["source_log"])
        if not source_log.is_absolute():
            source_log = Path.cwd() / source_log
        hanchan_json = read_json(hanchan_path)
        naga_json = read_json(naga_path)
        target_seat = int(row["target_seat"])
        seat_names = ["东家", "南家", "西家", "北家"]
        cases.append(
            {
                "case_index": row["case_index"],
                "kyoku_index": row["kyoku_index"],
                "target_model": row["target_model"],
                "target_seat": target_seat,
                "target_seat_name": seat_names[target_seat] if 0 <= target_seat < len(seat_names) else str(target_seat),
                "player_names": row["player_names"],
                "tags": row["tags"],
                "start_scores": row["start_scores"],
                "dealer": row["dealer"],
                "turns": row["turns"],
                "first_riichi_turn": row["first_riichi_turn"],
                "first_fuuro_turn": row["first_fuuro_turn"],
                "agari": row["agari"],
                "houjuu": row["houjuu"],
                "delta_score": row["delta_score"],
                "naga_url": row["url"],
                "hanchan_file": hanchan_path.name,
                "hanchan_url": tenhou6_url(hanchan_json),
                "hanchan_viewer_url": tenhou5_viewer_url(hanchan_json),
                "naga_url_computed": tenhou6_url(naga_json),
                "naga_viewer_url": tenhou5_viewer_url(naga_json),
                "mjai_events": extract_kyoku_events(source_log, int(row["kyoku_index"])),
            }
        )
    return cases


def write_compact_files(cases: list[dict[str, Any]], output_dir: Path) -> None:
    dst_hanchan = output_dir / "hanchan_tenhou6"
    dst_hanchan.mkdir(parents=True, exist_ok=True)
    seen_hanchan: set[str] = set()
    naga_blocks: list[str] = []
    for case in cases:
        hanchan_name = str(case["hanchan_file"]).removesuffix(".json") + ".url.txt"
        if hanchan_name not in seen_hanchan:
            (dst_hanchan / hanchan_name).write_text(str(case["hanchan_url"]) + "\n", encoding="utf-8")
            seen_hanchan.add(hanchan_name)
        naga_blocks.append(
            "\n".join(
                [
                    (
                        f"===== CASE {int(case['case_index']):03d} | kyoku={case['kyoku_index']} "
                        f"| seat={case['target_seat']}({case['target_seat_name']}) | tags={','.join(case['tags'])} ====="
                    ),
                    str(case["naga_url"]),
                    "",
                ]
            )
        )
    (output_dir / "naga_kyoku_blocks.txt").write_text("\n".join(naga_blocks), encoding="utf-8")


def build_html(cases: list[dict[str, Any]]) -> str:
    data = json.dumps(cases, ensure_ascii=False, separators=(",", ":"))
    escaped_data = data.replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>T1 Outcome Risk Review Cases</title>
<style>
:root {{
  color-scheme: light;
  font-family: Arial, "Microsoft YaHei", sans-serif;
  background: #f6f7f9;
  color: #15171a;
}}
* {{ box-sizing: border-box; }}
body {{ margin: 0; }}
header {{
  padding: 14px 18px;
  border-bottom: 1px solid #d8dde5;
  background: #ffffff;
  position: sticky;
  top: 0;
  z-index: 3;
}}
h1 {{ margin: 0 0 8px; font-size: 18px; }}
.toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
button, select {{
  height: 32px;
  border: 1px solid #b8c0cc;
  background: #fff;
  border-radius: 6px;
  padding: 0 10px;
  font-size: 13px;
}}
button.primary {{ background: #1f6feb; color: white; border-color: #1f6feb; }}
main {{
  display: grid;
  grid-template-columns: 310px minmax(0, 1fr);
  min-height: calc(100vh - 74px);
}}
aside {{
  border-right: 1px solid #d8dde5;
  background: #fff;
  overflow: auto;
  max-height: calc(100vh - 74px);
}}
.case-button {{
  width: 100%;
  min-height: 58px;
  border: 0;
  border-bottom: 1px solid #eef1f5;
  border-radius: 0;
  background: #fff;
  text-align: left;
  padding: 8px 10px;
}}
.case-button.active {{ background: #eaf2ff; }}
.case-title {{ font-weight: 700; }}
.case-meta {{ margin-top: 4px; font-size: 12px; color: #59636f; line-height: 1.35; }}
section {{ padding: 12px; min-width: 0; }}
.review-grid {{
  display: grid;
  grid-template-columns: minmax(0, 1fr) 360px;
  gap: 12px;
  align-items: start;
}}
.native-table {{
  width: 100%;
  min-height: calc(100vh - 104px);
  min-height: 640px;
  border: 1px solid #d8dde5;
  border-radius: 6px;
  background: #23624f;
  color: #fff;
  padding: 12px;
  display: grid;
  grid-template-rows: auto 1fr auto;
  gap: 10px;
}}
.table-controls {{
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
  background: rgba(0,0,0,.18);
  border-radius: 6px;
  padding: 8px;
}}
.table-controls input {{ flex: 1; min-width: 220px; }}
.table-controls button {{ background: #f8fafc; }}
.seat-grid {{
  display: grid;
  grid-template-columns: 1fr 1.1fr 1fr;
  grid-template-rows: auto 1fr auto;
  gap: 10px;
  align-items: stretch;
}}
.seat-card {{
  background: rgba(0,0,0,.2);
  border: 1px solid rgba(255,255,255,.18);
  border-radius: 6px;
  padding: 8px;
  min-height: 154px;
}}
.seat-card.target {{ outline: 3px solid #facc15; }}
.seat-north {{ grid-column: 2; grid-row: 1; }}
.seat-west {{ grid-column: 1; grid-row: 2; }}
.seat-center {{ grid-column: 2; grid-row: 2; align-self: center; justify-self: stretch; }}
.seat-east {{ grid-column: 3; grid-row: 2; }}
.seat-south {{ grid-column: 2; grid-row: 3; }}
.seat-head {{ display: flex; justify-content: space-between; gap: 8px; font-size: 13px; margin-bottom: 6px; }}
.seat-name {{ font-weight: 700; }}
.tile-row {{ display: flex; flex-wrap: wrap; gap: 3px; min-height: 24px; margin: 4px 0; }}
.tile {{
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 26px;
  height: 34px;
  border-radius: 4px;
  background: #fff;
  color: #111827;
  border: 1px solid #cbd5e1;
  font-weight: 700;
  font-size: 13px;
}}
.tile.red {{ color: #dc2626; }}
.tile.honor {{ color: #1d4ed8; }}
.tile.tsumogiri {{ opacity: .62; }}
.tile.reach {{ transform: rotate(90deg); margin: 0 5px; outline: 2px solid #facc15; }}
.meld {{
  padding: 2px 5px;
  border-radius: 4px;
  background: rgba(255,255,255,.2);
  font-size: 12px;
}}
.center-box {{
  background: rgba(0,0,0,.26);
  border-radius: 8px;
  padding: 12px;
  min-height: 150px;
}}
.center-box div {{ margin: 4px 0; }}
.event-line {{
  background: rgba(0,0,0,.18);
  border-radius: 6px;
  padding: 8px;
  min-height: 38px;
}}
.side-stack {{
  display: flex;
  flex-direction: column;
  gap: 12px;
  max-height: calc(100vh - 104px);
  overflow: auto;
}}
.summary {{
  display: grid;
  grid-template-columns: repeat(2, minmax(130px, 1fr));
  gap: 8px;
  margin-bottom: 12px;
}}
.metric {{
  background: #fff;
  border: 1px solid #d8dde5;
  border-radius: 6px;
  padding: 9px 10px;
}}
.metric label {{ display: block; font-size: 12px; color: #59636f; margin-bottom: 3px; }}
.metric div {{ font-size: 14px; font-weight: 700; word-break: break-word; }}
.panel {{
  background: #fff;
  border: 1px solid #d8dde5;
  border-radius: 6px;
  margin-top: 12px;
  overflow: hidden;
}}
.panel-head {{
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  padding: 10px;
  border-bottom: 1px solid #eef1f5;
}}
.panel-title {{ font-weight: 700; }}
textarea {{
  width: 100%;
  height: 230px;
  border: 0;
  resize: vertical;
  padding: 10px;
  font-family: Consolas, "Courier New", monospace;
  font-size: 12px;
  line-height: 1.45;
}}
.hint {{ font-size: 12px; color: #59636f; margin-left: 6px; }}
.tags {{ font-size: 12px; color: #39414d; line-height: 1.5; }}
.label-row {{ display: flex; flex-wrap: wrap; gap: 8px; padding: 10px; }}
.label-row button {{ min-width: 90px; }}
.label-row button.active {{ background: #1f6feb; color: #fff; border-color: #1f6feb; }}
.notes-box {{
  width: 100%;
  min-height: 78px;
  border: 1px solid #d8dde5;
  border-radius: 6px;
  padding: 8px;
  font-family: Arial, "Microsoft YaHei", sans-serif;
  font-size: 13px;
}}
.url-panel {{ display: none; }}
.url-panel.open {{ display: block; }}
@media (max-width: 850px) {{
  main {{ grid-template-columns: 1fr; }}
  aside {{ max-height: 260px; border-right: 0; border-bottom: 1px solid #d8dde5; }}
  .summary {{ grid-template-columns: repeat(2, minmax(0, 1fr)); }}
  .review-grid {{ grid-template-columns: 1fr; }}
  .native-table {{ min-height: 560px; }}
  .seat-grid {{ grid-template-columns: 1fr; grid-template-rows: none; }}
  .seat-north, .seat-west, .seat-center, .seat-east, .seat-south {{ grid-column: auto; grid-row: auto; }}
  .side-stack {{ max-height: none; }}
}}
</style>
</head>
<body>
<header>
  <h1>T1 风险切片 Review Cases</h1>
  <div class="toolbar">
    <button id="prevBtn">上一局</button>
    <button id="nextBtn">下一局</button>
    <select id="caseSelect"></select>
    <button class="primary" id="copyNagaBtn">复制 NAGA 单局 URL</button>
    <button class="primary" id="copyHanchanBtn">复制 4.1b 整半庄 URL</button>
    <button id="exportJsonBtn">导出标注 JSON</button>
    <button id="exportCsvBtn">导出标注 CSV</button>
    <span class="hint" id="statusText"></span>
  </div>
</header>
<main>
  <aside id="caseList"></aside>
  <section>
    <div class="review-grid">
      <div class="native-table">
        <div class="table-controls">
          <button id="stepStartBtn">开局</button>
          <button id="stepPrevBtn">上一步</button>
          <button id="stepNextBtn">下一步</button>
          <button id="stepEndBtn">结局</button>
          <input id="stepSlider" type="range" min="0" max="0" value="0">
          <span id="stepText"></span>
        </div>
        <div class="seat-grid" id="nativeTable"></div>
        <div class="event-line" id="eventLine"></div>
      </div>
      <div class="side-stack">
        <div class="summary" id="summary"></div>
        <div class="panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">人工标注</div>
              <div class="hint">A=明显坏模式，B=有争议，C=T1没问题，D=reviewer分歧/丢弃。</div>
            </div>
            <button id="clearLabelBtn">清空本局</button>
          </div>
          <div class="label-row" id="labelRow">
            <button data-label="A">A 坏模式</button>
            <button data-label="B">B 有争议</button>
            <button data-label="C">C 没问题</button>
            <button data-label="D">D 丢弃</button>
          </div>
          <div style="padding: 0 10px 10px;">
            <textarea class="notes-box" id="noteText" placeholder="备注：例如 reviewer 是否都认为过押、问题发生巡目、建议方向"></textarea>
          </div>
        </div>
        <div class="panel">
          <div class="panel-head">
            <div>
              <div class="panel-title">不确定时再看 URL</div>
              <div class="hint">默认在左侧牌桌标注；需要外部跑谱时展开复制。</div>
            </div>
            <button id="toggleUrlsBtn">显示/隐藏 URL</button>
          </div>
          <div id="urlPanel" class="url-panel">
            <div class="panel-head">
              <div class="panel-title">NAGA 单局 URL</div>
              <button id="selectNagaBtn">选中文本</button>
            </div>
            <textarea id="nagaText" spellcheck="false"></textarea>
            <div class="panel-head">
              <div class="panel-title">4.1b 整半庄 URL</div>
              <button id="selectHanchanBtn">选中文本</button>
            </div>
            <textarea id="hanchanText" spellcheck="false"></textarea>
          </div>
        </div>
      </div>
    </div>
  </section>
</main>
<script id="case-data" type="application/json">{html.escape(escaped_data, quote=False)}</script>
<script>
const cases = JSON.parse(document.getElementById('case-data').textContent);
let current = 0;
const caseList = document.getElementById('caseList');
const caseSelect = document.getElementById('caseSelect');
const summary = document.getElementById('summary');
const nagaText = document.getElementById('nagaText');
const hanchanText = document.getElementById('hanchanText');
const nativeTable = document.getElementById('nativeTable');
const eventLine = document.getElementById('eventLine');
const stepSlider = document.getElementById('stepSlider');
const stepText = document.getElementById('stepText');
const urlPanel = document.getElementById('urlPanel');
const statusText = document.getElementById('statusText');
const labelRow = document.getElementById('labelRow');
const noteText = document.getElementById('noteText');
const annotationKey = 't1_outcome_risk_review_annotations_v1';
let annotations = loadAnnotations();
let replayStep = 0;

function shortTags(c) {{
  return c.tags.slice(0, 3).join(', ') + (c.tags.length > 3 ? '...' : '');
}}

function renderList() {{
  caseList.innerHTML = '';
  caseSelect.innerHTML = '';
  cases.forEach((c, i) => {{
    const opt = document.createElement('option');
    opt.value = String(i);
    opt.textContent = `CASE ${{String(c.case_index).padStart(3, '0')}} | ${{c.target_seat_name}} | kyoku ${{c.kyoku_index}}`;
    caseSelect.appendChild(opt);

    const btn = document.createElement('button');
    btn.className = 'case-button' + (i === current ? ' active' : '');
    btn.innerHTML = `<div class="case-title">CASE ${{String(c.case_index).padStart(3, '0')}} · ${{c.target_seat_name}} · kyoku ${{c.kyoku_index}}</div>
      <div class="case-meta">${{c.hanchan_file}}<br>${{shortTags(c)}}</div>`;
    btn.onclick = () => showCase(i);
    caseList.appendChild(btn);
  }});
}}

function metric(label, value) {{
  return `<div class="metric"><label>${{label}}</label><div>${{value}}</div></div>`;
}}

function showCase(index) {{
  saveCurrentNote();
  current = (index + cases.length) % cases.length;
  const c = cases[current];
  caseSelect.value = String(current);
  [...caseList.children].forEach((el, i) => el.classList.toggle('active', i === current));
  summary.innerHTML = [
    metric('Case', String(c.case_index).padStart(3, '0')),
    metric('主视角', `${{c.target_model}} / ${{c.target_seat_name}} / seat ${{c.target_seat}}`),
    metric('局', `kyoku ${{c.kyoku_index}}, turns ${{c.turns}}`),
    metric('结果', `houjuu=${{c.houjuu}} / delta=${{c.delta_score}}`),
    metric('玩家', c.player_names.join(' / ')),
    metric('起始分', c.start_scores.join(' / ')),
    metric('立直/副露巡目', `${{c.first_riichi_turn ?? '-'}} / ${{c.first_fuuro_turn ?? '-'}}`),
    metric('标签', `<span class="tags">${{c.tags.join('<br>')}}</span>`),
  ].join('');
  nagaText.value = c.naga_url;
  hanchanText.value = c.hanchan_url;
  replayStep = Math.max(0, c.mjai_events.length - 1);
  stepSlider.max = String(Math.max(0, c.mjai_events.length - 1));
  renderNativeTable();
  renderAnnotation();
  statusText.textContent = '';
}}

function tileClass(tile) {{
  const text = String(tile || '');
  if (text.includes('r')) return 'tile red';
  if (/^[ESWNCFP]$/.test(text)) return 'tile honor';
  return 'tile';
}}

function tileHtml(tile, extra = '') {{
  return `<span class="${{tileClass(tile)}} ${{extra}}">${{String(tile).replace('r', '赤')}}</span>`;
}}

function removeTile(hand, pai) {{
  const exact = hand.indexOf(pai);
  if (exact >= 0) {{
    hand.splice(exact, 1);
    return;
  }}
  const normalized = String(pai).replace('r', '');
  const idx = hand.findIndex((t) => String(t).replace('r', '') === normalized);
  if (idx >= 0) hand.splice(idx, 1);
}}

function emptyState(c) {{
  return {{
    names: c.player_names,
    scores: c.start_scores.slice(),
    oya: 0,
    bakaze: '',
    kyoku: c.kyoku_index,
    honba: 0,
    kyotaku: 0,
    dora: [],
    hands: [[], [], [], []],
    discards: [[], [], [], []],
    melds: [[], [], [], []],
    riichiPending: [false, false, false, false],
    lastEvent: null,
    result: '',
  }};
}}

function applyEvent(state, event) {{
  const type = event.type;
  if (type === 'start_game') {{
    if (Array.isArray(event.names)) state.names = event.names.slice(0, 4);
    return;
  }}
  if (type === 'start_kyoku') {{
    state.oya = Number(event.oya ?? 0);
    state.bakaze = String(event.bakaze ?? '');
    state.kyoku = Number(event.kyoku ?? state.kyoku);
    state.honba = Number(event.honba ?? 0);
    state.kyotaku = Number(event.kyotaku ?? 0);
    state.scores = Array.isArray(event.scores) ? event.scores.slice(0, 4) : state.scores;
    state.dora = event.dora_marker ? [event.dora_marker] : [];
    state.hands = Array.isArray(event.tehais) ? event.tehais.map((h) => h.slice()) : state.hands;
    state.discards = [[], [], [], []];
    state.melds = [[], [], [], []];
    state.riichiPending = [false, false, false, false];
    state.result = '';
    return;
  }}
  const actor = Number(event.actor);
  if (type === 'tsumo' && actor >= 0 && actor < 4) {{
    state.hands[actor].push(event.pai);
  }} else if (type === 'dahai' && actor >= 0 && actor < 4) {{
    removeTile(state.hands[actor], event.pai);
    state.discards[actor].push({{ pai: event.pai, tsumogiri: Boolean(event.tsumogiri), reach: state.riichiPending[actor] }});
    state.riichiPending[actor] = false;
  }} else if (type === 'reach' && actor >= 0 && actor < 4) {{
    state.riichiPending[actor] = true;
  }} else if (['chi', 'pon', 'daiminkan', 'ankan', 'kakan'].includes(type) && actor >= 0 && actor < 4) {{
    const consumed = Array.isArray(event.consumed) ? event.consumed : [];
    consumed.forEach((p) => removeTile(state.hands[actor], p));
    state.melds[actor].push(`${{type}} ${{event.pai ?? ''}} ${{consumed.join(' ')}}`.trim());
  }} else if (type === 'hora') {{
    if (Array.isArray(event.deltas)) {{
      state.scores = state.scores.map((score, i) => Number(score) + Number(event.deltas[i] || 0));
    }}
    state.result = `和了 actor=${{event.actor}} target=${{event.target}} deltas=${{(event.deltas || []).join('/')}}`;
  }} else if (type === 'ryukyoku') {{
    if (Array.isArray(event.deltas)) {{
      state.scores = state.scores.map((score, i) => Number(score) + Number(event.deltas[i] || 0));
    }}
    state.result = `流局 deltas=${{(event.deltas || []).join('/')}}`;
  }}
  state.lastEvent = event;
}}

function buildState(c, step) {{
  const state = emptyState(c);
  for (let i = 0; i <= step && i < c.mjai_events.length; i += 1) {{
    applyEvent(state, c.mjai_events[i]);
  }}
  return state;
}}

function seatTitle(seat, state, c) {{
  const winds = ['东家', '南家', '西家', '北家'];
  const marker = seat === c.target_seat ? '主视角 ' : '';
  return `${{marker}}${{winds[seat]}} · ${{state.names[seat] || seat}}`;
}}

function seatCard(seat, cls, state, c) {{
  const discards = state.discards[seat].map((d) => tileHtml(d.pai, `${{d.tsumogiri ? 'tsumogiri' : ''}} ${{d.reach ? 'reach' : ''}}`)).join('');
  const hands = state.hands[seat].map((p) => tileHtml(p)).join('');
  const melds = state.melds[seat].map((m) => `<span class="meld">${{m}}</span>`).join('');
  return `<div class="seat-card ${{cls}} ${{seat === c.target_seat ? 'target' : ''}}">
    <div class="seat-head"><span class="seat-name">${{seatTitle(seat, state, c)}}</span><span>${{state.scores[seat] ?? ''}}</span></div>
    <div>手牌</div><div class="tile-row">${{hands}}</div>
    <div>副露</div><div class="tile-row">${{melds}}</div>
    <div>弃牌</div><div class="tile-row">${{discards}}</div>
  </div>`;
}}

function describeEvent(event) {{
  if (!event) return '';
  const body = Object.entries(event)
    .filter(([k]) => k !== 'type')
    .map(([k, v]) => `${{k}}=${{Array.isArray(v) ? v.join('/') : v}}`)
    .join(' ');
  return `${{event.type}} ${{body}}`;
}}

function renderNativeTable() {{
  const c = cases[current];
  replayStep = Math.max(0, Math.min(replayStep, c.mjai_events.length - 1));
  const state = buildState(c, replayStep);
  stepSlider.value = String(replayStep);
  stepText.textContent = `${{replayStep + 1}} / ${{c.mjai_events.length}}`;
  nativeTable.innerHTML = [
    seatCard(2, 'seat-north', state, c),
    seatCard(3, 'seat-west', state, c),
    `<div class="center-box seat-center">
      <div>场风: ${{state.bakaze || '-'}} ${{state.kyoku ?? ''}}局</div>
      <div>本场: ${{state.honba}} · 供托: ${{state.kyotaku}}</div>
      <div>亲: ${{state.names[state.oya] || state.oya}}</div>
      <div>宝牌: ${{state.dora.map((p) => tileHtml(p)).join('')}}</div>
      <div>结果: ${{state.result || '-'}}</div>
    </div>`,
    seatCard(1, 'seat-east', state, c),
    seatCard(0, 'seat-south', state, c),
  ].join('');
  eventLine.textContent = describeEvent(c.mjai_events[replayStep]);
}}

function caseId(c) {{
  return String(c.case_index).padStart(3, '0');
}}

function loadAnnotations() {{
  try {{
    return JSON.parse(localStorage.getItem(annotationKey) || '{{}}');
  }} catch (err) {{
    return {{}};
  }}
}}

function saveAnnotations() {{
  localStorage.setItem(annotationKey, JSON.stringify(annotations));
}}

function currentAnnotation() {{
  const id = caseId(cases[current]);
  if (!annotations[id]) {{
    annotations[id] = {{ label: '', note: '' }};
  }}
  return annotations[id];
}}

function saveCurrentNote() {{
  if (!cases.length) return;
  currentAnnotation().note = noteText.value;
  saveAnnotations();
}}

function setLabel(label) {{
  currentAnnotation().label = label;
  saveAnnotations();
  renderAnnotation();
  statusText.textContent = `CASE ${{caseId(cases[current])}} 标注为 ${{label}}`;
}}

function clearLabel() {{
  const id = caseId(cases[current]);
  delete annotations[id];
  saveAnnotations();
  renderAnnotation();
  statusText.textContent = `CASE ${{id}} 标注已清空`;
}}

function renderAnnotation() {{
  const ann = currentAnnotation();
  [...labelRow.querySelectorAll('button[data-label]')].forEach((btn) => {{
    btn.classList.toggle('active', btn.dataset.label === ann.label);
  }});
  noteText.value = ann.note || '';
}}

async function copyText(text, label) {{
  try {{
    await navigator.clipboard.writeText(text);
    statusText.textContent = `${{label}} 已复制`;
  }} catch (err) {{
    statusText.textContent = `复制失败，已选中文本，请 Ctrl+C`;
    const box = label.includes('NAGA') ? nagaText : hanchanText;
    box.focus();
    box.select();
  }}
}}

document.getElementById('prevBtn').onclick = () => showCase(current - 1);
document.getElementById('nextBtn').onclick = () => showCase(current + 1);
caseSelect.onchange = () => showCase(Number(caseSelect.value));
document.getElementById('copyNagaBtn').onclick = () => copyText(cases[current].naga_url, 'NAGA 单局 URL');
document.getElementById('copyHanchanBtn').onclick = () => copyText(cases[current].hanchan_url, '4.1b 整半庄 URL');
document.getElementById('selectNagaBtn').onclick = () => {{ nagaText.focus(); nagaText.select(); }};
document.getElementById('selectHanchanBtn').onclick = () => {{ hanchanText.focus(); hanchanText.select(); }};
document.getElementById('stepStartBtn').onclick = () => {{ replayStep = 0; renderNativeTable(); }};
document.getElementById('stepPrevBtn').onclick = () => {{ replayStep -= 1; renderNativeTable(); }};
document.getElementById('stepNextBtn').onclick = () => {{ replayStep += 1; renderNativeTable(); }};
document.getElementById('stepEndBtn').onclick = () => {{ replayStep = cases[current].mjai_events.length - 1; renderNativeTable(); }};
stepSlider.oninput = () => {{ replayStep = Number(stepSlider.value); renderNativeTable(); }};
document.getElementById('toggleUrlsBtn').onclick = () => urlPanel.classList.toggle('open');
document.getElementById('clearLabelBtn').onclick = clearLabel;
labelRow.onclick = (event) => {{
  const button = event.target.closest('button[data-label]');
  if (button) setLabel(button.dataset.label);
}};
noteText.oninput = () => {{
  currentAnnotation().note = noteText.value;
  saveAnnotations();
}};

function annotationRows() {{
  saveCurrentNote();
  return cases.map((c) => {{
    const id = caseId(c);
    const ann = annotations[id] || {{ label: '', note: '' }};
    return {{
      case: id,
      label: ann.label || '',
      note: ann.note || '',
      target_seat: c.target_seat,
      target_seat_name: c.target_seat_name,
      kyoku_index: c.kyoku_index,
      tags: c.tags.join(';'),
      hanchan_url: c.hanchan_url,
      naga_url: c.naga_url,
    }};
  }});
}}

function downloadText(filename, content, type) {{
  const blob = new Blob([content], {{ type }});
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}}

function csvEscape(value) {{
  const text = String(value ?? '');
  return '"' + text.replaceAll('"', '""') + '"';
}}

document.getElementById('exportJsonBtn').onclick = () => {{
  downloadText('t1_risk_case_annotations.json', JSON.stringify(annotationRows(), null, 2), 'application/json');
}};
document.getElementById('exportCsvBtn').onclick = () => {{
  const rows = annotationRows();
  const headers = Object.keys(rows[0]);
  const csv = [headers.join(','), ...rows.map((row) => headers.map((h) => csvEscape(row[h])).join(','))].join('\\n') + '\\n';
  downloadText('t1_risk_case_annotations.csv', csv, 'text/csv');
}};

renderList();
showCase(0);
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and not args.overwrite:
        raise SystemExit(f"output dir already exists, use --overwrite: {args.output_dir}")
    if args.output_dir.exists() and args.overwrite:
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cases = build_cases(args.case_dir)
    write_compact_files(cases, args.output_dir)
    (args.output_dir / "review_cases.html").write_text(build_html(cases), encoding="utf-8")
    print(f"cases: {len(cases)}")
    print(f"saved compact viewer: {args.output_dir / 'review_cases.html'}")


if __name__ == "__main__":
    main()
