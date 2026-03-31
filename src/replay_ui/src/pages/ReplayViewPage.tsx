// src/replay_ui/src/pages/ReplayViewPage.tsx
import { useState, useEffect, useCallback, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Loader2 } from 'lucide-react';
import type { ReplayData } from '../types/replay';
import type { DecisionLogEntry } from '../types/replay';
import { replayApi } from '../api/replayApi';
import { actionComparableKey, actionLabel, sameReplayAction } from '../utils/tileUtils';
import { CN_BAKAZE, SEAT_NAMES_CN } from '../utils/constants';
import { normalizeReplayPlayerNames, replayPlayerDisplayName } from '../utils/replayNames';

const TILE_BASE = '/tiles';
const TILE_SVG: Record<string, string> = {
  '1m':'Man1','2m':'Man2','3m':'Man3','4m':'Man4','5m':'Man5','6m':'Man6','7m':'Man7','8m':'Man8','9m':'Man9',
  '1p':'Pin1','2p':'Pin2','3p':'Pin3','4p':'Pin4','5p':'Pin5','6p':'Pin6','7p':'Pin7','8p':'Pin8','9p':'Pin9',
  '1s':'Sou1','2s':'Sou2','3s':'Sou3','4s':'Sou4','5s':'Sou5','6s':'Sou6','7s':'Sou7','8s':'Sou8','9s':'Sou9',
  '5mr':'Man5-Dora','5pr':'Pin5-Dora','5sr':'Sou5-Dora',
  'E':'Ton','S':'Nan','W':'Shaa','N':'Pei','P':'Haku','F':'Hatsu','C':'Chun',
};
const _TILE_ORDER: Record<string, number> = {
  '1m':0,'2m':1,'3m':2,'4m':3,'5m':4,'6m':5,'7m':6,'8m':7,'9m':8,'5mr':4,
  '1p':9,'2p':10,'3p':11,'4p':12,'5p':13,'6p':14,'7p':15,'8p':16,'9p':17,'5pr':13,
  '1s':18,'2s':19,'3s':20,'4s':21,'5s':22,'6s':23,'7s':24,'8s':25,'9s':26,'5sr':22,
  'E':27,'S':28,'W':29,'N':30,'P':31,'F':32,'C':33,
};
const MAX_BAR_H = 57;

function tileUrl(name: string) {
  return `${TILE_BASE}/${TILE_SVG[name] || 'Blank'}.svg`;
}

interface TileWithMeta {
  name: string;
  logit?: number;
  minLogit: number;
  logitRange: number;
  isTsumo: boolean;
}

function buildSortedTiles(entry: DecisionLogEntry): TileWithMeta[] {
  const hand = entry.hand || [];
  const tsumo_pai = entry.tsumo_pai || null;
  const tileLogit: Record<string, number> = {};
  let minLogit = 0, maxLogit = 1;
  if (entry.candidates && entry.candidates.length) {
    entry.candidates.forEach(c => {
      if (c.action && c.action.type === 'dahai' && c.action.pai) {
        tileLogit[c.action.pai] = c.logit;
      }
    });
    const vals = Object.values(tileLogit);
    if (vals.length > 1) { minLogit = Math.min(...vals); maxLogit = Math.max(...vals); }
    else if (vals.length === 1) { minLogit = maxLogit = vals[0]; }
  }
  const logitRange = Math.max(maxLogit - minLogit, 0.001);
  const others = hand.filter(t => t !== tsumo_pai);
  others.sort((a, b) => (_TILE_ORDER[a] ?? 99) - (_TILE_ORDER[b] ?? 99));
  const sorted = tsumo_pai ? [...others, tsumo_pai] : others;
  return sorted.map(name => ({
    name,
    logit: tileLogit[name],
    minLogit,
    logitRange,
    isTsumo: name === tsumo_pai,
  }));
}

function barHeight(tile: TileWithMeta): number {
  if (tile.logit === undefined) return 0;
  const pct = Math.max(1, (tile.logit - tile.minLogit) / tile.logitRange * 100);
  return Math.max(3, Math.round(pct / 100 * MAX_BAR_H));
}

function barColor(tile: TileWithMeta, entry: DecisionLogEntry) {
  const c = entry.chosen, g = entry.gt_action;
  if (tile.name === c?.pai && tile.name === g?.pai) return '#8e44ad';
  if (tile.name === c?.pai) return '#e74c3c';
  if (tile.name === g?.pai) return '#27ae60';
  return 'var(--accent)';
}

function tileCssClass(tile: TileWithMeta, entry: DecisionLogEntry): string {
  const c = entry.chosen, g = entry.gt_action;
  const b = tile.name === c?.pai, gg = tile.name === g?.pai;
  if (b && gg) return 'is-both';
  if (b) return 'is-bot';
  if (gg) return 'is-gt';
  return '';
}

// ---------------------------------------------------------------------------
// 候选评分表
// ---------------------------------------------------------------------------
function CandidateTable({
  candidates,
  chosen,
  gtAction,
}: {
  candidates: Array<{ action: import('../types/replay').Action; logit: number; beam_score?: number }>;
  chosen: import('../types/replay').Action | null;
  gtAction: import('../types/replay').Action | null;
}) {
  const hasBeam = candidates.some(c => c.beam_score !== undefined);
  const actKey = (a: import('../types/replay').Action | null) => actionComparableKey(a);
  const chosenKey = actKey(chosen);
  const gtKey = actKey(gtAction);
  return (
    <div style={{ marginTop: 6, fontSize: 11, overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', width: '100%' }}>
        <thead>
          <tr style={{ background: '#f1f5f9' }}>
            <th style={{ padding: '2px 8px', textAlign: 'left', color: '#475569', borderBottom: '1px solid #e2e8f0' }}>动作</th>
            <th style={{ padding: '2px 8px', textAlign: 'right', color: '#475569', borderBottom: '1px solid #e2e8f0', fontFamily: 'monospace' }}>Logit</th>
            {hasBeam && <th style={{ padding: '2px 8px', textAlign: 'right', color: '#475569', borderBottom: '1px solid #e2e8f0', fontFamily: 'monospace' }}>Beam</th>}
            <th style={{ padding: '2px 8px', borderBottom: '1px solid #e2e8f0' }}></th>
          </tr>
        </thead>
        <tbody>
          {candidates.map((c, i) => {
            const key = actKey(c.action);
            const isBot = key === chosenKey;
            const isGt  = key === gtKey && gtKey !== '';
            const isBoth = isBot && isGt;
            const bg = isBoth ? '#fdf4ff' : isBot ? '#fff0f0' : isGt ? '#f0fff4' : (i % 2 === 0 ? '#fff' : '#f9fafb');
            const color = isBoth ? '#6b21a8' : isBot ? '#c0392b' : isGt ? '#166534' : '#374151';
            const marks = (isBot ? '✓Bot ' : '') + (isGt ? '★玩家' : '');
            return (
              <tr key={i} style={{ background: bg, color, fontWeight: isBot || isBoth ? 600 : 400 }}>
                <td style={{ padding: '2px 8px', borderBottom: '1px solid #f1f5f9' }}>{actionLabel(c.action)}</td>
                <td style={{ padding: '2px 8px', textAlign: 'right', fontFamily: 'monospace', borderBottom: '1px solid #f1f5f9' }}>{c.logit >= 0 ? '+' : ''}{c.logit.toFixed(3)}</td>
                {hasBeam && <td style={{ padding: '2px 8px', textAlign: 'right', fontFamily: 'monospace', borderBottom: '1px solid #f1f5f9' }}>{c.beam_score !== undefined ? (c.beam_score >= 0 ? '+' : '') + c.beam_score.toFixed(3) : '—'}</td>}
                <td style={{ padding: '2px 8px', fontSize: 10, color: '#666', borderBottom: '1px solid #f1f5f9', whiteSpace: 'nowrap' }}>{marks}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 统计面板
// ---------------------------------------------------------------------------
export function StatsPanel({ data, onClose }: { data: ReplayData; onClose: () => void }) {
  const log = data.log.filter(e => !e.is_obs);
  const total = log.length;
  const match = log.filter((e) => sameReplayAction(e.chosen, e.gt_action)).length;
  const pct = total ? (match / total * 100).toFixed(1) : '0.0';
  const dahaiCount = log.filter(e => e.chosen?.type === 'dahai').length;
  const dahaiRate = total ? ((dahaiCount / total) * 100).toFixed(0) : '0';

  const byType: Record<string, { total: number; match: number }> = {};
  log.forEach(e => {
    const t = e.chosen?.type || '?';
    if (!byType[t]) byType[t] = { total: 0, match: 0 };
    byType[t].total++;
    if (sameReplayAction(e.chosen, e.gt_action)) byType[t].match++;
  });
  const labels: Record<string, string> = { dahai:'打牌', none:'过', reach:'立直', chi:'吃', pon:'碰', daiminkan:'大明杠', ankan:'暗杠', kakan:'加杠', hora:'胡', ryukyoku:'流局' };

  const byTile: Record<string, { total: number; match: number }> = {};
  log.forEach(e => {
    if (e.chosen?.type !== 'dahai') return;
    const t = e.chosen?.pai || '?';
    if (!byTile[t]) byTile[t] = { total: 0, match: 0 };
    byTile[t].total++;
    if (e.gt_action?.type === 'dahai' && sameReplayAction(e.chosen, e.gt_action)) byTile[t].match++;
  });

  return (
    <>
      <div className="stats-overlay open" onClick={onClose} />
      <div className="stats-panel open">
        <div className="stats-header">
          <span>决策统计</span>
          <button className="stats-close" onClick={onClose}>×</button>
        </div>
        <div className="stats-body">
          <div className="match-rate-bar">
            <div className="pct">{pct}%</div>
            <div className="sub">Bot 与玩家一致率 ({match} / {total})</div>
          </div>

          {data.rating !== null && data.rating !== undefined && (
            <div style={{ textAlign: 'center', margin: '8px 0 4px' }}>
              <span style={{ fontSize: 28, fontWeight: 700, color: data.rating >= 80 ? '#27ae60' : data.rating >= 60 ? '#3498db' : '#e74c3c' }}>
                {data.rating.toFixed(1)}
              </span>
              <span style={{ fontSize: 13, color: '#64748b', marginLeft: 4 }}>/ 100 Rating</span>
              <div style={{ fontSize: 11, color: '#94a3b8', marginTop: 2 }}>基于 logit 差距的相似度评分</div>
            </div>
          )}

          <div className="stats-summary">
            {[
              { val: total, lbl: '总决策数' },
              { val: match, lbl: '匹配数' },
              { val: data.kyoku_order?.length || 0, lbl: '总局数' },
              { val: `${dahaiRate}%`, lbl: '打牌占比' },
            ].map(item => (
              <div key={item.lbl} className="stats-card">
                <div className="val">{item.val}</div>
                <div className="lbl">{item.lbl}</div>
              </div>
            ))}
          </div>

          {Object.entries(byType).length > 0 && (
            <div className="stats-section-title">各动作准确率</div>
          )}
          {Object.entries(byType).map(([t, v]) => {
            const p = v.total ? Math.round(v.match / v.total * 100) : 0;
            const barClr = p > 70 ? '#27ae60' : p > 40 ? '#3498db' : '#e74c3c';
            return (
              <div key={t} className="stats-row">
                <span className="lbl">{labels[t] || t}</span>
                <div className="stats-bar-bg">
                  <div className="stats-bar-fg" style={{ width: `${p}%`, background: barClr }}>{p}%</div>
                </div>
                <span className="val">{v.match}/{v.total}</span>
              </div>
            );
          })}

          {Object.keys(byTile).length > 0 && (
            <div className="stats-section-title">打牌详情 Top10</div>
          )}
          {Object.entries(byTile).sort((a, b) => b[1].total - a[1].total).slice(0, 10).map(([t, v]) => {
            const p = v.total ? Math.round(v.match / v.total * 100) : 0;
            const barClr = p > 70 ? '#27ae60' : p > 40 ? '#3498db' : '#e74c3c';
            return (
              <div key={t} className="stats-row">
                <span className="lbl">{t}</span>
                <div className="stats-bar-bg">
                  <div className="stats-bar-fg" style={{ width: `${p}%`, background: barClr }}>{p}%</div>
                </div>
                <span className="val">{v.match}/{v.total}</span>
              </div>
            );
          })}
        </div>
      </div>
    </>
  );
}

// ---------------------------------------------------------------------------
// 单步卡片
// ---------------------------------------------------------------------------
function StepCard({ entry, playerNames }: { entry: DecisionLogEntry; playerNames: string[] }) {
  // obs 步：简化渲染
  if (entry.is_obs) {
    const k = entry.kyoku_key || entry;
    const kyokuLabel = `${CN_BAKAZE[k.bakaze] || k.bakaze}${k.kyoku}局 ${k.honba}本场`;
    const actor = entry.actor_to_move ?? entry.chosen?.actor ?? '?';
    return (
      <div className="step-card" style={{ opacity: 0.6, borderLeft: '3px solid #3498db' }}>
        <div className="step-top">
          <div className="step-header">{kyokuLabel} · Step {entry.step}</div>
          <div className="step-meta-right">
            <span className="action-chip none">
              {replayPlayerDisplayName(playerNames, Number(actor))}: {actionLabel(entry.chosen)}
            </span>
          </div>
        </div>
      </div>
    );
  }

  const tiles = buildSortedTiles(entry);
  const isDahai = entry.chosen?.type === 'dahai' || entry.chosen?.type === 'none';

  const chosenClass = (() => {
    const c = entry.chosen;
    if (!c) return 'none';
    const g = entry.gt_action;
    if (g && c.type === g.type && c.pai === g.pai) return 'both';
    if (c.type === 'dahai' || c.type === 'none') return 'bot';
    return 'gt';
  })();

  const gtClass = (() => {
    const c = entry.chosen, g = entry.gt_action;
    if (!g) return 'none';
    if (c && c.type === g.type && c.pai === g.pai) return 'both';
    return 'gt';
  })();

  const isMatch = sameReplayAction(entry.chosen, entry.gt_action);

  const k = entry.kyoku_key || entry;
  const kyokuLabel = `${CN_BAKAZE[k.bakaze] || k.bakaze}${k.kyoku}局 ${k.honba}本场`;

  const chipClass: Record<string, string> = {
    bot: 'action-chip bot',
    gt: 'action-chip gt',
    both: 'action-chip both',
    none: 'action-chip none',
  };

  return (
    <div className="step-card">
      {/* 顶栏 */}
      <div className="step-top">
        <div className="step-header">{kyokuLabel} · Step {entry.step}</div>
        <div className="step-meta-right">
          {entry.scores.map((s, i) => (
            <span
              key={i}
              className={`score-chip${entry.reached[i] ? ' reached' : ''}`}
            >
              P{i}{entry.reached[i] ? 'R' : ''}:{s}
            </span>
          ))}
          {entry.dora_markers?.length > 0 && (
            <span style={{ fontSize: 11, color: '#9ca3af', marginLeft: 4 }}>
              宝:{entry.dora_markers.map(d => (
                <img key={d} src={tileUrl(d)} width={20} className="dora" style={{ marginRight: 3 }} />
              ))}
            </span>
          )}
          <span className={chipClass[chosenClass]}>
            Bot: {actionLabel(entry.chosen)}
          </span>
          <span className={chipClass[gtClass]}>
            实际: {entry.gt_action ? actionLabel(entry.gt_action) : '—'}
          </span>
          {isMatch && (
            <span style={{ fontSize: 11, color: 'var(--success)', fontWeight: 600 }}>✓ Match</span>
          )}
        </div>
      </div>

      {/* 手牌 + 指示条 */}
      {isDahai && (
        <div className="hand-row">
          {tiles.map(tile => {
            const h = barHeight(tile);
            const cls = tileCssClass(tile, entry);
            const rClass = h >= 5 ? 'bar r5' : 'bar r3';
            return (
              <div key={tile.name} className={`tile-slot ${cls}${tile.isTsumo ? ' is-tsumo' : ''}`}>
                {tile.isTsumo && (
                  <span
                    style={{
                      position: 'absolute',
                      top: -14,
                      left: '50%',
                      transform: 'translateX(-50%)',
                      fontSize: 9,
                      fontWeight: 700,
                      color: 'var(--warning)',
                      background: 'var(--gold-bg)',
                      border: '1px solid var(--gold-border)',
                      borderRadius: 3,
                      padding: '0 3px',
                      lineHeight: 1.4,
                    }}
                  >
                    摸
                  </span>
                )}
                {h > 0 && (
                  <div className={`bar-wrap ${cls}`}>
                    <div className="bar-label">{tile.logit?.toFixed(2)}</div>
                    <div className={rClass} style={{ height: h, background: barColor(tile, entry) }} />
                  </div>
                )}
                <img
                  src={tileUrl(tile.name)}
                  width={38}
                  className="tile-img"
                  style={{
                    display: 'block',
                    borderRadius: 3,
                    border: cls ? `2px solid ${cls === 'is-bot' ? '#e74c3c' : cls === 'is-gt' ? '#27ae60' : '#8e44ad'}` : '1px solid #9ca3af',
                  }}
                />
              </div>
            );
          })}
        </div>
      )}

      {/* dahai 候选评分表 */}
      {isDahai && entry.candidates && entry.candidates.length > 0 && (
        <CandidateTable candidates={entry.candidates} chosen={entry.chosen} gtAction={entry.gt_action} />
      )}

      {/* 非 dahai */}
      {!isDahai && (
        <div className="nondahai-box">
          <div>Bot: <b>{entry.chosen?.type === 'none' ? '过' : actionLabel(entry.chosen)}</b></div>
          {entry.gt_action && (
            <div style={{ color: '#166534' }}>实际: <b>{entry.gt_action.type === 'none' ? '过' : actionLabel(entry.gt_action)}</b></div>
          )}
        </div>
      )}

      {/* 非 dahai 候选评分表 */}
      {!isDahai && entry.candidates && entry.candidates.length > 0 && (
        <CandidateTable candidates={entry.candidates} chosen={entry.chosen} gtAction={entry.gt_action} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 主页面
// ---------------------------------------------------------------------------
export function ReplayViewPage() {
  const location = useLocation();
  const navigate = useNavigate();

  const [data, setData] = useState<ReplayData | null>(null);
  const [curKyoku, setCurKyoku] = useState(0);
  const [showStats, setShowStats] = useState(false);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const stepRefs = useRef<Map<number, HTMLDivElement>>(new Map());
  const [scrollToStep, setScrollToStep] = useState<number | null>(null);
  const focusedStep = useRef<number>(-1);

  useEffect(() => {
    const state = location.state as { replayData?: ReplayData; replayId?: string } | null;
    if (state?.replayData) {
      setData(state.replayData);
      setLoading(false);
    } else if (state?.replayId) {
      replayApi.get(state.replayId).then(d => { setData(d); setLoading(false); }).catch(e => { setError(String(e)); setLoading(false); });
    } else {
      const params = new URLSearchParams(location.search);
      const replayId = params.get('id');
      if (replayId) {
        replayApi.get(replayId).then(d => { setData(d); setLoading(false); }).catch(e => { setError(String(e)); setLoading(false); });
      } else {
        setError('未找到回放数据，请从首页上传牌谱');
        setLoading(false);
      }
    }
  }, [location]);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      if (e.key === 'ArrowLeft' || e.key === 'h') { e.preventDefault(); setCurKyoku(c => Math.max(0, c - 1)); }
      if (e.key === 'ArrowRight' || e.key === 'l') { e.preventDefault(); if (data) setCurKyoku(c => Math.min(data.kyoku_order.length - 1, c + 1)); }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [data]);

  const visibleSteps = data ? data.log.filter(e => {
    const k = e.kyoku_key || e;
    const ko = data.kyoku_order[curKyoku];
    return k.bakaze === ko?.bakaze && k.kyoku === ko?.kyoku && k.honba === ko?.honba;
  }) : [];

  const isDiff = useCallback((e: DecisionLogEntry) =>
    data !== null &&
    !e.is_obs &&
    e.actor_to_move === data.player_id &&
    e.gt_action !== null &&
    !sameReplayAction(e.chosen, e.gt_action)
  , [data]);

  const jumpToPrevDiff = useCallback(() => {
    if (!data) return;
    const from = focusedStep.current >= 0 ? focusedStep.current - 1 : (data.log.length - 1);
    for (let i = from; i >= 0; i--) {
      const e = data.log[i];
      if (!isDiff(e)) continue;
      const ki = data.kyoku_order.findIndex(
        k => k.bakaze === e.kyoku_key.bakaze && k.kyoku === e.kyoku_key.kyoku && k.honba === e.kyoku_key.honba
      );
      if (ki < 0) continue;
      focusedStep.current = e.step;
      setCurKyoku(ki);
      setScrollToStep(e.step);
      return;
    }
  }, [data, isDiff]);

  const jumpToNextDiff = useCallback(() => {
    if (!data) return;
    const from = focusedStep.current >= 0 ? focusedStep.current + 1 : 0;
    for (let i = from; i < data.log.length; i++) {
      const e = data.log[i];
      if (!isDiff(e)) continue;
      const ki = data.kyoku_order.findIndex(
        k => k.bakaze === e.kyoku_key.bakaze && k.kyoku === e.kyoku_key.kyoku && k.honba === e.kyoku_key.honba
      );
      if (ki < 0) continue;
      focusedStep.current = e.step;
      setCurKyoku(ki);
      setScrollToStep(e.step);
      return;
    }
  }, [data, isDiff]);

  // scroll 到目标 step（在 curKyoku 切换并重新渲染后执行）
  useEffect(() => {
    if (scrollToStep === null) return;
    // defer until after DOM paint so new kyoku's cards are mounted
    const id = setTimeout(() => {
      const el = stepRefs.current.get(scrollToStep);
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
      setScrollToStep(null);
    }, 50);
    return () => clearTimeout(id);
  }, [scrollToStep, curKyoku]);



  const handleExportHtml = useCallback(() => {
    if (!data) return;
    const form = new FormData();
    form.append('data', JSON.stringify(data));
    fetch('/api/export-html', { method: 'POST', body: form })
      .then(r => r.blob())
      .then(blob => {
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'replay.html';
        a.click();
        URL.revokeObjectURL(url);
      });
  }, [data]);

  if (loading) {
    return (
      <div style={{ background: 'var(--page-bg)', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
        <Loader2 size={32} className="animate-spin" style={{ color: 'var(--accent)' }} />
        <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>加载回放数据中...</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div style={{ background: 'var(--page-bg)', height: '100%', display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
        <p style={{ color: 'var(--error)', fontSize: 14 }}>{error || '未找到回放数据'}</p>
        <button onClick={() => navigate('/')} style={{ color: 'var(--accent)', fontSize: 14, background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline' }}>
          返回首页
        </button>
      </div>
    );
  }

  const numKyoku = data.kyoku_order.length;
  const ko = data.kyoku_order[curKyoku];
  const kyokuLabel = ko ? `第${CN_BAKAZE[ko.bakaze] || ko.bakaze}${ko.kyoku}局 ${ko.honba}本场 (${curKyoku + 1}/${numKyoku})` : '';
  const pid = data.player_id || 0;
  const playerNames = normalizeReplayPlayerNames(data);

  return (
    <div style={{ background: 'var(--page-bg)', height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* 导航栏 */}
      <div className="nav-bar">
        <span className="nav-title">{SEAT_NAMES_CN[pid]}视角 · Replay Review</span>

        <button
          className="btn"
          onClick={() => setCurKyoku(c => Math.max(0, c - 1))}
          disabled={curKyoku === 0}
        >
          ◀ 上一局
        </button>
        <span style={{ color: 'rgba(255,255,255,0.85)', fontSize: 13, fontWeight: 600, minWidth: 180, textAlign: 'center' }}>
          {kyokuLabel}
        </span>
        <button
          className="btn"
          onClick={() => setCurKyoku(c => Math.min(numKyoku - 1, c + 1))}
          disabled={curKyoku === numKyoku - 1}
        >
          下一局 ▶
        </button>

        <span style={{ marginLeft: 8, display: 'flex', gap: 8 }}>
          <button className="btn" onClick={jumpToPrevDiff} title="上一个与Bot不同的决策">⏮差异</button>
          <button className="btn" onClick={jumpToNextDiff} title="下一个与Bot不同的决策">差异⏭</button>
          <button className="btn" onClick={() => setShowStats(true)}>📊统计</button>
          <button className="btn" onClick={handleExportHtml}>💾导出</button>
          <button className="btn" onClick={() => navigate('/game-replay', { state: { replayData: data } })} title="切换到牌桌视图">🀄牌桌</button>
        </span>

        <button className="btn" onClick={() => navigate('/')}>🏠</button>
        <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)', marginLeft: 8, display: 'none' }}>←/→ 换局</span>
      </div>

      {/* 统计面板 */}
      {showStats && <StatsPanel data={data} onClose={() => setShowStats(false)} />}

      {/* 步列表 */}
      <div style={{ flex: 1, overflowY: 'auto', padding: '12px 16px' }}>
        <div style={{ maxWidth: 860, margin: '0 auto' }}>
          {visibleSteps.map(entry => (
            <div key={entry.step} ref={el => { if (el) stepRefs.current.set(entry.step, el); else stepRefs.current.delete(entry.step); }}>
              <StepCard entry={entry} playerNames={playerNames} />
            </div>
          ))}
          {visibleSteps.length === 0 && (
            <p style={{ textAlign: 'center', color: '#9ca3af', fontSize: 14, padding: '48px 0' }}>
              该局暂无数据
            </p>
          )}
        </div>
      </div>
    </div>
  );
}
