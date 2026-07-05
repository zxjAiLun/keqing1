// src/replay_ui/src/pages/ReplayViewPage.tsx
import { useState, useEffect, useCallback, useRef } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Loader2 } from 'lucide-react';
import type { ReplayData } from '../types/replay';
import type { DecisionLogEntry } from '../types/replay';
import { replayApi } from '../api/replayApi';
import { actionLabel, isReplayPlayerDecision, isReplayReviewDiffForPlayer, sameReplayAction } from '../utils/tileUtils';
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
  prob?: number;
  minLogit: number;
  logitRange: number;
  isTsumo: boolean;
}

function softmaxProbabilities(scores: number[]): number[] {
  if (scores.length === 0) return [];
  const maxScore = Math.max(...scores);
  const exps = scores.map((score) => Math.exp(score - maxScore));
  const total = exps.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) return scores.map(() => 0);
  return exps.map((value) => value / total);
}

function candidateScore(c: { logit: number; beam_score?: number; final_score?: number }): number {
  return c.final_score ?? c.beam_score ?? c.logit;
}

function candidateProbabilities(candidates: Array<{ logit: number; beam_score?: number; final_score?: number; prob?: number }>): number[] {
  const fallback = softmaxProbabilities(candidates.map(candidateScore));
  return candidates.map((candidate, idx) => (
    typeof candidate.prob === 'number' && Number.isFinite(candidate.prob)
      ? candidate.prob
      : fallback[idx] ?? 0
  ));
}

function finiteNumber(value: unknown): number | null {
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function buildSortedTiles(entry: DecisionLogEntry): TileWithMeta[] {
  const hand = entry.hand || [];
  const tsumo_pai = entry.tsumo_pai || null;
  const tileLogit: Record<string, number> = {};
  const tileProb: Record<string, number> = {};
  let minLogit = 0, maxLogit = 1;
  if (entry.candidates && entry.candidates.length) {
    const probs = candidateProbabilities(entry.candidates);
    entry.candidates.forEach((c, idx) => {
      if (c.action && c.action.type === 'dahai' && c.action.pai) {
        tileLogit[c.action.pai] = c.logit;
        tileProb[c.action.pai] = probs[idx] ?? 0;
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
    prob: tileProb[name],
    minLogit,
    logitRange,
    isTsumo: name === tsumo_pai,
  }));
}

function barHeight(tile: TileWithMeta): number {
  if (tile.logit === undefined) return 0;
  const pct = tile.prob !== undefined
    ? Math.max(1, tile.prob * 100)
    : Math.max(1, (tile.logit - tile.minLogit) / tile.logitRange * 100);
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
  candidates: Array<{ action: import('../types/replay').Action; logit: number; beam_score?: number; final_score?: number; prob?: number }>;
  chosen: import('../types/replay').Action | null;
  gtAction: import('../types/replay').Action | null;
}) {
  const hasFinal = candidates.some(c => c.final_score !== undefined);
  const hasBeam = candidates.some(c => c.beam_score !== undefined);
  const probs = candidateProbabilities(candidates);
  return (
    <div style={{ marginTop: 6, fontSize: 11, overflowX: 'auto' }}>
      <table style={{ borderCollapse: 'collapse', width: '100%' }}>
        <thead>
          <tr style={{ background: '#f1f5f9' }}>
            <th style={{ padding: '2px 8px', textAlign: 'left', color: '#475569', borderBottom: '1px solid #e2e8f0' }}>动作</th>
            <th style={{ padding: '2px 8px', textAlign: 'right', color: '#475569', borderBottom: '1px solid #e2e8f0', fontFamily: 'monospace' }}>Logit</th>
            {hasFinal && <th style={{ padding: '2px 8px', textAlign: 'right', color: '#475569', borderBottom: '1px solid #e2e8f0', fontFamily: 'monospace' }}>Final</th>}
            {hasBeam && <th style={{ padding: '2px 8px', textAlign: 'right', color: '#475569', borderBottom: '1px solid #e2e8f0', fontFamily: 'monospace' }}>Beam</th>}
            <th style={{ padding: '2px 8px', textAlign: 'right', color: '#475569', borderBottom: '1px solid #e2e8f0', fontFamily: 'monospace' }}>P</th>
            <th style={{ padding: '2px 8px', borderBottom: '1px solid #e2e8f0' }}></th>
          </tr>
        </thead>
        <tbody>
          {candidates.map((c, i) => {
            const isBot = sameReplayAction(c.action, chosen);
            const isGt  = sameReplayAction(c.action, gtAction);
            const isBoth = isBot && isGt;
            const bg = isBoth ? '#fdf4ff' : isBot ? '#fff0f0' : isGt ? '#f0fff4' : (i % 2 === 0 ? '#fff' : '#f9fafb');
            const color = isBoth ? '#6b21a8' : isBot ? '#c0392b' : isGt ? '#166534' : '#374151';
            const marks = (isBot ? '✓Bot ' : '') + (isGt ? '★玩家' : '');
            return (
              <tr key={i} style={{ background: bg, color, fontWeight: isBot || isBoth ? 600 : 400 }}>
                <td style={{ padding: '2px 8px', borderBottom: '1px solid #f1f5f9' }}>{actionLabel(c.action)}</td>
                <td style={{ padding: '2px 8px', textAlign: 'right', fontFamily: 'monospace', borderBottom: '1px solid #f1f5f9' }}>{c.logit >= 0 ? '+' : ''}{c.logit.toFixed(3)}</td>
                {hasFinal && <td style={{ padding: '2px 8px', textAlign: 'right', fontFamily: 'monospace', borderBottom: '1px solid #f1f5f9' }}>{c.final_score !== undefined ? (c.final_score >= 0 ? '+' : '') + c.final_score.toFixed(3) : '—'}</td>}
                {hasBeam && <td style={{ padding: '2px 8px', textAlign: 'right', fontFamily: 'monospace', borderBottom: '1px solid #f1f5f9' }}>{c.beam_score !== undefined ? (c.beam_score >= 0 ? '+' : '') + c.beam_score.toFixed(3) : '—'}</td>}
                <td style={{ padding: '2px 8px', textAlign: 'right', fontFamily: 'monospace', borderBottom: '1px solid #f1f5f9' }}>{((probs[i] ?? 0) * 100).toFixed(1)}%</td>
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
  const log = data.log.filter(e => isReplayPlayerDecision(e, data.player_id));
  const fallbackTotal = log.length;
  const fallbackMatch = log.filter((e) => sameReplayAction(e.chosen, e.gt_action)).length;
  const teacherStats = (() => {
    const stats = new Map<string, {
      model: string;
      total: number;
      match: number;
      badMove: number;
      ratingScores: number[];
      similarityScores: number[];
    }>();

    const ensure = (model: string) => {
      const key = model || 'model';
      let item = stats.get(key);
      if (!item) {
        item = { model: key, total: 0, match: 0, badMove: 0, ratingScores: [], similarityScores: [] };
        stats.set(key, item);
      }
      return item;
    };

    for (const entry of log) {
      const reviews = entry.teacher_reviews && entry.teacher_reviews.length > 0
        ? entry.teacher_reviews
        : entry.teacher_review ? [entry.teacher_review] : [];
      for (const review of reviews) {
        const model = review.model || 'model';
        const item = ensure(model);
        const actual = review.actual_action ?? entry.gt_action;
        const expected = review.expected_action ?? review.top1?.action ?? null;
        const isImplicitPass = entry.gt_action == null && actual?.type === 'none';
        if (actual && expected && !isImplicitPass) {
          item.total += 1;
          if (sameReplayAction(expected, actual)) item.match += 1;
        }

        if (!actual || isImplicitPass) continue;
        const qValues: number[] = [];
        let actualQ = finiteNumber(review.actual_q);
        let actualProb = finiteNumber(review.actual_prob);
        let expectedProb = finiteNumber(review.expected_prob ?? review.top1?.prob ?? review.best_prob);
        for (const candidate of entry.candidates ?? []) {
          const teachers = candidate.teachers ?? (candidate.teacher ? [candidate.teacher] : []);
          const teacher = teachers.find((value) => value.model === model);
          const q = finiteNumber(teacher?.q_value);
          const prob = finiteNumber(teacher?.prob);
          if (q !== null) {
            qValues.push(q);
            if (actualQ === null && sameReplayAction(candidate.action, actual)) {
              actualQ = q;
            }
          }
          if (prob !== null) {
            if (actualProb === null && sameReplayAction(candidate.action, actual)) {
              actualProb = prob;
            }
            if (expected && expectedProb === null && sameReplayAction(candidate.action, expected)) {
              expectedProb = prob;
            }
          }
        }
        if (expected && actualProb !== null && expectedProb !== null) {
          const penalty = sameReplayAction(expected, actual) ? 0 : Math.abs(actualProb - expectedProb);
          item.similarityScores.push(Math.max(0, Math.min(1, 1 - penalty)));
          if (!sameReplayAction(expected, actual) && actualProb < 0.05) {
            item.badMove += 1;
          }
        }
        if (actualQ === null || qValues.length < 2) continue;
        if (!qValues.some((value) => value === actualQ)) qValues.push(actualQ);
        const minQ = Math.min(...qValues);
        const maxQ = Math.max(...qValues);
        const range = maxQ - minQ;
        if (range <= 0) continue;
        item.ratingScores.push((actualQ - minQ) / range);
      }
    }

    const modelOrder: Record<string, number> = { v4: 0, '70k.pth': 1, 'T1@71000': 2, 'gui_mortal.pth': 3 };
    return Array.from(stats.values()).map((item) => {
      const pct = item.total ? item.match / item.total * 100 : 0;
      const rating = item.ratingScores.length
        ? Math.round(1000 * 100 * Math.pow(item.ratingScores.reduce((sum, value) => sum + value, 0) / item.ratingScores.length, 2)) / 1000
        : null;
      const similarity = item.similarityScores.length
        ? item.similarityScores.reduce((sum, value) => sum + value, 0) / item.similarityScores.length * 100
        : null;
      return {
        ...item,
        pct,
        rating: rating === null ? null : Math.round(rating * 10) / 10,
        similarity: similarity === null ? null : Math.round(similarity * 10) / 10,
        badMoveRate: item.total ? item.badMove / item.total * 100 : 0,
      };
    }).sort((left, right) => (modelOrder[left.model] ?? 100) - (modelOrder[right.model] ?? 100));
  })();

  const hasTeacherStats = teacherStats.length > 0;
  const fallbackPct = fallbackTotal ? fallbackMatch / fallbackTotal * 100 : 0;

  return (
    <>
      <div className="stats-overlay open" onClick={onClose} />
      <div className="stats-panel open">
        <div className="stats-header">
          <span>决策统计</span>
          <button className="stats-close" onClick={onClose}>×</button>
        </div>
        <div className="stats-body">
          <div className="stats-summary">
            {[
              { val: hasTeacherStats ? teacherStats.length : 1, lbl: 'Review 模型数' },
              { val: fallbackTotal, lbl: '总决策数' },
              { val: data.kyoku_order?.length || 0, lbl: '总局数' },
            ].map(item => (
              <div key={item.lbl} className="stats-card">
                <div className="val">{item.val}</div>
                <div className="lbl">{item.lbl}</div>
              </div>
            ))}
          </div>

          <div className="stats-section-title">模型 Review 统计</div>
          <table style={{
            width: '100%',
            borderCollapse: 'collapse',
            fontSize: 13,
          }}>
            <thead>
              <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)' }}>
                <th style={statsThStyle}>模型</th>
                <th style={{ ...statsThStyle, textAlign: 'right' }}>类似度</th>
                <th style={{ ...statsThStyle, textAlign: 'right' }}>一致率</th>
                <th style={{ ...statsThStyle, textAlign: 'right' }}>恶手率</th>
                <th style={{ ...statsThStyle, textAlign: 'right' }}>Rating</th>
                <th style={{ ...statsThStyle, textAlign: 'right' }}>Match</th>
                <th style={{ ...statsThStyle, textAlign: 'right' }}>Total</th>
              </tr>
            </thead>
            <tbody>
              {(hasTeacherStats ? teacherStats : [{
                model: data.model_label || data.bot_type || 'Bot',
                total: fallbackTotal,
                match: fallbackMatch,
                pct: fallbackPct,
                similarity: null,
                rating: data.rating,
                badMove: 0,
                badMoveRate: null,
                ratingScores: [],
                similarityScores: [],
              }]).map((item) => {
                const pct = item.total ? item.match / item.total * 100 : 0;
                return (
                  <tr key={item.model} style={{ borderBottom: '1px solid var(--border)' }}>
                    <td style={statsTdStyle} title={item.model}>{item.model}</td>
                    <td style={{ ...statsTdStyle, textAlign: 'right', fontFamily: 'Menlo, Consolas, monospace' }}>
                      {item.similarity === null || item.similarity === undefined ? '—' : `${item.similarity.toFixed(1)}%`}
                    </td>
                    <td style={{ ...statsTdStyle, textAlign: 'right', fontFamily: 'Menlo, Consolas, monospace' }}>{pct.toFixed(1)}%</td>
                    <td style={{ ...statsTdStyle, textAlign: 'right', fontFamily: 'Menlo, Consolas, monospace' }}>
                      {item.badMoveRate === null || item.badMoveRate === undefined ? '—' : `${item.badMoveRate.toFixed(1)}%`}
                    </td>
                    <td style={{ ...statsTdStyle, textAlign: 'right', fontFamily: 'Menlo, Consolas, monospace' }}>
                      {item.rating === null || item.rating === undefined ? '—' : item.rating.toFixed(1)}
                    </td>
                    <td style={{ ...statsTdStyle, textAlign: 'right', fontFamily: 'Menlo, Consolas, monospace' }}>{item.match}</td>
                    <td style={{ ...statsTdStyle, textAlign: 'right', fontFamily: 'Menlo, Consolas, monospace' }}>{item.total}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </>
  );
}

const statsThStyle: React.CSSProperties = {
  padding: '7px 8px',
  fontSize: 11,
  fontWeight: 800,
  textTransform: 'uppercase',
  letterSpacing: '0.04em',
};

const statsTdStyle: React.CSSProperties = {
  padding: '8px',
  color: 'var(--text-primary)',
  fontWeight: 700,
};

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
    if (sameReplayAction(c, g)) return 'both';
    if (c.type === 'dahai' || c.type === 'none') return 'bot';
    return 'gt';
  })();

  const gtClass = (() => {
    const c = entry.chosen, g = entry.gt_action;
    if (!g) return 'none';
    if (sameReplayAction(c, g)) return 'both';
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
  const params = new URLSearchParams(location.search);
  const routeState = location.state as { replayData?: ReplayData; replayId?: string } | null;
  const replayIdFromQuery = params.get('id');
  const replayIdFromRoute = routeState?.replayId ?? null;
  const replayId = replayIdFromRoute ?? replayIdFromQuery;
  const playerIdFromQuery = Number(params.get('player_id') ?? '0');
  const requestedPlayerId = Number.isFinite(playerIdFromQuery) ? playerIdFromQuery : 0;
  const teacherReportFromQuery = params.get('teacher_report') || params.get('teacher_report_path');
  const teacherReportsFromQuery = params
    .getAll('teacher_reports')
    .flatMap((value) => value.split(','))
    .map((value) => value.trim())
    .filter(Boolean);
  const teacherReportsKey = teacherReportsFromQuery.join('\n');
  const initialData = routeState?.replayData && !replayIdFromQuery ? routeState.replayData : null;
  const initialError = initialData || replayId ? null : '未找到回放数据，请从首页上传牌谱';

  const [data, setData] = useState<ReplayData | null>(initialData);
  const [curKyoku, setCurKyoku] = useState(0);
  const [showStats, setShowStats] = useState(false);
  const [loading, setLoading] = useState(initialData === null && initialError === null);
  const [error, setError] = useState<string | null>(initialError);
  const stepRefs = useRef<Map<number, HTMLDivElement>>(new Map());
  const [scrollToStep, setScrollToStep] = useState<number | null>(null);
  const focusedStep = useRef<number>(-1);

  useEffect(() => {
    if (data || !replayId) return;
    let cancelled = false;
    const loadReplay = async () => {
      try {
        const loaded = await replayApi.get(replayId, requestedPlayerId, teacherReportFromQuery, teacherReportsFromQuery);
        if (!cancelled) {
          setData(loaded);
          setError(null);
          setLoading(false);
        }
      } catch (e) {
        if (!cancelled) {
          setError(String(e));
          setLoading(false);
        }
      }
    };
    loadReplay();
    return () => {
      cancelled = true;
    };
  }, [data, replayId, requestedPlayerId, teacherReportFromQuery, teacherReportsKey]);

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
    data !== null && isReplayReviewDiffForPlayer(e, data.player_id)
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
  const switchPerspective = (nextPid: number) => {
    const replayId = replayIdFromQuery ?? (location.state as { replayId?: string } | null)?.replayId;
    if (!replayId) return;
    navigate(`/replay?id=${encodeURIComponent(replayId)}&player_id=${nextPid}`);
  };

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
          <button
            className="btn"
            onClick={() => replayIdFromQuery
              ? navigate(`/game-replay?id=${encodeURIComponent(replayIdFromQuery)}&player_id=${pid}`)
              : navigate('/game-replay', { state: { replayData: data } })}
            title="切换到牌桌视图"
          >
            🀄牌桌
          </button>
        </span>

        <button className="btn" onClick={() => navigate('/')}>🏠</button>
        <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.5)', marginLeft: 8, display: 'none' }}>←/→ 换局</span>
      </div>

      {/* 统计面板 */}
      {showStats && <StatsPanel data={data} onClose={() => setShowStats(false)} />}

      {/* 步列表 */}
      <div style={{ flex: 1, overflow: 'hidden', display: 'flex' }}>
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
        <div
          style={{
            width: 220,
            borderLeft: '1px solid var(--border)',
            background: 'var(--sidebar-bg)',
            padding: 12,
            display: 'flex',
            flexDirection: 'column',
            gap: 8,
            flexShrink: 0,
          }}
        >
          <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>切换主视角</div>
          {playerNames.map((name, idx) => {
            const active = idx === pid;
            return (
              <button
                key={idx}
                onClick={() => switchPerspective(idx)}
                disabled={active}
                className="btn"
                style={{
                  justifyContent: 'flex-start',
                  opacity: active ? 1 : 0.92,
                  fontWeight: active ? 700 : 500,
                  outline: active ? '2px solid rgba(255,255,255,0.25)' : 'none',
                }}
                title={`切换到 ${name}`}
              >
                {SEAT_NAMES_CN[idx]} · {replayPlayerDisplayName(playerNames, idx)}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
