// src/replay_ui/src/components/DecisionPanel/ReplayDecisionPanel.tsx
import type { DecisionLogEntry } from '../../types/replay';
import { actionLabel, sameReplayAction } from '../../utils/tileUtils';
import { CN_BAKAZE } from '../../utils/constants';

interface ReplayDecisionPanelProps {
  entry: DecisionLogEntry | null;
  step: number;
  totalSteps: number;
  compact?: boolean;
  playerNames?: string[];
  currentPlayerId?: number;
  onSwitchPlayer?: (playerId: number) => void;
}

/** 候选动作的显示值：final_score（统一口径） */
function displayScore(c: { logit: number; beam_score?: number; final_score?: number }): number {
  return c.final_score ?? c.beam_score ?? c.logit;
}

function displayScoreLabel(c: { logit: number; beam_score?: number; final_score?: number }): string {
  return displayScore(c).toFixed(2);
}

function ensureVisibleCandidates(
  candidates: DecisionLogEntry['candidates'],
  chosen: DecisionLogEntry['chosen'],
  gtAction: DecisionLogEntry['gt_action'],
  limit = 12,
): DecisionLogEntry['candidates'] {
  const sorted = [...candidates].sort((a, b) => displayScore(b) - displayScore(a));
  const visible = sorted.slice(0, limit);

  const ensureAction = (action: DecisionLogEntry['chosen'] | DecisionLogEntry['gt_action']) => {
    if (!action) return;
    const alreadyVisible = visible.some((candidate) => sameReplayAction(candidate.action, action));
    if (alreadyVisible) return;
    const matched = sorted.find((candidate) => sameReplayAction(candidate.action, action));
    if (matched) {
      visible.push(matched);
    }
  };

  ensureAction(chosen);
  ensureAction(gtAction);

  return visible.sort((a, b) => displayScore(b) - displayScore(a));
}

function normalizedBarPercents(scores: number[]): number[] {
  if (scores.length === 0) return [];
  const maxScore = Math.max(...scores);
  const exps = scores.map((score) => Math.exp(score - maxScore));
  const total = exps.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) {
    return scores.map(() => 0);
  }
  return exps.map((value) => (value / total) * 100);
}

/** 动作的短名（用于表格第一列） */
function shortLabel(action: { type: string; pai?: string; consumed?: string[] }): string {
  switch (action.type) {
    case 'dahai':   return action.pai ?? '?';
    case 'reach':   return '立直';
    case 'none':    return '过';
    case 'hora':    return '和牌';
    case 'chi':     return `吃${action.pai ?? ''}`;
    case 'pon':     return `碰${action.pai ?? ''}`;
    case 'daiminkan': return `杠${action.pai ?? ''}`;
    case 'ankan':   return `暗杠`;
    case 'kakan':   return `加杠`;
    case 'ryukyoku': return '流局';
    default:        return action.type;
  }
}

export function ReplayDecisionPanel({
  entry,
  step,
  totalSteps,
  compact = false,
  playerNames = [],
  currentPlayerId,
  onSwitchPlayer,
}: ReplayDecisionPanelProps) {
  if (!entry) {
    return (
      <div style={panelStyle(compact)}>
        <div style={{ color: 'var(--text-muted)', fontSize: 12, padding: 16 }}>无数据</div>
      </div>
    );
  }

  const k = entry.kyoku_key ?? entry;
  const kyokuLabel = `${CN_BAKAZE[k.bakaze] ?? k.bakaze}${k.kyoku}局 ${k.honba}本场`;

  // obs 步：其他家操作，无 bot 推理数据
  if (entry.is_obs) {
    const actor = entry.actor_to_move;
    return (
      <div style={panelStyle(compact)}>
        <div style={sectionStyle}>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
            Step {step + 1} / {totalSteps} · {kyokuLabel}
          </div>
          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginBottom: 6 }}>
            {actor !== null && actor !== undefined ? (playerNames[actor] ?? `P${actor}`) : '未知玩家'} 的操作
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={badgeStyle('#3498db')}>动作</span>
            <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'Menlo, monospace' }}>
              {actionLabel(entry.chosen)}
            </span>
          </div>
        </div>
      </div>
    );
  }

  const { chosen, gt_action, candidates } = entry;

  // 按显示分值降序排序；即使超出前 12，也强制展示 Bot 选择和实际动作。
  const sorted = ensureVisibleCandidates(candidates, chosen, gt_action, 12);

  const barPercents = normalizedBarPercents(sorted.map((candidate) => displayScore(candidate)));

  const scoreTypeLabelShort = 'Final';

  const chosenIsGt = chosen && gt_action && sameReplayAction(chosen, gt_action);

  return (
    <div style={panelStyle(compact)}>
      {/* 步骤信息 */}
      <div style={sectionStyle}>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4 }}>
          Step {step + 1} / {totalSteps} · {kyokuLabel}
        </div>

        {/* Bot 选择 vs 实际 */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: 5 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={badgeStyle('#e74c3c')}>Bot</span>
            <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--text-primary)', fontFamily: 'Menlo, monospace' }}>
              {chosen ? actionLabel(chosen) : '—'}
            </span>
            {chosenIsGt && <span style={{ fontSize: 10, color: '#8e44ad' }}>✓ 一致</span>}
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={badgeStyle('#27ae60')}>实际</span>
            <span style={{
              fontSize: 13, fontWeight: 600,
              color: chosenIsGt ? 'var(--text-secondary)' : '#27ae60',
              fontFamily: 'Menlo, monospace',
            }}>
              {gt_action ? actionLabel(gt_action) : '—'}
            </span>
          </div>
        </div>
      </div>

      <div style={dividerStyle} />

      {/* 权重表格 */}
      <div style={{ ...sectionStyle, flex: 1, overflow: 'hidden', display: 'flex', flexDirection: 'column' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
          <span style={{ fontSize: 11, fontWeight: 700, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
            候选权重
          </span>
          <span style={{ fontSize: 10, color: 'var(--text-muted)', opacity: 0.7 }}>{scoreTypeLabelShort}</span>
        </div>

        {/* 表头 */}
        <div style={tableHeaderStyle}>
          <div style={{ width: COL1_W }}>牌名</div>
          <div style={{ flex: 1 }}>权重</div>
        </div>

        {/* 行 */}
        <div style={{ flex: 1, overflowY: 'auto' }}>
          {sorted.map((c, idx) => {
            const isChosen = sameReplayAction(c.action, chosen);
            const isGt = sameReplayAction(c.action, gt_action);
            const pct = Math.max(8, Math.round(barPercents[idx] ?? 0));

            const barColor = isChosen && isGt ? '#8e44ad'
              : isChosen ? '#e74c3c'
              : isGt ? '#27ae60'
              : 'var(--accent)';

            const rowBg = isChosen && isGt ? 'rgba(142,68,173,0.08)'
              : isChosen ? 'rgba(231,76,60,0.07)'
              : isGt ? 'rgba(39,174,96,0.07)'
              : idx % 2 === 0 ? 'transparent' : 'rgba(0,0,0,0.02)';

            return (
              <div key={idx} style={{
                display: 'flex', alignItems: 'center', gap: 0,
                padding: '3px 0',
                background: rowBg,
                borderRadius: 3,
              }}>
                {/* 第一列：牌名 + 标记 */}
                <div style={{
                  width: COL1_W, flexShrink: 0,
                  fontSize: 12, fontFamily: 'Menlo, monospace',
                  fontWeight: isChosen || isGt ? 700 : 400,
                  color: barColor,
                  display: 'flex', alignItems: 'center', gap: 3,
                }}>
                  {shortLabel(c.action)}
                  {isChosen && <span style={{ fontSize: 9, color: '#e74c3c' }}>★</span>}
                  {isGt && !isChosen && <span style={{ fontSize: 9, color: '#27ae60' }}>●</span>}
                </div>

                {/* 第二列：柱状图 + 数值 */}
                <div style={{ flex: 1, display: 'flex', alignItems: 'center', gap: 4 }}>
                  <div style={{ flex: 1, height: 10, background: 'var(--border)', borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{
                      width: `${pct}%`, height: '100%',
                      background: barColor,
                      borderRadius: 2,
                      transition: 'width 0.15s ease',
                    }} />
                  </div>
                  <div style={{
                    width: 38, textAlign: 'right', flexShrink: 0,
                    fontSize: 11, fontFamily: 'Menlo, monospace',
                    color: isChosen || isGt ? barColor : 'var(--text-muted)',
                    fontWeight: isChosen || isGt ? 700 : 400,
                  }}>
                    {displayScoreLabel(c)}
                  </div>
                </div>
              </div>
            );
          })}
          {sorted.length === 0 && (
            <div style={{ fontSize: 11, color: 'var(--text-muted)', paddingTop: 8 }}>无候选数据</div>
          )}
        </div>
      </div>

      {playerNames.length > 0 && onSwitchPlayer && currentPlayerId !== undefined && (
        <>
          <div style={dividerStyle} />
          <div style={{ ...sectionStyle, paddingTop: 8 }}>
            <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>切换主视角</div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              {playerNames.map((name, idx) => {
                const active = idx === currentPlayerId;
                return (
                  <button
                    key={idx}
                    onClick={() => onSwitchPlayer(idx)}
                    disabled={active}
                    style={{
                      ...switchBtnStyle,
                      borderColor: active ? 'var(--accent)' : 'var(--border)',
                      background: active ? 'rgba(52, 152, 219, 0.10)' : 'var(--card-bg)',
                      color: active ? 'var(--accent)' : 'var(--text-primary)',
                      cursor: active ? 'default' : 'pointer',
                    }}
                    title={`切换到 ${name}`}
                  >
                    <span style={{ fontWeight: 700 }}>P{idx}</span>
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{name}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 样式常量
// ---------------------------------------------------------------------------
const COL1_W = 52;

const panelStyle = (compact: boolean): React.CSSProperties => ({
  width: '100%',
  minHeight: compact ? 240 : undefined,
  maxHeight: compact ? 300 : undefined,
  flexShrink: 0,
  height: compact ? 'auto' : '100%',
  background: 'var(--sidebar-bg)',
  borderLeft: compact ? 'none' : '1px solid var(--border)',
  borderTop: compact ? '1px solid var(--border)' : 'none',
  display: 'flex',
  flexDirection: 'column',
  overflow: 'hidden',
  transition: 'background var(--transition)',
});

const sectionStyle: React.CSSProperties = {
  padding: '10px 12px',
};

const dividerStyle: React.CSSProperties = {
  height: 1,
  background: 'var(--border)',
  flexShrink: 0,
};

const tableHeaderStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  fontSize: 10,
  fontWeight: 700,
  color: 'var(--text-muted)',
  textTransform: 'uppercase',
  letterSpacing: '0.05em',
  paddingBottom: 4,
  borderBottom: '1px solid var(--border)',
  marginBottom: 4,
};

function badgeStyle(color: string): React.CSSProperties {
  return {
    fontSize: 10, fontWeight: 700,
    padding: '1px 5px', borderRadius: 3,
    background: color + '22',
    color,
    border: `1px solid ${color}44`,
    flexShrink: 0,
  };
}

const switchBtnStyle: React.CSSProperties = {
  width: '100%',
  borderRadius: 8,
  border: '1px solid var(--border)',
  padding: '8px 10px',
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  fontSize: 12,
  textAlign: 'left',
};
