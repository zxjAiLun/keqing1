// src/replay_ui/src/pages/GameBoardReplayPage.tsx
import { useState, useEffect, useCallback } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Loader2, ChevronLeft, ChevronRight, SkipBack, SkipForward } from 'lucide-react';
import { MahjongTable } from '../components/BattleBoard/MahjongTable';
import { ReplayDecisionPanel } from '../components/DecisionPanel/ReplayDecisionPanel';
import { StatsPanel } from './ReplayViewPage';
import { entryToBattleState, buildLogitData } from '../utils/replayAdapter';
import { useReplayPlayer } from '../hooks/useReplayPlayer';
import { replayApi } from '../api/replayApi';
import { CN_BAKAZE } from '../utils/constants';
import type { ReplayData } from '../types/replay';

const SPEED_OPTIONS = [
  { value: 0.5 as const, label: '0.5x' },
  { value: 1 as const, label: '1x' },
  { value: 2 as const, label: '2x' },
  { value: 4 as const, label: '4x' },
];

export function GameBoardReplayPage() {
  const location = useLocation();
  const navigate = useNavigate();

  const [data, setData] = useState<ReplayData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoHora, setAutoHora] = useState(true);
  const [noMeld, setNoMeld] = useState(false);
  const [autoTsumogiri, setAutoTsumogiri] = useState(false);
  const [showStats, setShowStats] = useState(false);

  // 加载数据
  useEffect(() => {
    const state = location.state as { replayData?: ReplayData; replayId?: string } | null;
    if (state?.replayData) {
      setData(state.replayData);
      setLoading(false);
    } else if (state?.replayId) {
      replayApi.get(state.replayId)
        .then(d => { setData(d); setLoading(false); })
        .catch(e => { setError(String(e)); setLoading(false); });
    } else {
      const params = new URLSearchParams(location.search);
      const replayId = params.get('id');
      if (replayId) {
        replayApi.get(replayId)
          .then(d => { setData(d); setLoading(false); })
          .catch(e => { setError(String(e)); setLoading(false); });
      } else {
        setError('未找到回放数据，请从首页上传牌谱');
        setLoading(false);
      }
    }
  }, [location]);

  const {
    currentStep, isPlaying, speed, totalSteps,
    currentEntry, currentKyoku, totalKyoku,
    togglePlay, stepForward, stepBackward,
    stepToNextAction, stepToPrevAction,
    goToStart, goToEnd, goToStep, goToKyoku, changeSpeed,
  } = useReplayPlayer(data);

  const isDiff = (e: import('../types/replay').DecisionLogEntry, pid: number) =>
    e.actor_to_move === pid && e.gt_action !== null &&
    (e.chosen.type !== e.gt_action.type || e.chosen.pai !== e.gt_action.pai);

  const jumpToPrevDiff = useCallback(() => {
    if (!data) return;
    const pid = data.player_id;
    for (let i = currentStep - 1; i >= 0; i--) {
      if (isDiff(data.log[i], pid)) { goToStep(i); return; }
    }
  }, [data, currentStep, goToStep]);

  const jumpToNextDiff = useCallback(() => {
    if (!data) return;
    const pid = data.player_id;
    for (let i = currentStep + 1; i < data.log.length; i++) {
      if (isDiff(data.log[i], pid)) { goToStep(i); return; }
    }
  }, [data, currentStep, goToStep]);

  // 键盘快捷键
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      const tag = (e.target as HTMLElement).tagName;
      if (tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT') return;
      // 提前拦截方向键和空格，防止页面滚动
      if (['ArrowLeft','ArrowRight','ArrowUp','ArrowDown',' ','h','j','k','l'].includes(e.key)) {
        e.preventDefault();
      }
      switch (e.key) {
        case 'ArrowLeft':  case 'h': stepBackward(); break;
        case 'ArrowRight': case 'l': stepForward(); break;
        case 'ArrowDown':  case 'j': stepToNextAction(); break;
        case 'ArrowUp':    case 'k': stepToPrevAction(); break;
        case ' ': togglePlay(); break;
      }
    };
    // useCapture=true 确保在其他元素之前捕获
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [stepBackward, stepForward, stepToNextAction, stepToPrevAction, togglePlay]);

  // 滚轮事件
  useEffect(() => {
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      if (e.deltaY > 0) stepToNextAction();
      else stepToPrevAction();
    };
    window.addEventListener('wheel', handler, { passive: false });
    return () => window.removeEventListener('wheel', handler);
  }, [stepToNextAction, stepToPrevAction]);

  // 优先使用后端返回的真实玩家名，fallback 到 P0/P1/P2/P3
  const playerNames = data?.player_names && data.player_names.length === 4
    ? data.player_names
    : [0, 1, 2, 3].map(i => `P${i}`);

  const viewPlayerId = data?.player_id ?? 0;

  // 适配数据
  const battleState = currentEntry ? entryToBattleState(currentEntry, playerNames, viewPlayerId) : null;
  const logitData = currentEntry ? buildLogitData(currentEntry) : undefined;

  const entry = currentEntry;
  const k = entry?.kyoku_key;
  const kyokuLabel = k ? `第${CN_BAKAZE[k.bakaze] ?? k.bakaze}${k.kyoku}局 ${k.honba}本场` : '';

  // 当前步 bot 选择和实际打出
  const chosenPai = entry?.chosen?.type === 'dahai' ? entry.chosen.pai ?? null : null;

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
        <button onClick={() => navigate('/review')} style={{ color: 'var(--accent)', fontSize: 14, background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline' }}>
          返回上传
        </button>
      </div>
    );
  }

  return (
    <div className="h-full flex flex-col" style={{ background: 'var(--page-bg)' }}>
      {/* ── 顶栏 ── */}
      <div style={{
        height: 40, flexShrink: 0,
        background: 'var(--sidebar-bg)',
        borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center',
        padding: '0 12px', gap: 10,
      }}>
        {/* 返回 */}
        <button onClick={() => navigate('/')} style={iconBtnStyle} title="返回">
          ←
        </button>

        <span style={{ color: 'var(--border)' }}>|</span>

        {/* 局号 */}
        <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', whiteSpace: 'nowrap' }}>
          {kyokuLabel}
        </span>

        {/* 小局跳转 */}
        {totalKyoku > 1 && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
            <button onClick={() => goToKyoku(Math.max(0, currentKyoku - 1))} disabled={currentKyoku === 0} style={smallBtnStyle}>
              ◀
            </button>
            <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>{currentKyoku + 1}/{totalKyoku}</span>
            <button onClick={() => goToKyoku(Math.min(totalKyoku - 1, currentKyoku + 1))} disabled={currentKyoku === totalKyoku - 1} style={smallBtnStyle}>
              ▶
            </button>
          </div>
        )}

        {/* 步骤进度 */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: 3, minWidth: 100 }}>
          <input
            type="range" min={0} max={totalSteps - 1} value={currentStep}
            onChange={e => goToStep(parseInt(e.target.value))}
            style={{ width: '100%', accentColor: 'var(--accent)' }}
          />
        </div>

        {/* 步进 */}
        <button onClick={goToStart} disabled={currentStep === 0} style={smallBtnStyle} title="第一步">
          <SkipBack size={13} />
        </button>
        <button onClick={stepBackward} disabled={currentStep === 0} style={smallBtnStyle} title="上一步 (←)">
          <ChevronLeft size={13} />
        </button>
        <button onClick={togglePlay} style={{ ...smallBtnStyle, fontWeight: 700 }}>
          {isPlaying ? '⏸' : '▶'}
        </button>
        <button onClick={stepForward} disabled={currentStep === totalSteps - 1} style={smallBtnStyle} title="下一步 (→)">
          <ChevronRight size={13} />
        </button>
        <button onClick={goToEnd} disabled={currentStep === totalSteps - 1} style={smallBtnStyle} title="最后一步">
          <SkipForward size={13} />
        </button>

        {/* 速度 */}
        <div style={{ display: 'flex', gap: 2 }}>
          {SPEED_OPTIONS.map(s => (
            <button key={s.label}
              onClick={() => changeSpeed(s.value)}
              style={{
                ...smallBtnStyle,
                fontWeight: speed === s.value ? 700 : 400,
                color: speed === s.value ? 'var(--accent)' : 'var(--text-muted)',
                fontSize: 10,
              }}
            >
              {s.label}
            </button>
          ))}
        </div>

        {/* step 数 */}
        <span style={{ fontSize: 11, color: 'var(--text-muted)', fontFamily: 'Menlo, monospace', flexShrink: 0 }}>
          {currentStep + 1}/{totalSteps}
        </span>

        <span style={{ color: 'var(--border)' }}>|</span>

        {/* 统计 */}
        <button onClick={() => setShowStats(true)} style={iconBtnStyle} title="统计">
          📊
        </button>

        {/* 跳转到上/下一不同 */}
        <button onClick={jumpToPrevDiff} style={iconBtnStyle} title="上一个与Bot不同的决策">
          ⏮差异
        </button>
        <button onClick={jumpToNextDiff} style={iconBtnStyle} title="下一个与Bot不同的决策">
          差异⏭
        </button>

        {/* 切换到决策列表 */}
        <button
          onClick={() => navigate('/replay', { state: { replayData: data } })}
          style={iconBtnStyle}
          title="切换到决策列表"
        >
          📋列表
        </button>
      </div>

      {/* 统计面板 */}
      {showStats && data && <StatsPanel data={data} onClose={() => setShowStats(false)} />}

      {/* ── 主内容：牌桌 + 右侧面板 ── */}
      <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
        {/* 牌桌 */}
        <div style={{ flex: 1, minWidth: 0 }}>
          {battleState ? (
            <MahjongTable
              state={battleState}
              onAction={() => {}}
              isMyTurn={false}
              selectedTile={chosenPai}
              selectedTileIdx={null}
              onTileSelect={() => {}}
              autoHora={autoHora} setAutoHora={setAutoHora}
              noMeld={noMeld} setNoMeld={setNoMeld}
              autoTsumogiri={autoTsumogiri} setAutoTsumogiri={setAutoTsumogiri}
              mode="replay"
              logitData={logitData}
            />
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
              <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>选择一个步骤</p>
            </div>
          )}
        </div>

        {/* 右侧面板 */}
        <ReplayDecisionPanel entry={currentEntry} step={currentStep} totalSteps={totalSteps} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 样式
// ---------------------------------------------------------------------------
const iconBtnStyle: React.CSSProperties = {
  padding: '3px 8px', borderRadius: 4, border: '1px solid var(--border)',
  background: 'var(--sidebar-hover-bg)', color: 'var(--text-secondary)',
  fontSize: 13, cursor: 'pointer', flexShrink: 0,
};

const smallBtnStyle: React.CSSProperties = {
  padding: 3, borderRadius: 4, border: '1px solid var(--border)',
  background: 'var(--sidebar-hover-bg)', color: 'var(--text-secondary)',
  cursor: 'pointer', flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center',
};
