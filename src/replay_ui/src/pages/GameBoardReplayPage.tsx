// src/replay_ui/src/pages/GameBoardReplayPage.tsx
import { useState, useEffect, useCallback, useMemo } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { Loader2, ChevronLeft, ChevronRight, SkipBack, SkipForward, PanelLeftClose, PanelLeftOpen, ChevronsUp, ChevronsDown } from 'lucide-react';
import { MahjongTable } from '../components/BattleBoard/MahjongTable';
import { Tile } from '../components/BattleBoard/Tile';
import { ReplayDecisionPanel } from '../components/DecisionPanel/ReplayDecisionPanel';
import { StatsPanel } from './ReplayViewPage';
import { entryToBattleState, buildLogitData, hasReplayPostAction, hasReplayReachPhase, type ReplayBoardPhase } from '../utils/replayAdapter';
import { useReplayPlayer } from '../hooks/useReplayPlayer';
import { replayApi } from '../api/replayApi';
import { CN_BAKAZE } from '../utils/constants';
import { sameReplayAction, TILE_ORDER } from '../utils/tileUtils';
import { normalizeReplayPlayerNames, replayPlayerDisplayName } from '../utils/replayNames';
import type { Action, ReplayData } from '../types/replay';

export function GameBoardReplayPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const routeState = location.state as { replayData?: ReplayData; replayId?: string } | null;
  const params = new URLSearchParams(location.search);
  const replayIdFromRoute = routeState?.replayId ?? params.get('id');
  const playerIdFromQuery = Number(params.get('player_id') ?? '0');
  const requestedPlayerId = Number.isFinite(playerIdFromQuery) ? playerIdFromQuery : 0;

  const [data, setData] = useState<ReplayData | null>(null);
  const [events, setEvents] = useState<Record<string, unknown>[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [autoHora, setAutoHora] = useState(true);
  const [noMeld, setNoMeld] = useState(false);
  const [autoTsumogiri, setAutoTsumogiri] = useState(false);
  const [showStats, setShowStats] = useState(false);
  const [narrowLayout, setNarrowLayout] = useState(false);
  const [boardPhase, setBoardPhase] = useState<ReplayBoardPhase>('pre');
  const [showOpponentHands, setShowOpponentHands] = useState(false);
  const [showPerspectiveDrawer, setShowPerspectiveDrawer] = useState(false);
  const [showTopToolbar, setShowTopToolbar] = useState(true);

  // 加载数据
  useEffect(() => {
    if (routeState?.replayData && !replayIdFromRoute) {
      setData(routeState.replayData);
      setLoading(false);
    } else if (routeState?.replayId) {
      replayApi.get(routeState.replayId, requestedPlayerId)
        .then(d => { setData(d); setLoading(false); })
        .catch(e => { setError(String(e)); setLoading(false); });
    } else {
      const replayId = replayIdFromRoute;
      if (replayId) {
        replayApi.get(replayId, requestedPlayerId)
          .then(d => { setData(d); setLoading(false); })
          .catch(e => { setError(String(e)); setLoading(false); });
      } else {
        setError('未找到回放数据，请从首页上传牌谱');
        setLoading(false);
      }
    }
  }, [location, replayIdFromRoute, requestedPlayerId, routeState]);

  useEffect(() => {
    if (!replayIdFromRoute) return;
    replayApi.getEvents(replayIdFromRoute, 0, 10000)
      .then((payload) => {
        setEvents((payload.events ?? []) as Record<string, unknown>[]);
      })
      .catch(() => {
        setEvents(null);
      });
  }, [replayIdFromRoute]);

  const {
    currentStep, totalSteps,
    currentEntry, currentKyoku, totalKyoku,
    stepForward, stepBackward,
    goToStart, goToEnd, goToStep, goToKyoku,
  } = useReplayPlayer(data);

  const currentHasPostPhase = hasReplayPostAction(currentEntry);
  const currentHasReachPhase = hasReplayReachPhase(currentEntry);

  const resetBoardPhase = useCallback(() => {
    setBoardPhase('pre');
  }, []);

  const moveBoardStep = useCallback((direction: 1 | -1) => {
    if (!data || !currentEntry) return;

    if (direction > 0) {
      if (boardPhase === 'pre' && currentHasReachPhase) {
        setBoardPhase('reach');
        return;
      }
      if ((boardPhase === 'pre' || boardPhase === 'reach') && currentHasPostPhase) {
        setBoardPhase('post');
        return;
      }
      for (let i = currentStep + 1; i < data.log.length; i++) {
        if (data.log[i]?.chosen) {
          goToStep(i);
          setBoardPhase('pre');
          return;
        }
      }
      return;
    }

    if (boardPhase === 'post') {
      if (currentHasReachPhase) {
        setBoardPhase('reach');
        return;
      }
      setBoardPhase('pre');
      return;
    }
    if (boardPhase === 'reach') {
      setBoardPhase('pre');
      return;
    }
    for (let i = currentStep - 1; i >= 0; i--) {
      if (!data.log[i]?.chosen) continue;
      goToStep(i);
      if (hasReplayPostAction(data.log[i])) {
        setBoardPhase('post');
      } else if (hasReplayReachPhase(data.log[i])) {
        setBoardPhase('reach');
      } else {
        setBoardPhase('pre');
      }
      return;
    }
  }, [data, currentEntry, boardPhase, currentHasPostPhase, currentHasReachPhase, currentStep, goToStep]);

  const handleStepForward = useCallback(() => {
    resetBoardPhase();
    setShowOpponentHands(false);
    stepForward();
  }, [resetBoardPhase, stepForward]);

  const handleStepBackward = useCallback(() => {
    resetBoardPhase();
    setShowOpponentHands(false);
    stepBackward();
  }, [resetBoardPhase, stepBackward]);

  const handleGoToStart = useCallback(() => {
    resetBoardPhase();
    setShowOpponentHands(false);
    goToStart();
  }, [goToStart, resetBoardPhase]);

  const handleGoToEnd = useCallback(() => {
    resetBoardPhase();
    setShowOpponentHands(false);
    goToEnd();
  }, [goToEnd, resetBoardPhase]);

  const handleGoToStep = useCallback((step: number) => {
    resetBoardPhase();
    setShowOpponentHands(false);
    goToStep(step);
  }, [goToStep, resetBoardPhase]);

  const handleGoToKyoku = useCallback((kyokuIdx: number) => {
    resetBoardPhase();
    setShowOpponentHands(false);
    goToKyoku(kyokuIdx);
  }, [goToKyoku, resetBoardPhase]);

  const isDiff = (e: import('../types/replay').DecisionLogEntry, pid: number) =>
    !e.is_obs &&
    e.actor_to_move === pid && e.gt_action !== null &&
    !sameReplayAction(e.chosen, e.gt_action);

  const jumpToPrevDiff = useCallback(() => {
    if (!data) return;
    const pid = data.player_id;
    for (let i = currentStep - 1; i >= 0; i--) {
      if (isDiff(data.log[i], pid)) {
        resetBoardPhase();
        setShowOpponentHands(false);
        goToStep(i);
        return;
      }
    }
  }, [data, currentStep, goToStep, resetBoardPhase]);

  const jumpToNextDiff = useCallback(() => {
    if (!data) return;
    const pid = data.player_id;
    for (let i = currentStep + 1; i < data.log.length; i++) {
      if (isDiff(data.log[i], pid)) {
        resetBoardPhase();
        setShowOpponentHands(false);
        goToStep(i);
        return;
      }
    }
  }, [data, currentStep, goToStep, resetBoardPhase]);

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
        case 'ArrowLeft':  case 'h': handleStepBackward(); break;
        case 'ArrowRight': case 'l': handleStepForward(); break;
        case 'ArrowDown':  case 'j': moveBoardStep(1); break;
        case 'ArrowUp':    case 'k': moveBoardStep(-1); break;
      }
    };
    // useCapture=true 确保在其他元素之前捕获
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [handleStepBackward, handleStepForward, moveBoardStep]);

  // 滚轮事件
  useEffect(() => {
    const handler = (e: WheelEvent) => {
      e.preventDefault();
      if (e.deltaY > 0) moveBoardStep(1);
      else moveBoardStep(-1);
    };
    window.addEventListener('wheel', handler, { passive: false });
    return () => window.removeEventListener('wheel', handler);
  }, [moveBoardStep]);

  useEffect(() => {
    const updateLayout = () => setNarrowLayout(window.innerWidth < 1120);
    updateLayout();
    window.addEventListener('resize', updateLayout);
    return () => window.removeEventListener('resize', updateLayout);
  }, []);

  // 优先使用后端返回的真实玩家名，fallback 到 P0/P1/P2/P3
  const playerNames = normalizeReplayPlayerNames(data);

  const viewPlayerId = data?.player_id ?? 0;
  const switchPerspective = (nextPid: number) => {
    if (!replayIdFromRoute) return;
    navigate(`/game-replay?id=${encodeURIComponent(replayIdFromRoute)}&player_id=${nextPid}`);
  };

  // 适配数据
  const battleState = useMemo(
    () => {
      if (!currentEntry || !data) return null;
      const prevEntry = currentStep > 0 ? data.log[currentStep - 1] : null;
      return entryToBattleState(currentEntry, playerNames, viewPlayerId, boardPhase, prevEntry);
    },
    [currentEntry, data, currentStep, playerNames, viewPlayerId, boardPhase],
  );
  // obs 步不显示柱状图
  const logitData = currentEntry && !currentEntry.is_obs ? buildLogitData(currentEntry) : undefined;

  const entry = currentEntry;
  const k = entry?.kyoku_key;
  const kyokuLabel = k ? `${CN_BAKAZE[k.bakaze] ?? k.bakaze}${k.kyoku}局 ${k.honba}本场` : '';

  // 当前步 bot 选择和实际打出
  const chosenPai = entry?.chosen?.type === 'dahai' ? entry.chosen.pai ?? null : null;
  const currentResultAction = (currentEntry?.gt_action ?? currentEntry?.chosen) as Action | undefined;
  const resultSummary = useMemo(
    () => {
      if (!currentEntry) return null;
      if ((currentResultAction?.type === 'hora' || currentResultAction?.type === 'ryukyoku') && boardPhase !== 'post') {
        return null;
      }
      return buildReplayResultSummary(events, currentEntry, playerNames);
    },
    [events, currentEntry, currentResultAction, playerNames, boardPhase],
  );
  const replayHands = useMemo(
    () => buildReplayHandsForBoard(events as ReplayEvent[] | null, data, currentStep, currentEntry ?? null, boardPhase),
    [events, data, currentStep, currentEntry, boardPhase],
  );

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
    <div
      style={{
        height: 'calc(100vh - var(--mobile-shell-offset, 0px))',
        background: 'var(--page-bg)',
        overflow: 'hidden',
      }}
    >
      {showStats && data && <StatsPanel data={data} onClose={() => setShowStats(false)} />}

      {battleState ? (
        <div style={{ display: 'flex', height: '100%', minHeight: 0 }}>
          <div style={{ flex: 1, minWidth: 0, minHeight: 0, position: 'relative' }}>
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
              revealedOpponentHands={showOpponentHands ? replayHands : null}
              onToggleOpponentHands={() => setShowOpponentHands((v) => !v)}
            />
            {resultSummary && <ReplayResultOverlay summary={resultSummary} playerNames={playerNames} />}

            <div style={floatingTopBarWrapStyle}>
              <button
                onClick={() => setShowTopToolbar((v) => !v)}
                style={toolbarToggleStyle}
                title={showTopToolbar ? '收起顶部工具栏' : '展开顶部工具栏'}
              >
                {showTopToolbar ? <ChevronsUp size={14} /> : <ChevronsDown size={14} />}
                <span>{showTopToolbar ? '收起' : '展开'}</span>
              </button>
              {showTopToolbar && (
                <div style={floatingTopBarStyle}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                    <button onClick={() => navigate('/')} style={iconBtnStyle} title="返回">←</button>
                    {totalKyoku > 1 ? (
                      <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                        <button onClick={() => handleGoToKyoku(Math.max(0, currentKyoku - 1))} disabled={currentKyoku === 0} style={smallBtnStyle}>◀</button>
                        <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.78)' }}>{kyokuLabel} · {currentKyoku + 1}/{totalKyoku}</span>
                        <button onClick={() => handleGoToKyoku(Math.min(totalKyoku - 1, currentKyoku + 1))} disabled={currentKyoku === totalKyoku - 1} style={smallBtnStyle}>▶</button>
                      </div>
                    ) : (
                      <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.78)' }}>{kyokuLabel}</span>
                    )}
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flex: 1, minWidth: 180 }}>
                    <input
                      type="range"
                      min={0}
                      max={totalSteps - 1}
                      value={currentStep}
                      onChange={e => handleGoToStep(parseInt(e.target.value))}
                      style={{ width: '100%', accentColor: 'var(--accent)' }}
                    />
                    <span style={{ fontSize: 11, color: 'rgba(255,255,255,0.72)', fontFamily: 'Menlo, monospace', whiteSpace: 'nowrap' }}>
                      {currentStep + 1}/{totalSteps}{boardPhase === 'post' ? ' · 后' : boardPhase === 'reach' ? ' · 立直' : ' · 前'}
                    </span>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, flexWrap: 'wrap' }}>
                    <button onClick={handleGoToStart} disabled={currentStep === 0 && boardPhase === 'pre'} style={smallBtnStyle} title="第一步"><SkipBack size={13} /></button>
                    <button onClick={handleStepBackward} disabled={currentStep === 0 && boardPhase === 'pre'} style={smallBtnStyle} title="上一步 (←)"><ChevronLeft size={13} /></button>
                    <button onClick={handleStepForward} disabled={currentStep === totalSteps - 1 && !currentHasPostPhase && !currentHasReachPhase} style={smallBtnStyle} title="下一步 (→)"><ChevronRight size={13} /></button>
                    <button onClick={handleGoToEnd} disabled={currentStep === totalSteps - 1 && boardPhase === 'pre'} style={smallBtnStyle} title="最后一步"><SkipForward size={13} /></button>

                    <button onClick={() => setShowStats(true)} style={iconBtnStyle} title="统计">📊</button>
                    <button onClick={jumpToPrevDiff} style={iconBtnStyle} title="上一个与Bot不同的决策">⏮差异</button>
                    <button onClick={jumpToNextDiff} style={iconBtnStyle} title="下一个与Bot不同的决策">差异⏭</button>
                    <button
                      onClick={() => replayIdFromRoute
                        ? navigate(`/replay?id=${encodeURIComponent(replayIdFromRoute)}&player_id=${viewPlayerId}`)
                        : navigate('/replay', { state: { replayData: data } })}
                      style={iconBtnStyle}
                      title="切换到决策列表"
                    >
                      📋列表
                    </button>
                  </div>
                </div>
              )}
            </div>

            <div style={floatingPerspectiveStyle}>
              <button
                onClick={() => setShowPerspectiveDrawer((v) => !v)}
                style={perspectiveToggleStyle}
                title="切换主视角"
              >
                {showPerspectiveDrawer ? <PanelLeftClose size={14} /> : <PanelLeftOpen size={14} />}
                <span>主视角</span>
              </button>
              {showPerspectiveDrawer && (
                <div style={perspectiveDrawerStyle}>
                  <div style={{ fontSize: 11, color: 'rgba(255,255,255,0.72)' }}>切换主视角</div>
                  {playerNames.map((name, idx) => (
                    <button
                      key={idx}
                      onClick={() => switchPerspective(idx)}
                      disabled={!replayIdFromRoute || idx === viewPlayerId}
                      style={{
                        ...floatingSwitchBtnStyle,
                        borderColor: idx === viewPlayerId ? 'var(--accent)' : 'rgba(255,255,255,0.12)',
                        background: idx === viewPlayerId ? 'rgba(52, 152, 219, 0.16)' : 'rgba(255,255,255,0.04)',
                        color: idx === viewPlayerId ? '#d6ebff' : '#f3f4f6',
                        cursor: idx === viewPlayerId ? 'default' : 'pointer',
                      }}
                      title={`切换到 ${name}`}
                    >
                      {name}
                    </button>
                  ))}
                </div>
              )}
            </div>
          </div>

          {!narrowLayout && (
            <div
              style={{
                width: 198,
                minWidth: 198,
                height: '100%',
                borderLeft: '1px solid var(--border)',
                background: 'var(--page-bg)',
              }}
            >
              <ReplayDecisionPanel
                entry={currentEntry}
                step={currentStep}
                totalSteps={totalSteps}
                compact={false}
                playerNames={playerNames}
              />
            </div>
          )}

        </div>
      ) : (
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
          <p style={{ color: 'var(--text-muted)', fontSize: 14 }}>选择一个步骤</p>
        </div>
      )}
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

const floatingSwitchBtnStyle: React.CSSProperties = {
  width: '100%',
  borderRadius: 8,
  border: '1px solid rgba(255,255,255,0.12)',
  padding: '7px 9px',
  textAlign: 'left',
  fontSize: 12,
  whiteSpace: 'nowrap',
  overflow: 'hidden',
  textOverflow: 'ellipsis',
};

const perspectiveToggleStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  borderRadius: 10,
  border: '1px solid rgba(255,255,255,0.12)',
  background: 'rgba(17, 24, 39, 0.86)',
  color: '#f3f4f6',
  padding: '8px 10px',
  fontSize: 12,
  cursor: 'pointer',
  boxShadow: '0 8px 24px rgba(0,0,0,0.28)',
  backdropFilter: 'blur(10px)',
};

const perspectiveDrawerStyle: React.CSSProperties = {
  position: 'absolute',
  left: 0,
  bottom: 'calc(100% + 8px)',
  zIndex: 30,
  background: 'rgba(17, 24, 39, 0.92)',
  border: '1px solid rgba(255,255,255,0.12)',
  borderRadius: 12,
  padding: '10px 10px 8px',
  display: 'flex',
  flexDirection: 'column',
  gap: 6,
  minWidth: 170,
  boxShadow: '0 8px 24px rgba(0,0,0,0.28)',
  backdropFilter: 'blur(10px)',
};

const floatingTopBarWrapStyle: React.CSSProperties = {
  position: 'absolute',
  top: 12,
  left: 168,
  right: 12,
  zIndex: 34,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'flex-start',
  gap: 8,
  pointerEvents: 'none',
};

const toolbarToggleStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 6,
  padding: '7px 10px',
  borderRadius: 999,
  border: '1px solid rgba(255,255,255,0.12)',
  background: 'rgba(12, 16, 20, 0.52)',
  color: '#f3f4f6',
  fontSize: 12,
  cursor: 'pointer',
  boxShadow: '0 12px 24px rgba(0,0,0,0.18)',
  backdropFilter: 'blur(10px)',
  pointerEvents: 'auto',
};

const floatingTopBarStyle: React.CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 10,
  flexWrap: 'wrap',
  padding: '10px 12px',
  borderRadius: 16,
  width: 'min(980px, calc(100% - 32px))',
  background: 'linear-gradient(90deg, rgba(12, 16, 20, 0.78) 0%, rgba(12, 16, 20, 0.72) 58%, rgba(12, 16, 20, 0.24) 78%, rgba(12, 16, 20, 0.04) 100%)',
  border: '1px solid rgba(255,255,255,0.08)',
  boxShadow: '0 16px 32px rgba(0,0,0,0.18)',
  backdropFilter: 'blur(14px)',
  pointerEvents: 'auto',
};

const floatingPerspectiveStyle: React.CSSProperties = {
  position: 'absolute',
  left: 12,
  bottom: 12,
  zIndex: 30,
  display: 'flex',
  flexDirection: 'column',
  alignItems: 'flex-start',
  gap: 6,
};

type ReplayEvent = Record<string, unknown>;
type ResultSummary = {
  type: 'hora' | 'ryukyoku';
  title: string;
  subtitle: string;
  honba: number;
  kyotaku: number;
  doraMarkers: string[];
  uraMarkers: string[];
  yakuLines: string[];
  scoreLines: Array<{ pid: number; label: string; name: string; before: number; delta: number; after: number }>;
  winnerHand: string[];
  winnerMelds: Array<{ type: string; pai: string; consumed: string[] }>;
  winner: number | null;
  winTile: string | null;
};

const YAKU_LABELS: Record<string, string> = {
  Riichi: '立直',
  Ippatsu: '一发',
  'Menzen Tsumo': '门前清自摸和',
  Tsumo: '门前清自摸和',
  Pinfu: '平和',
  Tanyao: '断幺九',
  Dora: '宝牌',
  'Aka Dora': '赤宝牌',
  'Ura Dora': '里宝牌',
  Iipeiko: '一杯口',
  Ryanpeikou: '二杯口',
  Ittsu: '一气通贯',
  Chanta: '混全带幺九',
  Junchan: '纯全带幺九',
  Toitoi: '对对和',
  Sanankou: '三暗刻',
  Sankantsu: '三杠子',
  Shousangen: '小三元',
  Honroutou: '混老头',
  Honitsu: '混一色',
  Chinitsu: '清一色',
  Chiitoitsu: '七对子',
  'Sanshoku Doukou': '三色同刻',
  'Sanshoku Doujun': '三色同顺',
  'Yakuhai (chun)': '役牌 中',
  'Yakuhai (haku)': '役牌 白',
  'Yakuhai (hatsu)': '役牌 发',
  'Yakuhai (east)': '役牌 东',
  'Yakuhai (south)': '役牌 南',
  'Yakuhai (west)': '役牌 西',
  'Yakuhai (north)': '役牌 北',
};

const RESULT_LEVEL_LABELS: Record<string, string> = {
  mangan: '满贯',
  haneman: '跳满',
  baiman: '倍满',
  sanbaiman: '三倍满',
  yakuman: '役满',
};

function getSeatLabel(oya: number, pid: number): string {
  const order = ['东', '南', '西', '北'];
  return order[(pid - oya + 4) % 4];
}

function removeTileOnce(hand: string[], tile?: string) {
  if (!tile) return hand;
  const idx = hand.indexOf(tile);
  if (idx >= 0) {
    hand.splice(idx, 1);
    return hand;
  }
  const norm = tile.endsWith('r') ? tile.slice(0, 2) : tile;
  const normIdx = hand.findIndex((h) => h === norm || (h.endsWith('r') ? h.slice(0, 2) : h) === norm);
  if (normIdx >= 0) hand.splice(normIdx, 1);
  return hand;
}

function normalizeTileForDora(tile: string): string {
  return tile.endsWith('r') ? tile.slice(0, 2) : tile;
}

function indicatorToDoraTile(tile: string): string | null {
  const normalized = normalizeTileForDora(tile);
  const num = normalized[0];
  const suit = normalized[1];
  const honorCycle: Record<string, string> = {
    E: 'S',
    S: 'W',
    W: 'N',
    N: 'E',
    P: 'F',
    F: 'C',
    C: 'P',
  };
  if (normalized in honorCycle) {
    return honorCycle[normalized];
  }
  if (suit === 'm' || suit === 'p' || suit === 's') {
    const next = num === '9' ? '1' : String(Number(num) + 1);
    return `${next}${suit}`;
  }
  return null;
}

function countMarkerDora(allTiles: string[], markers: string[]): number {
  let total = 0;
  for (const marker of markers) {
    const doraTile = indicatorToDoraTile(marker);
    if (!doraTile) continue;
    const normalizedDora = normalizeTileForDora(doraTile);
    total += allTiles.filter((tile) => normalizeTileForDora(tile) === normalizedDora).length;
  }
  return total;
}

function buildFallbackYakuDetails(
  yakuList: string[],
  winnerTiles: string[],
  winnerMelds: Array<{ type: string; pai: string; consumed: string[] }>,
  doraMarkers: string[],
  uraMarkers: string[],
): Array<{ key: string; name: string; han: number }> {
  const doraLike = new Set(['Dora', 'Ura Dora', 'Aka Dora']);
  const grouped = new Map<string, number>();
  for (const name of yakuList) {
    if (doraLike.has(name)) continue;
    grouped.set(name, (grouped.get(name) ?? 0) + 1);
  }

  const allTiles = [
    ...winnerTiles,
    ...winnerMelds.flatMap((meld) => [...meld.consumed, meld.pai].filter(Boolean)),
  ];
  const details = Array.from(grouped.entries()).map(([name, han]) => ({ key: name, name, han }));
  const doraCount = countMarkerDora(allTiles, doraMarkers);
  const uraCount = countMarkerDora(allTiles, uraMarkers);
  const akaCount = allTiles.filter((tile) => tile.endsWith('r')).length;

  if (doraCount > 0) details.push({ key: 'Dora', name: 'Dora', han: doraCount });
  if (uraCount > 0) details.push({ key: 'Ura Dora', name: 'Ura Dora', han: uraCount });
  if (akaCount > 0) details.push({ key: 'Aka Dora', name: 'Aka Dora', han: akaCount });

  return details;
}

function sortTilesForResult(tiles: string[], winTile?: string | null): string[] {
  const sorted = [...tiles].sort((a, b) => (TILE_ORDER[a] ?? 99) - (TILE_ORDER[b] ?? 99));
  if (!winTile) return sorted;
  const exactIdx = sorted.findIndex((tile) => tile === winTile);
  if (exactIdx >= 0) {
    const [tile] = sorted.splice(exactIdx, 1);
    sorted.push(tile);
    return sorted;
  }
  const norm = winTile.endsWith('r') ? winTile.slice(0, 2) : winTile;
  const normIdx = sorted.findIndex((tile) => (tile.endsWith('r') ? tile.slice(0, 2) : tile) === norm);
  if (normIdx >= 0) {
    const [tile] = sorted.splice(normIdx, 1);
    sorted.push(tile);
  }
  return sorted;
}

function sameConsumed(a: string[] = [], b: string[] = []): boolean {
  if (a.length !== b.length) return false;
  return [...a].sort().join(',') === [...b].sort().join(',');
}

function eventMatchesAction(ev: ReplayEvent, action: Action): boolean {
  const type = String(ev.type ?? '');
  if (type !== action.type) return false;
  if (Number(ev.actor ?? -1) !== Number(action.actor ?? -1)) return false;
  if (action.pai !== undefined && String(ev.pai ?? '') !== String(action.pai ?? '')) return false;
  if (action.target !== undefined && Number(ev.target ?? -1) !== Number(action.target ?? -1)) return false;
  if (action.consumed && !sameConsumed(((ev.consumed as string[] | undefined) ?? []).map(String), action.consumed)) return false;
  return true;
}

function applyReplayEventToHands(hands: string[][], ev: ReplayEvent) {
  const type = String(ev.type ?? '');
  const actor = Number(ev.actor ?? -1);
  if (actor < 0) return;
  if (type === 'tsumo') {
    hands[actor].push(String(ev.pai ?? ''));
    return;
  }
  if (type === 'dahai') {
    removeTileOnce(hands[actor], String(ev.pai ?? ''));
    return;
  }
  if (['chi', 'pon', 'daiminkan', 'ankan'].includes(type)) {
    const consumed = ((ev.consumed as string[] | undefined) ?? []).map(String);
    for (const tile of consumed) removeTileOnce(hands[actor], tile);
    return;
  }
  if (type === 'kakan' || type === 'kakan_accepted') {
    removeTileOnce(hands[actor], String(ev.pai ?? ''));
  }
}

function findKyokuEventRange(events: ReplayEvent[], entry: ReplayData['log'][number]) {
  const kyokuKey = entry.kyoku_key;
  let startIdx = -1;
  let endIdx = events.length;
  for (let i = 0; i < events.length; i++) {
    const ev = events[i];
    if (
      ev.type === 'start_kyoku' &&
      ev.bakaze === kyokuKey.bakaze &&
      ev.kyoku === kyokuKey.kyoku &&
      ev.honba === kyokuKey.honba
    ) {
      startIdx = i;
      continue;
    }
    if (startIdx >= 0 && i > startIdx && ev.type === 'start_kyoku') {
      endIdx = i;
      break;
    }
  }
  return startIdx >= 0 ? { startIdx, endIdx } : null;
}

function applyEntryEventsToHands(
  kyokuEvents: ReplayEvent[],
  startCursor: number,
  hands: string[][],
  action: Action | null | undefined,
  phase: ReplayBoardPhase,
): number {
  if (!action || ['none', 'hora', 'ryukyoku'].includes(action.type)) {
    return startCursor;
  }

  let cursor = startCursor;
  while (cursor < kyokuEvents.length) {
    const ev = kyokuEvents[cursor];
    const type = String(ev.type ?? '');

    if (type === 'dora' || type === 'reach_accepted') {
      cursor += 1;
      continue;
    }

    if (type === 'tsumo') {
      const tsumoActor = Number(ev.actor ?? -1);
      if ((action.type === 'dahai' || action.type === 'reach') && tsumoActor === action.actor) {
        applyReplayEventToHands(hands, ev);
        cursor += 1;
        if (action.type === 'reach') {
          if (phase !== 'post') return cursor;
          continue;
        }
        if (phase !== 'post') return cursor;
        continue;
      }
      return cursor;
    }

    if (eventMatchesAction(ev, action)) {
      if (phase === 'post') {
        applyReplayEventToHands(hands, ev);
        cursor += 1;
      }
      return cursor;
    }

    return cursor;
  }

  return cursor;
}

function buildReplayHandsForBoard(
  events: ReplayEvent[] | null,
  data: ReplayData | null,
  currentStep: number,
  currentEntry: ReplayData['log'][number] | null,
  boardPhase: ReplayBoardPhase,
): string[][] | null {
  if (!events || !data || !currentEntry) return null;
  const range = findKyokuEventRange(events, currentEntry);
  if (!range) return null;

  const kyokuEvents = events.slice(range.startIdx, range.endIdx);
  const startEv = kyokuEvents[0];
  const hands = ((startEv.tehais as string[][] | undefined) ?? [[], [], [], []]).map((tiles) => [...tiles]);
  let eventCursor = 1;

  const currentKyokuEntries = data.log.filter((entry) =>
    entry.kyoku_key.bakaze === currentEntry.kyoku_key.bakaze &&
    entry.kyoku_key.kyoku === currentEntry.kyoku_key.kyoku &&
    entry.kyoku_key.honba === currentEntry.kyoku_key.honba,
  );

  for (const entry of currentKyokuEntries) {
    const action = (entry.gt_action ?? entry.chosen) as Action | null | undefined;
    if (entry.step < currentStep) {
      eventCursor = applyEntryEventsToHands(kyokuEvents, eventCursor, hands, action, 'post');
      continue;
    }
    if (entry.step === currentStep) {
      eventCursor = applyEntryEventsToHands(
        kyokuEvents,
        eventCursor,
        hands,
        action,
        boardPhase === 'post' ? 'post' : 'pre',
      );
      break;
    }
  }

  return hands.map((tiles) => [...tiles]);
}

function buildReplayResultSummary(
  events: ReplayEvent[] | null,
  entry: ReplayData['log'][number] | undefined,
  playerNames: string[],
): ResultSummary | null {
  if (!events || !entry) return null;
  const action = (entry.gt_action ?? entry.chosen) as Action | undefined;
  if (!action || (action.type !== 'hora' && action.type !== 'ryukyoku')) return null;
  const kyokuKey = entry.kyoku_key;
  if (!kyokuKey) return null;

  let startIdx = -1;
  let endIdx = events.length;
  for (let i = 0; i < events.length; i++) {
    const ev = events[i];
    if (
      ev.type === 'start_kyoku' &&
      ev.bakaze === kyokuKey.bakaze &&
      ev.kyoku === kyokuKey.kyoku &&
      ev.honba === kyokuKey.honba
    ) {
      startIdx = i;
      continue;
    }
    if (startIdx >= 0 && i > startIdx && ev.type === 'start_kyoku') {
      endIdx = i;
      break;
    }
  }
  if (startIdx < 0) return null;
  const kyokuEvents = events.slice(startIdx, endIdx);
  const resultEvent = kyokuEvents.find((ev) => {
    if (ev.type !== action.type) return false;
    if (action.type === 'hora') {
      return ev.actor === action.actor && ev.pai === action.pai && ev.target === action.target;
    }
    return true;
  }) as ReplayEvent | undefined;
  if (!resultEvent) return null;

  const startEv = kyokuEvents[0];
  const hands = ((startEv.tehais as string[][] | undefined) ?? [[], [], [], []]).map((tiles) => [...tiles]);
  const melds = [[], [], [], []] as Array<Array<{ type: string; pai: string; consumed: string[] }>>;
  const doraMarkers = [String(startEv.dora_marker ?? '')].filter(Boolean);
  let inferredKyotaku = Number(startEv.kyotaku ?? 0);
  const inferredHonba = Number(startEv.honba ?? kyokuKey.honba ?? 0);
  let resultIndex = kyokuEvents.indexOf(resultEvent);
  for (let i = 1; i < resultIndex; i++) {
    const ev = kyokuEvents[i];
    const type = String(ev.type ?? '');
    const actor = Number(ev.actor ?? -1);
    if (type === 'tsumo' && actor >= 0) {
      hands[actor].push(String(ev.pai ?? ''));
    } else if (type === 'dahai' && actor >= 0) {
      removeTileOnce(hands[actor], String(ev.pai ?? ''));
    } else if (['chi', 'pon', 'daiminkan', 'ankan'].includes(type) && actor >= 0) {
      const consumed = ((ev.consumed as string[] | undefined) ?? []).map(String);
      for (const tile of consumed) removeTileOnce(hands[actor], tile);
      melds[actor].push({ type, pai: String(ev.pai ?? ''), consumed });
    } else if (type === 'kakan_accepted' && actor >= 0) {
      removeTileOnce(hands[actor], String(ev.pai ?? ''));
      const meld = melds[actor].find((m) => m.type === 'pon' && m.pai === String(ev.pai ?? ''));
      if (meld) meld.type = 'kakan';
    } else if (type === 'dora') {
      doraMarkers.push(String(ev.dora_marker ?? ''));
    } else if (type === 'reach_accepted') {
      if (typeof ev.kyotaku === 'number') inferredKyotaku = Number(ev.kyotaku);
      else inferredKyotaku += 1;
    }
  }

  if (action.type === 'hora' && action.actor !== undefined && action.pai && !action.is_tsumo) {
    hands[action.actor].push(action.pai);
  }

  const afterScores = ((resultEvent.scores as number[] | undefined) ?? entry.scores ?? [0, 0, 0, 0]).map(Number);
  const deltas = ((resultEvent.deltas as number[] | undefined) ?? [0, 0, 0, 0]).map(Number);
  const beforeScores = afterScores.map((score, idx) => score - (deltas[idx] ?? 0));
  const honba = Number(resultEvent.honba ?? inferredHonba);
  const kyotaku = Number(resultEvent.kyotaku ?? inferredKyotaku);

  if (action.type === 'ryukyoku') {
    return {
      type: 'ryukyoku',
      title: '流局',
      subtitle: '本局无和牌',
      honba,
      kyotaku,
      doraMarkers,
      uraMarkers: ((resultEvent.ura_dora_markers as string[] | undefined) ?? []).map(String),
      yakuLines: [],
      scoreLines: afterScores.map((score, pid) => ({
        pid,
        label: getSeatLabel(entry.oya, pid),
        name: replayPlayerDisplayName(playerNames, pid),
        before: beforeScores[pid],
        delta: deltas[pid] ?? 0,
        after: score,
      })),
      winnerHand: [],
      winnerMelds: [],
      winner: null,
      winTile: null,
    };
  }

  const winner = Number(action.actor);
  const winnerHand = winner >= 0 ? sortTilesForResult(hands[winner], String(action.pai ?? '')) : [];
  const winnerMelds = winner >= 0 ? melds[winner] : [];
  const cost = (resultEvent.cost as Action['cost']) ?? action.cost ?? {};
  const yakuList = ((resultEvent.yaku as string[] | undefined) ?? action.yaku ?? []).map(String);
  const structuredYaku =
    ((resultEvent.yaku_details as Action['yaku_details'] | undefined) ?? action.yaku_details ?? []).map((detail) => ({
      key: String(detail.key),
      name: String(detail.name),
      han: Number(detail.han ?? 0),
    }));
  const yakuDetails =
    structuredYaku.length > 0
      ? structuredYaku
      : buildFallbackYakuDetails(
          yakuList,
          winnerHand,
          winnerMelds,
          doraMarkers,
          ((resultEvent.ura_dora_markers as string[] | undefined) ?? (action.ura_dora_markers ?? [])).map(String),
        );
  const yakuLines = yakuDetails.map((detail) => {
    const label = YAKU_LABELS[detail.name] ?? detail.name;
    return `${label} · ${detail.han}翻`;
  });
  const levelKey = String(cost?.yaku_level ?? '').toLowerCase();
  const baseRonPoints = Number(cost?.main ?? 0);
  const baseTsumoMain = Number(cost?.main ?? 0);
  const baseTsumoAdditional = Number(cost?.additional ?? 0);
  const title = levelKey
    ? action.is_tsumo
      ? baseTsumoAdditional > 0 && baseTsumoAdditional !== baseTsumoMain
        ? `${RESULT_LEVEL_LABELS[levelKey] ?? cost?.yaku_level} ${baseTsumoAdditional}-${baseTsumoMain}点`
        : `${RESULT_LEVEL_LABELS[levelKey] ?? cost?.yaku_level} ${baseTsumoMain}点`
      : `${RESULT_LEVEL_LABELS[levelKey] ?? cost?.yaku_level} ${baseRonPoints || Number(cost?.total ?? 0)}点`
    : action.is_tsumo
      ? `${Number(resultEvent.han ?? action.han ?? 0)}翻 ${Number(resultEvent.fu ?? action.fu ?? 0)}符 ${baseTsumoAdditional}-${baseTsumoMain}点`
      : `${Number(resultEvent.han ?? action.han ?? 0)}翻 ${Number(resultEvent.fu ?? action.fu ?? 0)}符 ${baseRonPoints}点`;
  const subtitle = action.is_tsumo ? '自摸和了' : '荣和';

  return {
    type: 'hora',
    title,
    subtitle,
    honba,
    kyotaku,
    doraMarkers,
    uraMarkers: ((resultEvent.ura_dora_markers as string[] | undefined) ?? (action.ura_dora_markers ?? [])).map(String),
    yakuLines,
    scoreLines: afterScores.map((score, pid) => ({
      pid,
      label: getSeatLabel(entry.oya, pid),
      name: replayPlayerDisplayName(playerNames, pid),
      before: beforeScores[pid],
      delta: deltas[pid] ?? 0,
      after: score,
    })),
    winnerHand,
    winnerMelds,
    winner,
    winTile: String(action.pai ?? ''),
  };
}

function ReplayResultOverlay({ summary, playerNames }: { summary: ResultSummary; playerNames: string[] }) {
  return (
    <div
      style={{
        position: 'absolute',
        inset: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        pointerEvents: 'none',
        zIndex: 40,
        background: 'rgba(8, 12, 16, 0.28)',
        backdropFilter: 'blur(2px)',
      }}
    >
      <div
        style={{
          width: 900,
          maxWidth: '92%',
          minHeight: 360,
          borderRadius: 24,
          background: 'linear-gradient(180deg, rgba(22,26,31,0.96) 0%, rgba(12,16,20,0.96) 100%)',
          border: '1px solid rgba(212,168,83,0.28)',
          boxShadow: '0 24px 60px rgba(0,0,0,0.35)',
          padding: '22px 26px 18px',
          color: '#f4efe3',
          display: 'grid',
          gridTemplateRows: 'auto auto auto 1fr auto',
          gap: 14,
        }}
      >
        <div style={{ display: 'grid', gridTemplateColumns: '1fr auto', alignItems: 'start', gap: 16 }}>
          <div>
            <div style={{ fontSize: 30, lineHeight: 1.1, fontWeight: 900, color: 'var(--gold)' }}>{summary.title}</div>
            <div style={{ marginTop: 6, fontSize: 14, color: 'rgba(244,239,227,0.74)' }}>{summary.subtitle}</div>
          </div>
          <div style={{ display: 'grid', gap: 4, textAlign: 'right' }}>
            <div style={{ fontSize: 13, color: 'rgba(244,239,227,0.7)' }}>本场: {summary.honba}</div>
            <div style={{ fontSize: 13, color: 'rgba(244,239,227,0.7)' }}>供托: {summary.kyotaku}</div>
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
          <IndicatorRow title="宝牌指示牌" tiles={summary.doraMarkers} />
          <IndicatorRow title="里宝牌指示牌" tiles={summary.uraMarkers} emptyLabel="未记录" />
        </div>

        <div style={{ display: 'grid', gap: 12, alignItems: 'start' }}>
          <div style={{ display: 'grid', gap: 10 }}>
            <div style={{ fontSize: 13, fontWeight: 800, color: 'rgba(244,239,227,0.78)' }}>
              {summary.winner !== null ? `和牌手牌 · ${replayPlayerDisplayName(playerNames, summary.winner)}` : '局结果'}
            </div>
            {summary.winner !== null ? (
              <>
                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                  {summary.winnerHand.map((tile, idx) => (
                    <Tile key={`${tile}-${idx}`} tile={tile} size="normal" highlighted={tile === summary.winTile && idx === summary.winnerHand.lastIndexOf(tile)} />
                  ))}
                </div>
                {summary.winnerMelds.length > 0 && (
                  <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
                    {summary.winnerMelds.map((meld, idx) => (
                      <div key={`${meld.type}-${idx}`} style={{ display: 'flex', gap: 4, alignItems: 'center' }}>
                        <span style={{ fontSize: 11, color: 'rgba(244,239,227,0.6)' }}>{meld.type}</span>
                        {[...meld.consumed, meld.pai].filter(Boolean).map((tile, tileIdx) => (
                          <Tile key={`${tile}-${tileIdx}`} tile={tile} size="small" />
                        ))}
                      </div>
                    ))}
                  </div>
                )}
              </>
            ) : (
              <div style={{ fontSize: 13, color: 'rgba(244,239,227,0.72)' }}>流局，没有和牌者手牌展示。</div>
            )}
          </div>

          <div style={{ display: 'grid', gap: 8, alignContent: 'start' }}>
            <div style={{ fontSize: 13, fontWeight: 800, color: 'rgba(244,239,227,0.78)' }}>役种 / 结算明细</div>
            <div
              style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(2, minmax(0, 1fr))',
                gap: '8px 20px',
                alignItems: 'start',
              }}
            >
              {summary.yakuLines.length > 0 ? summary.yakuLines.map((line, idx) => (
                <div key={`${line}-${idx}`} style={{ fontSize: 14, color: '#f4efe3' }}>{line}</div>
              )) : (
                <div style={{ fontSize: 13, color: 'rgba(244,239,227,0.66)' }}>无役种列表</div>
              )}
            </div>
          </div>
        </div>

        <div
          style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(4, 1fr)',
            gap: 10,
            borderTop: '1px solid rgba(255,255,255,0.08)',
            paddingTop: 14,
          }}
        >
          {summary.scoreLines.map((row) => (
            <div key={row.pid} style={{ padding: '10px 12px', borderRadius: 14, background: 'rgba(255,255,255,0.04)' }}>
              <div style={{ fontSize: 12, color: 'rgba(244,239,227,0.62)', marginBottom: 4 }}>
                {row.label} {row.name}
              </div>
              <div style={{ fontFamily: 'Menlo, monospace', fontSize: 13, color: '#f4efe3' }}>
                {row.before.toLocaleString()} {row.delta >= 0 ? '+' : ''}{row.delta.toLocaleString()}
              </div>
              <div style={{ marginTop: 4, fontSize: 12, color: row.delta >= 0 ? '#8ce98c' : '#ff9898' }}>
                {'=>'} {row.after.toLocaleString()}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

function IndicatorRow({
  title,
  tiles,
  emptyLabel = '无',
}: {
  title: string;
  tiles: string[];
  emptyLabel?: string;
}) {
  return (
    <div style={{ display: 'grid', gap: 8 }}>
      <div style={{ fontSize: 12, fontWeight: 800, color: 'rgba(244,239,227,0.72)' }}>{title}</div>
      <div style={{ display: 'flex', gap: 6, minHeight: 46, alignItems: 'center' }}>
        {tiles.length > 0 ? tiles.map((tile, idx) => <Tile key={`${tile}-${idx}`} tile={tile} size="normal" />) : (
          <div style={{ fontSize: 12, color: 'rgba(244,239,227,0.5)' }}>{emptyLabel}</div>
        )}
      </div>
    </div>
  );
}
