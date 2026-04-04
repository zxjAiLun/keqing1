// src/replay_ui/src/pages/BattlePage.tsx
import { useState, useCallback, useEffect, useRef } from "react";
import { MahjongTable } from "../components/BattleBoard/MahjongTable";
import { startBattle, doAction, closeBattle, fetchWithTimeout } from "../api/battleApi";
import { useAutoActions } from "../hooks/useAutoActions";
import { useBattlePolling } from "../hooks/useBattlePolling";
import { useConnectionManager } from "../hooks/useConnectionManager";
import type { BattleState, Action, StartBattleRequest } from "../types/battle";

export function BattlePage() {
  const [gameId, setGameId] = useState<string | null>(null);
  const [state, setState] = useState<BattleState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedTile, setSelectedTile] = useState<string | null>(null);
  const [selectedTileIdx, setSelectedTileIdx] = useState<number | null>(null);
  const [playerName, setPlayerName] = useState("玩家");
  const [botModel, setBotModel] = useState("keqingv1");
  const [autoHora, setAutoHora] = useState(true);       // 自动胡牌，默认开
  const [noMeld, setNoMeld] = useState(false);           // 不响应附露，默认关
  const [autoTsumogiri, setAutoTsumogiri] = useState(false); // 自动摸切，默认关
  const pendingActionRef = useRef(false);

  const startNewGame = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const req: StartBattleRequest = { player_name: playerName, bot_count: 3, bot_model: botModel };
      const res = await startBattle(req);
      setGameId(res.game_id);
      setState(res.state);
    } catch (e) {
      setError(e instanceof Error ? e.message : "启动失败");
    } finally {
      setLoading(false);
    }
  }, [playerName, botModel]);

  const handleAction = useCallback(
    async (action: Action) => {
      if (!gameId) return;
      if (pendingActionRef.current) return;
      pendingActionRef.current = true;
      setLoading(true);
      try {
        const res = await doAction({ game_id: gameId, action });
        setState(res.state);
      } catch (e) {
        setError(e instanceof Error ? e.message : "操作失败");
      } finally {
        setLoading(false);
        pendingActionRef.current = false;
      }
    },
    [gameId]
  );

  const downloadExport = useCallback(async (format: "mjai" | "tenhou6") => {
    if (!gameId) return;
    try {
      const endpoint = format === "mjai" ? "export_mjai" : "export_tenhou6";
      const res = await fetchWithTimeout(`/api/battle/${endpoint}/${gameId}`);
      if (!res.ok) throw new Error("导出失败");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `${gameId}_${format}.${format === "mjai" ? "jsonl" : "json"}`;
      a.click();
      URL.revokeObjectURL(url);
    } catch {
      alert("导出失败");
    }
  }, [gameId]);

  const { isMyTurn, legalActions, hasHora, hasMeld, hasDahai, onlyNoneResponse, needsDecision } = useAutoActions({
    state,
    autoHora,
    noMeld,
    autoTsumogiri,
  });

  useEffect(() => {
    if (isMyTurn) return;
    setSelectedTile(null);
    setSelectedTileIdx(null);
  }, [isMyTurn, state?.game_id, state?.actor_to_move, state?.last_discard]);

  // 自动化：自动胡牌 / 不响应附露 / 自动摸切
  useEffect(() => {
    if (!isMyTurn || !gameId || pendingActionRef.current) return;
    if (onlyNoneResponse) {
      const noneAction = legalActions.find(a => a.type === "none");
      if (noneAction) { handleAction(noneAction); return; }
    }
    if (autoHora && hasHora) {
      const horaAction = legalActions.find(a => a.type === "hora");
      if (horaAction) { handleAction(horaAction); return; }
    }
    if (noMeld && hasMeld && !hasHora) {
      const noneAction = legalActions.find(a => a.type === "none");
      if (noneAction) { handleAction(noneAction); return; }
    }
    if (autoTsumogiri && hasDahai) {
      const tsumogiriAction = legalActions.find(a => a.type === "dahai" && a.tsumogiri);
      if (tsumogiriAction) { handleAction(tsumogiriAction); return; }
    }
  }, [isMyTurn, gameId, state?.legal_actions, autoHora, noMeld, autoTsumogiri, hasHora, hasMeld, hasDahai, onlyNoneResponse, legalActions, handleAction]);

  useBattlePolling({
    gameId,
    state,
    needsDecision,
    pendingActionRef,
    onStateUpdate: setState,
  });

  // 退出对局
  const [showQuitConfirm, setShowQuitConfirm] = useState(false);
  const handleQuit = useCallback(async () => {
    if (!gameId) return;
    try {
      await fetch(`/api/battle/quit/${gameId}`, { method: 'POST' });
    } catch { /* ignore */ }
    setGameId(null);
    setState(null);
    setShowQuitConfirm(false);
  }, [gameId]);

  // 连接管理（心跳/断线/重连）
  const { status: connStatus, reconnectAttempts } = useConnectionManager({
    gameId: state?.phase === 'playing' ? gameId : null,
    playerId: state?.human_player_id ?? 0,
    onStateUpdate: setState,
    onGiveUp: () => { setGameId(null); setState(null); },
  });

  if (!state) {
    return (
      <div
        style={{
          background: '#f0f2f5',
          minHeight: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 24,
        }}
      >
        {/* 麻将图标 + 旋转动画 */}
        <div
          style={{
            width: 64,
            height: 64,
            borderRadius: 16,
            background: 'linear-gradient(135deg, #1e4a7a 0%, #0f2d4a 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 8px 24px rgba(30,74,122,0.3)',
            marginBottom: 8,
            animation: loading ? "spinIcon 1.5s linear infinite" : "floatIcon 3s ease-in-out infinite",
          }}
        >
          <span style={{ color: '#fff', fontWeight: 700, fontSize: 24 }}>麻</span>
        </div>
        <h1 style={{ fontSize: 28, fontWeight: 700, color: '#1f2937' }}>立直麻将对战</h1>

        <div
          style={{
            background: '#fff',
            border: '1px solid #e5e7eb',
            borderRadius: 12,
            padding: 20,
            width: 320,
            boxShadow: '0 4px 16px rgba(0,0,0,0.07)',
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
            <input
              type="text"
              placeholder="你的名字"
              value={playerName}
              onChange={(e) => setPlayerName(e.target.value)}
              style={{
                width: '100%',
                padding: '9px 12px',
                border: '1px solid #d1d5db',
                borderRadius: 8,
                fontSize: 14,
                background: '#f9fafb',
                color: '#1f2937',
                outline: 'none',
                transition: 'border-color 0.15s',
              }}
              onFocus={(e) => (e.target.style.borderColor = '#3498db')}
              onBlur={(e) => (e.target.style.borderColor = '#d1d5db')}
            />
            <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
              <label style={{ fontSize: 12, color: '#6b7280', fontWeight: 500 }}>对手模型</label>
              <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                {['keqingv1', 'keqingv2', 'keqingv3', 'rulebase'].map(m => (
                  <button
                    key={m}
                    onClick={() => setBotModel(m)}
                    style={{
                      padding: '5px 12px', borderRadius: 6, fontSize: 13, fontWeight: 500,
                      border: `2px solid ${botModel === m ? '#1e4a7a' : '#d1d5db'}`,
                      background: botModel === m ? '#1e4a7a' : '#f9fafb',
                      color: botModel === m ? '#fff' : '#374151',
                      cursor: 'pointer', transition: 'all 0.15s',
                    }}
                  >{m}</button>
                ))}
              </div>
            </div>
            <button
              onClick={startNewGame}
              disabled={loading}
              style={{
                width: '100%',
                padding: '10px',
                borderRadius: 8,
                border: 'none',
                background: loading
                  ? 'linear-gradient(135deg, #9ca3af 0%, #8b9298 100%)'
                  : 'linear-gradient(135deg, #1e4a7a 0%, #0f2d4a 100%)',
                color: '#fff',
                fontSize: 15,
                fontWeight: 600,
                cursor: loading ? 'not-allowed' : 'pointer',
                boxShadow: loading ? 'none' : '0 4px 12px rgba(30,74,122,0.25)',
                transition: 'all 0.2s',
              }}
            >
              {loading ? (
                <span style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}>
                  <span style={{ animation: "dotPulse 1.2s ease-in-out infinite" }}>●</span>
                  洗牌中...
                </span>
              ) : '开始对战'}
            </button>
            {error && (
              <div style={{
                fontSize: 13, textAlign: 'center', color: '#dc2626',
                background: '#fef2f2', padding: '6px 10px', borderRadius: 6, border: '1px solid #fecaca'
              }}>
                {error}
              </div>
            )}
          </div>
        </div>

        <p style={{ fontSize: 13, color: '#9ca3af' }}>你将与 3 个 Bot 对战</p>

        <style>{`
          @keyframes spinIcon { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
          @keyframes floatIcon { 0%, 100% { transform: translateY(0); } 50% { transform: translateY(-6px); } }
          @keyframes dotPulse { 0%, 100% { opacity: 0.3; } 50% { opacity: 1; } }
        `}</style>
      </div>
    );
  }

  if (state.phase === "ended") {
    const winner = state.winner;
    const isWinner = winner === state.human_player_id;

    return (
      <div
        style={{
          background: 'var(--page-bg)',
          minHeight: '100%',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 24,
        }}
      >
        {/* 天凤风格终局面板 */}
        <div
          style={{
            background: 'var(--result-panel-bg)',
            border: '1px solid var(--result-panel-border)',
            borderRadius: 8,
            boxShadow: 'var(--result-panel-shadow)',
            padding: '24px 28px 20px',
            width: 480,
            maxWidth: '92%',
            display: 'flex',
            flexDirection: 'column',
            gap: 16,
          }}
        >
          {/* 标题行 */}
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--result-title)' }}>
              {isWinner ? '和了' : `和了 · ${state.player_info[winner ?? 0]?.name}`}
            </div>
            <div style={{
              width: 36, height: 36, borderRadius: '50%',
              background: isWinner
                ? 'linear-gradient(135deg, #d4a853 0%, #b8922e 100%)'
                : 'rgba(255,255,255,0.1)',
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              flexShrink: 0,
            }}>
              <span style={{ color: '#fff', fontWeight: 700, fontSize: 16 }}>
                {isWinner ? '勝' : '負'}
              </span>
            </div>
          </div>

          {/* 排名列表 */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {[...state.scores.entries()].sort(([, a], [, b]) => b - a).map(([pid, score], rank) => {
              const isHuman = pid === state.human_player_id;
              const delta = score - (state.scores[state.human_player_id] ?? 0);
              const rankLabel = ['1位', '2位', '3位', '4位'][rank];
              return (
                <div key={pid} style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: 10,
                  padding: '7px 10px',
                  borderRadius: 5,
                  background: isHuman ? 'rgba(255,255,255,0.06)' : 'transparent',
                  border: isHuman ? '1px solid rgba(255,255,255,0.1)' : '1px solid transparent',
                }}>
                  <span style={{ fontSize: 11, color: 'var(--result-muted)', width: 24, flexShrink: 0 }}>{rankLabel}</span>
                  <span style={{ fontSize: 13, color: isHuman ? 'var(--result-title)' : 'var(--result-muted)', flex: 1 }}>
                    {state.player_info[pid]?.name ?? `Player ${pid}`}
                  </span>
                  <span style={{
                    fontFamily: 'Menlo, monospace',
                    fontSize: 13,
                    color: pid === state.human_player_id
                      ? (delta >= 0 ? 'var(--result-positive)' : 'var(--result-negative)')
                      : 'var(--result-muted)',
                  }}>
                    {pid === state.human_player_id
                      ? `${delta >= 0 ? '+' : ''}${delta.toLocaleString()} → `
                      : ''}{score.toLocaleString()}
                  </span>
                </div>
              );
            })}
          </div>
        </div>

        {/* 操作按钮 */}
        <div style={{ display: 'flex', gap: 8 }}>
          <button
            onClick={async () => {
              if (gameId) await closeBattle(gameId);
              setState(null);
              setGameId(null);
              setSelectedTile(null);
            }}
            style={{
              padding: '8px 20px',
              borderRadius: 6,
              border: '1px solid var(--result-panel-border)',
              background: 'var(--result-panel-bg)',
              color: 'var(--result-muted)',
              fontSize: 13,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            再来一局
          </button>
          <button
            onClick={() => downloadExport("mjai")}
            style={{
              padding: '8px 16px',
              borderRadius: 6,
              border: '1px solid rgba(52,152,219,0.4)',
              background: 'transparent',
              color: 'rgba(52,152,219,0.8)',
              fontSize: 13,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Mjai Log
          </button>
          <button
            onClick={() => downloadExport("tenhou6")}
            style={{
              padding: '8px 16px',
              borderRadius: 6,
              border: '1px solid rgba(39,174,96,0.4)',
              background: 'transparent',
              color: 'rgba(39,174,96,0.8)',
              fontSize: 13,
              fontWeight: 600,
              cursor: 'pointer',
            }}
          >
            Tenhou6
          </button>
        </div>
      </div>
    );
  }

  return (
    <div style={{ height: '100%', padding: 16, background: '#f0f2f5', position: 'relative' }}>
      <MahjongTable
        state={state}
        onAction={handleAction}
        isMyTurn={isMyTurn}
        selectedTile={selectedTile}
        selectedTileIdx={selectedTileIdx}
        onTileSelect={(tile, idx) => { setSelectedTile(tile); setSelectedTileIdx(idx ?? null); }}
        autoHora={autoHora} setAutoHora={setAutoHora}
        noMeld={noMeld} setNoMeld={setNoMeld}
        autoTsumogiri={autoTsumogiri} setAutoTsumogiri={setAutoTsumogiri}
        actionPending={loading}
      />

      {/* 退出按钮 */}
      <button
        onClick={() => setShowQuitConfirm(true)}
        style={{
          position: 'absolute', top: 24, right: 24, zIndex: 100,
          padding: '6px 14px', borderRadius: 6, border: '1px solid #e74c3c',
          background: 'rgba(255,255,255,0.9)', color: '#e74c3c',
          fontSize: 13, fontWeight: 600, cursor: 'pointer',
        }}
      >
        退出对局
      </button>

      {/* 退出确认对话框 */}
      {showQuitConfirm && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.5)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 200,
        }}>
          <div style={{
            background: '#fff', borderRadius: 12, padding: 28, minWidth: 280,
            boxShadow: '0 8px 32px rgba(0,0,0,0.2)', textAlign: 'center',
          }}>
            <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 8 }}>确认退出对局？</div>
            <div style={{ fontSize: 13, color: '#666', marginBottom: 20 }}>退出后该座位将由 Bot 接管，对局继续。</div>
            <div style={{ display: 'flex', gap: 12, justifyContent: 'center' }}>
              <button onClick={() => setShowQuitConfirm(false)}
                style={{ padding: '8px 20px', borderRadius: 6, border: '1px solid #ccc', background: '#fff', cursor: 'pointer', fontSize: 13 }}>
                取消
              </button>
              <button onClick={handleQuit}
                style={{ padding: '8px 20px', borderRadius: 6, border: 'none', background: '#e74c3c', color: '#fff', cursor: 'pointer', fontSize: 13, fontWeight: 600 }}>
                确认退出
              </button>
            </div>
          </div>
        </div>
      )}

      {/* 断线/重连提示覆盖层 */}
      {connStatus === 'disconnected' && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,0.6)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 300,
        }}>
          <div style={{
            background: '#fff', borderRadius: 12, padding: 32, minWidth: 300,
            boxShadow: '0 8px 32px rgba(0,0,0,0.3)', textAlign: 'center',
          }}>
            <div style={{ fontSize: 32, marginBottom: 12 }}>⚠️</div>
            <div style={{ fontSize: 16, fontWeight: 700, marginBottom: 8 }}>连接已断开</div>
            <div style={{ fontSize: 13, color: '#666', marginBottom: 4 }}>正在尝试重连... ({reconnectAttempts}/5)</div>
            <div style={{ fontSize: 12, color: '#999' }}>若长时间无法重连，可以放弃本局。</div>
            <button onClick={() => { setGameId(null); setState(null); }}
              style={{ marginTop: 20, padding: '8px 20px', borderRadius: 6, border: '1px solid #ccc', background: '#fff', cursor: 'pointer', fontSize: 13 }}>
              放弃本局
            </button>
          </div>
        </div>
      )}

      {connStatus === 'reconnecting' && (
        <div style={{
          position: 'fixed', bottom: 24, left: '50%', transform: 'translateX(-50%)',
          background: 'rgba(0,0,0,0.75)', color: '#fff', borderRadius: 20,
          padding: '8px 20px', fontSize: 13, zIndex: 300,
        }}>
          重连中... ({reconnectAttempts}/5)
        </div>
      )}
    </div>
  );
}
