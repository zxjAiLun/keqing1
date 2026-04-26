// src/replay_ui/src/pages/BotBattlePage.tsx
import { useState, useEffect, useRef } from "react";
import { MahjongTable } from "../components/BattleBoard/MahjongTable";
import { fetchWithTimeout } from "../api/battleApi";
import { PageHeader, PageShell, SectionTitle } from "../components/Layout/PageScaffold";
import { subtleButtonStyle } from "../components/Layout/layoutStyles";
import type { BattleState } from "../types/battle";
import type { BotType } from "../types/bot";
import { BOT_CATALOG, DEFAULT_BOT_TYPE, getBotCatalogEntry } from "../utils/botCatalog";

export function BotBattlePage() {
  const [gameId, setGameId] = useState<string | null>(null);
  const [state, setState] = useState<BattleState | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [botModel, setBotModel] = useState<BotType>(DEFAULT_BOT_TYPE);
  const pollingRef = useRef<number | null>(null);
  const mountedRef = useRef(true);

  const startBotBattle = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetchWithTimeout("/api/battle/start_4bot", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ bot_model: botModel }),
      });
      if (!res.ok) throw new Error("启动失败");
      const data = await res.json();
      setGameId(data.game_id);
      setState(data.state);
    } catch (e) {
      setError(e instanceof Error ? e.message : "启动失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    mountedRef.current = true;
    return () => { mountedRef.current = false; };
  }, []);

  useEffect(() => {
    if (!gameId) return;
    if (pollingRef.current) clearInterval(pollingRef.current);
    pollingRef.current = window.setInterval(async () => {
      try {
        const res = await fetchWithTimeout(`/api/battle/state/${gameId}?player_id=0`);
        if (!res.ok) return;
        const data = await res.json();
        if (mountedRef.current) {
          setState(data.state);
        }
      } catch { /* ignore */ }
    }, 500);
    return () => { if (pollingRef.current) clearInterval(pollingRef.current); };
  }, [gameId]);

  const downloadExport = async (format: "mjai" | "tenhou6") => {
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
  };

  const selectedBot = getBotCatalogEntry(botModel);

  if (!state) {
    return (
      <PageShell width={720}>
        <PageHeader
          eyebrow="Bot Arena"
          title="4 Bot 对战"
          description="用于观察当前活跃模型线之间的完整对局流程。适合快速回看回合推进、兼容面表现和导出实验牌谱。"
        />
        <div
          style={{
            background: "var(--page-bg)",
            minHeight: "60vh",
            display: "flex",
            flexDirection: "column",
            alignItems: "center",
            justifyContent: "center",
            gap: 24,
          }}
        >
        <div
          style={{
            width: 64,
            height: 64,
            borderRadius: 16,
            background: "linear-gradient(135deg, #8e44ad 0%, #7d3c9e 100%)",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            boxShadow: "0 8px 24px rgba(142,68,173,0.3)",
            marginBottom: 8,
          }}
        >
          <span style={{ color: "#fff", fontWeight: 700, fontSize: 24 }}>🤖</span>
        </div>
        <h1 style={{ fontSize: 28, fontWeight: 700, color: "var(--text-primary)" }}>
          4 Bot 对战
        </h1>

        <div className="card" style={{ maxWidth: 320 }}>
          <SectionTitle title="开始一局自动对战" description="启动后会持续轮询局面，结束后可导出实验结果。" />
          <div style={{ display: "flex", flexDirection: "column", gap: 6, marginBottom: 12 }}>
            <label style={{ fontSize: 12, color: "var(--text-muted)", fontWeight: 500 }}>
              Bot 类型
            </label>
            <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
              {BOT_CATALOG.map((bot) => (
                <button
                  key={bot.value}
                  onClick={() => setBotModel(bot.value)}
                  style={{
                    padding: "10px 12px",
                    borderRadius: 8,
                    fontSize: 13,
                    fontWeight: 500,
                    border: `2px solid ${botModel === bot.value ? "#8e44ad" : "var(--border-muted)"}`,
                    background: botModel === bot.value ? "rgba(142,68,173,0.08)" : "var(--surface-subtle)",
                    color: "var(--text-primary)",
                    cursor: "pointer",
                    transition: "all var(--transition)",
                    textAlign: "left",
                  }}
                >
                  <div style={{ display: "flex", justifyContent: "space-between", gap: 10, alignItems: "center" }}>
                    <span style={{ fontWeight: 700 }}>{bot.label}</span>
                    <span style={{ fontSize: 11, color: botModel === bot.value ? "#8e44ad" : "var(--text-muted)" }}>{bot.badge}</span>
                  </div>
                  <div style={{ marginTop: 3, fontSize: 12, color: "var(--text-muted)" }}>{bot.description}</div>
                </button>
              ))}
            </div>
            <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
              当前选择：{selectedBot.label}，{selectedBot.description}
            </div>
          </div>
          <button
            onClick={startBotBattle}
            disabled={loading}
            style={{
              width: "100%",
              padding: "10px",
              borderRadius: "var(--radius-sm)",
              border: "none",
              background: loading ? "var(--text-muted)" : "#8e44ad",
              color: "#fff",
              fontSize: 15,
              fontWeight: 600,
              cursor: loading ? "not-allowed" : "pointer",
              transition: "background var(--transition)",
            }}
          >
            {loading ? "启动中..." : "开始对战"}
          </button>
          {error && (
            <div style={{ fontSize: 13, textAlign: "center", color: "var(--error)", marginTop: 8 }}>
              {error}
            </div>
          )}
        </div>

        <p style={{ fontSize: 13, color: "var(--text-muted)" }}>
          默认观察 xmodel1 主线；需要 backup/baseline 对照时再切换。
        </p>
        </div>
      </PageShell>
    );
  }

  return (
    <PageShell width={1400}>
      <PageHeader
        eyebrow="Bot Arena"
        title="4 Bot 对战"
        description="实时观察自动对战流程。结束后可以直接导出 Mjai 或 Tenhou6 牌谱。当前默认模型线为 xmodel1。"
      />
      {state.phase === "ended" && (
        <div
          style={{
            display: "flex",
            gap: 12,
            marginBottom: 12,
            justifyContent: "center",
            flexWrap: "wrap",
          }}
        >
          <button
            onClick={() => downloadExport("mjai")}
            style={{
              padding: "8px 16px",
              borderRadius: "var(--radius-sm)",
              border: "none",
              background: "var(--accent)",
              color: "#fff",
              fontSize: 13,
              fontWeight: 600,
              cursor: "pointer",
              transition: "background var(--transition)",
            }}
          >
            导出 Mjai Log
          </button>
          <button
            onClick={() => downloadExport("tenhou6")}
            style={{
              padding: "8px 16px",
              borderRadius: "var(--radius-sm)",
              border: "none",
              background: "var(--success)",
              color: "#fff",
              fontSize: 13,
              fontWeight: 600,
              cursor: "pointer",
              transition: "background var(--transition)",
            }}
          >
            导出 Tenhou6
          </button>
          <button
            onClick={() => { setState(null); setGameId(null); }}
            style={subtleButtonStyle}
          >
            再来一局
          </button>
        </div>
      )}
      <div style={{ height: "calc(100dvh - 180px)", minHeight: 640, background: "var(--page-bg)" }}>
        <MahjongTable
          state={state}
          onAction={() => {}}
          isMyTurn={false}
          selectedTile={null}
          onTileSelect={() => {}}
          autoHora={false} setAutoHora={() => {}}
          noMeld={false} setNoMeld={() => {}}
          autoTsumogiri={false} setAutoTsumogiri={() => {}}
        />
      </div>
    </PageShell>
  );
}
