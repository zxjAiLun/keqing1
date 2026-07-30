// src/replay_ui/src/pages/PlayWithYouPage.tsx
// "Play with you" — summon Mortal-weight bot accounts into a Tenhou private
// room (e.g. L2147) as NoName guests, driven from the GUI, similar to
// mjai.ekyu.moe's Play with you.
import { useCallback, useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { PageShell, SectionTitle } from "../components/Layout/PageScaffold";
import {
  startPlayWithYou,
  stopPlayWithYou,
  getPlayWithYouStatus,
  type NetworkId,
  type SpeedId,
  type DeviceId,
  type BotInfo,
  type PlayWithYouStatus,
} from "../api/playwithyouApi";

const ACCENT = "#8e44ad";

const NETWORK_OPTIONS: Array<{ value: NetworkId; label: string; hint: string }> = [
  { value: "none", label: "none", hint: "不呼出" },
  { value: "mortal", label: "Mortal candidate", hint: "V2@74000 or 70k fallback" },
  { value: "70k", label: "Mortal 70k", hint: "70k anchor" },
  { value: "ext_mortal", label: "External Mortal", hint: "external reference" },
  { value: "custom", label: "自定义权重", hint: "绝对 .pth 路径" },
];

const SPEED_OPTIONS: Array<{ value: SpeedId; label: string }> = [
  { value: "slow", label: "Slow" },
  { value: "normal", label: "Normal" },
  { value: "fast", label: "Fast" },
  { value: "turbo", label: "Turbo" },
];

const DEVICE_OPTIONS: Array<{ value: DeviceId; label: string }> = [
  { value: "cuda", label: "CUDA" },
  { value: "cpu", label: "CPU" },
];

function shortSpec(spec: string): string {
  if (!spec) return spec;
  if (spec.includes("/") || spec.includes("\\")) {
    return spec.split(/[\\/]/).pop() || spec;
  }
  return spec;
}

function Segmented<T extends string>({
  options,
  value,
  onChange,
  disabled,
}: {
  options: Array<{ value: T; label: string }>;
  value: T;
  onChange: (v: T) => void;
  disabled?: boolean;
}) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: `repeat(${options.length}, 1fr)`, gap: 6 }}>
      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            disabled={disabled}
            onClick={() => onChange(opt.value)}
            style={{
              height: 32,
              borderRadius: 6,
              fontSize: 13,
              fontWeight: 700,
              border: `1px solid ${active ? ACCENT : "var(--border)"}`,
              background: active ? "rgba(142,68,173,0.08)" : "var(--surface-subtle)",
              color: active ? ACCENT : "var(--text-primary)",
              cursor: disabled ? "not-allowed" : "pointer",
              opacity: disabled ? 0.5 : 1,
            }}
          >
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}

export function PlayWithYouPage() {
  const [lobbyId, setLobbyId] = useState<string>("2147");
  const [speed, setSpeed] = useState<SpeedId>("normal");
  const [device, setDevice] = useState<DeviceId>("cuda");
  const [quantity, setQuantity] = useState<number>(3);
  const [networks, setNetworks] = useState<string[]>(["mortal", "70k", "ext_mortal", "none"]);
  const [customPaths, setCustomPaths] = useState<Record<number, string>>({});

  const [status, setStatus] = useState<PlayWithYouStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const logRef = useRef<HTMLDivElement | null>(null);
  const pollingRef = useRef<number | null>(null);

  const activeBotCount = networks
    .slice(0, quantity)
    .filter((n) => n && n !== "none").length;

  const stopPolling = useCallback(() => {
    if (pollingRef.current) {
      window.clearInterval(pollingRef.current);
      pollingRef.current = null;
    }
  }, []);

  const refresh = useCallback(async () => {
    try {
      const s = await getPlayWithYouStatus();
      setStatus(s);
      if (!s.running) stopPolling();
    } catch {
      /* ignore transient */
    }
  }, [stopPolling]);

  const start = async () => {
    setLoading(true);
    setError(null);
    try {
      const req = {
        lobby_id: lobbyId,
        speed,
        quantity,
        networks: [...networks],
        custom_paths: customPaths,
        device,
      };
      const s = await startPlayWithYou(req);
      setStatus(s);
      // Begin polling for logs / liveness.
      stopPolling();
      pollingRef.current = window.setInterval(refresh, 2000);
    } catch (e) {
      setError(e instanceof Error ? e.message : "启动失败");
    } finally {
      setLoading(false);
    }
  };

  const stop = async () => {
    setLoading(true);
    setError(null);
    try {
      const s = await stopPlayWithYou();
      setStatus(s);
      stopPolling();
    } catch (e) {
      setError(e instanceof Error ? e.message : "停止失败");
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => stopPolling, [stopPolling]);

  // Auto-scroll the log viewer to the bottom on update.
  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [status?.log_tail]);

  const isRunning = status?.running ?? false;
  // Tenhou's individual-room URL takes the numeric lobby id; the `L` prefix
  // belongs to the gateway protocol/display name, not the browser query.
  const joinUrl = `https://tenhou.net/0/?${status?.lobby_id ?? lobbyId}`;

  return (
    <PageShell width={1120}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          gap: 12,
          flexWrap: "wrap",
          marginBottom: 12,
        }}
      >
        <div>
          <div style={{ fontSize: 20, fontWeight: 800, color: "var(--text-primary)" }}>
            Play with you · 天凤在线呼出
          </div>
          <div style={{ fontSize: 12, color: "var(--text-muted)" }}>
            呼出最多 4 个 Mortal 权重账号进入指定天凤个室（如 L2147），你自己从天凤客户端加入对战。
          </div>
        </div>
        {isRunning ? (
          <button
            onClick={stop}
            disabled={loading}
            className="btn-primary"
            style={{ height: 34, padding: "0 16px", fontSize: 13, background: loading ? "var(--text-muted)" : "#c0392b" }}
          >
            {loading ? "处理中..." : "停止呼出"}
          </button>
        ) : (
          <button
            onClick={start}
            disabled={loading}
            className="btn-primary"
            style={{ height: 34, padding: "0 16px", fontSize: 13, background: loading ? "var(--text-muted)" : ACCENT }}
          >
            {loading ? "呼出中..." : "呼出账号"}
          </button>
        )}
      </div>

      {error && (
        <div style={{ fontSize: 13, color: "var(--error)", marginBottom: 10 }}>{error}</div>
      )}

      <div className="card" style={{ padding: 14, marginBottom: 12 }}>
        <SectionTitle title="设置" description="选择天凤个室、出牌速度、以及每个 AI 座位使用的 Mortal 网络。" />

        {/* Lobby ID + quantity */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 12 }}>
          <div>
            <label style={labelStyle}>Tenhou Lobby ID</label>
            <input
              value={lobbyId}
              onChange={(e) => setLobbyId(e.target.value)}
              placeholder="2147"
              disabled={isRunning}
              style={inputStyle}
            />
            <div style={hintStyle}>将进入 L{lobbyId || "2147"} 个室（半庄，4 人）。</div>
          </div>
          <div>
            <label style={labelStyle}>Selected quantity（呼出数量）</label>
            <Segmented
              options={[
                { value: "1", label: "1" },
                { value: "2", label: "2" },
                { value: "3", label: "3" },
                { value: "4", label: "4" },
              ]}
              value={String(quantity) as "1" | "2" | "3" | "4"}
              onChange={(v) => setQuantity(Number(v))}
              disabled={isRunning}
            />
            <div style={hintStyle}>将呼出前 {quantity} 个非 none 的 AI 座位（当前 {activeBotCount} 个）。</div>
          </div>
        </div>

        {/* Speed + device */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14, marginBottom: 12 }}>
          <div>
            <label style={labelStyle}>Speed（出牌速度）</label>
            <Segmented options={SPEED_OPTIONS} value={speed} onChange={setSpeed} disabled={isRunning} />
            <div style={hintStyle}>在自己回合前的思考停顿，便于观战。Turbo 即瞬时出牌。</div>
          </div>
          <div>
            <label style={labelStyle}>Device（推理设备）</label>
            <Segmented options={DEVICE_OPTIONS} value={device} onChange={setDevice} disabled={isRunning} />
            <div style={hintStyle}>无 GPU 时可切 CPU；launcher 会在 CUDA 不可用时自动回退。</div>
          </div>
        </div>

        {/* Per-AI network selectors */}
        <label style={labelStyle}>Mortal network（每个 AI 座位）</label>
        <div style={{ display: "grid", gap: 8 }}>
          {[0, 1, 2, 3].map((slot) => {
            const disabled = slot >= quantity || isRunning;
            const net = (networks[slot] as NetworkId) || "none";
            return (
              <div
                key={slot}
                style={{
                  display: "grid",
                  gridTemplateColumns: "84px 1fr",
                  gap: 10,
                  alignItems: "center",
                  opacity: disabled ? 0.45 : 1,
                }}
              >
                <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)" }}>AI #{slot + 1}</span>
                <div>
                  <select
                    value={net}
                    disabled={disabled}
                    onChange={(e) => {
                      const next = [...networks];
                      next[slot] = e.target.value;
                      setNetworks(next);
                    }}
                    style={{ ...inputStyle, width: "100%" }}
                  >
                    {NETWORK_OPTIONS.map((o) => (
                      <option key={o.value} value={o.value}>
                        {o.label} · {o.hint}
                      </option>
                    ))}
                  </select>
                  {net === "custom" && !disabled && (
                    <input
                      value={customPaths[slot] || ""}
                      onChange={(e) =>
                        setCustomPaths((prev) => ({ ...prev, [slot]: e.target.value }))
                      }
                      placeholder="绝对路径，例如 C:/models/my_new.pth"
                      style={{ ...inputStyle, marginTop: 6, width: "100%" }}
                    />
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* Status / live log */}
      {status && (
        <div className="card" style={{ padding: 14 }}>
          <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 10, flexWrap: "wrap" }}>
            <span
              style={{
                padding: "4px 10px",
                borderRadius: 6,
                fontSize: 12,
                fontWeight: 700,
                color: "#fff",
                background: isRunning ? "#27ae60" : "#7f8c8d",
              }}
            >
              {isRunning ? "运行中" : "已结束"}
            </span>
            <span style={{ fontSize: 13, color: "var(--text-muted)" }}>
              个室 L{status.lobby_id} · Speed {status.speed} · {status.device}
            </span>
            {status.lobby_id && (
              <a
                href={joinUrl}
                target="_blank"
                rel="noreferrer"
                style={{ fontSize: 13, color: ACCENT, fontWeight: 700 }}
              >
                打开天凤加入 L{status.lobby_id} ↗
              </a>
            )}
          </div>

          {status.bots.length > 0 && (
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 10 }}>
              {status.bots.map((b: BotInfo) => (
                <span
                  key={b.name}
                  style={{
                    fontSize: 12,
                    padding: "3px 8px",
                    borderRadius: 6,
                    border: `1px solid var(--border)`,
                    background: "var(--surface-subtle)",
                    color: "var(--text-primary)",
                  }}
                >
                  {b.name} = {shortSpec(b.spec)}
                </span>
              ))}
            </div>
          )}

          <div
            ref={logRef}
            style={{
              height: 320,
              overflow: "auto",
              background: "#0f1115",
              color: "#d6dde6",
              borderRadius: 8,
              padding: 10,
              fontFamily: "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace",
              fontSize: 12,
              lineHeight: 1.5,
              whiteSpace: "pre-wrap",
              wordBreak: "break-word",
            }}
          >
            {status.log_tail.length > 0
              ? status.log_tail.join("\n")
              : "（暂无日志，呼出后这里会实时滚动显示 bot / gateway 输出）"}
          </div>
        </div>
      )}
    </PageShell>
  );
}

const labelStyle: CSSProperties = {
  fontSize: 12,
  color: "var(--text-muted)",
  fontWeight: 600,
  display: "block",
  marginBottom: 5,
};

const hintStyle: CSSProperties = {
  fontSize: 11,
  color: "var(--text-muted)",
  marginTop: 4,
};

const inputStyle: CSSProperties = {
  height: 34,
  borderRadius: 6,
  border: "1px solid var(--border)",
  background: "var(--surface-subtle)",
  color: "var(--text-primary)",
  padding: "0 10px",
  fontSize: 13,
  width: "100%",
  boxSizing: "border-box",
};
