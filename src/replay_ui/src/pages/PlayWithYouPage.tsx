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
  listLadderCaptures,
  confirmLadderCapture,
  ignoreLadderCapture,
  retryPublishLadderCapture,
  type NetworkId,
  type SpeedId,
  type DeviceId,
  type BotInfo,
  type PlayWithYouStatus,
  type LadderCaptureEntry,
  type ParticipantBindingRequest,
} from "../api/playwithyouApi";
import { participantsApi } from "../api/participantsApi";
import type { Account as ParticipantAccount } from "../types/participants";
import { useNavigate } from "react-router-dom";
import { routes } from "../routes";

const ACCENT = "#8e44ad";

type RosterBinding = {
  account_id: string;
  controller_type: string;
  model_identity_id: string;
  model_artifact_id: string;
  launched: boolean;
  expected_raw_name: string;
};

const CONTROLLER_OPTIONS = [
  { value: "human_ui", label: "真人" },
  { value: "local_model", label: "本地模型" },
  { value: "external_agent", label: "外部代理" },
  { value: "manual_only", label: "仅登记" },
];

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
  const navigate = useNavigate();
  const [lobbyId, setLobbyId] = useState<string>("2147");
  const [speed, setSpeed] = useState<SpeedId>("normal");
  const [device, setDevice] = useState<DeviceId>("cuda");
  const [quantity, setQuantity] = useState<number>(3);
  const [networks, setNetworks] = useState<string[]>(["mortal", "70k", "ext_mortal", "none"]);
  const [customPaths, setCustomPaths] = useState<Record<number, string>>({});

  const [status, setStatus] = useState<PlayWithYouStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // R9-3 正式天梯捕获绑定（呼出前选择；开局后冻结）
  const [captureEnabled, setCaptureEnabled] = useState(false);
  const [captureSeason, setCaptureSeason] = useState("official-ladder-v1");
  const [captureHuman, setCaptureHuman] = useState("nick@01");
  const [captureBots, setCaptureBots] = useState(["70k@01", "70k@02", "70k@03"]);
  const [captures, setCaptures] = useState<LadderCaptureEntry[]>([]);
  const [captureBusy, setCaptureBusy] = useState<string | null>(null);

  // R10-E：通用四人阵容模式（预期四人阵容与 launcher 数量分离）
  const [rosterMode, setRosterMode] = useState(false);
  const [roster, setRoster] = useState<RosterBinding[]>([
    { account_id: "nick@01", controller_type: "human_ui", model_identity_id: "", model_artifact_id: "", launched: false, expected_raw_name: "" },
    { account_id: "70k@01", controller_type: "local_model", model_identity_id: "", model_artifact_id: "", launched: true, expected_raw_name: "NoName-1" },
    { account_id: "70k@02", controller_type: "local_model", model_identity_id: "", model_artifact_id: "", launched: true, expected_raw_name: "NoName-2" },
    { account_id: "", controller_type: "external_agent", model_identity_id: "", model_artifact_id: "", launched: false, expected_raw_name: "" },
  ]);
  const [accounts, setAccounts] = useState<ParticipantAccount[]>([]);

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
      const launchedSlots = roster
        .map((entry, index) => (entry.launched ? index : null))
        .filter((slot): slot is number => slot !== null);
      const rosterPayload: ParticipantBindingRequest[] | undefined = rosterMode
        ? roster.map((entry, index) => ({
            account_id: entry.account_id,
            controller_type: entry.controller_type,
            model_identity_id: entry.model_identity_id || null,
            model_artifact_id: entry.model_artifact_id || null,
            launcher_slot: entry.launched ? index : null,
            expected_raw_name: entry.expected_raw_name || null,
            resolution_required: !entry.account_id,
          }))
        : undefined;
      const req = {
        lobby_id: lobbyId,
        speed,
        // P1-3：rosterMode=false 时沿用旧 quantity 选择器，不得被 roster 草稿覆盖
        quantity: rosterMode ? launchedSlots.length : quantity,
        networks: [...networks],
        custom_paths: customPaths,
        device,
        roster: rosterPayload,
        ladder_capture: captureEnabled && !rosterMode
          ? {
              enabled: true,
              season_id: captureSeason,
              human_account_id: captureHuman,
              bot_account_ids: [...captureBots],
              mode: "confirm" as const,
            }
          : undefined,
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

  const refreshCaptures = useCallback(async () => {
    try {
      const data = await listLadderCaptures();
      setCaptures(data.captures);
    } catch {
      /* transient */
    }
  }, []);

  const confirmCapture = async (captureId: string) => {
    setCaptureBusy(captureId);
    setError(null);
    try {
      await confirmLadderCapture(captureId);
      await refreshCaptures();
    } catch (e) {
      setError(e instanceof Error ? e.message : "确认失败");
    } finally {
      setCaptureBusy(null);
    }
  };

  const ignoreCapture = async (captureId: string) => {
    setCaptureBusy(captureId);
    setError(null);
    try {
      await ignoreLadderCapture(captureId);
      await refreshCaptures();
    } catch (e) {
      setError(e instanceof Error ? e.message : "忽略失败");
    } finally {
      setCaptureBusy(null);
    }
  };

  const retryPublish = async (captureId: string) => {
    setCaptureBusy(captureId);
    setError(null);
    try {
      await retryPublishLadderCapture(captureId);
      await refreshCaptures();
    } catch (e) {
      setError(e instanceof Error ? e.message : "重试失败");
    } finally {
      setCaptureBusy(null);
    }
  };

  useEffect(() => {
    refreshCaptures();
    const timer = window.setInterval(refreshCaptures, 5000);
    return () => window.clearInterval(timer);
  }, [refreshCaptures]);

  // R10-E：加载 participants 账号（roster 账号选择用）
  useEffect(() => {
    const controller = new AbortController();
    participantsApi
      .listAccounts(controller.signal)
      .then((resp) => setAccounts(resp.accounts))
      .catch(() => {});
    return () => controller.abort();
  }, []);

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

        {/* R9-3 正式天梯捕获绑定 */}
        <div
          style={{
            marginTop: 14,
            padding: 12,
            borderRadius: 8,
            border: `1px solid ${captureEnabled ? ACCENT : "var(--border)"}`,
            background: captureEnabled ? "rgba(142,68,173,0.04)" : "var(--surface-subtle)",
          }}
        >
          <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={captureEnabled}
              disabled={isRunning || rosterMode}
              onChange={(e) => setCaptureEnabled(e.target.checked)}
            />
            <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)" }}>
              ☑ 计入正式天梯（tenhou_rank_progression）
            </span>
          </label>
          <div style={hintStyle}>
            开启后对局结束后确认录入 official-ladder-v1 赛季；开局后绑定冻结，不可中途修改。
          </div>
          {captureEnabled && (
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginTop: 10 }}>
              <div>
                <label style={labelStyle}>赛季</label>
                <input
                  value={captureSeason}
                  disabled={isRunning}
                  onChange={(e) => setCaptureSeason(e.target.value)}
                  style={inputStyle}
                />
              </div>
              <div>
                <label style={labelStyle}>人类账号</label>
                <input
                  value={captureHuman}
                  disabled={isRunning}
                  onChange={(e) => setCaptureHuman(e.target.value)}
                  style={inputStyle}
                />
              </div>
              <div style={{ gridColumn: "1 / -1" }}>
                <label style={labelStyle}>Bot 账号（3 个，顺序对应 AI 槽位）</label>
                <div style={{ display: "flex", gap: 8 }}>
                  {captureBots.map((bot, index) => (
                    <input
                      key={index}
                      value={bot}
                      disabled={isRunning}
                      onChange={(e) => {
                        const next = [...captureBots];
                        next[index] = e.target.value;
                        setCaptureBots(next);
                      }}
                      style={{ ...inputStyle, flex: 1 }}
                    />
                  ))}
                </div>
              </div>
            </div>
          )}
        </div>

        {/* R10-E 通用四人阵容（预期四人阵容与 launcher 数量分离） */}
        <div
          style={{
            marginTop: 14,
            padding: 12,
            borderRadius: 8,
            border: `1px solid ${rosterMode ? ACCENT : "var(--border)"}`,
            background: rosterMode ? "rgba(142,68,173,0.04)" : "var(--surface-subtle)",
          }}
        >
          <label style={{ display: "flex", alignItems: "center", gap: 8, cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={rosterMode}
              disabled={isRunning || captureEnabled}
              onChange={(e) => setRosterMode(e.target.checked)}
            />
            <span style={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)" }}>
              通用四人阵容（可选：任意四人，非 1 人类 + 3 bot）
            </span>
          </label>
          <div style={hintStyle}>
            勾选「呼出」的座位由本系统启动（数量 = launcher 数，如 2 个本地 bot + 外部 Mortal 合法）；
            赛后在天凤牌谱导入页按 session 自动解析 NoName 与会话绑定的模型版本。
          </div>
          {rosterMode && (
            <div style={{ display: "grid", gap: 8, marginTop: 10 }}>
              {roster.map((entry, index) => (
                <div key={index} style={{ display: "flex", gap: 8, alignItems: "center" }}>
                  <span style={{ width: 24, fontWeight: 800, color: "var(--text-muted)" }}>{["東", "南", "西", "北"][index]}</span>
                  <select
                    value={entry.account_id}
                    disabled={isRunning}
                    onChange={(e) => {
                      const next = [...roster];
                      next[index] = { ...next[index], account_id: e.target.value };
                      setRoster(next);
                    }}
                    style={{ ...inputStyle, flex: 1 }}
                  >
                    <option value="">选择账号…</option>
                    {accounts.map((a) => (
                      <option key={a.account_id} value={a.account_id}>
                        {a.display_name}（{a.account_id}）
                      </option>
                    ))}
                  </select>
                  <select
                    value={entry.controller_type}
                    disabled={isRunning}
                    onChange={(e) => {
                      const next = [...roster];
                      next[index] = { ...next[index], controller_type: e.target.value };
                      setRoster(next);
                    }}
                    style={inputStyle}
                  >
                    {CONTROLLER_OPTIONS.map((option) => (
                      <option key={option.value} value={option.value}>{option.label}</option>
                    ))}
                  </select>
                  <label style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, whiteSpace: "nowrap" }}>
                    <input
                      type="checkbox"
                      checked={entry.launched}
                      disabled={isRunning}
                      onChange={(e) => {
                        const next = [...roster];
                        next[index] = { ...next[index], launched: e.target.checked };
                        setRoster(next);
                      }}
                    />
                    呼出
                  </label>
                  {entry.launched && (
                    <input
                      value={entry.expected_raw_name}
                      disabled={isRunning}
                      onChange={(e) => {
                        const next = [...roster];
                        next[index] = { ...next[index], expected_raw_name: e.target.value };
                        setRoster(next);
                      }}
                      placeholder={`NoName-${index + 1}`}
                      style={{ ...inputStyle, width: 110 }}
                    />
                  )}
                </div>
              ))}
            </div>
          )}
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

          {/* 冻结的正式天梯绑定（开局后不可修改） */}
          {status.ladder_capture?.enabled && (
            <div
              style={{
                marginTop: 10,
                fontSize: 12,
                padding: "8px 10px",
                borderRadius: 6,
                border: "1px solid rgba(142,68,173,0.4)",
                background: "rgba(142,68,173,0.06)",
                color: "var(--text-secondary)",
              }}
            >
              ☑ 正式天梯（已冻结）：
              <b style={{ color: ACCENT }}>{status.ladder_capture.season_id}</b> ·
              人类 <b>{status.ladder_capture.human_account_id}</b> ·
              Bot {status.ladder_capture.bot_account_ids.join(" / ")} ·
              模式 {status.ladder_capture.mode}
            </div>
          )}
        </div>
      )}

      {/* R9-3 已捕获正式天梯对局（等待确认） */}
      {captures.length > 0 && (
        <div className="card" style={{ padding: 14, marginTop: 12 }}>
          <SectionTitle
            title="已捕获正式天梯对局"
            description="对局结束后确认录入 official-ladder-v1；状态：waiting_start / in_game / pending_confirmation / published / ignored / incomplete / conflict / accepted_publish_failed"
          />
          {captures.map((c) => (
            <div
              key={c.capture_id}
              style={{
                border: "1px solid var(--border)",
                borderRadius: 8,
                padding: 12,
                marginBottom: 10,
                background: "var(--surface-subtle)",
              }}
            >
              <div style={{ display: "flex", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
                <div style={{ fontSize: 12, color: "var(--text-secondary)" }}>
                  <b style={{ color: "var(--text-primary)" }}>{c.match?.match_id || c.capture_id}</b>
                  <span style={{ color: "var(--text-muted)" }}> · {c.season_id}</span>
                  {c.tenhou_log_url && (
                    <a
                      href={c.tenhou_log_url}
                      target="_blank"
                      rel="noreferrer"
                      style={{ marginLeft: 8, color: ACCENT, fontWeight: 700 }}
                    >
                      打开天凤牌谱 ↗
                    </a>
                  )}
                </div>
                <span
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    padding: "3px 8px",
                    borderRadius: 6,
                    border: "1px solid var(--border)",
                    color: c.state === "pending_confirmation" ? "#27ae60" : "var(--text-muted)",
                  }}
                >
                  {c.state}
                </span>
              </div>
              {c.match?.players && c.match.players.length > 0 && (
                <div style={{ marginTop: 8 }}>
                  {[...c.match.players]
                    .sort((a, b) => b.final_score - a.final_score || a.seat - b.seat)
                    .map((p, index) => (
                      <div key={p.account_id} style={{ fontSize: 12, padding: "2px 0" }}>
                        <span style={{ display: "inline-block", width: 120, color: "var(--text-primary)", fontWeight: 700 }}>
                          {p.account_id}
                        </span>
                        <span style={{ display: "inline-block", width: 80, color: "var(--text-muted)" }}>
                          {p.final_score}
                        </span>
                        <span style={{ color: "var(--text-secondary)" }}>
                          {index + 1} 位（seat {p.seat}）
                        </span>
                      </div>
                    ))}
                </div>
              )}
              <div style={{ marginTop: 8, fontSize: 11, color: "var(--text-muted)" }}>
                验证 observer：{(c.score_observers ?? []).join(" / ") || "—"}
                {c.state === "accepted_publish_failed" && (
                  <span style={{ color: "var(--error)" }}> · 已确认但发布失败，可重试</span>
                )}
                {c.evidence_warning && (
                  <span style={{ color: "#e67e22" }}> · 证据警告：{c.evidence_warning}</span>
                )}
              </div>
              <div style={{ display: "flex", gap: 8, marginTop: 8 }}>
                {c.state === "pending_confirmation" && (
                  <>
                    <button
                      type="button"
                      onClick={() => confirmCapture(c.capture_id)}
                      disabled={captureBusy === c.capture_id}
                      style={{ height: 30, padding: "0 12px", background: "#27ae60", color: "#fff", borderRadius: 6, fontSize: 12, cursor: "pointer" }}
                    >
                      确认录入并发布
                    </button>
                    <button
                      type="button"
                      onClick={() => ignoreCapture(c.capture_id)}
                      disabled={captureBusy === c.capture_id}
                      style={{ height: 30, padding: "0 12px", background: "transparent", color: "var(--text-muted)", border: "1px solid var(--border)", borderRadius: 6, fontSize: 12, cursor: "pointer" }}
                    >
                      忽略本局
                    </button>
                  </>
                )}
                {c.state === "awaiting_import" && c.tenhou_log_url && (
                  <button
                    type="button"
                    onClick={() => {
                      const params = new URLSearchParams({ url: c.tenhou_log_url ?? "" });
                      if (c.session_id) params.set("session_id", c.session_id);
                      navigate(`${routes.matchImport}?${params.toString()}`);
                    }}
                    style={{ height: 30, padding: "0 12px", background: ACCENT, color: "#fff", borderRadius: 6, fontSize: 12, cursor: "pointer" }}
                  >
                    导入对局
                  </button>
                )}
                {c.state === "accepted_publish_failed" && (
                  <button
                    type="button"
                    onClick={() => retryPublish(c.capture_id)}
                    disabled={captureBusy === c.capture_id}
                    style={{ height: 30, padding: "0 12px", background: ACCENT, color: "#fff", borderRadius: 6, fontSize: 12, cursor: "pointer" }}
                  >
                    重新发布
                  </button>
                )}
              </div>
            </div>
          ))}
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
