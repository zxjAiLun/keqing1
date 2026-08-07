// src/replay_ui/src/pages/PlayWithYouPage.tsx
// "Play with you" — 四人对局配置（R10 UX Repair：roster 唯一启动模式）。
// 每个 seat 直选 账号 / 控制器 / 是否由本系统呼出 / 模型身份 / 模型产物；
// 正式天梯不再作为启动模式，改由赛后 Match intake/confirm/revise 决定。
import { useCallback, useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import { PageShell, SectionTitle } from "../components/Layout/PageScaffold";
import {
  startPlayWithYou,
  stopPlayWithYou,
  getPlayWithYouStatus,
  type SpeedId,
  type DeviceId,
  type BotInfo,
  type PlayWithYouStatus,
  type ParticipantBindingRequest,
} from "../api/playwithyouApi";
import { participantsApi } from "../api/participantsApi";
import type { Account as ParticipantAccount, ModelIdentity } from "../types/participants";

const ACCENT = "#8e44ad";

type RosterBinding = {
  account_id: string;
  controller_type: string;
  model_identity_id: string;
  model_artifact_id: string;
  launched: boolean;
  expected_raw_name: string;
};

const SEAT_WINDS = ["東", "南", "西", "北"];

const CONTROLLER_OPTIONS = [
  { value: "human_ui", label: "真人" },
  { value: "local_model", label: "本地模型" },
  { value: "external_agent", label: "外部代理" },
  { value: "manual_only", label: "仅登记" },
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

  const [status, setStatus] = useState<PlayWithYouStatus | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // R10 UX Repair：四人对局配置是唯一启动模式；seat 直选账号/控制器/模型。
  const [roster, setRoster] = useState<RosterBinding[]>([
    { account_id: "nick@01", controller_type: "human_ui", model_identity_id: "", model_artifact_id: "", launched: false, expected_raw_name: "" },
    { account_id: "70k@01", controller_type: "local_model", model_identity_id: "", model_artifact_id: "", launched: true, expected_raw_name: "NoName-1" },
    { account_id: "70k@02", controller_type: "local_model", model_identity_id: "", model_artifact_id: "", launched: true, expected_raw_name: "NoName-2" },
    { account_id: "", controller_type: "external_agent", model_identity_id: "", model_artifact_id: "", launched: false, expected_raw_name: "" },
  ]);
  const [accounts, setAccounts] = useState<ParticipantAccount[]>([]);
  const [identities, setIdentities] = useState<ModelIdentity[]>([]);

  const logRef = useRef<HTMLDivElement | null>(null);
  const pollingRef = useRef<number | null>(null);

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

  const updateSeat = (index: number, patch: Partial<RosterBinding>) => {
    setRoster((prev) => prev.map((entry, i) => (i === index ? { ...entry, ...patch } : entry)));
  };

  const start = async () => {
    setLoading(true);
    setError(null);
    try {
      const launchedSlots = roster
        .map((entry, index) => (entry.launched ? index : null))
        .filter((slot): slot is number => slot !== null);
      if (launchedSlots.length === 0) {
        throw new Error("至少需要一个「由本系统呼出」的座位");
      }
      // P1-2：launched 且 local_model 的 seat 必须明确选择模型身份 + 产物
      for (const index of launchedSlots) {
        const entry = roster[index];
        if (entry.controller_type === "local_model" && (!entry.model_identity_id || !entry.model_artifact_id)) {
          throw new Error(`座位「${SEAT_WINDS[index]}」选择了本地模型，请选择模型身份与产物`);
        }
        if (!entry.account_id) {
          throw new Error(`座位「${SEAT_WINDS[index]}」由本系统呼出，必须选择账号`);
        }
      }
      const rosterPayload: ParticipantBindingRequest[] = roster.map((entry, index) => ({
        account_id: entry.account_id,
        controller_type: entry.controller_type,
        model_identity_id: entry.model_identity_id || null,
        model_artifact_id: entry.model_artifact_id || null,
        launcher_slot: entry.launched ? index : null,
        expected_raw_name: entry.expected_raw_name || (entry.launched ? `NoName-${index + 1}` : null),
        resolution_required: !entry.account_id,
      }));
      const s = await startPlayWithYou({
        lobby_id: lobbyId,
        speed,
        quantity: launchedSlots.length,
        device,
        roster: rosterPayload,
      });
      setStatus(s);
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

  useEffect(() => {
    const controller = new AbortController();
    Promise.all([
      participantsApi.listAccounts(controller.signal),
      participantsApi.listModels(controller.signal),
    ])
      .then(([accResp, modelResp]) => {
        setAccounts(accResp.accounts);
        setIdentities(modelResp.identities);
      })
      .catch(() => {});
    return () => controller.abort();
  }, []);

  useEffect(() => stopPolling, [stopPolling]);

  useEffect(() => {
    if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
  }, [status?.log_tail]);

  const isRunning = status?.running ?? false;
  const joinUrl = `https://tenhou.net/0/?${status?.lobby_id ?? lobbyId}`;
  const launchedCount = roster.filter((entry) => entry.launched).length;

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
            配置四人对局：每个座位直选账号 / 控制器 / 模型。正式天梯在赛后 Match 确认时决定，不再是启动选项。
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
        <SectionTitle title="四人对局配置" description="四个座位永远存在；勾选「由本系统呼出」的座位启动本地模型/外部代理。" />

        {/* Lobby + speed + device */}
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
            <label style={labelStyle}>Speed / Device</label>
            <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
              <Segmented options={SPEED_OPTIONS} value={speed} onChange={setSpeed} disabled={isRunning} />
              <Segmented options={DEVICE_OPTIONS} value={device} onChange={setDevice} disabled={isRunning} />
            </div>
            <div style={hintStyle}>本系统呼出 {launchedCount} 个座位；Speed 在自己回合前的思考停顿。</div>
          </div>
        </div>

        {/* Roster rows */}
        <div style={{ display: "grid", gap: 8 }}>
          {roster.map((entry, index) => {
            const relevantIdentities = identities.filter(
              (m) => m.account_id === entry.account_id || m.account_id == null,
            );
            const chosenIdentity = relevantIdentities.find(
              (m) => m.model_identity_id === entry.model_identity_id,
            );
            const artifacts = chosenIdentity?.artifacts ?? [];
            const showModel = entry.controller_type === "local_model" && entry.launched;
            return (
              <div
                key={index}
                style={{
                  display: "flex",
                  gap: 8,
                  alignItems: "center",
                  flexWrap: "wrap",
                  border: "1px solid var(--border)",
                  borderRadius: 8,
                  padding: "8px 10px",
                  background: entry.launched ? "rgba(142,68,173,0.04)" : "var(--surface-subtle)",
                }}
              >
                <span style={{ width: 24, fontWeight: 800, color: "var(--text-muted)" }}>{SEAT_WINDS[index]}</span>
                <select
                  value={entry.account_id}
                  disabled={isRunning}
                  onChange={(e) => {
                    const accountId = e.target.value;
                    const acc = accounts.find((a) => a.account_id === accountId);
                    updateSeat(index, {
                      account_id: accountId,
                      controller_type: acc?.default_controller ?? entry.controller_type,
                      model_identity_id: "",
                      model_artifact_id: "",
                    });
                  }}
                  style={{ ...inputStyle, flex: 1, minWidth: 120 }}
                >
                  <option value="">选择账号…</option>
                  {accounts.map((a) => (
                    <option key={a.account_id} value={a.account_id}>
                      {a.display_name}（{a.account_id}）{a.enabled ? "" : " · 停用"}
                    </option>
                  ))}
                </select>
                <select
                  value={entry.controller_type}
                  disabled={isRunning}
                  onChange={(e) =>
                    updateSeat(index, { controller_type: e.target.value, model_identity_id: "", model_artifact_id: "" })
                  }
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
                    onChange={(e) =>
                      updateSeat(index, {
                        launched: e.target.checked,
                        expected_raw_name: e.target.checked && !entry.expected_raw_name ? `NoName-${index + 1}` : entry.expected_raw_name,
                      })
                    }
                  />
                  由本系统呼出
                </label>
                {entry.launched && (
                  <input
                    value={entry.expected_raw_name}
                    disabled={isRunning}
                    onChange={(e) => updateSeat(index, { expected_raw_name: e.target.value })}
                    placeholder={`NoName-${index + 1}`}
                    style={{ ...inputStyle, width: 110 }}
                  />
                )}
                {showModel && (
                  <>
                    <select
                      value={entry.model_identity_id}
                      disabled={isRunning}
                      onChange={(e) =>
                        updateSeat(index, { model_identity_id: e.target.value, model_artifact_id: "" })
                      }
                      style={{ ...inputStyle, flex: 1, minWidth: 130 }}
                    >
                      <option value="">选择模型身份…</option>
                      {relevantIdentities.map((m) => (
                        <option key={m.model_identity_id} value={m.model_identity_id}>
                          {m.label}{m.account_id == null ? "（全局）" : ""}
                        </option>
                      ))}
                    </select>
                    <select
                      value={entry.model_artifact_id}
                      disabled={isRunning || artifacts.length === 0}
                      onChange={(e) => updateSeat(index, { model_artifact_id: e.target.value })}
                      style={{ ...inputStyle, flex: 1, minWidth: 130 }}
                    >
                      <option value="">
                        {artifacts.length === 0 ? "该身份无产物" : "选择模型产物…"}
                      </option>
                      {artifacts.map((art) => (
                        <option key={art.model_artifact_id} value={art.model_artifact_id}>
                          {art.label}{art.is_current ? "（当前）" : ""} · {shortSpec(art.artifact_path ?? "")}
                        </option>
                      ))}
                    </select>
                  </>
                )}
              </div>
            );
          })}
        </div>
        <div style={hintStyle}>
          由本系统呼出 = false 的座位为真人 / 外部 Mortal / 朋友：赛后通过 Tenhou name / session 别名 / 人工 resolve 绑定。
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
                    border: "1px solid var(--border)",
                    background: "var(--surface-subtle)",
                    color: "var(--text-primary)",
                  }}
                >
                  {b.name} = {shortSpec(b.spec)}
                </span>
              ))}
            </div>
          )}

          {/* 已配置阵容（呼出后展示 NoName ↔ 账号 ↔ 模型） */}
          {isRunning && (
            <div
              style={{
                marginTop: 4,
                marginBottom: 10,
                fontSize: 12,
                padding: "8px 10px",
                borderRadius: 6,
                border: "1px solid rgba(142,68,173,0.3)",
                background: "rgba(142,68,173,0.05)",
                color: "var(--text-secondary)",
              }}
            >
              <div style={{ fontWeight: 700, color: "var(--text-primary)", marginBottom: 4 }}>已配置阵容（session 冻结）</div>
              {roster.map((entry, index) => {
                if (!entry.launched) return null;
                const identity = identities.find((m) => m.model_identity_id === entry.model_identity_id);
                const artifact = identity?.artifacts.find((a) => a.model_artifact_id === entry.model_artifact_id);
                return (
                  <div key={index} style={{ padding: "2px 0" }}>
                    {SEAT_WINDS[index]} · <b>{entry.account_id}</b> → {entry.expected_raw_name || `NoName-${index + 1}`}
                    {identity && (
                      <span style={{ color: "var(--text-muted)" }}>
                        {" "}· {identity.label}
                        {artifact ? ` / ${artifact.label}` : ""}
                      </span>
                    )}
                  </div>
                );
              })}
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
  boxSizing: "border-box",
};
