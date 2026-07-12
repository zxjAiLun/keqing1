// src/replay_ui/src/api/playwithyouApi.ts
// Client for the "Play with you" backend: summon Mortal-weight bot accounts
// into a Tenhou private room (e.g. L2147) as NoName guests.
import { fetchWithTimeout } from "./battleApi";

const BASE = "/api/battle/playwithyou";

export type NetworkId = "none" | "mortal" | "70k" | "weak_mortal" | "custom";
export type SpeedId = "slow" | "normal" | "fast" | "turbo";
export type DeviceId = "cuda" | "cpu";

export interface StartPlayWithYouRequest {
  lobby_id: string;
  speed: SpeedId;
  quantity: number;
  networks: string[]; // length 4, slot 0..3
  custom_paths: Record<number, string>; // slot -> absolute .pth path
  device: DeviceId;
  name_prefix?: string;
  tenhou_cookie?: string;
}

export interface BotInfo {
  name: string;
  spec: string;
}

export interface PlayWithYouStatus {
  session_id: string | null;
  running: boolean;
  lobby_id: string | null;
  speed: string | null;
  device: string | null;
  bots: BotInfo[];
  log_tail: string[];
  started_at: number | null;
}

export async function startPlayWithYou(req: StartPlayWithYouRequest): Promise<PlayWithYouStatus> {
  const res = await fetchWithTimeout(`${BASE}/start`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const data = await res.json();
      detail = data.detail || detail;
    } catch {
      /* ignore */
    }
    throw new Error(detail);
  }
  return res.json();
}

export async function getPlayWithYouStatus(): Promise<PlayWithYouStatus> {
  const res = await fetchWithTimeout(`${BASE}/status`);
  if (!res.ok) throw new Error(`status ${res.statusText}`);
  return res.json();
}

export async function stopPlayWithYou(): Promise<PlayWithYouStatus> {
  const res = await fetchWithTimeout(`${BASE}/stop`, { method: "POST" });
  if (!res.ok) throw new Error(`stop ${res.statusText}`);
  return res.json();
}
