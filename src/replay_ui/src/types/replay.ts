// src/replay_ui/src/types/replay.ts

export interface DiscardEntry {
  pai: string;
  tsumogiri: boolean;
  /** 后端实际不返回此字段，需兼容处理 */
  reach_declared?: boolean;
}

export interface MeldEntry {
  type: string;
  pai: string;
  consumed: string[];
  target: number;
}

export interface Action {
  type: ActionType;
  actor: number;
  pai?: string;
  target?: number;
  consumed?: string[];
  tsumogiri?: boolean;
}

export type ActionType =
  | 'dahai'
  | 'reach'
  | 'reach_accepted'
  | 'chi'
  | 'pon'
  | 'daiminkan'
  | 'ankan'
  | 'kakan'
  | 'hora'
  | 'ryukyoku'
  | 'none';

export interface KyokuInfo {
  bakaze: string;
  kyoku: number;
  honba: number;
}

/** /api/replay 返回的 decision_log 条目结构 */
export interface DecisionLogEntry {
  step: number;
  bakaze: string;
  kyoku: number;
  honba: number;
  scores: number[];
  reached: boolean[];
  dora_markers: string[];
  hand: string[];
  discards: Record<number, DiscardEntry[]>;
  melds: Record<number, MeldEntry[]>;
  actor_to_move: number;
  tsumo_pai: string | null;
  last_discard: { actor: number; pai: string } | null;
  /** 当前视角 Bot 的决策 */
  chosen: Action;
  /** 所有合法动作候选 + logit */
  candidates: Array<{ action: Action; logit: number }>;
  /** 当前视角 Bot 的 value loss 预测 */
  value?: number;
  /** ground truth：玩家实际动作 */
  gt_action: Action | null;
  /** 供前端按小局过滤 */
  kyoku_key: KyokuInfo;
}

export interface ReplayData {
  log: DecisionLogEntry[];
  kyoku_order: KyokuInfo[];
  total_ops: number;
  match_count: number;
  player_id: number;
}

export interface ReplayMeta {
  replay_id: string;
  created_at: string;
  bot_type: 'keqingv1' | 'v5model';
  kyoku_count: number;
  total_steps: number;
  player_names: string[];
  final_scores: number[];
}

export interface ReplaySubmitRequest {
  input_type: 'tenhou_url' | 'tenhou6_json' | 'mjson_file' | 'mjson_text';
  content: string;
  bot_type: 'keqingv1' | 'v5model';
  player_ids: number[];
  checkpoint?: string;
}

export interface BotDecision {
  action: Action;
  logit: number;
  is_correct: boolean | null;
  value: number | null;
}

export interface StepEntry {
  step: number;
  bakaze: string;
  kyoku: number;
  honba: number;
  scores: number[];
  reached: boolean[];
  dora_markers: string[];
  hand: string[];
  discards: Record<number, DiscardEntry[]>;
  melds: Record<number, MeldEntry[]>;
  actor_to_move: number;
  tsumo_pai: string | null;
  last_discard: { actor: number; pai: string } | null;
  bot_decisions: Record<number, BotDecision>;
  gt_action: Action | null;
  kyoku_key: KyokuInfo;
}

export interface ReplaySubmitResponse {
  replay_id: string;
  status: 'pending' | 'running' | 'done' | 'failed';
  progress: number;
  error?: string;
}

export type PlayerMode =
  | { type: 'replay'; replay_id: string }
  | { type: 'realtime'; game_id: string };
