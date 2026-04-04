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
  is_tsumo?: boolean;
  deltas?: number[];
  scores?: number[];
  han?: number;
  fu?: number;
  yaku?: string[];
  yaku_details?: Array<{
    key: string;
    name: string;
    han: number;
  }>;
  ura_dora_markers?: string[];
  cost?: {
    main?: number;
    main_bonus?: number;
    additional?: number;
    additional_bonus?: number;
    kyoutaku_bonus?: number;
    total?: number;
    yaku_level?: string;
  };
  honba?: number;
  kyotaku?: number;
  tenpai_players?: number[];
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
  oya: number;
  scores: number[];
  reached: boolean[];
  dora_markers: string[];
  hand: string[];
  discards: Record<number, DiscardEntry[]> | DiscardEntry[][];
  melds?: Record<number, MeldEntry[]> | MeldEntry[][];
  actor_to_move: number;
  tsumo_pai: string | null;
  last_discard: { actor: number; pai: string; pai_raw?: string } | null;
  /** 是否为其他家的观察步（无 bot 推理数据） */
  is_obs: boolean;
  /** 当前视角 Bot 的决策（obs 步为他家实际动作） */
  chosen: Action;
  /** 所有合法动作候选；final_score 为默认展示/统计口径，旧版本回放兼容 beam_score/logit */
  candidates: Array<{ action: Action; logit: number; beam_score?: number; final_score?: number }>;
  /** 当前视角 Bot 的 value loss 预测 */
  value?: number;
  /** ground truth：玩家实际动作 */
  gt_action: Action | null;
  /** 观察步类型，仅 is_obs=true 时有意义 */
  obs_kind?: 'discard' | 'meld' | 'reach' | 'terminal';
  /** 当前棋盘快照语义，默认 after_action */
  board_phase?: 'before_action' | 'after_action';
  /** 对应 events.jsonl 的事件序号 */
  source_event_index?: number;
  /** 供前端按小局过滤 */
  kyoku_key: KyokuInfo;
}

export interface ReplayData {
  replay_id?: string;
  log: DecisionLogEntry[];
  kyoku_order: KyokuInfo[];
  total_ops: number;
  match_count: number;
  rating: number | null;
  player_id: number;
  player_names?: string[];
  bot_type?: 'keqingv1' | 'keqingv2' | 'keqingv3';
}

export interface ReplayMeta {
  replay_id: string;
  created_at: string;
  bot_type: 'keqingv1' | 'keqingv2' | 'keqingv3';
  kyoku_count: number;
  total_steps: number;
  player_names: string[];
  final_scores: number[];
}

export interface ReplaySubmitRequest {
  input_type: 'tenhou_url' | 'tenhou6_json' | 'mjson_file' | 'mjson_text';
  content: string;
  bot_type: 'keqingv1' | 'keqingv2' | 'keqingv3';
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
  oya: number;
  scores: number[];
  reached: boolean[];
  dora_markers: string[];
  hand: string[];
  discards: Record<number, DiscardEntry[]>;
  melds: Record<number, MeldEntry[]>;
  actor_to_move: number;
  tsumo_pai: string | null;
  last_discard: { actor: number; pai: string; pai_raw?: string } | null;
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

export interface SelfplayAnomalyReplayItem {
  game_id: number;
  anomaly_score: number;
  score_components: Record<string, number>;
  interest?: number;
  scores?: number[];
  ranks?: number[];
  turns?: number;
  rounds?: number;
  mjson: string;
  meta: string;
  replay_id?: string;
  replay_player_id?: number;
  replay_bot_type?: string;
  replay_view_url?: string;
  game_board_url?: string;
  replay_ui_error?: string | null;
}

export interface SelfplayAnomalyReplayGroup {
  output_dir: string;
  output_dir_path: string;
  manifest_path: string;
  collection_type?: string;
  updated_at: number;
  stats: {
    games?: number;
    completed_games?: number;
    error_games?: number;
    seconds_per_game?: number;
    avg_turns?: number;
  } | null;
  items: SelfplayAnomalyReplayItem[];
}

export type PlayerMode =
  | { type: 'replay'; replay_id: string }
  | { type: 'realtime'; game_id: string };
