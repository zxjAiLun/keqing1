// src/replay_ui/src/types/battle.ts

export interface PlayerInfo {
  player_id: number;
  name: string;
  type: "human" | "bot";
}

export interface DiscardEntry {
  pai: string;
  tsumogiri: boolean;
  /** 后端实际不返回此字段，需兼容处理 */
  reach_declared?: boolean;
}

export interface MeldEntry {
  type: "pon" | "chi" | "daiminkan" | "ankan" | "kakan";
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
}

export type ActionType =
  | "dahai"
  | "reach"
  | "pon"
  | "chi"
  | "daiminkan"
  | "ankan"
  | "kakan"
  | "hora"
  | "ryukyoku"
  | "none";

export interface BattleState {
  game_id: string;
  phase: "waiting" | "playing" | "ended";
  winner: number | null;
  bakaze: string;
  kyoku: number;
  honba: number;
  kyotaku: number;
  oya: number;
  scores: number[];
  dora_markers: string[];
  actor_to_move: number | null;
  /** 后端不返回 pai_raw，忽略此字段 */
  last_discard: { actor: number; pai: string; pai_raw?: string } | null;
  hand: string[];
  tsumo_pai: string | null;
  discards: DiscardEntry[][];
  melds: MeldEntry[][];
  reached: boolean[];
  pending_reach: boolean[];
  legal_actions: Action[];
  remaining_wall: number;
  human_player_id: number;
  player_info: PlayerInfo[];
  /** 仅回放使用：当前处于摸牌前态的玩家，用于展示他家第 14 张手牌 */
  replay_draw_actor?: number | null;
}

export interface StartBattleRequest {
  player_name: string;
  bot_count?: number;
  seed?: number;
  bot_model?: string; // "modelv5" | "modelv5_naga" | "keqingv1" | "keqingv2"
}

export interface StartBattleResponse {
  game_id: string;
  state: BattleState;
}

export interface ActionRequest {
  game_id: string;
  action: Action;
}

export interface ActionResponse {
  success: boolean;
  state: BattleState;
  bot_action?: Action;
}
