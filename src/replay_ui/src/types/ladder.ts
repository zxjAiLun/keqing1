// src/replay_ui/src/types/ladder.ts
// Model Ladder & Account Profiles 类型契约（与 /api/ladder/* 响应对齐）。

export type LadderReadinessState = 'ready' | 'not_published' | 'invalid' | 'registry_mismatch';

export interface LadderSeasonReadiness {
  state: LadderReadinessState;
  code: string;
  message: string;
  detail?: string;
  retryable: boolean;
}

export type LadderDefaultSource = 'registry' | 'single_season' | null;

export interface LadderSeasonsResponse {
  schema: string;
  default_season_id: string | null;
  default_source: LadderDefaultSource;
  seasons: LadderSeason[];
}

export interface LadderSeasonScoring {
  pt_profile?: string;
  pt_rank_deltas?: number[];
  pt_initial?: number;
  pt_target?: number;
  rating_initial?: number;
  rating_formula?: string;
  rank_name?: string;
}

export interface LadderSeason {
  season_id: string;
  title?: string;
  status?: string;
  games_expected?: number;
  notes?: string;
  models?: string[];
  accounts?: number;
  games?: number;
  data_ready?: boolean;
  report_schema?: string;
  /** 当前只读快照目录名（如 20260801-203000） */
  snapshot_id?: string;
  /** account_summary.json 的 mtime（epoch 秒） */
  updated_at?: number;
  scoring?: LadderSeasonScoring;
  /** 是否默认赛季（目录派生的规范化标记） */
  is_default?: boolean;
  /** 未就绪原因（结构化 readiness 契约；就绪时 state="ready"） */
  readiness?: LadderSeasonReadiness;
}

export interface LadderAccountRow {
  account_id: string;
  display_name: string;
  model_id: string;
  checkpoint?: string | null;
  games: number;
  rank_name?: string | null;
  pt_current: number;
  pt_target: number;
  pt_gap: number;
  rating: number;
  rank_1: number;
  rank_2: number;
  rank_3: number;
  rank_4: number;
  rank_1_rate: number | null;
  rank_4_rate: number | null;
  avg_rank: number | null;
  avg_rank_pt: number | null;
  agari_rate: number | null;
  houjuu_rate: number | null;
  fuuro_rate: number | null;
  riichi_rate: number | null;
  agari_rate_after_fuuro: number | null;
  houjuu_rate_after_fuuro: number | null;
  agari_rate_after_riichi: number | null;
  houjuu_rate_after_riichi: number | null;
  avg_point_per_agari: number | null;
  total_delta_score: number | null;
  rank_position?: number;
}

export interface LadderModelSummary {
  model_id: string;
  accounts: number;
  games: number;
  avg_pt: number | null;
  avg_rating: number | null;
  avg_rank: number | null;
  avg_rank_pt: number | null;
}

export interface LadderResponse {
  season: LadderSeason;
  sort: string;
  accounts: LadderAccountRow[];
  models: LadderModelSummary[];
}

export interface LadderCurvePoint {
  games: number;
  rating: number;
  pt: number;
}

export interface LadderRecentGame {
  game_index: number;
  rank: number;
  final_score: number;
  score_delta: number;
  pt_delta: number;
  pt_after: number;
  rating_after: number;
  source_log?: string | null;
}

export interface LadderAccountDetail {
  season: LadderSeason;
  account: LadderAccountRow;
  rank_distribution: number[];
  curve: LadderCurvePoint[];
  recent_games: LadderRecentGame[];
}

export interface LadderModelDetail {
  season: LadderSeason;
  model: {
    model_id: string;
    checkpoint?: string | null;
    accounts: LadderAccountRow[];
    summary: LadderModelSummary | null;
  };
  league_summary: {
    schema?: string;
    games_total?: number;
    lineups?: string[];
    model: Record<string, unknown>;
  } | null;
}
