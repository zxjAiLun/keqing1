// src/replay_ui/src/types/participants.ts
//
// R10 参赛身份 / 四人阵容 / 统一对局账本 —— 与后端 src/participants/schemas.py 对应。

export type AccountType = 'human' | 'managed_bot' | 'external_bot';
export type ControllerType = 'human_ui' | 'local_model' | 'external_agent' | 'manual_only';
export type SourceType = 'native' | 'imported' | 'manual';
export type DataCompleteness = 'result_only' | 'hand_summary' | 'full_replay';
export type GameLength = 'tonpu' | 'hanchan';
export type SeatNo = 0 | 1 | 2 | 3;

export interface Account {
  account_id: string;
  display_name: string;
  account_type: AccountType;
  enabled: boolean;
  default_controller: ControllerType;
  avatar?: string | null;
  note?: string | null;
  migrated_from_replay?: boolean;
  created_at: string;
  updated_at: string;
}

export interface ModelArtifact {
  model_artifact_id: string;
  label: string;
  model_identity_id: string;
  artifact_path?: string | null;
  hash?: string | null;
  is_current: boolean;
  created_at: string;
  retired_at?: string | null;
}

export interface ModelIdentity {
  model_identity_id: string;
  label: string;
  kind: 'local_model' | 'external_agent' | 'none';
  account_id?: string | null;
  is_current: boolean;
  created_at: string;
  retired_at?: string | null;
  note?: string | null;
  artifacts: ModelArtifact[];
}

export interface MatchSeat {
  seat: SeatNo;
  account_id: string;
  controller_type?: ControllerType | null;
  model_identity_id?: string | null;
  model_artifact_id?: string | null;
}

export interface Match {
  schema: string;
  match_id: string;
  occurred_at: string;
  game_length: GameLength;
  rule_set: string;
  starting_points: number;
  initial_oya: number;
  source: SourceType;
  source_ref?: string | null;
  note?: string | null;
  data_completeness: DataCompleteness;
  replay_id?: string | null;
  seats: MatchSeat[];
  final_scores: number[];
  ranks: number[];
  status: 'active' | 'void';
  void_reason?: string | null;
  revision: number;
  latest_revision_id: string;
  created_at: string;
  updated_at: string;
  created_by: 'manual' | 'migration' | 'system';
}

export interface ValidationIssue {
  code: string;
  message: string;
}

export interface RevisionSummary {
  revision_id: string;
  match_id: string;
  revision: number;
  action: 'create' | 'revise' | 'void';
  created_at: string;
  by: string;
  force: boolean;
  reason?: string | null;
  validation: { passed: boolean; issues: ValidationIssue[] };
}

export interface AccountCreate {
  account_id?: string | null;
  display_name: string;
  account_type: AccountType;
  default_controller?: ControllerType | null;
  avatar?: string | null;
  note?: string | null;
}

export interface AccountUpdate {
  display_name?: string | null;
  enabled?: boolean | null;
  default_controller?: ControllerType | null;
  avatar?: string | null;
  note?: string | null;
}

export interface ModelIdentityCreate {
  model_identity_id?: string | null;
  label: string;
  kind: 'local_model' | 'external_agent' | 'none';
  account_id?: string | null;
  artifact_path?: string | null;
  note?: string | null;
}

export interface ModelArtifactCreate {
  model_artifact_id?: string | null;
  label: string;
  artifact_path?: string | null;
}

export interface MatchCreate {
  occurred_at: string;
  game_length: GameLength;
  rule_set?: string;
  starting_points?: number;
  initial_oya?: number;
  source?: SourceType;
  source_ref?: string | null;
  note?: string | null;
  data_completeness?: DataCompleteness;
  replay_id?: string | null;
  seats: MatchSeat[];
  final_scores: number[];
  force?: boolean;
  reason?: string | null;
}

export interface MatchRevise {
  occurred_at?: string | null;
  game_length?: GameLength | null;
  rule_set?: string | null;
  starting_points?: number | null;
  initial_oya?: number | null;
  note?: string | null;
  data_completeness?: DataCompleteness | null;
  seats?: MatchSeat[] | null;
  final_scores?: number[] | null;
  force?: boolean;
  reason?: string | null;
}

export interface MatchResponse {
  match: Match;
  revisions: RevisionSummary[];
}

export interface MatchListResponse {
  schema: string;
  matches: Match[];
  total: number;
}

export interface AccountsResponse {
  schema: string;
  accounts: Account[];
}

export interface ModelsResponse {
  schema: string;
  identities: ModelIdentity[];
}
