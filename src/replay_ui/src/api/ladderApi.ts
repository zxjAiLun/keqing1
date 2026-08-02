// src/replay_ui/src/api/ladderApi.ts
import { ApiError } from './replayApi';
import type {
  LadderAccountDetail,
  LadderModelDetail,
  LadderResponse,
  LadderSeason,
} from '../types/ladder';

const API_BASE = '/api';

async function api<T>(path: string): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`);
  if (!res.ok) {
    const body = await res.json().catch(() => null);
    throw new ApiError(res.status, body);
  }
  return res.json();
}

export const ladderApi = {
  /** 已注册的评测赛季 */
  listSeasons: (): Promise<{ seasons: LadderSeason[] }> =>
    api('/ladder/seasons'),

  /** 赛季天梯榜：账号排名 + 模型展示性聚合 */
  getLadder: (seasonId: string, sort = 'pt'): Promise<LadderResponse> =>
    api(`/ladder/seasons/${encodeURIComponent(seasonId)}?sort=${encodeURIComponent(sort)}`),

  /** 账号详情：聚合指标 + 曲线 + 最近对局 */
  getAccount: (seasonId: string, accountId: string, recentGames = 50): Promise<LadderAccountDetail> =>
    api(`/ladder/seasons/${encodeURIComponent(seasonId)}/accounts/${encodeURIComponent(accountId)}?recent_games=${recentGames}`),

  /** 模型详情：账号横向对比 + 可选联赛聚合 */
  getModel: (seasonId: string, modelId: string): Promise<LadderModelDetail> =>
    api(`/ladder/seasons/${encodeURIComponent(seasonId)}/models/${encodeURIComponent(modelId)}`),
};
