// src/replay_ui/src/utils/replayAdapter.ts
// DecisionLogEntry → BattleState 适配，供 MahjongTable 消费

import type { DecisionLogEntry } from '../types/replay';
import type { BattleState, DiscardEntry, MeldEntry } from '../types/battle';
import { TILE_ORDER } from './tileUtils';

// ---------------------------------------------------------------------------
// 柱状图数据结构
// ---------------------------------------------------------------------------
export interface LogitTileData {
  pai: string;
  /** beam_score 优先，否则用 logit；undefined 表示该牌无候选权重 */
  score: number | undefined;
  /** 相对百分比 0-100，用于柱高 */
  pct: number;
  isChosen: boolean;
  isGt: boolean;
  isTsumo: boolean;
}

/** 从 DecisionLogEntry 提取手牌柱状图数据（dahai 类决策才有意义） */
export function buildLogitData(entry: DecisionLogEntry): LogitTileData[] {
  const hand = entry.hand ?? [];
  const tsumo = entry.tsumo_pai ?? null;
  const chosenPai = entry.chosen?.type === 'dahai' ? (entry.chosen.pai ?? null) : null;
  const gtPai = entry.gt_action?.type === 'dahai' ? (entry.gt_action.pai ?? null) : null;

  // 构建 pai → score 映射（beam_score 优先）
  const scoreMap: Record<string, number> = {};
  for (const c of entry.candidates ?? []) {
    if (c.action?.type === 'dahai' && c.action.pai) {
      scoreMap[c.action.pai] = c.beam_score ?? c.logit;
    }
  }

  const scores = Object.values(scoreMap);
  const maxScore = scores.length ? Math.max(...scores) : 1;
  const minScore = scores.length ? Math.min(...scores) : 0;
  const range = Math.max(maxScore - minScore, 0.001);

  // 排序：摸切排最后，其余按 TILE_ORDER
  const others = hand.filter(t => t !== tsumo);
  others.sort((a, b) => (TILE_ORDER[a] ?? 99) - (TILE_ORDER[b] ?? 99));
  const sorted = tsumo ? [...others, tsumo] : others;

  return sorted.map(pai => {
    const score = scoreMap[pai];
    const pct = score !== undefined
      ? Math.max(2, ((score - minScore) / range) * 100)
      : 0;
    return {
      pai,
      score,
      pct,
      isChosen: pai === chosenPai,
      isGt: pai === gtPai,
      isTsumo: pai === tsumo,
    };
  });
}

/** DecisionLogEntry 中非 dahai 候选的权重列表（beam_score 优先，降序） */
export interface CandidateScore {
  action: DecisionLogEntry['candidates'][number]['action'];
  score: number;
  isChosen: boolean;
  isGt: boolean;
}

export function buildCandidateScores(entry: DecisionLogEntry): CandidateScore[] {
  const candidates = entry.candidates ?? [];
  const chosen = entry.chosen;
  const gt = entry.gt_action;

  const list: CandidateScore[] = candidates.map(c => ({
    action: c.action,
    score: c.beam_score ?? c.logit,
    isChosen: chosen?.type === c.action.type && chosen?.pai === c.action.pai,
    isGt: gt?.type === c.action.type && gt?.pai === c.action.pai,
  }));

  list.sort((a, b) => b.score - a.score);
  return list;
}

// ---------------------------------------------------------------------------
// DecisionLogEntry → BattleState
// ---------------------------------------------------------------------------
export function entryToBattleState(
  entry: DecisionLogEntry,
  playerNames: string[] = ['P0', 'P1', 'P2', 'P3'],
  viewPlayerId: number = 0,
): BattleState {
  // Record<number, X[]> → X[][]（4家，空补齐）
  function toArray<T>(rec: Record<number, T[]> | undefined): T[][] {
    return [0, 1, 2, 3].map(i => rec?.[i] ?? []);
  }

  const discards = toArray(entry.discards) as DiscardEntry[][];
  const melds = toArray(entry.melds) as MeldEntry[][];

  return {
    game_id: 'replay',
    phase: 'playing',
    winner: null,
    bakaze: entry.bakaze,
    kyoku: entry.kyoku,
    honba: entry.honba,
    kyotaku: 0,
    oya: entry.kyoku - 1,
    scores: entry.scores,
    dora_markers: entry.dora_markers ?? [],
    actor_to_move: entry.actor_to_move,
    last_discard: entry.last_discard,
    hand: entry.hand ?? [],
    tsumo_pai: entry.tsumo_pai,
    discards,
    melds,
    reached: entry.reached ?? [false, false, false, false],
    pending_reach: [false, false, false, false],
    legal_actions: [],
    remaining_wall: 0,
    human_player_id: viewPlayerId,
    player_info: playerNames.map((name, id) => ({ player_id: id, name, type: id === 0 ? 'human' : 'bot' })),
  };
}
