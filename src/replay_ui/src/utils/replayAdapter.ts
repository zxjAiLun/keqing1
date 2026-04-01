// src/replay_ui/src/utils/replayAdapter.ts
// DecisionLogEntry → BattleState 适配，供 MahjongTable 消费

import type { DecisionLogEntry } from '../types/replay';
import type { BattleState, DiscardEntry, MeldEntry } from '../types/battle';
import { TILE_ORDER } from './tileUtils';
import { sameReplayAction } from './tileUtils';

// ---------------------------------------------------------------------------
// 柱状图数据结构
// ---------------------------------------------------------------------------
export interface LogitTileData {
  pai: string;
  /** final_score 优先，兼容 beam_score/logit；undefined 表示该牌无候选权重 */
  score: number | undefined;
  /** 相对百分比 0-100，用于柱高 */
  pct: number;
  isChosen: boolean;
  isGt: boolean;
  isTsumo: boolean;
}

function normalizedPercentages(scores: number[]): number[] {
  if (scores.length === 0) return [];
  const maxScore = Math.max(...scores);
  const exps = scores.map((score) => Math.exp(score - maxScore));
  const total = exps.reduce((sum, value) => sum + value, 0);
  if (!Number.isFinite(total) || total <= 0) {
    return scores.map(() => 0);
  }
  return exps.map((value) => (value / total) * 100);
}

/** 从 DecisionLogEntry 提取手牌柱状图数据（dahai 类决策才有意义） */
export function buildLogitData(entry: DecisionLogEntry): LogitTileData[] {
  const hand = entry.hand ?? [];
  const tsumo = entry.tsumo_pai ?? null;
  const chosenPai = entry.chosen?.type === 'dahai' ? (entry.chosen.pai ?? null) : null;
  const gtPai = entry.gt_action?.type === 'dahai' ? (entry.gt_action.pai ?? null) : null;

  // 构建 pai → score 映射（final_score 优先，兼容 beam/logit）
  const scoreMap: Record<string, number> = {};
  for (const c of entry.candidates ?? []) {
    if (c.action?.type === 'dahai' && c.action.pai) {
      scoreMap[c.action.pai] = c.final_score ?? c.beam_score ?? c.logit;
    }
  }

  const normalizedPcts = normalizedPercentages(Object.values(scoreMap));
  const scorePctMap = Object.fromEntries(
    Object.keys(scoreMap).map((pai, idx) => [pai, normalizedPcts[idx] ?? 0]),
  );

  // 排序：摸切排最后，其余按 TILE_ORDER
  const others = hand.filter(t => t !== tsumo);
  others.sort((a, b) => (TILE_ORDER[a] ?? 99) - (TILE_ORDER[b] ?? 99));
  const sorted = tsumo ? [...others, tsumo] : others;

  return sorted.map(pai => {
    const score = scoreMap[pai];
    const pct = score !== undefined
      ? Math.max(6, scorePctMap[pai] ?? 0)
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

/** DecisionLogEntry 中非 dahai 候选的权重列表（final_score 优先，降序） */
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
    score: c.final_score ?? c.beam_score ?? c.logit,
    isChosen: sameReplayAction(chosen, c.action),
    isGt: sameReplayAction(gt, c.action),
  }));

  list.sort((a, b) => b.score - a.score);
  return list;
}

export type ReplayBoardPhase = 'pre' | 'reach' | 'post';

function mergeReplayEntryWithPrevious(
  entry: DecisionLogEntry,
  prevEntry?: DecisionLogEntry | null,
): DecisionLogEntry {
  if (!prevEntry) return entry;
  return {
    ...prevEntry,
    ...entry,
    discards: entry.discards ?? prevEntry.discards,
    melds: entry.melds ?? prevEntry.melds,
    dora_markers: entry.dora_markers ?? prevEntry.dora_markers,
    reached: entry.reached ?? prevEntry.reached,
    scores: entry.scores ?? prevEntry.scores,
  };
}

function cloneDiscards(discards: DiscardEntry[][]): DiscardEntry[][] {
  return discards.map(row => row.map(item => ({ ...item })));
}

function cloneMelds(melds: MeldEntry[][]): MeldEntry[][] {
  return melds.map(row => row.map(item => ({ ...item, consumed: [...item.consumed] })));
}

function removeTileOnce(hand: string[], pai?: string): string[] {
  if (!pai) return [...hand];
  const idx = hand.findIndex(tile => tile === pai);
  if (idx < 0) return [...hand];
  const next = [...hand];
  next.splice(idx, 1);
  return next;
}

function popLastDiscardIfMatches(
  discards: DiscardEntry[][],
  actor: number,
  pai?: string,
): { discards: DiscardEntry[][]; removed?: DiscardEntry } {
  const next = cloneDiscards(discards);
  const actorDiscards = [...(next[actor] ?? [])];
  const last = actorDiscards[actorDiscards.length - 1];
  if (last && (!pai || last.pai === pai)) {
    actorDiscards.pop();
    next[actor] = actorDiscards;
    return { discards: next, removed: last };
  }
  return { discards: next };
}

function popLastMeld(
  melds: MeldEntry[][],
  actor: number,
): { melds: MeldEntry[][]; removed?: MeldEntry } {
  const next = cloneMelds(melds);
  const actorMelds = [...(next[actor] ?? [])];
  const removed = actorMelds.pop();
  next[actor] = actorMelds;
  return { melds: next, removed };
}

function supportsPostActionPhase(entry: DecisionLogEntry): boolean {
  const action = entry.gt_action ?? entry.chosen;
  return ['dahai', 'chi', 'pon', 'daiminkan', 'ankan', 'kakan', 'hora', 'ryukyoku'].includes(action?.type ?? '');
}

function supportsReachPhase(entry: DecisionLogEntry): boolean {
  const action = entry.gt_action ?? entry.chosen;
  return action?.type === 'reach';
}

export function hasReplayPostAction(entry: DecisionLogEntry | null | undefined): boolean {
  return Boolean(entry && supportsPostActionPhase(entry));
}

export function hasReplayReachPhase(entry: DecisionLogEntry | null | undefined): boolean {
  return Boolean(entry && supportsReachPhase(entry));
}

// ---------------------------------------------------------------------------
// DecisionLogEntry → BattleState
// ---------------------------------------------------------------------------
export function entryToBattleState(
  entry: DecisionLogEntry,
  playerNames: string[] = ['P0', 'P1', 'P2', 'P3'],
  viewPlayerId: number = 0,
  phase: ReplayBoardPhase = 'pre',
  prevEntry?: DecisionLogEntry | null,
): BattleState {
  const mergedEntry = mergeReplayEntryWithPrevious(entry, prevEntry);
  // Record<number, X[]> → X[][]（4家，空补齐）
  function toArray<T>(rec: Record<number, T[]> | undefined): T[][] {
    return [0, 1, 2, 3].map(i => rec?.[i] ?? []);
  }

  const discards = Array.isArray(mergedEntry.discards)
    ? mergedEntry.discards as DiscardEntry[][]
    : toArray(mergedEntry.discards) as DiscardEntry[][];
  const melds = Array.isArray(mergedEntry.melds)
    ? mergedEntry.melds as MeldEntry[][]
    : toArray(mergedEntry.melds) as MeldEntry[][];
  const action = mergedEntry.gt_action ?? mergedEntry.chosen;
  const isPreDiscardPhase = phase === 'pre' && action?.type === 'dahai';
  const pendingDiscardActorFromSnapshot =
    mergedEntry.actor_to_move !== null && mergedEntry.actor_to_move !== undefined && mergedEntry.last_discard === null
      ? mergedEntry.actor_to_move
      : null;
  const showsReplayDraw = isPreDiscardPhase ? action.actor : pendingDiscardActorFromSnapshot;
  const normalizedTsumoPai =
    isPreDiscardPhase && action.actor === viewPlayerId
      ? (mergedEntry.tsumo_pai ?? (action.tsumogiri ? action.pai ?? null : null))
      : mergedEntry.tsumo_pai;
  const normalizedActorToMove =
    isPreDiscardPhase
      ? action.actor
      : mergedEntry.actor_to_move;
  const normalizedLastDiscard =
    isPreDiscardPhase && mergedEntry.last_discard?.actor === action.actor
      ? null
      : mergedEntry.last_discard;

  const baseState: BattleState = {
    game_id: 'replay',
    phase: 'playing',
    winner: null,
    bakaze: mergedEntry.bakaze,
    kyoku: mergedEntry.kyoku,
    honba: mergedEntry.honba,
    kyotaku: 0,
    oya: mergedEntry.oya,
    scores: mergedEntry.scores,
    dora_markers: mergedEntry.dora_markers ?? [],
    actor_to_move: normalizedActorToMove,
    last_discard: normalizedLastDiscard,
    hand: mergedEntry.hand ?? [],
    tsumo_pai: normalizedTsumoPai,
    discards,
    melds,
    reached: mergedEntry.reached ?? [false, false, false, false],
    pending_reach: [false, false, false, false],
    legal_actions: [],
    remaining_wall: 0,
    human_player_id: viewPlayerId,
    player_info: playerNames.map((name, id) => ({ player_id: id, name, type: id === 0 ? 'human' : 'bot' })),
    replay_draw_actor: showsReplayDraw,
  };

  if (mergedEntry.is_obs) {
    const obsBoardPhase =
      action?.type === 'dahai'
        ? (phase === 'post' ? 'after_action' : 'before_action')
        : (mergedEntry.board_phase ?? 'after_action');
    if (obsBoardPhase === 'after_action') {
      return {
        ...baseState,
        replay_draw_actor: null,
      };
    }

    const nextState: BattleState = {
      ...baseState,
      hand: [...baseState.hand],
      discards: cloneDiscards(baseState.discards),
      melds: cloneMelds(baseState.melds),
      reached: [...baseState.reached],
      pending_reach: [...baseState.pending_reach],
      last_discard: baseState.last_discard ? { ...baseState.last_discard } : null,
      actor_to_move: baseState.actor_to_move,
      replay_draw_actor: null,
    };

    if (action.type === 'dahai') {
      const { discards: prevDiscards, removed } = popLastDiscardIfMatches(
        baseState.discards,
        action.actor,
        action.pai,
      );
      nextState.discards = prevDiscards;
      nextState.last_discard = null;
      nextState.actor_to_move = action.actor;
      nextState.replay_draw_actor = action.actor;
      if (removed?.reach_declared) {
        nextState.reached[action.actor] = false;
        nextState.pending_reach[action.actor] = true;
      }
      return nextState;
    }

    if (['chi', 'pon', 'daiminkan', 'ankan', 'kakan'].includes(action.type)) {
      const { melds: prevMelds } = popLastMeld(baseState.melds, action.actor);
      nextState.melds = prevMelds;
      nextState.actor_to_move = action.actor;
      nextState.last_discard = action.type === 'ankan'
        ? null
        : action.pai
          ? { actor: action.target ?? action.actor, pai: action.pai, pai_raw: action.pai }
          : null;
      return nextState;
    }

    return baseState;
  }

  if (phase === 'pre') {
    return baseState;
  }

  const nextState: BattleState = {
    ...baseState,
    hand: [...baseState.hand],
    discards: cloneDiscards(baseState.discards),
    melds: cloneMelds(baseState.melds),
    reached: [...baseState.reached],
    pending_reach: [...baseState.pending_reach],
    last_discard: baseState.last_discard ? { ...baseState.last_discard } : null,
    actor_to_move: baseState.actor_to_move,
    replay_draw_actor: null,
  };

  if (phase === 'reach' && supportsReachPhase(mergedEntry)) {
    nextState.pending_reach[action.actor] = true;
    nextState.actor_to_move = action.actor;
    return nextState;
  }

  if (!supportsPostActionPhase(mergedEntry)) {
    return baseState;
  }

  if (action.type === 'dahai') {
    const declaresReach = mergedEntry.gt_action?.type === 'reach' && mergedEntry.gt_action.actor === action.actor;
    if (action.actor === viewPlayerId) {
      nextState.hand = removeTileOnce(nextState.hand, action.pai);
      nextState.tsumo_pai = null;
    }
    if (declaresReach) {
      nextState.reached[action.actor] = true;
      nextState.pending_reach[action.actor] = false;
    }
    nextState.discards[action.actor] = [
      ...(nextState.discards[action.actor] ?? []),
      {
        pai: action.pai ?? '?',
        tsumogiri: Boolean(action.tsumogiri),
        reach_declared: declaresReach,
      },
    ];
    nextState.last_discard = action.pai
      ? { actor: action.actor, pai: action.pai, pai_raw: action.pai }
      : null;
    nextState.actor_to_move = null;
    return nextState;
  }

  if (['chi', 'pon', 'daiminkan', 'ankan', 'kakan'].includes(action.type)) {
    const meldType = action.type as MeldEntry['type'];
    if (action.actor === viewPlayerId) {
      let hand = [...nextState.hand];
      for (const tile of action.consumed ?? []) {
        hand = removeTileOnce(hand, tile);
      }
      nextState.hand = hand;
      nextState.tsumo_pai = null;
    }
    nextState.melds[action.actor] = [
      ...(nextState.melds[action.actor] ?? []),
      {
        type: meldType,
        pai: action.pai ?? '',
        consumed: [...(action.consumed ?? [])],
        target: action.target ?? action.actor,
      },
    ];
    nextState.last_discard = null;
    nextState.actor_to_move = action.actor;
    return nextState;
  }

  return baseState;
}
