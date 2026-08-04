// src/replay_ui/src/utils/replayAdapter.ts
// DecisionLogEntry → BattleState 适配，供 MahjongTable 消费

import type { Action, DecisionLogEntry } from '../types/replay';
import type { BattleState, DiscardEntry, MeldEntry } from '../types/battle';
import { TILE_ORDER } from './tileUtils.ts';
import { sameReplayAction } from './tileUtils.ts';

// ---------------------------------------------------------------------------
// 柱状图数据结构
// ---------------------------------------------------------------------------
export interface LogitTileData {
  pai: string;
  /** final_score 优先，兼容 beam_score/logit；undefined 表示该牌无候选权重 */
  score: number | undefined;
  /** Q value 按 tau=1 softmax 后的概率 */
  prob?: number;
  /** 概率百分比 0-100，用于柱高 */
  pct: number;
  isChosen: boolean;
  isGt: boolean;
  isTsumo: boolean;
  teacherBars?: Array<{
    model: string;
    qValue?: number | null;
    prob?: number | null;
    pct: number;
    rank?: number | null;
  }>;
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
  const probMap: Record<string, number> = {};
  const teacherMap: Record<string, LogitTileData['teacherBars']> = {};
  for (const c of entry.candidates ?? []) {
    if (c.action?.type === 'dahai' && c.action.pai) {
      scoreMap[c.action.pai] = c.final_score ?? c.beam_score ?? c.logit;
      if (typeof c.prob === 'number' && Number.isFinite(c.prob)) {
        probMap[c.action.pai] = c.prob;
      }
      if (c.teachers?.length) {
        teacherMap[c.action.pai] = c.teachers.map((teacher) => ({
          model: teacher.model,
          qValue: teacher.q_value,
          prob: teacher.prob,
          pct: typeof teacher.prob === 'number' && Number.isFinite(teacher.prob)
            ? teacher.prob * 100
            : 0,
          rank: teacher.rank,
        }));
      }
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
    const prob = probMap[pai];
    const pct = score !== undefined
      ? prob !== undefined ? prob * 100 : scorePctMap[pai] ?? 0
      : 0;
    return {
      pai,
      score,
      prob,
      pct,
      isChosen: pai === chosenPai,
      isGt: pai === gtPai,
      isTsumo: pai === tsumo,
      teacherBars: teacherMap[pai],
    };
  });
}

/** DecisionLogEntry 中非 dahai 候选的权重列表（final_score 优先，降序） */
export interface CandidateScore {
  action: DecisionLogEntry['candidates'][number]['action'];
  score: number;
  prob?: number;
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
    prob: c.prob,
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
    is_obs: entry.is_obs === true,
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

const MELD_ACTION_TYPES = ['chi', 'pon', 'daiminkan', 'ankan', 'kakan'] as const;

function normalizedTileList(tiles: string[]): string {
  return [...tiles].sort().join('|');
}

function sameMeldShape(a: MeldEntry, b: MeldEntry): boolean {
  return a.type === b.type
    && a.pai === b.pai
    && a.target === b.target
    && normalizedTileList(a.consumed) === normalizedTileList(b.consumed);
}

function meldShapeForAction(action: Action): MeldEntry {
  return {
    type: action.type as MeldEntry['type'],
    pai: action.pai ?? '',
    consumed: [...(action.consumed ?? [])],
    target: action.target ?? action.actor,
  };
}

function needsMeldRewind(state: BattleState, action: Action): boolean {
  if (!(MELD_ACTION_TYPES as readonly string[]).includes(action?.type ?? '')) return false;
  const actorMelds = state.melds[action.actor] ?? [];
  return actorMelds.some((meld) => sameMeldShape(meld, meldShapeForAction(action)));
}

function addTileOnce(hand: string[], tile?: string): string[] {
  if (!tile) return [...hand];
  return [...hand, tile];
}

/** 把副露动作回退为动作前状态（pre phase 专用）。 */
function rewindMeldAction(
  state: BattleState,
  action: Action,
  viewPlayerId: number,
): BattleState {
  const melds = cloneMelds(state.melds);
  const actorMelds = [...(melds[action.actor] ?? [])];
  const shape = meldShapeForAction(action);

  // 匹配此次动作新增的 meld（type/pai/target/consumed），不是简单 popLast
  let removeIndex = -1;
  for (let i = actorMelds.length - 1; i >= 0; i--) {
    if (sameMeldShape(actorMelds[i], shape)) { removeIndex = i; break; }
  }
  const targetIndex = removeIndex >= 0 ? removeIndex : actorMelds.length - 1;
  const removed = actorMelds[targetIndex];
  const remaining = actorMelds.filter((_, i) => i !== targetIndex);

  let hand = [...state.hand];
  if (action.type === 'kakan' && removed && removed.consumed.length >= 4) {
    // kakan 前态是原 pon：consumed = [手牌1, 手牌2, 被鸣, 加杠]，加杠牌放回手中
    remaining.push({
      type: 'pon',
      pai: removed.consumed[2] ?? removed.pai,
      consumed: removed.consumed.slice(0, 2),
      target: removed.target,
    });
    if (action.actor === viewPlayerId) {
      for (const tile of removed.consumed.slice(3)) hand = addTileOnce(hand, tile);
    }
  } else if (action.actor === viewPlayerId) {
    for (const tile of action.consumed ?? []) hand = addTileOnce(hand, tile);
  }

  const lastDiscard =
    action.type === 'ankan'
      ? null
      : action.pai
        ? { actor: action.target ?? action.actor, pai: action.pai, pai_raw: action.pai }
        : null;

  return {
    ...state,
    hand,
    melds: { ...melds, [action.actor]: remaining },
    last_discard: lastDiscard,
    actor_to_move: action.actor,
    tsumo_pai: null,
  };
}

function supportsPostActionPhase(entry: DecisionLogEntry): boolean {
  const action = entry.gt_action ?? entry.chosen;
  return ['dahai', 'chi', 'pon', 'daiminkan', 'ankan', 'kakan', 'hora', 'ryukyoku'].includes(action?.type ?? '');
}

function supportsReachPhase(entry: DecisionLogEntry): boolean {
  const action = entry.gt_action ?? entry.chosen;
  return action?.type === 'reach';
}

function isResponseNoneAction(entry: DecisionLogEntry | null | undefined): boolean {
  const action = entry?.gt_action ?? entry?.chosen;
  return action?.type === 'none';
}

function hasNonNoneCandidates(entry: DecisionLogEntry | null | undefined): boolean {
  return Boolean(entry?.candidates?.some((candidate) => candidate.action?.type !== 'none'));
}

export function isCollapsibleResponsePassStep(
  entry: DecisionLogEntry | null | undefined,
  prevEntry?: DecisionLogEntry | null,
): boolean {
  const prevAction = prevEntry ? (prevEntry.gt_action ?? prevEntry.chosen) : null;
  return Boolean(
    entry
    && prevEntry
    && isResponseNoneAction(entry)
    && prevEntry.is_obs
    && prevAction?.type === 'dahai'
    && !hasNonNoneCandidates(entry)
  );
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
  if (isCollapsibleResponsePassStep(entry, prevEntry)) {
    return entryToBattleState(prevEntry!, playerNames, viewPlayerId, 'post');
  }
  const mergedEntry = mergeReplayEntryWithPrevious(entry, prevEntry);
  const previousAction = prevEntry ? (prevEntry.gt_action ?? prevEntry.chosen) : null;
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
  const suppressSnapshotDrawActor = action?.type === 'none';
  const pendingDiscardActorFromSnapshot =
    mergedEntry.actor_to_move !== null && mergedEntry.actor_to_move !== undefined && mergedEntry.last_discard === null
      ? mergedEntry.actor_to_move
      : null;
  const showsReplayDraw = isPreDiscardPhase
    ? action.actor
    : suppressSnapshotDrawActor
      ? null
      : pendingDiscardActorFromSnapshot;
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
      if (!needsMeldRewind(baseState, action)) {
        // snapshot 已是动作前：无需回退
        return baseState;
      }
      return rewindMeldAction(baseState, action, viewPlayerId);
    }

    return baseState;
  }

  if (phase === 'pre') {
    if (action && needsMeldRewind(baseState, action)) {
      // 动作前：把 snapshot 中的副露动作回退（恢复 consumed、正确 meld、last_discard）
      return rewindMeldAction(baseState, action, viewPlayerId);
    }
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
    const declaresReach =
      (mergedEntry.gt_action?.type === 'reach' && mergedEntry.gt_action.actor === action.actor)
      || (previousAction?.type === 'reach' && previousAction.actor === action.actor);
    if (action.actor === viewPlayerId) {
      const handWithDraw = baseState.tsumo_pai ? [...nextState.hand, baseState.tsumo_pai] : [...nextState.hand];
      nextState.hand = removeTileOnce(handWithDraw, action.pai);
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
    if (needsMeldRewind(baseState, action)) {
      // snapshot 已经是动作后状态（meld 已在、consumed 已移除）：不再前向应用
      return baseState;
    }
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
