// src/replay_ui/src/hooks/useAutoActions.ts
// 返回值中所有 has* 字段均已包含 isMyTurn 前提（isMyTurn=false 时全为 false）
import type { BattleState } from "../types/battle";

interface UseAutoActionsOptions {
  state: BattleState | null;
  autoHora: boolean;
  noMeld: boolean;
  autoTsumogiri: boolean;
}

export function useAutoActions({
  state,
  autoHora,
  noMeld,
  autoTsumogiri,
}: UseAutoActionsOptions) {
  const isMyTurn = state?.actor_to_move === state?.human_player_id && state?.phase === "playing";
  const legalActions = state?.legal_actions ?? [];

  const hasHora    = isMyTurn && legalActions.some(a => a.type === "hora");
  const hasMeld    = isMyTurn && legalActions.some(
    a => a.type === "pon" || a.type === "chi" || a.type === "daiminkan"
  );
  const hasDahai   = isMyTurn && legalActions.some(a => a.type === "dahai");
  const hasNone    = isMyTurn && legalActions.some(a => a.type === "none");
  const hasRealChoice = isMyTurn && legalActions.some(
    a => a.type === "pon" || a.type === "chi" || a.type === "daiminkan"
      || a.type === "hora" || a.type === "reach"
      || a.type === "ankan" || a.type === "kakan"
  );

  // 需要玩家手动决策：有实质选择，且自动化规则无法覆盖
  const needsDecision = hasRealChoice && !(
    (autoHora && hasHora) ||
    (noMeld && hasMeld && !hasHora) ||
    (autoTsumogiri && hasDahai && !hasHora)
  );

  return {
    isMyTurn,
    legalActions,
    hasHora,
    hasMeld,
    hasDahai,
    hasNone,
    hasRealChoice,
    needsDecision,
  };
}
