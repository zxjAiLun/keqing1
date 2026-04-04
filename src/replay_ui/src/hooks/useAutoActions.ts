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
  const isMyTurn = Boolean(state?.needs_input);
  const legalActions = isMyTurn ? (state?.legal_actions ?? []) : [];

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
  const onlyNoneResponse = isMyTurn && legalActions.length === 1 && legalActions[0]?.type === "none";

  // 需要玩家手动决策：只要当前存在合法动作且未命中显式自动化，就暂停轮询等待用户点击。
  const needsDecision = isMyTurn && legalActions.length > 0 && !(
    onlyNoneResponse ||
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
    onlyNoneResponse,
    needsDecision,
  };
}
