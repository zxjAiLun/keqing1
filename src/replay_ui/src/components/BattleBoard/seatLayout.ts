import type { MeldEntry } from "../../types/battle";

// ── 自家底部固定区域常量（Commit A 收口）──────────────────────────────
// 最坏组合：4 组最宽副露（daiminkan，south large：66 + 3×48 + 3×4 = 222）≈ 897px，
// 加暗手 1 张 + 摸牌 = 2 张可见（2×48 + 1 + 4 = 101px），再计固定 24px gap → 总 ≥ 1022。
// shell 取 1120，保证 0~4 副露、13/14 摸打时 hand lane 恒 ≥ 手牌内容宽度（不裁切、不重叠）。
export const SELF_SEAT_SHELL_WIDTH_PX = 1120;
export const SELF_SEAT_SIDE_MARGIN = 40;
export const SELF_HAND_MELD_GAP = 24;

export type SeatPosition = "south" | "north" | "east" | "west";
export type LayoutAxis = "row" | "column";
export type RelativeCallSide = "left" | "across" | "right" | "self";

export interface SeatModel {
  position: SeatPosition;
  tileOrientation: 0 | 90 | 180 | 270;
  concealedAxis: LayoutAxis;
  concealedReverse: boolean;
  meldAxis: LayoutAxis;
  meldPlacement: "before" | "after";
  labelPlacement: "top" | "bottom" | "left" | "right";
}

export interface MeldDisplayTile {
  tile: string;
  rotated: boolean;
  stackedOn?: number;
  hidden?: boolean;
}

const SEAT_MODELS: Record<SeatPosition, SeatModel> = {
  south: {
    position: "south",
    tileOrientation: 0,
    concealedAxis: "row",
    concealedReverse: false,
    meldAxis: "row",
    meldPlacement: "after",
    labelPlacement: "bottom",
  },
  north: {
    position: "north",
    tileOrientation: 180,
    concealedAxis: "row",
    concealedReverse: true,
    meldAxis: "row",
    meldPlacement: "before",
    labelPlacement: "top",
  },
  east: {
    position: "east",
    tileOrientation: 270,
    concealedAxis: "column",
    concealedReverse: true,
    meldAxis: "column",
    meldPlacement: "after",
    labelPlacement: "right",
  },
  west: {
    position: "west",
    tileOrientation: 90,
    concealedAxis: "column",
    concealedReverse: false,
    meldAxis: "column",
    meldPlacement: "before",
    labelPlacement: "left",
  },
};

export function getSeatModel(position: SeatPosition): SeatModel {
  return SEAT_MODELS[position];
}

export function getRelativeCallSide(actor: number, target: number | null | undefined): RelativeCallSide {
  if (target == null || target === actor) return "self";
  const delta = (target - actor + 4) % 4;
  if (delta === 1) return "right";
  if (delta === 2) return "across";
  if (delta === 3) return "left";
  return "self";
}

function getClaimedTileIndex(tileCount: number, side: RelativeCallSide): number {
  if (tileCount <= 1) return 0;
  if (side === "left") return 0;
  if (side === "right") return tileCount - 1;
  if (side === "across") return Math.floor((tileCount - 1) / 2);
  return tileCount - 1;
}

function buildCalledMeldTiles(
  actor: number,
  target: number | null | undefined,
  handTiles: string[],
  calledTile: string,
): MeldDisplayTile[] {
  const claimedIndex = getClaimedTileIndex(handTiles.length + 1, getRelativeCallSide(actor, target));
  const displayTiles: MeldDisplayTile[] = [];
  let handIndex = 0;
  for (let idx = 0; idx < handTiles.length + 1; idx++) {
    if (idx === claimedIndex) {
      displayTiles.push({ tile: calledTile, rotated: true });
    } else {
      displayTiles.push({ tile: handTiles[handIndex], rotated: false });
      handIndex += 1;
    }
  }
  return displayTiles;
}

export function validateMeldTiles(meld: MeldEntry): boolean {
  // 空牌/空字符串禁止渲染成空白牌框
  if (meld.consumed.some((tile) => !tile)) {
    if (typeof console !== "undefined") console.error("Malformed meld (empty consumed tile)", meld);
    return false;
  }
  let ok = true;
  switch (meld.type) {
    case "chi":
    case "pon":
      ok = meld.consumed.length === 2 && Boolean(meld.pai);
      break;
    case "daiminkan":
      ok = meld.consumed.length === 3 && Boolean(meld.pai);
      break;
    case "ankan":
      ok = meld.consumed.length === 4;
      break;
    case "kakan":
      ok = meld.consumed.length === 4;
      break;
    default:
      ok = false;
  }
  if (!ok && typeof console !== "undefined") console.error("Malformed meld (tile count)", meld);
  return ok;
}

export function buildMeldDisplayTiles(actor: number, meld: MeldEntry): MeldDisplayTile[] {
  // 校验失败即拒绝渲染：不产生空白牌框或 undefined 牌
  if (!validateMeldTiles(meld)) return [];
  if (meld.type === "kakan") {
    // kakan consumed = [手牌1, 手牌2, 被鸣, 加杠]：前 3 张为原 pon，第 4 张叠在被鸣牌上
    const baseHandTiles = meld.consumed.slice(0, 2);
    const calledTile = meld.consumed[2] ?? meld.pai;
    const addedTile = meld.consumed[3];
    const baseDisplayTiles = buildCalledMeldTiles(actor, meld.target, baseHandTiles, calledTile);
    const claimedIndex = baseDisplayTiles.findIndex(tile => tile.rotated);
    return [...baseDisplayTiles, { tile: addedTile, rotated: false, stackedOn: claimedIndex }];
  }

  if (meld.type === "ankan") {
    const tiles = meld.consumed.slice(0, 4);
    return tiles.map((tile, idx) => ({
      tile,
      rotated: false,
      hidden: idx === 0 || idx === tiles.length - 1,
    }));
  }

  return buildCalledMeldTiles(actor, meld.target, [...meld.consumed], meld.pai);
}

/** 自家手牌区实际渲染宽度（含摸牌）。与 MahjongTable 的 flex 布局保持一致。 */
export function computeSelfHandWidth(
  handCount: number,
  hasTsumoPai: boolean,
  tileWidth: number,
  tileGap: number,
  drawGap: number,
): number {
  const tileCount = handCount + (hasTsumoPai ? 1 : 0);
  // flex 子元素数 = tileCount，普通 gap 数 = tileCount - 1；��牌另加 drawGap margin
  const normalGapCount =
    Math.max(0, handCount - 1)
    + (hasTsumoPai && handCount > 0 ? 1 : 0);
  return tileCount * tileWidth + normalGapCount * tileGap + (hasTsumoPai ? drawGap : 0);
}

/**
 * 自家底部固定区域的确定性几何（Commit A 收口）。
 *
 * 布局契约：
 *   - shell 拥有确定宽度（SELF_SEAT_SHELL_WIDTH_PX），右吸附于牌桌右侧；
 *   - SelfHandLane 宽度 = shell − meldLaneWidth − handMeldGap（可计算，非随暗手伸缩）；
 *   - 手牌与副露之间 gap 恒定 = handMeldGap；
 *   - SelfMeldLane 右吸附 → meldRight = sideMargin + shellWidth，对任意 0~4 副露不变。
 *
 * 手牌内容（left-aligned）恒小于 lane 宽度：手牌张数与副露数反比
 * （14 − 3×meldCount），因此 0~4 副露、13/14 张摸打都不裁切。
 */
export interface SelfSeatGeometry {
  shellWidth: number;
  sideMargin: number;
  handMeldGap: number;
  handLaneWidth: number;
  meldLaneWidth: number;
  handLeft: number;
  handRight: number;
  meldLeft: number;
  meldRight: number;
}

export function computeSelfSeatGeometry(params: {
  shellWidth: number;
  sideMargin: number;
  handMeldGap: number;
  meldLaneWidth: number;
}): SelfSeatGeometry {
  const { shellWidth, sideMargin, handMeldGap, meldLaneWidth } = params;
  const handLaneWidth = Math.max(shellWidth - meldLaneWidth - handMeldGap, 0);
  const handLeft = sideMargin;
  const handRight = handLeft + handLaneWidth;
  const meldLeft = handRight + handMeldGap;
  const meldRight = meldLeft + meldLaneWidth;
  return {
    shellWidth,
    sideMargin,
    handMeldGap,
    handLaneWidth,
    meldLaneWidth,
    handLeft,
    handRight,
    meldLeft,
    meldRight,
  };
}
