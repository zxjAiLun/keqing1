// src/replay_ui/src/components/BattleBoard/MahjongTable.tsx
import { useState, useCallback, useMemo, useEffect, useRef } from "react";
import { Tile, TileBack, TILE_SIZES } from "./Tile";
import { ActionBar } from "./ActionBar";
import type { BattleState, Action, DiscardEntry, MeldEntry } from "../../types/battle";
import type { LogitTileData } from "../../utils/replayAdapter";
import { BAKAZE_CN, JIKAZE_CN } from "../../utils/constants";
import { sortHand } from "../../utils/tileUtils";
import { buildMeldDisplayTiles, getSeatModel, type SeatPosition } from "./seatLayout";

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------
const SEAT_COLORS = ["var(--seat-0)", "var(--seat-1)", "var(--seat-2)", "var(--seat-3)"];

const DISC_COLS = 6;

// ---------------------------------------------------------------------------
// 工具
// ---------------------------------------------------------------------------
function chunkDiscards(discards: DiscardEntry[], cols: number): DiscardEntry[][] {
  const rows: DiscardEntry[][] = [];
  for (let i = 0; i < discards.length; i += cols)
    rows.push(discards.slice(i, i + cols));
  return rows;
}



// ---------------------------------------------------------------------------
// 弃牌堆 tile 尺寸
// ---------------------------------------------------------------------------
const DISC_W = TILE_SIZES.normal.w;
const DISC_GAP = 2; // 间距
const BASE_TABLE_WIDTH = 1280;
const BASE_TABLE_HEIGHT = 900;
const SELF_HAND_GAP = 2;
const SELF_HAND_DRAW_GAP = 6;
const SELF_HAND_RESERVED_WIDTH = TILE_SIZES.large.w * 14 + SELF_HAND_GAP * 12 + SELF_HAND_DRAW_GAP;
// 6张一列宽度（用于中心正方形边长）
export const CENTER_SIZE = DISC_COLS * DISC_W + (DISC_COLS - 1) * DISC_GAP; // 142px

function normalizeOrientation(orientation: number): 0 | 90 | 180 | 270 {
  const normalized = ((orientation % 360) + 360) % 360;
  if (normalized === 90 || normalized === 180 || normalized === 270) return normalized;
  return 0;
}

function getTileBox(size: "small" | "normal" | "large", orientation: 0 | 90 | 180 | 270) {
  const dim = TILE_SIZES[size];
  const sideways = orientation === 90 || orientation === 270;
  return { width: sideways ? dim.h : dim.w, height: sideways ? dim.w : dim.h };
}

function OrientedTile({
  tile,
  size,
  orientation,
  dimmed,
  outlined,
  outlineColor,
}: {
  tile: string;
  size: "small" | "normal" | "large";
  orientation: 0 | 90 | 180 | 270;
  dimmed?: boolean;
  outlined?: boolean;
  outlineColor?: string;
}) {
  const { width, height } = getTileBox(size, orientation);
  return (
    <div style={{ width, height, position: "relative", flexShrink: 0 }}>
      <div
        style={{
          position: "absolute",
          top: "50%",
          left: "50%",
          transform: `translate(-50%, -50%) rotate(${orientation}deg)`,
          filter: dimmed ? "brightness(0.6)" : undefined,
          outline: outlined ? `2px solid ${outlineColor ?? "var(--gold)"}` : undefined,
          outlineOffset: "1px",
          borderRadius: outlined ? 3 : undefined,
        }}
      >
        <Tile tile={tile} size={size} />
      </div>
    </div>
  );
}

function DiscardTile({ d, position, zIndex }: { d: DiscardEntry; position: SeatPosition; zIndex?: number }) {
  const modelOrientation = getSeatModel(position).tileOrientation;
  const baseOrientation =
    position === "east" ? 90
    : position === "west" ? 270
    : modelOrientation;
  const orientation = normalizeOrientation(baseOrientation + (d.reach_declared ? 90 : 0));
  return (
    <div style={{ position: "relative", zIndex }}>
      <OrientedTile
        tile={d.pai}
        size="normal"
        orientation={orientation}
        dimmed={d.tsumogiri}
        outlined={d.reach_declared}
      />
    </div>
  );
}

// 自家（底部）：左下角起，行左→右，新行向下
function DiscardPondSouth({ discards }: { discards: DiscardEntry[] }) {
  const rows = chunkDiscards(discards, DISC_COLS);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: DISC_GAP, alignItems: "flex-start" }}>
      {rows.map((row, ri) => (
        <div key={ri} style={{ display: "flex", flexDirection: "row", gap: DISC_GAP }}>
          {row.map((d, di) => {
            const globalIndex = ri * DISC_COLS + di;
            return (
              <DiscardTile
                key={`${globalIndex}-${d.pai}-${d.tsumogiri ? "t" : "d"}-${d.reach_declared ? "r" : "n"}`}
                d={d}
                position="south"
              />
            );
          })}
        </div>
      ))}
    </div>
  );
}

// 对家（顶部）：右上角起，行右→左，新行向上（column-reverse）
function DiscardPondNorth({ discards }: { discards: DiscardEntry[] }) {
  const rows = chunkDiscards(discards, DISC_COLS);
  return (
    <div style={{ display: "flex", flexDirection: "column-reverse", gap: DISC_GAP, alignItems: "flex-end" }}>
      {rows.map((row, ri) => (
        <div key={ri} style={{ display: "flex", flexDirection: "row-reverse", gap: DISC_GAP }}>
          {row.map((d, di) => {
            const globalIndex = ri * DISC_COLS + di;
            return (
              <DiscardTile
                key={`${globalIndex}-${d.pai}-${d.tsumogiri ? "t" : "d"}-${d.reach_declared ? "r" : "n"}`}
                d={d}
                position="north"
              />
            );
          })}
        </div>
      ))}
    </div>
  );
}

// 左家（上家）：牌旋转90°，列从上→下，新列向左
function DiscardPondLeft({ discards }: { discards: DiscardEntry[] }) {
  const cols = chunkDiscards(discards, DISC_COLS);
  return (
    <div style={{ display: "flex", flexDirection: "row-reverse", gap: DISC_GAP, alignItems: "flex-start" }}>
      {cols.map((col, ci) => (
        <div key={ci} style={{ display: "flex", flexDirection: "column", gap: DISC_GAP, position: "relative", zIndex: cols.length - ci }}>
          {col.map((d, di) => {
            const globalIndex = ci * DISC_COLS + di;
            return (
              <DiscardTile
                key={`${globalIndex}-${d.pai}-${d.tsumogiri ? "t" : "d"}-${d.reach_declared ? "r" : "n"}`}
                d={d}
                position="east"
                zIndex={di + 1}
              />
            );
          })}
        </div>
      ))}
    </div>
  );
}

// 右家（下家）：牌旋转90°+180°=270°，列从下→上，新列向右
function DiscardPondRight({ discards }: { discards: DiscardEntry[] }) {
  const cols = chunkDiscards(discards, DISC_COLS);
  return (
    <div style={{ display: "flex", flexDirection: "row", gap: DISC_GAP, alignItems: "flex-end" }}>
      {cols.map((col, ci) => (
        <div key={ci} style={{ display: "flex", flexDirection: "column-reverse", gap: DISC_GAP, position: "relative", zIndex: cols.length - ci }}>
          {col.map((d, di) => {
            const globalIndex = ci * DISC_COLS + di;
            return (
              <DiscardTile
                key={`${globalIndex}-${d.pai}-${d.tsumogiri ? "t" : "d"}-${d.reach_declared ? "r" : "n"}`}
                d={d}
                position="west"
                zIndex={di + 1}
              />
            );
          })}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 单家玩家区
// ---------------------------------------------------------------------------
function getFlexDirection(axis: "row" | "column", reverse: boolean) {
  if (axis === "row") return reverse ? "row-reverse" : "row";
  return reverse ? "column-reverse" : "column";
}

function getConcealedCountForSeat(melds: MeldEntry[]): number {
  return Math.max(0, 13 - melds.length * 3);
}

function hasPendingExtraTile(state: BattleState, pid: number): boolean {
  if (state.replay_draw_actor === pid) return true;
  return state.actor_to_move === pid && !state.last_discard;
}

type DiscardHole = number | "draw" | null;

function deterministicHoleIndex(pid: number, handLength: number, discards: DiscardEntry[]): number {
  const last = discards[discards.length - 1];
  const seedText = `${pid}:${handLength}:${discards.length}:${last?.pai ?? ""}:${last?.tsumogiri ? 1 : 0}`;
  let hash = 0;
  for (let i = 0; i < seedText.length; i++) {
    hash = (hash * 31 + seedText.charCodeAt(i)) >>> 0;
  }
  return handLength > 0 ? hash % (handLength + 1) : 0;
}

function getDisplayedOpponentTiles(position: SeatPosition, hand: string[]): string[] {
  const tiles = sortHand([...hand], null);
  return position === "east" || position === "west" ? tiles.reverse() : tiles;
}

function getOpponentDiscardHole(
  position: SeatPosition,
  hand: string[],
  discards: DiscardEntry[],
  lastDiscard: BattleState["last_discard"],
  pid: number,
  fallbackLength: number,
): DiscardHole {
  if (!lastDiscard || lastDiscard.actor !== pid) return null;
  const last = discards[discards.length - 1];
  if (!last || last.pai !== lastDiscard.pai) return null;
  if (last.tsumogiri) return "draw";
  if (hand.length === 0) return deterministicHoleIndex(pid, fallbackLength, discards);

  const displayed = getDisplayedOpponentTiles(position, hand);
  const withDiscard = getDisplayedOpponentTiles(position, [...hand, lastDiscard.pai]);
  const preferredIndex = withDiscard.indexOf(lastDiscard.pai);
  if (preferredIndex >= 0) return preferredIndex;
  return deterministicHoleIndex(pid, displayed.length, discards);
}

function MeldBlock({ pid, meld, position }: { pid: number; meld: MeldEntry; position: SeatPosition }) {
  const model = getSeatModel(position);
  const meldOrientation: 0 | 90 | 180 | 270 =
    position === "east" ? 90
    : position === "west" ? 270
    : model.tileOrientation;
  const displayTiles = buildMeldDisplayTiles(pid, meld);
  const flowDirection = getFlexDirection(model.meldAxis, false);
  const stackOffset =
    position === "south" ? "translate(-3px, -7px)"
    : position === "north" ? "translate(3px, 7px)"
    : position === "east" ? "translate(7px, 3px)"
    : "translate(-7px, -3px)";
  return (
    <div style={{ display: "flex", flexDirection: flowDirection, gap: 4 }}>
      {displayTiles.map((entry, idx) => {
        const orientation = normalizeOrientation(meldOrientation + (entry.rotated ? 90 : 0));
        const stackedTile = displayTiles.find((candidate) => candidate.stackedOn === idx);
        const { width, height } = getTileBox("small", orientation);
        return (
          <div key={`${entry.tile}-${idx}`} style={{ width, height, position: "relative", flexShrink: 0 }}>
            {entry.hidden ? (
              <TileBack size="normal" orientation={meldOrientation} />
            ) : (
              <OrientedTile tile={entry.tile} size="normal" orientation={orientation} />
            )}
            {stackedTile && (
              <div style={{ position: "absolute", inset: 0, transform: stackOffset, pointerEvents: "none" }}>
                <OrientedTile tile={stackedTile.tile} size="normal" orientation={meldOrientation} />
              </div>
            )}
          </div>
        );
      }).filter((_, idx) => displayTiles[idx].stackedOn === undefined)}
    </div>
  );
}

function MeldArea({ pid, melds, position }: { pid: number; melds: MeldEntry[]; position: SeatPosition }) {
  if (melds.length === 0) return null;
  const model = getSeatModel(position);
  const flowDirection = getFlexDirection(model.meldAxis, model.meldPlacement === "before");
  return (
    <div style={{ display: "flex", flexDirection: flowDirection, gap: 10, flexShrink: 0 }}>
      {melds.map((meld, idx) => (
        <MeldBlock key={`${meld.type}-${meld.pai}-${idx}`} pid={pid} meld={meld} position={position} />
      ))}
    </div>
  );
}

function ConcealedHand({
  position,
  count = 13,
  showDrawTile = false,
  discardHole = null,
  onClick,
}: {
  position: SeatPosition;
  count?: number;
  showDrawTile?: boolean;
  discardHole?: DiscardHole;
  onClick?: () => void;
}) {
  const model = getSeatModel(position);
  const concealedOrientation: 0 | 90 | 180 | 270 =
    position === "east" ? 90
    : position === "west" ? 270
    : model.tileOrientation;
  const concealedAxis = model.concealedAxis;
  const concealedReverse = model.concealedReverse;
  const flowDirection = getFlexDirection(concealedAxis, concealedReverse);
  const gap = position === "north" ? 1 : 2;
  const drawGap = 12;
  const { width, height } = getTileBox("normal", concealedOrientation);
  const drawGapStyle =
    concealedAxis === "row"
      ? { [concealedReverse ? "marginRight" : "marginLeft"]: drawGap }
      : { [concealedReverse ? "marginBottom" : "marginTop"]: drawGap };
  const reservedMainSpan = count * (concealedAxis === "row" ? width : height) + Math.max(count - 1, 0) * gap;
  const reservedCrossSpan = concealedAxis === "row" ? height : width;
  const reservedTotalSpan = reservedMainSpan + drawGap + (concealedAxis === "row" ? width : height);
  const frameStyle =
    concealedAxis === "row"
      ? { width: reservedTotalSpan, height: reservedCrossSpan }
      : { width: reservedCrossSpan, height: reservedTotalSpan };
  return (
    <div
      style={{
        display: "flex",
        justifyContent: concealedReverse ? "flex-end" : "flex-start",
        alignItems: concealedAxis === "row" ? "center" : concealedReverse ? "flex-end" : "flex-start",
        padding:
          position === "north" ? "2px 8px 6px"
          : position === "east" ? "8px 2px 6px 8px"
          : position === "west" ? "8px 8px 6px 2px"
          : "2px 0 4px",
        cursor: onClick ? "pointer" : undefined,
      }}
      onClick={onClick}
    >
      <div style={{ ...frameStyle, position: "relative", flexShrink: 0 }}>
        <div
          style={{
            display: "flex",
            flexDirection: flowDirection,
            gap,
            width: "100%",
            height: "100%",
            justifyContent: concealedReverse ? "flex-end" : "flex-start",
            alignItems: concealedAxis === "row" ? "center" : concealedReverse ? "flex-end" : "flex-start",
          }}
        >
          {discardHole !== "draw" && Array.from({ length: count + (discardHole !== null ? 1 : 0) }, (_, idx) => {
            const isHole = discardHole !== null && idx === discardHole;
            const tileZIndex = concealedReverse ? idx + 1 : count - idx;
            return (
              <div key={idx} style={{ width, height, flexShrink: 0, position: "relative", zIndex: tileZIndex, opacity: isHole ? 0 : 1 }}>
                {!isHole && <TileBack size="normal" orientation={concealedOrientation} />}
              </div>
            );
          })}
          {discardHole === "draw" && Array.from({ length: count }, (_, idx) => {
            const tileZIndex = concealedReverse ? idx + 1 : count - idx;
            return (
              <div key={idx} style={{ width, height, flexShrink: 0, position: "relative", zIndex: tileZIndex }}>
                <TileBack size="normal" orientation={concealedOrientation} />
              </div>
            );
          })}
          {showDrawTile && (
            <div style={{ ...drawGapStyle, position: "relative", flexShrink: 0 }}>
              <div style={{ filter: "drop-shadow(0 0 8px rgba(212,168,83,0.22))" }}>
                <TileBack size="normal" orientation={concealedOrientation} />
              </div>
            </div>
          )}
          {discardHole === "draw" && !showDrawTile && (
            <div style={{ ...drawGapStyle, width, height, flexShrink: 0 }} />
          )}
        </div>
      </div>
    </div>
  );
}

function RevealedOpponentHand({
  position,
  hand,
  showDrawTile = false,
  discardHole = null,
  onClick,
}: {
  position: SeatPosition;
  hand: string[];
  showDrawTile?: boolean;
  discardHole?: DiscardHole;
  onClick?: () => void;
}) {
  const model = getSeatModel(position);
  const revealedOrientation: 0 | 90 | 180 | 270 =
    position === "east" ? 90
    : position === "west" ? 270
    : model.tileOrientation;
  const sortedTiles = getDisplayedOpponentTiles(position, hand);
  const drawTile = showDrawTile && hand.length > 0 ? hand[hand.length - 1] : null;
  const mainTiles = showDrawTile ? getDisplayedOpponentTiles(position, hand.slice(0, -1)) : sortedTiles;
  const revealedAxis = model.concealedAxis;
  const revealedReverse = model.concealedReverse;
  const flowDirection = getFlexDirection(revealedAxis, revealedReverse);
  const gap = 2;
  const drawGap = 12;
  const { width, height } = getTileBox("normal", revealedOrientation);
  const drawGapStyle =
    revealedAxis === "row"
      ? { [revealedReverse ? "marginRight" : "marginLeft"]: drawGap }
      : { [revealedReverse ? "marginBottom" : "marginTop"]: drawGap };
  return (
    <div
      onClick={onClick}
      style={{
        display: "flex",
        flexDirection: flowDirection,
        gap,
        padding:
          position === "north" ? "2px 8px 6px"
          : position === "east" ? "8px 2px 6px 8px"
          : position === "west" ? "8px 8px 6px 2px"
          : "2px 0 4px",
        cursor: onClick ? "pointer" : undefined,
      }}
    >
      {discardHole !== "draw" && Array.from({ length: mainTiles.length + (discardHole !== null ? 1 : 0) }, (_, idx) => {
        const isHole = discardHole !== null && idx === discardHole;
        const tile = isHole ? null : mainTiles[idx - (discardHole !== null && idx > discardHole ? 1 : 0)];
        return (
          <div key={`${tile ?? "hole"}-${idx}`} style={{ width, height, flexShrink: 0 }}>
            {tile && <OrientedTile tile={tile} size="normal" orientation={revealedOrientation} />}
          </div>
        );
      })}
      {discardHole === "draw" && mainTiles.map((tile, idx) => (
        <OrientedTile
          key={`${tile}-${idx}`}
          tile={tile}
          size="normal"
          orientation={revealedOrientation}
        />
      ))}
      {showDrawTile && drawTile && (
        <div style={{ ...drawGapStyle, flexShrink: 0 }}>
          <OrientedTile tile={drawTile} size="normal" orientation={revealedOrientation} />
        </div>
      )}
      {discardHole === "draw" && !showDrawTile && (
        <div style={{ ...drawGapStyle, width, height, flexShrink: 0 }} />
      )}
    </div>
  );
}

function PlayerZone({
  pid,
  position,
  playerName,
  hand,
  tsumoPai,
  // discards: 弃牌已移至中央 DiscardPond，这里不再使用
  // eslint-disable-next-line @typescript-eslint/no-unused-vars
  discards: _discards,
  melds,
  reached,
  isActive,
  highlightIdx,
  onTileClick,
  logitData,
  compact,
  showReplayDrawTile,
  concealedCount,
  revealedHand,
  discardHole,
  onOpponentHandToggle,
}: {
  pid: number;
  position: "south" | "north" | "east" | "west";
  playerName: string;
  hand: string[];
  tsumoPai?: string | null;
  discards: DiscardEntry[];
  melds: MeldEntry[];
  reached: boolean;
  isActive: boolean;
  isHuman?: boolean;
  highlightTile?: string | null;
  highlightIdx?: number | null;
  onTileClick?: (tile: string, idx: number) => void;
  logitData?: LogitTileData[];
  compact?: boolean;
  showReplayDrawTile?: boolean;
  concealedCount?: number;
  revealedHand?: string[] | null;
  discardHole?: DiscardHole;
  onOpponentHandToggle?: () => void;
}) {
  const color   = SEAT_COLORS[pid];
  const jikaze = JIKAZE_CN[pid];
  const model = getSeatModel(position);

  // ── 自家（south）──
  if (position === "south") {
    // 去掉 tsumoPai 后的手牌，用 useMemo 缓存排序避免摸牌时重排闪烁
    // eslint-disable-next-line react-hooks/rules-of-hooks
    const sortedHand = useMemo(() => {
      const arr = [...hand];
      if (tsumoPai) {
        const idx = arr.lastIndexOf(tsumoPai);
        if (idx >= 0) arr.splice(idx, 1);
      }
      return sortHand(arr, null);
    // 只依赖 hand 的内容（不含 tsumo），tsumo 变化不触发重排
    // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [hand.filter(t => t !== tsumoPai).join(",")]);

    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 5 }}>

        <div style={{ display: "flex", flexDirection: compact ? "column" : "row", alignItems: compact ? "center" : "flex-end", gap: compact ? 8 : 12 }}>
          <div style={{ position: "relative", width: SELF_HAND_RESERVED_WIDTH }}>
          {/* 柱状图层（回放模式，绝对定位在手牌上方） */}
          {logitData && logitData.length > 0 && (
            <div style={{
              position: "absolute", bottom: "100%", left: 0,
              display: "flex", gap: 2, paddingBottom: 3,
              pointerEvents: "none", alignItems: "flex-end",
              width: SELF_HAND_RESERVED_WIDTH,
            }}>
              {[...sortedHand, ...(tsumoPai ? [tsumoPai] : [])].map((tile, i) => {
                const d = logitData.find(x => x.pai === tile && x.isTsumo === (tsumoPai ? i === sortedHand.length : false))
                  ?? logitData.find(x => x.pai === tile);
                const h = d ? Math.max(3, Math.round(d.pct / 100 * 48)) : 2;
                const barColor = d?.isChosen && d?.isGt ? "#8e44ad"
                  : d?.isChosen ? "#e74c3c"
                  : d?.isGt ? "#27ae60"
                  : d?.score !== undefined ? "var(--accent)"
                  : "rgba(150,150,150,0.25)";
                return (
                  <div key={`bar-${tile}-${i}`} style={{
                    width: TILE_SIZES.large.w,
                    marginLeft: i === sortedHand.length && tsumoPai ? SELF_HAND_DRAW_GAP : 0,
                    display: "flex", alignItems: "flex-end", justifyContent: "center", flexShrink: 0,
                  }}>
                    <div style={{
                      width: 24, height: h,
                      background: barColor,
                      borderRadius: "2px 2px 0 0",
                      transition: "height 0.15s ease",
                    }} />
                  </div>
                );
              })}
            </div>
          )}
          {/* 手牌 */}
          <div style={{ display: "flex", gap: SELF_HAND_GAP, flexWrap: "nowrap", transform: "translateY(-2px)", width: SELF_HAND_RESERVED_WIDTH }}>
            {sortedHand.map((tile, i) => (
              <div key={`${tile}-${i}`} style={{ width: TILE_SIZES.large.w, height: TILE_SIZES.large.h, flexShrink: 0 }}>
                <Tile tile={tile} size="large" selected={i === highlightIdx} onClick={() => onTileClick?.(tile, i)} />
              </div>
            ))}
            {tsumoPai && (
              <div
                style={{ width: TILE_SIZES.large.w, height: TILE_SIZES.large.h, outline: "2px solid var(--gold)", outlineOffset: "1px", borderRadius: 4, boxShadow: "0 0 10px rgba(212,168,83,0.45)", flexShrink: 0, cursor: onTileClick ? "pointer" : undefined, marginLeft: SELF_HAND_DRAW_GAP }}
                onClick={() => onTileClick?.(tsumoPai, sortedHand.length)}
              >
                <Tile tile={tsumoPai} size="large" selected={sortedHand.length === highlightIdx} />
              </div>
            )}
          </div>
          </div>
          <div style={{ alignSelf: compact ? "flex-end" : undefined }}>
            <MeldArea pid={pid} melds={melds} position={position} />
          </div>
        </div>

        {/* 舍牌已移至中央弃牌堆 */}

        {/* 玩家标签 */}
        <div style={{ display: "flex", alignItems: "center", gap: 4, marginTop: 2 }}>
          {isActive && <div style={{
            width: 8, height: 8, borderRadius: "50%", background: color,
            animation: "seatPulse 1.5s ease-in-out infinite",
          }} />}
          <span style={{ fontWeight: 700, fontSize: 12, color }}>{playerName}</span>
          <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{jikaze}</span>
          {reached && (
            <span style={{
              fontSize: 9, padding: "1px 4px", borderRadius: 3,
              background: "var(--gold-bg)", color: "var(--gold)",
              border: "1px solid var(--gold-border)", fontWeight: 700,
              animation: "riichiPulse 2s ease-in-out infinite",
            }}>立</span>
          )}
        </div>
      </div>
    );
  }

  // ── 对面（north）── 整体旋转180deg
  if (position === "north") {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, paddingTop: 4 }}>
        <PlayerLabel color={color} playerName={playerName} jikaze={jikaze} reached={reached} isActive={isActive} />
        <div style={{ display: "flex", flexDirection: "row", alignItems: "flex-end", gap: 12 }}>
          {model.meldPlacement === "before" && <MeldArea pid={pid} melds={melds} position={position} />}
          {revealedHand ? (
            <RevealedOpponentHand position={position} hand={revealedHand} showDrawTile={showReplayDrawTile} discardHole={discardHole} onClick={onOpponentHandToggle} />
          ) : (
            <ConcealedHand
              position={position}
              count={concealedCount}
              showDrawTile={showReplayDrawTile}
              discardHole={discardHole}
              onClick={onOpponentHandToggle}
            />
          )}
          {model.meldPlacement === "after" && <MeldArea pid={pid} melds={melds} position={position} />}
        </div>
      </div>
    );
  }

  // ── 左家（east）── 竖列暗牌，副露贴右手边
  if (position === "east") {
    return (
      <div style={{ display: "flex", flexDirection: "row", alignItems: "flex-start", gap: 8, paddingLeft: 4 }}>
        <div style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "center" }}>
          {revealedHand ? (
            <RevealedOpponentHand position={position} hand={revealedHand} showDrawTile={showReplayDrawTile} discardHole={discardHole} onClick={onOpponentHandToggle} />
          ) : (
            <ConcealedHand
              position={position}
              count={concealedCount}
              showDrawTile={showReplayDrawTile}
              discardHole={discardHole}
              onClick={onOpponentHandToggle}
            />
          )}
          <MeldArea pid={pid} melds={melds} position={position} />
        </div>
        <div style={{ paddingTop: 4 }}>
          <PlayerLabel color={color} playerName={playerName} jikaze={jikaze} reached={reached} isActive={isActive} />
        </div>
      </div>
    );
  }

  // ── 右家（west）── 竖列暗牌，副露贴右手边
  return (
    <div style={{ display: "flex", flexDirection: "row", alignItems: "flex-start", gap: 8, paddingRight: 4 }}>
      <div style={{ paddingTop: 4 }}>
        <PlayerLabel color={color} playerName={playerName} jikaze={jikaze} reached={reached} isActive={isActive} />
      </div>
      <div style={{ display: "flex", flexDirection: "column", gap: 4, alignItems: "center" }}>
        <MeldArea pid={pid} melds={melds} position={position} />
        {revealedHand ? (
          <RevealedOpponentHand position={position} hand={revealedHand} showDrawTile={showReplayDrawTile} discardHole={discardHole} onClick={onOpponentHandToggle} />
        ) : (
          <ConcealedHand
            position={position}
            count={concealedCount}
            showDrawTile={showReplayDrawTile}
            discardHole={discardHole}
            onClick={onOpponentHandToggle}
          />
        )}
      </div>
    </div>
  );
}

// 玩家标签子组件
function PlayerLabel({ color, playerName, jikaze, reached, isActive }: {
  color: string; playerName: string; jikaze: string; reached: boolean; isActive: boolean;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
      {isActive && <div style={{ width: 7, height: 7, borderRadius: "50%", background: color, animation: "seatPulse 1.5s ease-in-out infinite" }} />}
      <span style={{ fontWeight: 700, fontSize: 11, color }}>{playerName}</span>
      <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{jikaze}</span>
      {reached && <span style={{ fontSize: 9, padding: "1px 4px", borderRadius: 3, background: "var(--gold-bg)", color: "var(--gold)", border: "1px solid var(--gold-border)", fontWeight: 700 }}>立</span>}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 中心面板
// ---------------------------------------------------------------------------
function formatScoreDelta(delta: number): string {
  if (delta === 0) return "±0";
  return delta > 0 ? `+${delta.toLocaleString()}` : delta.toLocaleString();
}

function ScoreMarker({
  value,
  color,
  isDealer = false,
  style,
}: {
  value: string;
  color: string;
  isDealer?: boolean;
  style: React.CSSProperties;
}) {
  return (
    <div
      style={{
        position: "absolute",
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "3px 6px",
        borderRadius: 8,
        background: "rgba(255,255,255,0.88)",
        border: "1px solid rgba(32, 41, 58, 0.08)",
        boxShadow: "0 2px 8px rgba(20, 28, 45, 0.08)",
        backdropFilter: "blur(10px)",
        pointerEvents: "none",
        ...style,
      }}
    >
      {isDealer && (
        <span style={{
          fontSize: 9,
          fontWeight: 800,
          color: "var(--gold)",
          padding: "1px 4px",
          borderRadius: 999,
          border: "1px solid var(--gold-border)",
          background: "var(--gold-bg)",
          lineHeight: 1.1,
        }}>
          庄
        </span>
      )}
      <span style={{ fontSize: 11, fontWeight: 700, color, fontFamily: "Menlo, monospace" }}>
        {value}
      </span>
    </div>
  );
}

function RiichiStickCluster({ count }: { count: number }) {
  return (
    <div style={{ display: "flex", gap: 3, alignItems: "center" }}>
      {Array.from({ length: Math.min(count, 4) }, (_, idx) => (
        <div
          key={idx}
          style={{
            width: 18,
            height: 4,
            borderRadius: 999,
            background: "linear-gradient(90deg, #fff7ea 0%, #f4e0ba 100%)",
            border: "1px solid rgba(138, 96, 28, 0.35)",
            boxShadow: "0 1px 2px rgba(0,0,0,0.12)",
          }}
        />
      ))}
      <span style={{ fontSize: 10, fontWeight: 700, color: "var(--text-muted)", fontFamily: "Menlo, monospace" }}>
        x{count}
      </span>
    </div>
  );
}

function BoardInfoCorner({
  dora_markers,
  honba,
  kyotaku,
}: {
  dora_markers: string[];
  honba: number;
  kyotaku: number;
}) {
  return (
    <div
      style={{
        position: "absolute",
        top: 18,
        left: 18,
        zIndex: 24,
        display: "grid",
        gap: 8,
        padding: "10px 12px",
        borderRadius: 14,
        background: "rgba(255,255,255,0.84)",
        border: "1px solid rgba(32,41,58,0.08)",
        boxShadow: "0 10px 24px rgba(15,23,42,0.12)",
        backdropFilter: "blur(10px)",
      }}
    >
      <div style={{ display: "flex", gap: 4 }}>
        {Array.from({ length: 5 }, (_, i) => (
          <div
            key={i}
            style={{
              width: TILE_SIZES.small.w,
              height: TILE_SIZES.small.h,
              borderRadius: 4,
              background: i < dora_markers.length ? "transparent" : "rgba(255,255,255,0.42)",
              border: i < dora_markers.length ? "none" : "1px dashed rgba(55, 73, 96, 0.24)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {i < dora_markers.length && <Tile tile={dora_markers[i]} size="small" />}
          </div>
        ))}
      </div>
      <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
        <span style={{
          fontSize: 10,
          fontWeight: 700,
          color: "var(--gold)",
          padding: "1px 5px",
          borderRadius: 999,
          background: "var(--gold-bg)",
          border: "1px solid var(--gold-border)",
          fontFamily: "Menlo, monospace",
        }}>
          本 {honba}
        </span>
        <RiichiStickCluster count={kyotaku} />
      </div>
    </div>
  );
}

function CenterPanel({
  bakaze,
  scores,
  oya,
  humanId,
  topPid,
  leftPid,
  rightPid,
  showScoreDiff,
  onToggleScoreMode,
}: {
  bakaze: string;
  scores: number[];
  oya: number;
  humanId: number;
  topPid: number;
  leftPid: number;
  rightPid: number;
  showScoreDiff: boolean;
  onToggleScoreMode: () => void;
}) {
  const baseScore = scores[humanId] ?? 0;
  const scoreText = (pid: number) =>
    showScoreDiff
      ? formatScoreDelta((scores[pid] ?? 0) - baseScore)
      : (scores[pid] ?? 0).toLocaleString();

  return (
    <div
      onClick={onToggleScoreMode}
      title={showScoreDiff ? "点击切换为四家点数" : "点击切换为与当前主视角的分差"}
      style={{
        position: "absolute",
        top: "50%", left: "50%",
        transform: "translate(-50%, -50%)",
        width: CENTER_SIZE, height: CENTER_SIZE,
        background: "var(--card-bg)",
        border: "2px solid var(--table-border)",
        borderRadius: 12,
        boxShadow: "0 4px 20px rgba(0,0,0,0.12), inset 0 1px 0 rgba(255,255,255,0.8)",
        display: "flex", flexDirection: "column",
        alignItems: "center", justifyContent: "center",
        gap: 4, zIndex: 20,
        backdropFilter: "blur(12px)",
        cursor: "pointer",
        userSelect: "none",
      }}
    >
      <ScoreMarker
        value={scoreText(topPid)}
        color={SEAT_COLORS[topPid]}
        isDealer={topPid === oya}
        style={{ top: 6, left: "50%", transform: "translateX(-50%)" }}
      />
      <ScoreMarker
        value={scoreText(leftPid)}
        color={SEAT_COLORS[leftPid]}
        isDealer={leftPid === oya}
        style={{ left: 4, top: "50%", transform: "translateY(-50%) rotate(90deg)", transformOrigin: "left center" }}
      />
      <ScoreMarker
        value={scoreText(rightPid)}
        color={SEAT_COLORS[rightPid]}
        isDealer={rightPid === oya}
        style={{ right: 4, top: "50%", transform: "translateY(-50%) rotate(270deg)", transformOrigin: "right center" }}
      />
      <ScoreMarker
        value={scoreText(humanId)}
        color={SEAT_COLORS[humanId]}
        isDealer={humanId === oya}
        style={{ bottom: 6, left: "50%", transform: "translateX(-50%)" }}
      />
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <div style={{
          padding: "2px 8px",
          background: "var(--accent)",
          color: "#fff",
          borderRadius: 4,
          fontSize: 12,
          fontWeight: 700,
          fontFamily: "Menlo, monospace",
        }}>
          {BAKAZE_CN[bakaze] ?? bakaze}
        </div>
      </div>
      <span style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 2 }}>
        {showScoreDiff ? "分差" : "点数"} · 点击切换
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// 主组件
// ---------------------------------------------------------------------------
const TABLECLOTH_OPTIONS = [
  { id: "default", label: "默认", color: "var(--table-bg)" },
  { id: "green",   label: "浅绿", color: "#e8f5e9" },
  { id: "blue",    label: "浅蓝", color: "#e3f2fd" },
  { id: "beige",   label: "米黄", color: "#f5f0e0" },
] as const;

export function MahjongTable({
  state, onAction, isMyTurn, selectedTile, selectedTileIdx, onTileSelect,
  autoHora, setAutoHora, noMeld, setNoMeld, autoTsumogiri, setAutoTsumogiri,
  actionPending,
  mode = "battle",
  logitData,
  revealedOpponentHands,
  onToggleOpponentHands,
}: {
  state: BattleState;
  onAction: (action: Action) => void;
  isMyTurn: boolean;
  selectedTile: string | null;
  selectedTileIdx?: number | null;
  onTileSelect: (tile: string | null, idx?: number | null) => void;
  autoHora: boolean;
  setAutoHora: (v: boolean) => void;
  noMeld: boolean;
  setNoMeld: (v: boolean) => void;
  autoTsumogiri: boolean;
  setAutoTsumogiri: (v: boolean) => void;
  actionPending?: boolean;
  mode?: "battle" | "replay";
  logitData?: LogitTileData[];
  revealedOpponentHands?: string[][] | null;
  onToggleOpponentHands?: () => void;
}) {
  const [reachPending, setReachPending] = useState(false);
  const [tableScale, setTableScale] = useState(1);
  const [showScoreDiff, setShowScoreDiff] = useState(false);
  const boardViewportRef = useRef<HTMLDivElement | null>(null);
  const {
    player_info, hand, tsumo_pai, discards, melds, reached,
    dora_markers, scores, actor_to_move, bakaze, kyoku, honba, kyotaku, oya,
  } = state;

  const humanId  = state.human_player_id;
  const oppRight = (humanId + 1) % 4;
  const oppTop   = (humanId + 2) % 4;
  const oppLeft  = (humanId + 3) % 4;

  const [tablecloth, setTablecloth]             = useState<"default" | "green" | "blue" | "beige">("default");
  const [showTableclothPicker, setShowTableclothPicker] = useState(false);
  const compactTable = tableScale < 0.9;
  const showActionBar = mode === "battle" && isMyTurn && !reachPending && !state.pending_reach[humanId];
  const showReachBanner = mode === "battle" && isMyTurn && (reachPending || state.pending_reach[humanId]);
  const replayPendingReachActor = mode === "replay"
    ? state.pending_reach.findIndex(Boolean)
    : -1;
  const topConcealedCount = getConcealedCountForSeat(melds[oppTop] || []);
  const leftConcealedCount = getConcealedCountForSeat(melds[oppLeft] || []);
  const rightConcealedCount = getConcealedCountForSeat(melds[oppRight] || []);
  const topShowExtraTile = hasPendingExtraTile(state, oppTop);
  const leftShowExtraTile = hasPendingExtraTile(state, oppLeft);
  const rightShowExtraTile = hasPendingExtraTile(state, oppRight);
  const topDiscardHole = getOpponentDiscardHole("north", revealedOpponentHands?.[oppTop] ?? [], discards[oppTop] || [], state.last_discard, oppTop, topConcealedCount);
  const leftDiscardHole = getOpponentDiscardHole("east", revealedOpponentHands?.[oppLeft] ?? [], discards[oppLeft] || [], state.last_discard, oppLeft, leftConcealedCount);
  const rightDiscardHole = getOpponentDiscardHole("west", revealedOpponentHands?.[oppRight] ?? [], discards[oppRight] || [], state.last_discard, oppRight, rightConcealedCount);
  const canToggleOpponentHands = mode === "replay" && Boolean(onToggleOpponentHands);

  // 赤宝牌归一化：5mr->5m, 5pr->5p, 5sr->5s
  const normTile = (t: string) => t === "5mr" ? "5m" : t === "5pr" ? "5p" : t === "5sr" ? "5s" : t;
  // 找与 tile 对应的 dahai legal action（赤宝牌兼容匹配）
  const findDahaiAction = useCallback((tile: string) => {
    const norm = normTile(tile);
    return state.legal_actions.find(
      x => x.type === "dahai" && (x.pai === tile || normTile(x.pai ?? "") === norm)
    );
  }, [state.legal_actions]);

  const handleTileClick = useCallback((tile: string, idx: number) => {
    if (!isMyTurn) return;
    if (reachPending) {
      // 立直选牌模式：只允许选合法 dahai 牌
      const a = findDahaiAction(tile);
      if (a) { onAction(a); setReachPending(false); onTileSelect(null, null); }
      return;
    }
    if (selectedTileIdx === idx && selectedTile === tile) {
      // 二次点击同一张牌（同tile同index）=> 立即打出
      const a = findDahaiAction(tile);
      if (a) { onAction(a); onTileSelect(null, null); }
    } else {
      if (findDahaiAction(tile)) {
        onTileSelect(tile, idx);
      }
    }
  }, [isMyTurn, reachPending, selectedTile, selectedTileIdx, findDahaiAction, onAction, onTileSelect]);

  const handleAction = useCallback((action: Action) => {
    if (action.type === "reach") {
      // 点立直：先发 reach 动作，然后进入立直选牌模式
      onAction(action);
      setReachPending(true);
      onTileSelect(null, null);
      return;
    }
    if (selectedTile && action.type === "dahai" && action.pai !== selectedTile) {
      const a = findDahaiAction(selectedTile);
      if (a) { onAction(a); onTileSelect(null, null); return; }
    }
    onAction(action);
    onTileSelect(null, null);
    setReachPending(false);
  }, [selectedTile, findDahaiAction, onAction, onTileSelect]);

  useEffect(() => {
    if (mode !== "battle") {
      setReachPending(false);
      return;
    }
    if (!isMyTurn) {
      setReachPending(false);
      return;
    }
    if (state.pending_reach[humanId]) {
      setReachPending(true);
      return;
    }
    if (!actionPending) {
      setReachPending(false);
    }
  }, [mode, isMyTurn, state.pending_reach, humanId, actionPending]);

  useEffect(() => {
    const node = boardViewportRef.current;
    if (!node) return;
    const updateScale = () => {
      const widthScale = node.clientWidth / BASE_TABLE_WIDTH;
      const heightScale = node.clientHeight / BASE_TABLE_HEIGHT;
      setTableScale(Math.min(widthScale, heightScale, 1));
    };
    updateScale();
    const observer = new ResizeObserver(updateScale);
    observer.observe(node);
    return () => observer.disconnect();
  }, []);

  const tableBg = tablecloth === "default"
    ? "var(--table-bg)"
    : TABLECLOTH_OPTIONS.find(t => t.id === tablecloth)?.color ?? "var(--table-bg)";

  // 桌面坐标 — 全部改为百分比，适配任意尺寸

  return (
    <div className="flex flex-col h-full overflow-hidden" style={{ background: "var(--page-bg)" }}>

      {/* 顶部工具栏 */}
      <div style={{
        minHeight: 44,
        background: "var(--sidebar-bg)",
        borderBottom: "1px solid var(--sidebar-border)",
        display: "flex",
        alignItems: "center",
        padding: compactTable ? "6px 12px" : "6px 16px",
        gap: compactTable ? 10 : 16,
        flexWrap: "wrap",
        flexShrink: 0,
        backdropFilter: "blur(12px)",
        transition: "background var(--transition), border-color var(--transition)",
      }}>
        <span style={{ fontWeight: 700, fontSize: 14, color: "var(--text-primary)" }}>
          {BAKAZE_CN[bakaze] ?? bakaze}局
        </span>
        <span style={{ color: "var(--text-muted)", fontSize: 13 }}>{kyoku}圈</span>
        <span style={{
          fontSize: 11, padding: "1px 6px", borderRadius: 4,
          background: "var(--gold-bg)", color: "var(--gold)",
          border: "1px solid var(--gold-border)",
          fontWeight: 600,
          transition: "background var(--transition), color var(--transition)",
        }}>
          {honba}本场
        </span>
        {kyotaku > 0 && (
          <span style={{
            fontSize: 11, padding: "1px 6px", borderRadius: 4,
            background: "var(--error)", color: "#fff",
            fontWeight: 700,
            transition: "background var(--transition)",
          }}>
            立直棒 {kyotaku}
          </span>
        )}

        {/* 桌布切换 */}
        <div style={{ position: "relative", marginLeft: "auto" }}>
          <button
            onClick={() => setShowTableclothPicker(!showTableclothPicker)}
            style={{
              fontSize: 12, color: "var(--text-muted)",
              background: "none", border: "none", cursor: "pointer",
              padding: "4px 8px", borderRadius: 4,
              transition: "color var(--transition)",
            }}
          >
            桌布 ▾
          </button>
          {showTableclothPicker && (
            <div style={{
              position: "absolute", top: "100%", right: 0, marginTop: 4,
              background: "var(--card-bg)",
              border: "1px solid var(--card-border)",
              borderRadius: 8,
              padding: 8, display: "flex", gap: 8,
              boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
              zIndex: 50,
              backdropFilter: "blur(12px)",
            }}>
              {TABLECLOTH_OPTIONS.map(opt => (
                <button
                  key={opt.id}
                  onClick={() => { setTablecloth(opt.id); setShowTableclothPicker(false); }}
                  style={{
                    display: "flex", flexDirection: "column", alignItems: "center", gap: 4,
                    padding: 8, borderRadius: 8, cursor: "pointer",
                    background: tablecloth === opt.id ? "var(--accent-bg)" : "transparent",
                    border: `2px solid ${tablecloth === opt.id ? "var(--accent)" : "transparent"}`,
                    transition: "background var(--transition), border-color var(--transition)",
                  }}
                >
                  <div style={{
                    width: 28, height: 28, borderRadius: "50%",
                    background: opt.color,
                    boxShadow: "inset 0 1px 3px rgba(0,0,0,0.1)",
                  }} />
                  <span style={{ fontSize: 10, color: "var(--text-muted)" }}>{opt.label}</span>
                </button>
              ))}
            </div>
          )}
        </div>

        {/* 得分 */}
        <div style={{ display: "flex", gap: compactTable ? 10 : 16, marginLeft: compactTable ? 0 : 8, flexWrap: "wrap", minWidth: 0 }}>
          {scores.map((score, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 4, minWidth: 0 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: SEAT_COLORS[i] }} />
              <span style={{ fontSize: 12, fontWeight: 600, color: SEAT_COLORS[i], maxWidth: compactTable ? 84 : 120, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                {player_info[i]?.name || `P${i}`}
              </span>
              <span style={{ fontSize: 11, color: "var(--text-muted)", fontFamily: "Menlo, monospace" }}>
                {score.toLocaleString()}
              </span>
            </div>
          ))}
        </div>
      </div>

      {/* 牌桌主体 */}
      <div ref={boardViewportRef} style={{
        flex: 1, minHeight: 0, display: "flex", alignItems: "center", justifyContent: "center",
        padding: 12, position: "relative",
        background: "var(--page-bg)",
        transition: "background var(--transition)",
      }}>
        <div style={{
          width: BASE_TABLE_WIDTH, height: BASE_TABLE_HEIGHT,
          background: tableBg,
          borderRadius: 16,
          border: "2px solid var(--table-border)",
          boxShadow: "var(--table-shadow)",
          position: "relative", overflow: "hidden",
          transform: `scale(${tableScale})`,
          transformOrigin: "center center",
          flexShrink: 0,
          transition: "background var(--transition), border-color var(--transition), box-shadow var(--transition)",
        }}>

          <BoardInfoCorner dora_markers={dora_markers} honba={honba} kyotaku={kyotaku} />

          {/* 中心面板 */}
          <CenterPanel
            bakaze={bakaze}
            scores={scores}
            oya={oya}
            humanId={humanId}
            topPid={oppTop}
            leftPid={oppLeft}
            rightPid={oppRight}
            showScoreDiff={showScoreDiff}
            onToggleScoreMode={() => setShowScoreDiff((v) => !v)}
          />

          {/* 中央弃牌堆 — 以自家为主视角，紧贴 CenterPanel 四角，全部用 top/left */}
          {/* 自家（底部）：正方形左下角，行左→右，向下扩展 */}
          <div style={{ position: "absolute", left: `calc(50% - ${CENTER_SIZE/2}px)`, top: `calc(50% + ${CENTER_SIZE/2}px)` }}>
            <DiscardPondSouth discards={discards[humanId] || []} />
          </div>
          {/* 对家（顶部）：第一张牌右下角对齐正方形右上角，行右→左，新行向上 */}
          <div style={{ position: "absolute", left: `calc(50% + ${CENTER_SIZE/2}px)`, top: `calc(50% - ${CENTER_SIZE/2}px)`, transform: "translate(-100%, -100%)" }}>
            <DiscardPondNorth discards={discards[oppTop] || []} />
          </div>
          {/* 左家（上家）：第一张牌左上角对齐正方形左上角，列上→下，新列向左 */}
          <div style={{ position: "absolute", left: `calc(50% - ${CENTER_SIZE/2}px)`, top: `calc(50% - ${CENTER_SIZE/2}px)`, transform: "translateX(-100%)" }}>
            <DiscardPondLeft discards={discards[oppLeft] || []} />
          </div>
          {/* 右家（下家）：第一张牌左上角对齐正方形右下角，横向，向上扩展 */}
          <div style={{ position: "absolute", left: `calc(50% + ${CENTER_SIZE/2}px)`, top: `calc(50% + ${CENTER_SIZE/2}px)`, transform: "translateY(-100%)" }}>
            <DiscardPondRight discards={discards[oppRight] || []} />
          </div>


          {/* 南（自家） */}
          <div style={{ position: "absolute", left: "50%", bottom: 24, transform: "translateX(-50%)" }}>
            <PlayerZone
              pid={humanId} position="south"
              playerName={player_info[humanId]?.name || `P${humanId}`}
              hand={hand} tsumoPai={tsumo_pai}
              discards={discards[humanId] || []} melds={melds[humanId] || []}
              reached={reached[humanId]} isActive={actor_to_move === humanId}
              isHuman={true} highlightTile={selectedTile} highlightIdx={selectedTileIdx}
              onTileClick={mode === "replay" ? undefined : handleTileClick}
              logitData={logitData}
              compact={compactTable}
            />
          </div>

          {/* 北（对面） */}
          <div style={{ position: "absolute", left: "50%", top: 20, transform: "translateX(-50%)" }}>
            <PlayerZone
              pid={oppTop} position="north"
              playerName={player_info[oppTop]?.name || `P${oppTop}`}
              hand={[]} discards={discards[oppTop] || []} melds={melds[oppTop] || []}
              reached={reached[oppTop]} isActive={actor_to_move === oppTop}
              isHuman={false}
              concealedCount={topConcealedCount}
              showReplayDrawTile={topShowExtraTile}
              revealedHand={revealedOpponentHands?.[oppTop] ?? null}
              discardHole={topDiscardHole}
              onOpponentHandToggle={canToggleOpponentHands ? onToggleOpponentHands : undefined}
            />
          </div>

          {/* 东（左） */}
          <div style={{ position: "absolute", left: 18, top: "50%", transform: "translateY(-50%)" }}>
            <PlayerZone
              pid={oppLeft} position="east"
              playerName={player_info[oppLeft]?.name || `P${oppLeft}`}
              hand={[]} discards={discards[oppLeft] || []} melds={melds[oppLeft] || []}
              reached={reached[oppLeft]} isActive={actor_to_move === oppLeft}
              isHuman={false}
              concealedCount={leftConcealedCount}
              showReplayDrawTile={leftShowExtraTile}
              revealedHand={revealedOpponentHands?.[oppLeft] ?? null}
              discardHole={leftDiscardHole}
              onOpponentHandToggle={canToggleOpponentHands ? onToggleOpponentHands : undefined}
            />
          </div>

          {/* 西（右） */}
          <div style={{ position: "absolute", right: 18, top: "50%", transform: "translateY(-50%)" }}>
            <PlayerZone
              pid={oppRight} position="west"
              playerName={player_info[oppRight]?.name || `P${oppRight}`}
              hand={[]} discards={discards[oppRight] || []} melds={melds[oppRight] || []}
              reached={reached[oppRight]} isActive={actor_to_move === oppRight}
              isHuman={false}
              concealedCount={rightConcealedCount}
              showReplayDrawTile={rightShowExtraTile}
              revealedHand={revealedOpponentHands?.[oppRight] ?? null}
              discardHole={rightDiscardHole}
              onOpponentHandToggle={canToggleOpponentHands ? onToggleOpponentHands : undefined}
            />
          </div>

          {/* 行动中指示（呼吸脉冲动画）：game 模式且 bot 摸牌打牌回合才显示 */}
          {mode === "battle" && actor_to_move !== null && actor_to_move !== humanId &&
           (!state.last_discard || state.last_discard.actor === actor_to_move) && (
            <div style={{
              position: "absolute", top: "50%", left: "50%",
              transform: "translate(-50%, -50%)", zIndex: 30, pointerEvents: "none",
            }}>
              <div
                style={{
                  padding: "5px 16px", borderRadius: 20, fontSize: 12, fontWeight: 700,
                  background: "rgba(0,0,0,0.6)", color: SEAT_COLORS[actor_to_move],
                  border: `2px solid ${SEAT_COLORS[actor_to_move]}`,
                  whiteSpace: "nowrap",
                  animation: "thinkingPulse 2s ease-in-out infinite",
                  boxShadow: `0 0 12px ${SEAT_COLORS[actor_to_move]}40`,
                }}
              >
                {player_info[actor_to_move]?.name || `P${actor_to_move}`} 思考中...
              </div>
            </div>
          )}

          {mode === "replay" && replayPendingReachActor >= 0 && (
            <div style={{
              position: "absolute",
              top: "50%",
              left: "50%",
              transform: "translate(-50%, -50%)",
              zIndex: 32,
              pointerEvents: "none",
            }}>
              <div style={{
                padding: "10px 20px",
                borderRadius: 18,
                background: "rgba(24, 18, 7, 0.86)",
                border: "1px solid var(--gold-border)",
                boxShadow: "0 0 20px rgba(212,168,83,0.24)",
                display: "flex",
                alignItems: "center",
                gap: 12,
                backdropFilter: "blur(10px)",
              }}>
                <div style={{
                  width: 34,
                  height: 34,
                  borderRadius: "50%",
                  background: "linear-gradient(135deg, var(--gold) 0%, #c99731 100%)",
                  color: "#23180a",
                  fontWeight: 900,
                  fontSize: 18,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  boxShadow: "0 4px 12px rgba(212,168,83,0.24)",
                }}>
                  立
                </div>
                <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
                  <div style={{ fontSize: 13, fontWeight: 800, color: "var(--gold)" }}>
                    {player_info[replayPendingReachActor]?.name || `P${replayPendingReachActor}`} 宣告立直
                  </div>
                  <div style={{ fontSize: 11, color: "rgba(255,255,255,0.78)" }}>
                    下一子步将进入宣言后的打牌或后续结算
                  </div>
                </div>
              </div>
            </div>
          )}

        </div>
      </div>

      {/* 动作栏槽位 */}
      {showActionBar && (
        <div style={{
          minHeight: 56,
          padding: "6px 16px 2px",
          display: "flex", justifyContent: "center", alignItems: "center",
          flexShrink: 0,
        }}>
          <div style={{ width: "min(100%, 920px)" }}>
            <ActionBar legalActions={state.legal_actions} onAction={handleAction} disabled={!isMyTurn} />
          </div>
        </div>
      )}
      {showReachBanner && (
        <div style={{
          minHeight: 56,
          padding: "6px 16px 2px",
          display: "flex", justifyContent: "center", alignItems: "center",
          flexShrink: 0,
        }}>
          <div style={{
            fontSize: 12, color: "var(--gold)", fontWeight: 700,
            padding: "4px 14px", borderRadius: 6,
            background: "rgba(0,0,0,0.7)", border: "1px solid var(--gold-border)",
          }}>
            立直宣言中 — 请点击要打出的牌
          </div>
        </div>
      )}
      {mode === "battle" && !showActionBar && !showReachBanner && (
        <div style={{ minHeight: 8, flexShrink: 0 }} />
      )}

      {/* 底部固定设置栏 — battle 模式始终占位，replay 模式隐藏 */}
      {mode === "replay" ? <div style={{ height: 8, flexShrink: 0 }} /> : null}
      {mode === "battle" && <div style={{
        minHeight: 44, flexShrink: 0,
        borderTop: "1px solid var(--border)",
        background: "var(--sidebar-bg)",
        display: "flex", alignItems: "center", justifyContent: "center",
        gap: 8, padding: "6px 16px", flexWrap: "wrap",
        backdropFilter: "blur(12px)",
      }}>
        {[
          { label: "自动胡牌", value: autoHora, set: setAutoHora, defaultOn: true },
          { label: "不响应附露", value: noMeld, set: setNoMeld, defaultOn: false },
          { label: "自动摸切", value: autoTsumogiri, set: setAutoTsumogiri, defaultOn: false },
        ].map(({ label, value, set }) => (
          <button
            key={label}
            onClick={() => set(!value)}
            style={{
              padding: "4px 12px",
              borderRadius: 6,
              border: `1px solid ${value ? "var(--accent)" : "var(--border)"}`,
              background: value ? "var(--accent)" : "var(--card-bg)",
              color: value ? "#fff" : "var(--text-muted)",
              fontSize: 12, fontWeight: 600, cursor: "pointer",
              transition: "all 0.15s",
              display: "flex", alignItems: "center", gap: 5,
            }}
          >
            <span style={{
              width: 8, height: 8, borderRadius: "50%",
              background: value ? "#4ade80" : "#6b7280",
              display: "inline-block", flexShrink: 0,
            }} />
            {label}
          </button>
        ))}
      </div>}

      {/* 全局 CSS 动画 */}
      <style>{`
        @keyframes seatPulse {
          0%, 100% { opacity: 1; transform: scale(1); }
          50% { opacity: 0.5; transform: scale(0.85); }
        }
        @keyframes riichiPulse {
          0%, 100% { box-shadow: 0 0 0 0 rgba(252,211,77,0.4); }
          50% { box-shadow: 0 0 0 4px rgba(252,211,77,0); }
        }
        @keyframes thinkingPulse {
          0%, 100% { opacity: 0.85; transform: translate(-50%, -50%) scale(1); }
          50% { opacity: 1; transform: translate(-50%, -50%) scale(1.03); }
        }
      `}</style>
    </div>
  );
}
