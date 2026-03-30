// src/replay_ui/src/components/BattleBoard/MahjongTable.tsx
import { useState, useCallback, useMemo } from "react";
import { Tile, TileBack } from "./Tile";
import { ActionBar } from "./ActionBar";
import type { BattleState, Action, DiscardEntry } from "../../types/battle";
import type { LogitTileData } from "../../utils/replayAdapter";
import { BAKAZE_CN, JIKAZE_CN } from "../../utils/constants";
import { sortHand } from "../../utils/tileUtils";

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
const DISC_W = 22;  // 单张弃牌宽度（正常大小）
const DISC_H = 30;  // 单张弃牌高度
const DISC_GAP = 2; // 间距
// 6张一列宽度（用于中心正方形边长）
export const CENTER_SIZE = DISC_COLS * DISC_W + (DISC_COLS - 1) * DISC_GAP; // 142px

function DiscardTile({ d, rotated }: { d: DiscardEntry; rotated?: boolean }) {
  // rotated: 左右家固有旋转（牌横向摆放）
  // reachRotated: 立直宣言牌横置
  //   - South/North（rotated=false）：宣言牌额外旋转90° → 容器变宽
  //   - East/West（rotated=true）：宣言牌取消旋转 → 容器变回竖
  const isExtraRotated = !rotated && d.reach_declared;  // South/North 宣言牌旋转
  const isCancelRotated = rotated && d.reach_declared;  // East/West 宣言牌取消旋转

  let cw: number, ch: number;
  let transform: string | undefined;

  if (isCancelRotated) {
    // 取消旋转：变回竖向
    cw = DISC_W; ch = DISC_H;
    transform = undefined;
  } else if (isExtraRotated) {
    // 额外旋转90°（South/North 宣言牌）
    cw = DISC_H; ch = DISC_W;
    transform = `rotate(90deg) translate(0, -${cw}px)`;
  } else if (rotated) {
    cw = DISC_H; ch = DISC_W;
    transform = `rotate(90deg) translate(0, -${cw}px)`;
  } else {
    cw = DISC_W; ch = DISC_H;
    transform = undefined;
  }

  return (
    <div style={{ width: cw, height: ch, position: "relative", flexShrink: 0, overflow: "hidden" }}>
      <div style={{
        width: 22, height: 30,
        transformOrigin: "top left",
        transform,
        position: "absolute", top: 0, left: 0,
        filter: d.tsumogiri ? "brightness(0.6)" : undefined,
        outline: d.reach_declared ? "2px solid var(--gold)" : undefined,
        outlineOffset: "1px",
        borderRadius: d.reach_declared ? 3 : undefined,
      }}>
        <Tile tile={d.pai} size="small" />
      </div>
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
          {row.map((d, di) => <DiscardTile key={di} d={d} />)}
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
          {row.map((d, di) => <DiscardTile key={di} d={d} />)}
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
        <div key={ci} style={{ display: "flex", flexDirection: "column", gap: DISC_GAP }}>
          {col.map((d, di) => <DiscardTile key={di} d={d} rotated />)}
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
        <div key={ci} style={{ display: "flex", flexDirection: "column-reverse", gap: DISC_GAP }}>
          {col.map((d, di) => (
            // 宣言牌：取消内部旋转后外层180°= 牌倒置竖放；改为外层0°+ rotated=false → 竖置正常，需特殊处理
            d.reach_declared ? (
              // 宣言牌在右家：270°方向立直 = 竖置倒转（180°），外层不转，内部不转
              <div key={di} style={{ width: DISC_W, height: DISC_H, position: "relative", flexShrink: 0, overflow: "hidden" }}>
                <div style={{ width: 22, height: 30, position: "absolute", top: 0, left: 0,
                  transform: "rotate(180deg)", transformOrigin: "center center",
                  filter: d.tsumogiri ? "brightness(0.6)" : undefined,
                  outline: "2px solid var(--gold)", outlineOffset: "1px", borderRadius: 3,
                }}>
                  <Tile tile={d.pai} size="small" />
                </div>
              </div>
            ) : (
              <div key={di} style={{ transform: "rotate(180deg)" }}>
                <DiscardTile d={d} rotated />
              </div>
            )
          ))}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// 单家玩家区
// ---------------------------------------------------------------------------
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
}: {
  pid: number;
  position: "south" | "north" | "east" | "west";
  playerName: string;
  hand: string[];
  tsumoPai?: string | null;
  discards: DiscardEntry[];
  melds: { consumed: string[] }[];
  reached: boolean;
  isActive: boolean;
  isHuman?: boolean;
  highlightTile?: string | null;
  highlightIdx?: number | null;
  onTileClick?: (tile: string, idx: number) => void;
  logitData?: LogitTileData[];
}) {
  const color   = SEAT_COLORS[pid];
  const jikaze = JIKAZE_CN[pid];

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

        {/* 明手（微微上浮营造立体感）+ 可选柱状图叠加 */}
        <div style={{ position: "relative" }}>
          {/* 柱状图层（回放模式，绝对定位在手牌上方） */}
          {logitData && logitData.length > 0 && (
            <div style={{
              position: "absolute", bottom: "100%", left: 0,
              display: "flex", gap: 2, paddingBottom: 3,
              pointerEvents: "none", alignItems: "flex-end",
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
                    width: i === sortedHand.length ? 34 : 34,
                    marginLeft: i === sortedHand.length && tsumoPai ? 6 : 0,
                    display: "flex", alignItems: "flex-end", justifyContent: "center", flexShrink: 0,
                  }}>
                    <div style={{
                      width: 20, height: h,
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
          <div style={{ display: "flex", gap: 2, flexWrap: "nowrap", transform: "translateY(-2px)" }}>
            {sortedHand.map((tile, i) => (
              <div key={`${tile}-${i}`} style={{ width: 34, height: 46, flexShrink: 0 }}>
                <Tile tile={tile} size="normal" selected={i === highlightIdx} onClick={() => onTileClick?.(tile, i)} />
              </div>
            ))}
            {tsumoPai && (
              <div
                style={{ width: 34, height: 46, outline: "2px solid var(--gold)", outlineOffset: "1px", borderRadius: 4, boxShadow: "0 0 10px rgba(212,168,83,0.45)", flexShrink: 0, cursor: onTileClick ? "pointer" : undefined, marginLeft: 6 }}
                onClick={() => onTileClick?.(tsumoPai, sortedHand.length)}
              >
                <Tile tile={tsumoPai} size="normal" selected={sortedHand.length === highlightIdx} />
              </div>
            )}
          </div>
        </div>

        {/* 副露 */}
        {melds.length > 0 && (
          <div style={{ display: "flex", gap: 3 }}>
            {melds.map((m, mi) => (
              <div key={mi} style={{ display: "flex", gap: 1 }}>
                {m.consumed.map((t, ci) => (
                  <div key={ci} style={{ width: 26, height: 34, flexShrink: 0 }}>
                    <Tile tile={t} size="small" />
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}

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
      <div style={{ transform: "rotate(180deg)", display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
        {/* 玩家标签 */}
        <PlayerLabel color={color} playerName={playerName} jikaze={jikaze} reached={reached} isActive={isActive} />
        {/* 暗手 + 副露（横向） */}
        <div style={{ display: "flex", gap: 2, alignItems: "flex-end" }}>
          {melds.length > 0 && (
            <div style={{ display: "flex", gap: 3, marginRight: 4 }}>
              {[...melds].reverse().map((m, mi) => (
                <div key={mi} style={{ display: "flex", gap: 1 }}>
                  {m.consumed.map((t, ci) => (
                    <div key={ci} style={{ width: 22, height: 30, flexShrink: 0 }}><Tile tile={t} size="small" /></div>
                  ))}
                </div>
              ))}
            </div>
          )}
          {Array.from({ length: 13 }, (_, i) => (
            <div key={i} style={{ width: 28, height: 38, flexShrink: 0 }}><TileBack size="normal" /></div>
          ))}
        </div>
      </div>
    );
  }

  // ── 左家（east）── 竖列暗牌贴左边，名称显示在牌右侧偏上（不旋转）
  if (position === "east") {
    return (
      <div style={{ position: "relative", display: "flex", flexDirection: "row" }}>
        {/* 竖列暗牌 */}
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {melds.length > 0 && (
            <div style={{ display: "flex", flexDirection: "column", gap: 2, marginBottom: 4 }}>
              {[...melds].reverse().map((m, mi) => (
                <div key={mi} style={{ display: "flex", flexDirection: "row", gap: 1 }}>
                  {m.consumed.map((t, ci) => (
                    <div key={ci} style={{ width: 22, height: 30, flexShrink: 0 }}><Tile tile={t} size="small" /></div>
                  ))}
                </div>
              ))}
            </div>
          )}
          {Array.from({ length: 13 }, (_, i) => (
            <div key={i} style={{ width: 38, height: 28, flexShrink: 0 }}><TileBack size="normal" rotated /></div>
          ))}
        </div>
        {/* 名称：正向显示，在牌右侧偏上 */}
        <div style={{ position: "absolute", left: "100%", top: 4, marginLeft: 4 }}>
          <PlayerLabel color={color} playerName={playerName} jikaze={jikaze} reached={reached} isActive={isActive} />
        </div>
      </div>
    );
  }

  // ── 右家（west）── 竖列暗牌贴右边，名称显示在牌左侧偏上（不旋转）
  return (
    <div style={{ position: "relative", display: "flex", flexDirection: "row" }}>
      {/* 名称：正向显示，在牌左侧偏上 */}
      <div style={{ position: "absolute", right: "100%", top: 4, marginRight: 4 }}>
        <PlayerLabel color={color} playerName={playerName} jikaze={jikaze} reached={reached} isActive={isActive} />
      </div>
      {/* 竖列暗牌 */}
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {melds.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 2, marginBottom: 4 }}>
            {[...melds].reverse().map((m, mi) => (
              <div key={mi} style={{ display: "flex", flexDirection: "row", gap: 1 }}>
                {m.consumed.map((t, ci) => (
                  <div key={ci} style={{ width: 22, height: 30, flexShrink: 0 }}><Tile tile={t} size="small" /></div>
                ))}
              </div>
            ))}
          </div>
        )}
          {Array.from({ length: 13 }, (_, i) => (
          <div key={i} style={{ width: 38, height: 28, flexShrink: 0 }}><TileBack size="normal" rotated /></div>
        ))}
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
function CenterPanel({ bakaze, honba, kyotaku, dora_markers }: {
  bakaze: string; honba: number; kyotaku: number; dora_markers: string[];
}) {
  return (
    <div style={{
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
    }}>
      {/* 场风 / 本场 / 立直棒 */}
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
        <div style={{
          padding: "2px 6px",
          background: "var(--gold-bg)",
          color: "var(--gold)",
          border: "1px solid var(--gold-border)",
          borderRadius: 4,
          fontSize: 11,
          fontWeight: 700,
          fontFamily: "Menlo, monospace",
        }}>
          {honba}
        </div>
        {kyotaku > 0 && (
          <div style={{
            padding: "2px 6px",
            background: "var(--error)",
            color: "#fff",
            borderRadius: 4,
            fontSize: 11,
            fontWeight: 700,
          }}>
            {kyotaku}
          </div>
        )}
      </div>
      <div style={{ width: "80%", height: 1, background: "var(--border)" }} />
      {/* Dora */}
      <div style={{ display: "flex", gap: 3, flexWrap: "wrap", justifyContent: "center" }}>
        {dora_markers.slice(0, 4).map((tile, i) => (
          <div key={i} style={{ width: 26, height: 34 }}>
            <Tile tile={tile} size="small" />
          </div>
        ))}
      </div>
      <span style={{ fontSize: 9, color: "var(--text-muted)", marginTop: -2 }}>宝牌</span>
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
  mode = "battle",
  logitData,
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
  mode?: "battle" | "replay";
  logitData?: LogitTileData[];
}) {
  const [reachPending, setReachPending] = useState(false);
  const {
    player_info, hand, tsumo_pai, discards, melds, reached,
    dora_markers, scores, actor_to_move, bakaze, kyoku, honba, kyotaku,
  } = state;

  const humanId  = state.human_player_id;
  const oppRight = (humanId + 1) % 4;
  const oppTop   = (humanId + 2) % 4;
  const oppLeft  = (humanId + 3) % 4;

  const [tablecloth, setTablecloth]             = useState<"default" | "green" | "blue" | "beige">("default");
  const [showTableclothPicker, setShowTableclothPicker] = useState(false);

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

  const tableBg = tablecloth === "default"
    ? "var(--table-bg)"
    : TABLECLOTH_OPTIONS.find(t => t.id === tablecloth)?.color ?? "var(--table-bg)";

  // 桌面坐标 — 全部改为百分比，适配任意尺寸

  return (
    <div className="flex flex-col h-full overflow-hidden" style={{ background: "var(--page-bg)" }}>

      {/* 顶部工具栏 */}
      <div style={{
        height: 44,
        background: "var(--sidebar-bg)",
        borderBottom: "1px solid var(--sidebar-border)",
        display: "flex",
        alignItems: "center",
        padding: "0 16px",
        gap: 16,
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
        <div style={{ display: "flex", gap: 16, marginLeft: 8 }}>
          {scores.map((score, i) => (
            <div key={i} style={{ display: "flex", alignItems: "center", gap: 4 }}>
              <div style={{ width: 8, height: 8, borderRadius: "50%", background: SEAT_COLORS[i] }} />
              <span style={{ fontSize: 12, fontWeight: 600, color: SEAT_COLORS[i] }}>
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
      <div style={{
        flex: 1, display: "flex", alignItems: "stretch", justifyContent: "stretch",
        padding: 0, position: "relative",
        background: "var(--page-bg)",
        transition: "background var(--transition)",
      }}>
        <div style={{
          width: "100%", height: "100%",
          background: tableBg,
          borderRadius: 16,
          border: "2px solid var(--table-border)",
          boxShadow: "var(--table-shadow)",
          position: "relative", overflow: "hidden",
          transition: "background var(--transition), border-color var(--transition), box-shadow var(--transition)",
        }}>

          {/* 中心面板 */}
          <CenterPanel bakaze={bakaze} honba={honba} kyotaku={kyotaku} dora_markers={dora_markers} />

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
          <div style={{ position: "absolute", left: "50%", bottom: 0, transform: "translateX(-50%)" }}>
            <PlayerZone
              pid={humanId} position="south"
              playerName={player_info[humanId]?.name || `P${humanId}`}
              hand={hand} tsumoPai={tsumo_pai}
              discards={discards[humanId] || []} melds={melds[humanId] || []}
              reached={reached[humanId]} isActive={actor_to_move === humanId}
              isHuman={true} highlightTile={selectedTile} highlightIdx={selectedTileIdx}
              onTileClick={mode === "replay" ? undefined : handleTileClick}
              logitData={logitData}
            />
          </div>

          {/* 北（对面） */}
          <div style={{ position: "absolute", left: "50%", top: 0, transform: "translateX(-50%)" }}>
            <PlayerZone
              pid={oppTop} position="north"
              playerName={player_info[oppTop]?.name || `P${oppTop}`}
              hand={[]} discards={discards[oppTop] || []} melds={melds[oppTop] || []}
              reached={reached[oppTop]} isActive={actor_to_move === oppTop}
              isHuman={false}
            />
          </div>

          {/* 东（左） */}
          <div style={{ position: "absolute", left: 0, top: "50%", transform: "translateY(-50%)" }}>
            <PlayerZone
              pid={oppLeft} position="east"
              playerName={player_info[oppLeft]?.name || `P${oppLeft}`}
              hand={[]} discards={discards[oppLeft] || []} melds={melds[oppLeft] || []}
              reached={reached[oppLeft]} isActive={actor_to_move === oppLeft}
              isHuman={false}
            />
          </div>

          {/* 西（右） */}
          <div style={{ position: "absolute", right: 0, top: "50%", transform: "translateY(-50%)" }}>
            <PlayerZone
              pid={oppRight} position="west"
              playerName={player_info[oppRight]?.name || `P${oppRight}`}
              hand={[]} discards={discards[oppRight] || []} melds={melds[oppRight] || []}
              reached={reached[oppRight]} isActive={actor_to_move === oppRight}
              isHuman={false}
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

        </div>
      </div>

      {/* 动作栏 overlay — 绝对定位在牌桌底部，不压缩牌桌空间 */}
      {mode === "battle" && isMyTurn && !reachPending && !state.pending_reach[humanId] && (
        <div style={{
          position: "absolute", bottom: 44, left: 0, right: 0,
          display: "flex", justifyContent: "center", zIndex: 50, pointerEvents: "none",
        }}>
          <div style={{ pointerEvents: "auto" }}>
            <ActionBar legalActions={state.legal_actions} onAction={handleAction} disabled={!isMyTurn} />
          </div>
        </div>
      )}
      {mode === "battle" && isMyTurn && (reachPending || state.pending_reach[humanId]) && (
        <div style={{
          position: "absolute", bottom: 44, left: 0, right: 0,
          display: "flex", justifyContent: "center", zIndex: 50,
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

      {/* 底部固定设置栏 — battle 模式始终占位，replay 模式隐藏 */}
      {mode === "replay" ? <div style={{ height: 8, flexShrink: 0 }} /> : null}
      {mode === "battle" && <div style={{
        height: 44, flexShrink: 0,
        borderTop: "1px solid var(--border)",
        background: "var(--sidebar-bg)",
        display: "flex", alignItems: "center", justifyContent: "center",
        gap: 12, padding: "0 16px",
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
