// src/replay_ui/src/components/BattleBoard/MahjongTable.tsx
import { useState, useCallback } from "react";
import { Tile } from "./Tile";
import { TileBack } from "./Tile";
import { ActionBar } from "./ActionBar";
import type { BattleState, Action, DiscardEntry } from "../../types/battle";
import { BAKAZE_CN, JIKAZE_CN } from "../../utils/constants";
import { sortHand } from "../../utils/tileUtils";

// ---------------------------------------------------------------------------
// 常量
// ---------------------------------------------------------------------------
const SEAT_COLORS = ["var(--seat-0)", "var(--seat-1)", "var(--seat-2)", "var(--seat-3)"];

const TABLE_W = 780;
const TABLE_H = 500;
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
// 中央弃牌堆（每家6张一行，最多3行）
// ---------------------------------------------------------------------------
function DiscardPond({ discards }: { discards: DiscardEntry[] }) {
  const rows = chunkDiscards(discards, DISC_COLS);
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
      {rows.map((row, ri) => (
        <div key={ri} style={{ display: "flex", gap: 1 }}>
          {row.map((d, ci) => (
            <div key={ci} style={{ width: 22, height: 30, position: "relative", flexShrink: 0 }}>
              <Tile tile={d.pai} size="small" />
              {d.tsumogiri && (
                <div style={{
                  position: "absolute", top: 1, right: 1,
                  width: 3, height: 5,
                  background: "var(--gold)",
                  borderRadius: 1,
                }} />
              )}
            </div>
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
  discards,
  melds,
  reached,
  isActive,
  highlightIdx,
  onTileClick,
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
}) {
  const color   = SEAT_COLORS[pid];
  const jikaze = JIKAZE_CN[pid];

  // ── 自家（south）──
  if (position === "south") {
    const sortedHand = sortHand(hand, tsumoPai);

    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 5 }}>

        {/* 明手（微微上浮营造立体感） */}
        <div style={{ display: "flex", gap: 2, flexWrap: "nowrap", transform: "translateY(-2px)" }}>
          {sortedHand.map((tile, i) => (
            <div key={`${tile}-${i}`} style={{ width: 34, height: 46, flexShrink: 0 }}>
              <Tile tile={tile} size="normal" selected={i === highlightIdx} onClick={() => onTileClick?.(tile, i)} />
            </div>
          ))}
          {tsumoPai && (
            <div
              style={{ width: 34, height: 46, border: "2px solid var(--gold)", borderRadius: 4, boxShadow: "0 0 10px rgba(212,168,83,0.45)", flexShrink: 0, cursor: onTileClick ? "pointer" : undefined }}
              onClick={() => onTileClick?.(tsumoPai, sortedHand.length)}
            >
              <Tile tile={tsumoPai} size="normal" selected={sortedHand.length === highlightIdx} />
            </div>
          )}
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

  // ── 对面（north）──
  if (position === "north") {
    return (
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 5 }}>

        {/* 玩家标签 */}
        <div style={{ display: "flex", alignItems: "center", gap: 4, marginBottom: 2 }}>
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

        {/* 舍牌已移至中央弃牌堆 */}

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

        {/* 暗手 */}
        <div style={{ display: "flex", gap: 2 }}>
          {Array.from({ length: 13 }, (_, i) => (
            <div key={i} style={{ width: 34, height: 46, flexShrink: 0 }}>
              <TileBack size="normal" />
            </div>
          ))}
        </div>
      </div>
    );
  }

  // ── 左家（east）──
  if (position === "east") {
    return (
      <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: 6 }}>

        {/* 玩家标签 */}
        <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
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

        {/* 舍牌已移至中央弃牌堆 */}

        {/* 副露 */}
        {melds.length > 0 && (
          <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
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

        {/* 暗手（竖向，旋转90度） */}
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
          {Array.from({ length: 13 }, (_, i) => (
            <div key={i} style={{ width: 34, height: 46, flexShrink: 0 }}>
              <TileBack size="normal" rotated />
            </div>
          ))}
        </div>
      </div>
    );
  }

  // ── 右家（west）──
  return (
    <div style={{ display: "flex", flexDirection: "row", alignItems: "center", gap: 6 }}>

      {/* 暗手（竖向，旋转90度） */}
      <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
        {Array.from({ length: 13 }, (_, i) => (
          <div key={i} style={{ width: 34, height: 46, flexShrink: 0 }}>
            <TileBack size="normal" rotated />
          </div>
        ))}
      </div>

      {/* 副露 */}
      {melds.length > 0 && (
        <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
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

      {/* 舍牌（横向，最新牌在内侧） */}
      {discards.length > 0 && (
        <div style={{ display: "flex", flexDirection: "row", gap: 2, flexWrap: "wrap", maxWidth: 168, alignItems: "center" }}>
          {[...discards].reverse().map((d, ci) => (
            <div key={ci} style={{ width: 26, height: 34, position: "relative", flexShrink: 0 }}>
              <Tile tile={d.pai} size="small" />
              {d.tsumogiri && (
                <div style={{
                  position: "absolute", top: 1, right: 1,
                  width: 3, height: 6,
                  background: "var(--gold)",
                  borderRadius: 1,
                  boxShadow: "0 0 3px rgba(251,191,36,0.6)",
                }} />
              )}
            </div>
          ))}
        </div>
      )}

      {/* 玩家标签 */}
      <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 4 }}>
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
      width: 118, height: 118,
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
}: {
  state: BattleState;
  onAction: (action: Action) => void;
  isMyTurn: boolean;
  selectedTile: string | null;
  selectedTileIdx?: number | null;
  onTileSelect: (tile: string | null, idx?: number | null) => void;
}) {
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
    if (selectedTileIdx === idx && selectedTile === tile) {
      // 二次点击同一张牌（同tile同index）=> 立即打出
      const a = findDahaiAction(tile);
      if (a) { onAction(a); onTileSelect(null, null); }
    } else {
      if (findDahaiAction(tile)) {
        onTileSelect(tile, idx);
      }
    }
  }, [isMyTurn, selectedTile, selectedTileIdx, findDahaiAction, onAction, onTileSelect]);

  const handleAction = useCallback((action: Action) => {
    if (selectedTile && action.type === "dahai" && action.pai !== selectedTile) {
      const a = findDahaiAction(selectedTile);
      if (a) { onAction(a); onTileSelect(null); return; }
    }
    onAction(action);
    onTileSelect(null);
  }, [selectedTile, findDahaiAction, onAction, onTileSelect]);

  const tableBg = tablecloth === "default"
    ? "var(--table-bg)"
    : TABLECLOTH_OPTIONS.find(t => t.id === tablecloth)?.color ?? "var(--table-bg)";

  // 桌面坐标
  const southX = TABLE_W / 2;   const southY = TABLE_H - 55;
  const northX = TABLE_W / 2;   const northY = 55;
  const eastX  = 70;             const eastY  = TABLE_H / 2;
  const westX  = TABLE_W - 70;  const westY  = TABLE_H / 2;

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
        flex: 1, display: "flex", alignItems: "center", justifyContent: "center",
        padding: 16,
        background: "var(--page-bg)",
        transition: "background var(--transition)",
      }}>
        <div style={{
          width: TABLE_W, height: TABLE_H,
          background: tableBg,
          borderRadius: 16,
          border: "2px solid var(--table-border)",
          boxShadow: "var(--table-shadow)",
          position: "relative", overflow: "hidden",
          transition: "background var(--transition), border-color var(--transition), box-shadow var(--transition)",
        }}>

          {/* 中心面板 */}
          <CenterPanel bakaze={bakaze} honba={honba} kyotaku={kyotaku} dora_markers={dora_markers} />

          {/* 中央弃牌堆 */}
          {/* 南家弃牌（中心下方） */}
          <div style={{ position: "absolute", left: "50%", top: "57%", transform: "translateX(-50%)" }}>
            <DiscardPond discards={discards[humanId] || []} />
          </div>
          {/* 北家弃牌（中心上方） */}
          <div style={{ position: "absolute", left: "50%", top: "18%", transform: "translateX(-50%) rotate(180deg)" }}>
            <DiscardPond discards={discards[oppTop] || []} />
          </div>
          {/* 西家弃牌（中心右方） */}
          <div style={{ position: "absolute", left: "63%", top: "50%", transform: "translateY(-50%) rotate(90deg)" }}>
            <DiscardPond discards={discards[oppRight] || []} />
          </div>
          {/* 东家弃牌（中心左方） */}
          <div style={{ position: "absolute", left: "24%", top: "50%", transform: "translateY(-50%) rotate(-90deg)" }}>
            <DiscardPond discards={discards[oppLeft] || []} />
          </div>

          {/* 南（自家） */}
          <div style={{ position: "absolute", left: southX, top: southY, transform: "translateX(-50%)" }}>
            <PlayerZone
              pid={humanId} position="south"
              playerName={player_info[humanId]?.name || `P${humanId}`}
              hand={hand} tsumoPai={tsumo_pai}
              discards={discards[humanId] || []} melds={melds[humanId] || []}
              reached={reached[humanId]} isActive={actor_to_move === humanId}
              isHuman={true} highlightTile={selectedTile} highlightIdx={selectedTileIdx} onTileClick={handleTileClick}
            />
          </div>

          {/* 北（对面） */}
          <div style={{ position: "absolute", left: northX, top: northY, transform: "translateX(-50%)" }}>
            <PlayerZone
              pid={oppTop} position="north"
              playerName={player_info[oppTop]?.name || `P${oppTop}`}
              hand={[]} discards={discards[oppTop] || []} melds={melds[oppTop] || []}
              reached={reached[oppTop]} isActive={actor_to_move === oppTop}
              isHuman={false}
            />
          </div>

          {/* 东（左） */}
          <div style={{ position: "absolute", left: eastX, top: eastY, transform: "translateY(-50%)" }}>
            <PlayerZone
              pid={oppLeft} position="east"
              playerName={player_info[oppLeft]?.name || `P${oppLeft}`}
              hand={[]} discards={discards[oppLeft] || []} melds={melds[oppLeft] || []}
              reached={reached[oppLeft]} isActive={actor_to_move === oppLeft}
              isHuman={false}
            />
          </div>

          {/* 西（右） */}
          <div style={{ position: "absolute", left: westX, top: westY, transform: "translateY(-50%)" }}>
            <PlayerZone
              pid={oppRight} position="west"
              playerName={player_info[oppRight]?.name || `P${oppRight}`}
              hand={[]} discards={discards[oppRight] || []} melds={melds[oppRight] || []}
              reached={reached[oppRight]} isActive={actor_to_move === oppRight}
              isHuman={false}
            />
          </div>

          {/* 行动中指示（呼吸脉冲动画） */}
          {actor_to_move !== null && actor_to_move !== humanId && (
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

      {/* 动作栏 */}
      {isMyTurn && (
        <div style={{
          borderTop: "1px solid var(--border)",
          background: "var(--sidebar-bg)",
          display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center",
          padding: "8px 24px",
          backdropFilter: "blur(12px)",
          transition: "background var(--transition), border-color var(--transition)",
          gap: 4,
        }}>
          {state.pending_reach[humanId] && (
            <div style={{
              fontSize: 12, color: "var(--gold)", fontWeight: 700,
              padding: "2px 10px", borderRadius: 4,
              background: "var(--gold-bg)", border: "1px solid var(--gold-border)",
            }}>
              立直宣言中 — 请点击要打出的牌
            </div>
          )}
          <div style={{ width: "100%", maxWidth: 560 }}>
            <ActionBar legalActions={state.legal_actions} onAction={handleAction} disabled={!isMyTurn} />
          </div>
        </div>
      )}

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
