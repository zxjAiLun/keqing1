// src/replay_ui/src/components/BattleBoard/Tile.tsx
import { TILE_SVG_NAME } from "../../utils/tileUtils";
import { TILE_SIZES } from "./tileSizes";

interface TileProps {
  tile: string;
  size?: "small" | "normal" | "large";
  selected?: boolean;
  highlighted?: boolean;
  onClick?: () => void;
  className?: string;
}

export function Tile({ tile, size = "normal", selected, highlighted, onClick, className = "" }: TileProps) {
  const dim = TILE_SIZES[size];
  const tileSvgName = TILE_SVG_NAME[tile];

  // 选中时增强发光效果
  const ringGlow = selected
    ? "0 0 0 3px var(--gold), 0 0 12px rgba(212,168,83,0.5)"
    : highlighted
    ? "0 0 0 2px #e85a5a, 0 0 8px rgba(232,90,90,0.4)"
    : "0 0 0 1px rgba(0,0,0,0.12)";

  const liftY = selected ? -8 : highlighted ? -4 : -2;

  return (
    <div
      className={`relative inline-flex items-center justify-center cursor-pointer ${className}`}
      style={{
        width: dim.w,
        height: dim.h,
        transform: `translateY(${liftY}px)`,
        transition: "transform var(--table-transition-mid), box-shadow var(--table-transition-mid), opacity var(--table-transition-fast)",
        boxShadow: [
          // 底部硬阴影（天凤风格）
          "0 4px 0px #9e9888",
          "0 7px 0px #8a8478",
          "0 10px 4px rgba(0,0,0,0.35)",
          ringGlow,
        ].join(", "),
        borderRadius: 4,
        background: "var(--tile-bg)",
        flexShrink: 0,
      }}
      onClick={onClick}
    >
      {/* 顶部白边（3D 厚度感） */}
      <div style={{
        position: "absolute", top: 0, left: 0, right: 0, height: 3,
        background: "linear-gradient(90deg, rgba(255,255,255,0.15) 0%, rgba(255,255,255,0.98) 25%, rgba(255,255,255,1) 100%)",
        borderRadius: "4px 4px 0 0",
        pointerEvents: "none",
      }} />
      {/* 右边白边（3D 厚度感） */}
      <div style={{
        position: "absolute", top: 0, right: 0, bottom: 0, width: 3,
        background: "linear-gradient(180deg, rgba(255,255,255,0.98) 0%, rgba(255,255,255,0.7) 30%, rgba(255,255,255,0.4) 100%)",
        borderRadius: "0 4px 4px 0",
        pointerEvents: "none",
      }} />
      {/* 牌图 */}
      {tileSvgName ? (
        <img
          src={`/tiles/${tileSvgName}.svg`}
          alt={tile}
          style={{ width: dim.w - 6, height: dim.h - 6, display: "block" }}
        />
      ) : (
        <div style={{ width: dim.w - 6, height: dim.h - 6, background: "#ddd8cc", borderRadius: 3 }} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------

interface TileBackProps {
  size?: "small" | "normal" | "large";
  className?: string;
  orientation?: 0 | 90 | 180 | 270;
}

export function TileBack({ size = "normal", className = "", orientation = 0 }: TileBackProps) {
  const dim = TILE_SIZES[size];
  const rotated = orientation === 90 || orientation === 270;
  const isFlipped = orientation === 180 || orientation === 270;

  const shadowOffsetX = orientation === 90 ? -3 : orientation === 270 ? 3 : 3;
  const shadowOffsetY = isFlipped ? -4 : 4;

  // 旋转后内部高光方向跟着转
  const topGrad = rotated
    ? "linear-gradient(180deg, rgba(255,255,255,0.0) 0%, rgba(255,255,255,0.45) 20%, rgba(255,255,255,0.75) 100%)"
    : "linear-gradient(90deg, rgba(255,255,255,0.0) 0%, rgba(255,255,255,0.85) 20%, rgba(255,255,255,1) 100%)";
  const topGradPos = { top: 0, left: 0, right: 0, height: 3 };
  const rightGrad = rotated
    ? "linear-gradient(270deg, rgba(255,255,255,0.75) 0%, rgba(255,255,255,0.45) 30%, rgba(255,255,255,0.2) 100%)"
    : "linear-gradient(180deg, rgba(255,255,255,0.88) 0%, rgba(255,255,255,0.55) 30%, rgba(255,255,255,0.25) 100%)";
  const rightGradPos = { top: 0, right: 0, bottom: 0, width: 3 };

  return (
    <div
      className={`relative ${className}`}
      style={{
        width: rotated ? dim.h : dim.w,
        height: rotated ? dim.w : dim.h,
        borderRadius: 4,
        background: "var(--tile-back-bg)",
        boxShadow: [
          `${shadowOffsetX}px ${shadowOffsetY}px 0px #1a1e2a`,
          `${shadowOffsetX * 2}px ${shadowOffsetY + 2}px 0px #141820`,
          `${shadowOffsetX * 2 + 2}px ${shadowOffsetY + 4}px 6px rgba(0,0,0,0.5)`,
        ].join(", "),
        flexShrink: 0,
        overflow: "hidden",
        transform: orientation ? `rotate(${orientation}deg)` : undefined,
        transition: "transform var(--table-transition-mid), box-shadow var(--table-transition-mid), filter var(--table-transition-mid)",
      }}
    >
      {/* 顶部/右边高光 */}
      <div style={{ position: "absolute", ...topGradPos, background: topGrad, borderRadius: rotated ? "0 4px 0 0" : "4px 4px 0 0", zIndex: 1 }} />
      <div style={{ position: "absolute", ...rightGradPos, background: rightGrad, borderRadius: rotated ? "0 0 4px 0" : "0 4px 4px 0", zIndex: 1 }} />

      {/* 内部格纹纹理（天凤风格细密格） */}
      <div style={{
        position: "absolute", inset: 4,
        background: `
          repeating-linear-gradient(45deg,
            rgba(255,255,255,0.06) 0px,
            rgba(255,255,255,0.06) 1px,
            transparent 1px,
            transparent 8px
          ),
          repeating-linear-gradient(-45deg,
            rgba(255,255,255,0.06) 0px,
            rgba(255,255,255,0.06) 1px,
            transparent 1px,
            transparent 8px
          )
        `,
        borderRadius: 2,
      }} />

      {/* 中心圆徽 */}
      <div style={{
        position: "absolute",
        top: "50%", left: "50%",
        transform: "translate(-50%, -50%)",
        width: rotated ? dim.w * 0.45 : dim.h * 0.45,
        height: rotated ? dim.w * 0.45 : dim.h * 0.45,
        borderRadius: "50%",
        background: "radial-gradient(circle, rgba(255,255,255,0.10) 0%, transparent 70%)",
        border: "1px solid rgba(255,255,255,0.07)",
      }} />
    </div>
  );
}
