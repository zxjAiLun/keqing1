// Tenhou 比例估算（从截图结构反推）：
// 14张自家手牌 + gap 应占底部约 65% 宽度 = ~830px
// → large牌宽 ≈ 830/14 ≈ 59px，实取 56px（留少量gap余地）
// 对家手牌约为此尺寸的 0.45 倍
export const TILE_SIZES = {
  small: { w: 20, h: 28 },
  normal: { w: 26, h: 36 },
  large: { w: 56, h: 76 },
} as const;
