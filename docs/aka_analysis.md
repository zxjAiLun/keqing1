现在我可以完整回答了。

---

## Red 5 在 Chi 中的处理

### 1. Tile 编码体系

天凤的 tile ID 分布：
```
0-8:   1m-9m（索子）
9-17:  1p-9p（筒子）
18-26: 1s-9s（万子）
27-33: E,S,W,N,P,F,C（字牌）
34-36: 5mr, 5pr, 5sr（赤5）
```

关键操作：
- `deaka()`：赤5 → 普通5（`5mr → 5m`）
- `is_aka()`：是否是赤5

### 2. ChiType 判断 — 不区分赤5

```rust
impl ChiType {
    pub fn new(consumed: [Tile; 2], tile: Tile) -> Self {
        let a = consumed[0].deaka().as_u8();  // ← 赤5被抹平
        let b = consumed[1].deaka().as_u8();  // ← 赤5被抹平
        let min = a.min(b);
        let max = a.max(b);
        let tile_id = tile.deaka().as_u8();   // ← called tile也抹平
        if tile_id < min { Self::Low }
        else if tile_id < max { Self::Mid }
        else { Self::High }
    }
}
```

`ChiType::new` 对所有 tile 做 `deaka()`，所以 **Low/Mid/High 的分类和赤5完全无关**。

### 3. `can_chi_*` 的判断 — 同样不区分赤5

`set_can_chi_from_tile` 检查的是 `self.tehai[tile_id]` 数组，tile_id 是 deaka 后的索引：

```rust
// 例：chi-low on 7s
// tehai[6] = 5s 的数量 (包含普通5s和赤5sr的总和)
// tehai[8] = 7s 的数量
if literal_num <= 7 && self.tehai[tile_id + 1] > 0 && self.tehai[tile_id + 2] > 0
```

**`tehai` 数组把普通5和赤5视为同一个 slot**（因为都用 deaka 后的 tile_id 索引）。如果你手里有 `5s` 和 `5sr` 各一张，计数是 2，但代码只看 `> 0`，不区分是哪种。

### 4. 动作标签（Label）— 同样不区分

```rust
Event::Chi { actor, pai, consumed, .. } if actor == self.player_id
  => match ChiType::new(consumed, pai) {
      ChiType::Low  => Some(38),   // ← 只分三类
      ChiType::Mid  => Some(39),
      ChiType::High => Some(40),
  }
```

**结论：动作标签只区分 chi 类型（Low/Mid/High），完全不区分是否用了赤5。**

### 5. Observation 编码 — 赤5信息存在，但被压缩

Fuuro（副露）编码在 `obs_repr.rs` 第 303-320 行：

```rust
for tile in f {  // f 是副露的牌组
    let tile_id = tile.deaka().as_usize();  // 牌本身
    // 找到第一个空位存入
    let i = (0..4).find(|&i| self.arr.get(self.idx + i, tile_id) == 0.).unwrap();
    self.arr.assign(self.idx + i, tile_id, 1.);
    
    // 赤5 标志
    if tile.is_aka() {
        self.arr.fill(self.idx + 4, 1.);  // ← aka 标记
    }
}
self.idx += 5;  // 每个副露占 5 个 channel
```

所以 **fuuro 里记录了"这个副露吃过赤5"**，但：

```rust
// 鸣牌后的河底牌编码（第737行）
if let Some(cp) = &k.chi_pon {
    let a = cp.consumed[0].deaka().as_usize();
    let b = cp.consumed[1].deaka().as_usize();
    let min = a.min(b);   // ← 只存min/max（都是deaka后的）
    let max = a.max(b);
    self.arr.assign(self.idx, min, 1.);
    self.arr.assign(self.idx + 1, max, 1.);
    // aka信息在 fuuro_overview 里
}
```

河底的 chi/pon 信息只存 min/max（deaka 后的），不区分哪个是赤5。

### 6. 核心矛盾

**举例**：手牌 `4m, 5m, 5mr, 7m`，对 6m 吃。

- Chi-low 可选：`4m+5m` 或 `5mr+7m` → 都输出 **label=38**
- Mask 38 = true，无法从 action 区分用的是哪组

在训练时，advantage 会在两种 option 之间**平均/混合**，模型学不到"优先用普通5还是赤5"。

### 7. 模型怎么决定实际吃哪组？

模型只输出 38/39/40。实际执行由**游戏引擎**决定，`validate_logs.rs` 第 108 行验证 chi 是否合法（调用 `ensure_tiles_in_hand(&consumed)`），但不决定用哪组 —— mjai 日志里已经记录了具体 `consumed` 数组。

所以 **Mortal v4 对 chi 的设计是"粗粒度"的**：动作只区分 Low/Mid/High，不做 tile 级别的选择。赤5 vs 普通5 的区别存在于 observation（fuuro 的 aka flag）但不存在于 action label，模型无法学到选择策略。

