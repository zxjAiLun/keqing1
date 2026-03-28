现在我可以完整地解释 chi 在 Mortal 项目中的处理流程了：

---

## Chi 处理流程：Tenhoulog → Mjailog → Encoding → 模型

### 1. Tenhoulog → Mjailog

原始数据来自天凤（tenhou.net）的牌谱记录。Mortal 项目不直接处理 tenhou 的原始 XML/二进制格式，而是使用已经转换为 **mjailog JSON 格式**的预处理日志（`.gz` 文件，每行一个 JSON event）。

Chi 事件的 mjailog 格式定义在 `libriichi/src/mjai/event.rs`：

```json
{"type":"chi","actor":1,"target":0,"pai":"6s","consumed":["5sr","7s"]}
```

| 字段 | 含义 |
|------|------|
| `actor` | 吃牌者 |
| `target` | 被鸣家（打、出铳的人） |
| `pai` | 摸到的牌（杠的牌） |
| `consumed` | 鸣牌时打出的两张牌 |

### 2. ChiType 分类（Low/Mid/High）

`libriichi/src/chi_type.rs` 定义了 chi 的三种类型：

```rust
pub enum ChiType { Low, Mid, High }

impl ChiType {
    pub fn new(consumed: [Tile; 2], tile: Tile) -> Self {
        let a = consumed[0].deaka().as_u8();
        let b = consumed[1].deaka().as_u8();
        let min = a.min(b); let max = a.max(b);
        let tile_id = tile.deaka().as_u8();
        if tile_id < min { Self::Low }
        else if tile_id < max { Self::Mid }
        else { Self::High }
    }
}
```

**原理**：假设吃 `6s`，手里的牌是 `4s,5s` → tile_id=5, consumed={3,4}，min=3,max=4，tile_id在min和max之间 → **Mid**

### 3. 数据处理（Dataset → Label）

`libriichi/src/dataset/gameplay.rs` 的 `GameplayLoader` 加载 mjailog 日志，将每个 event 映射为模型训练的 label。Chi 对应的 label：

| ChiType | Action ID |
|---------|-----------|
| Low | **38** |
| Mid | **39** |
| High | **40** |

```rust
Event::Chi { actor, pai, consumed, .. } if actor == self.player_id
  => match ChiType::new(consumed, pai) {
      ChiType::Low  => Some(38),
      ChiType::Mid  => Some(39),
      ChiType::High => Some(40),
  }
```

### 4. 状态编码（State Encoding）

`libriichi/src/state/obs_repr.rs` 构建observation。在 feature vector 中，chi 相关的编码：

```rust
// can_chi_low/mid/high 由 set_can_chi_from_tile() 计算
if cans.can_chi_low {
    self.arr.fill(self.idx, 1.);    // 特征标记
    self.mask[38] = true;            // 允许选择 chi_low
}
if cans.can_chi_mid {
    self.arr.fill(self.idx + 1, 1.);
    self.mask[39] = true;
}
if cans.can_chi_high {
    self.arr.fill(self.idx + 2, 1.);
    self.mask[40] = true;
}
```

`can_chi_*` 的判断逻辑（`libriichi/src/state/update.rs`）：

```rust
fn set_can_chi_from_tile(&mut self, tile: Tile) {
    let tile_id = tile.deaka().as_usize();
    let literal_num = tile_id % 9 + 1;

    // Low: tile+1, tile+2 (e.g., chi 14 from 1234)
    if literal_num <= 7 && self.tehai[tile_id + 1] > 0 && self.tehai[tile_id + 2] > 0 {
        // 还要检查鸣了之后手牌是否还有用
        self.last_cans.can_chi_low = tehai_after.iter().any(|&t| t > 0);
    }

    // Mid: tile-1, tile+1 (e.g., chi 5 from 456)
    if matches!(literal_num, 2..=8) && self.tehai[tile_id - 1] > 0 && self.tehai[tile_id + 1] > 0 {
        self.last_cans.can_chi_mid = ...;
    }

    // High: tile-2, tile-1 (e.g., chi 7 from 6789)
    if literal_num >= 3 && self.tehai[tile_id - 2] > 0 && self.tehai[tile_id - 1] > 0 {
        self.last_cans.can_chi_high = ...;
    }
}
```

### 5. V4 模型推理

Action Space 结构（`libriichi/src/consts.rs`）：

```
0-36   : 37种切牌 (dicard | kan select)
37     : 立直 (Riichi)
38     : 吃 Low (Chi Low)
39     : 吃 Mid (Chi Mid)
40     : 吃 High (Chi High)
41     : 碰 (Pon)
42     : 杠 (Kan / Daiminkan / Ankan / Kakan)
43     : 和了 (Agari / Ron)
44     : 留局 (Ryukyoku)
45     : 过 (Pass)
```

V4 模型（`mortal/model.py`）：

```python
class DQN(nn.Module):
    def __init__(self, *, version=1):
        # v4: 单一线性层输出 1 + 46 = 47 维
        self.net = nn.Linear(1024, 1 + ACTION_SPACE)  # ACTION_SPACE = 46

    def forward(self, phi, mask):
        v, a = self.net(phi).split((1, ACTION_SPACE), dim=-1)  # DuDQN
        a_sum = a.masked_fill(~mask, 0.).sum(-1, keepdim=True)
        a_mean = a_sum / mask.sum(-1, keepdim=True)
        q = (v + a - a_mean).masked_fill(~mask, -torch.inf)
        return q
```

V4 用的 Brain 是 ResNet + ChannelAttention，v4 特有的是：
- BatchNorm momentum=0.01, eps=1e-3
- DuDQN head（value + advantage 分离）
- v4 的 `obs_shape` = (1012, 34)，比 v3 的 (934, 34) 大，多出来的维度主要编码 expected value (EV) table 信息

**关键**：mask 机制保证无效的 chi 类型不会被选中 —— 即使 policy head 给出了一个 Q 值，只要 `mask[38/39/40] = false`，该动作就会被标记为 `-inf` 不会被执行。