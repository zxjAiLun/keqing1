Mortal 有完整的 yaku 检测系统，但主要用在**得分计算**而非动作选择上。

---

## Yaku 检测实现

核心在 `libriichi/src/alalgo/agari.rs`，主要结构是 `AgariCalculator`：

```rust
pub fn search_yakus<const RETURN_IF_ANY: bool>(&self) -> Option<Agari> {
    // 返回 None (无役) 或 Some(Agari::Normal { fu, han }) 或 Some(Agari::Yakuman(n))
}
```

它用了一个预计算的 `agari.bin.gz` 表（来自 `mahjong-helper`）来做标准型的判断，然后对特殊型（国士、七对子、九莲宝燈等）做专项检测。

主要检测的 yaku：

| Yaku | 检测逻辑 |
|------|---------|
| 平和 (Pinfu) | menzen_shuntsu.len()==4 && 顺子边张搭子条件 |
| 断幺九 (Tanyao) | 所有面子/对子都是数牌1-8，不含幺九牌 |
| 七对子 (Chiitoitsu) | div.has_chitoi |
| 一杯口 (Iipeikou) | 两个相同顺子 |
| 二杯口 (Ryanpeikou) | has_ryanpeikou |
| 混一色/清一色 | isou_kind + 字牌有无 |
| 三色同顺/同刻 | s_counter/k_counter 检测 |
| 对对和 (Toitoi) | 无顺子 && 无吃 |
| 三暗刻/四暗刻 | ankous_count |
| 三槓子/四槓子 | kans_count |
| 役牌/大三元/大四喜 | 字牌刻子检测 |
| 国士无双 | shanten::calc_kokushi() |
| 九莲宝燈 | div.has_chuuren |
| 绿一色 | 全是索子23468+字牌F |
| 混老頭/清老頭 | 全部是幺九牌 |
| ... | 还有很多 |

---

## 在训练中如何使用

### Observation 中的 EV 编码

`obs_repr.rs` 第 613-619 行：

```rust
// Use the minimal tsumo agari point as the max EV. It is
// ...actually used to make the EV less optimistic.
let min_tsumo_agari = state.agari_points(cans.can_ron_agari, &[])?;
self.encode_ev(min_tsumo_agari);
```

`agari_points()` 最终调用 `AgariCalculator` 的 `agari()` 方法，返回 `Point`（含 fu、han、yakuman），用于编码 **期望收益** 到 observation 中。

所以 yaku 信息通过 **EV (Expected Value)** 间接影响模型训练 —— EV 越高，模型越倾向于做能成胡的决策。

### Reward 计算

`reward_calculator.py` 的 `calc_delta_pt`：

```python
def calc_delta_pt(self, player_id, grp_feature, rank_by_player):
    rank_prob = self.calc_rank_prob(...)  # 基于GRP网络的排名预测
    exp_pts = rank_prob @ self.pts       # [3, 1, -1, -3] 的期望排名分
    reward = exp_pts[1:] - exp_pts[:-1]  # 每步的阶段reward
```

**排名 reward**，不是点值 reward。所以 yaku 本身（断幺vs平和vs一杯口）不影响 reward —— 只要最后**赢**了就行。

### Action Space 不变

Action 43 = agari，无论有没有役、役有多少。所以模型不知道也不关心你成的是什么 yaku。

---

## 总结

- **Yaku 检测**：完整实现，用于计算最终得分和 EV
- **Observation**：EV 编码了当前手牌的理论得分上限（受 yaku 影响）
- **Reward**：基于排名，不是具体 yaku
- **Action**：不区分 yaku 种类，43 号动作就是"和了"

也就是说，模型学到的是"什么时候该荣/自摸"，但学不到"断幺平和比一杯口更好"这种价值判断 —— 这由 `reward_calculator` 的排名机制决定，而不是 yaku 本身。