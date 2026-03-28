# 2026-03-22 修改总结

## 一、今日修改

### 1. features.py 索引 16 - 非役字牌倾向

**位置**：`src/v4model/features.py` 索引 16

**逻辑**：无人立直时才考虑打出字牌

| 条件 | 倾向值 |
|------|--------|
| 早巡 + 字牌非宝牌 | 0.95 (极高) |
| 早巡 + 字牌是宝牌 | 0.05 (极低) |
| 尾盘 + 手持>=2张字牌 | 0.1 (极低) |
| 其他巡目 + 向听<=2 | 0.9 (极高) |
| 其他 | 0.5 |
| 有人立直 | 0.0 |

### 2. features.py 索引 19 - 行动模式

**位置**：`src/v4model/features.py` 索引 19

**逻辑**：

| 条件 | 模式 | 值 |
|------|------|-----|
| 有人立直 + 自己向听>2 | 防守 | 1.0 |
| 其他家fuuro宝牌>=3 | 防守 | 1.0 |
| 早巡 或 自己宝牌>=2张 | 进攻 | 0.5 |
| 其他 | 平衡 | 0.0 |

### 3. features.py ukeire/speedRef 重构

- 重构了 `_calc_ukeire` 方法，正确区分13张/14张手牌
- 实现了 `_calc_ukeire_after_discard` 计算打牌后进张数
- 实现了 `_calc_best_ukeire_for_14tile` 计算14张手牌的最佳进张数
- 向听依赖的归一化：1向听系数20，2向听系数30，3+向听系数50

---

## 二、问题分析与解决方案

### 问题1 (对应 thought 22-25行)：如何让模型不轻易拆面子/雀头

**背景**：用户希望通过speed_ref奖励/惩罚机制，让模型学会"不轻易拆掉面子/雀头"

**解决方案：Reward Shaping + 训练数据过滤**

#### 方案A：训练样本过滤（推荐）

在 `build_supervised_samples` 或 `actor_filter` 阶段：

```python
def filter_samples(events):
    filtered = []
    for event in events:
        if event.type == 'discard':
            # 计算原14张向听数
            shanten_before = calculate_shanten(event.hand_14)
            # 计算打出的牌
            discarded = event.discarded_tile
            hand_13 = [t for t in event.hand_14 if t != discarded]
            shanten_after = calculate_shanten(hand_13)

            # 如果向听倒退，降低权重或过滤
            if shanten_after > shanten_before:
                event.weight *= 0.5  # 降低权重
                # 或直接跳过: continue
        filtered.append(event)
    return filtered
```

#### 方案B：Reward Shaping（用于强化学习训练）

```python
def compute_reward(action, obs_before, obs_after):
    # 基础reward
    reward = 0

    # 1. Speed Ref 奖励/惩罚 (仅进攻模式)
    if obs.mode == 'attack':
        speed_before = obs.speed_ref
        speed_after = compute_speed_ref(obs_after.hand)
        reward += (speed_after - speed_before) * speed_weight  # 0.1~0.3

    # 2. 防守模式惩罚
    elif obs.mode == 'defense':
        # 打出立直家打过的牌 -> 正reward
        if action.tile in obs.riichi_player_discards:
            reward += defense_reward
        # 放铳 -> 大量负reward
        if action.is_ron:
            reward -= big_penalty

    return reward
```

#### 方案C：辅助Loss（监督学习）

在原有SL loss基础上添加辅助loss：

```python
def auxiliary_loss(pred_discard, true_discard, hand_14, shanten_before):
    """
    如果打出的牌导致向听倒退，增加额外loss
    """
    hand_after = remove_tile(hand_14, true_discard)
    shanten_after = calculate_shanten(hand_after)

    if shanten_after > shanten_before:
        # 向听倒退惩罚
        return ce_loss * 2.0  # 加倍惩罚
    return ce_loss
```

---

### 问题2 (对应 thought 26-31行)：模型做牌/附录判断问题

**问题描述**：
1. 模型不会判断向听数，不会判断手里的牌有没有向胡牌推进
2. 附录判断完全不对

**解决方案**：

#### 1. 特征增强

已在索引15、18中添加：
- 索引15：`ukeire` (有效进张数) - 衡量"进张"能力
- 索引18：`speed_ref` - 衡量"速度"

#### 2. 附录( Fuuro ) 倾向控制

**方案**：降低附录相关特征的初始值，让模型"默认不附录"，只在特定条件下学习附录：

```python
# 索引19目前设为0（暂不使用）

# 未来可添加更精细的判断：
# - 役牌刻子（自风/场风/三元牌）
# - 断幺九（tenpai时）
# - 早期做大国的时间窗口
```

#### 3. 七对子/国士无双判断

当前 `calculate_shanten` 可能未考虑七对子/国士，需要确认：

```python
# 检查riichienv的calculate_shanten是否支持
from riichienv import calculate_shanten
# 如果不支持，需要单独实现
```

---

## 三、TODO

### 高优先级

- [ ] **P0**: 验证 `riichienv.calculate_shanten` 是否支持七对子/国士无双向听计算
- [ ] **P0**: 实现训练样本过滤：过滤/降权"向听倒退"的打牌样本
- [ ] **P0**: 实现 speed_ref 辅助loss，确保模型学会"速度管理"

### 中优先级

- [ ] **P1**: 实现防守模式的 reward shaping（立直家打过的牌 +放铳惩罚）
- [ ] **P1**: 添加役牌刻子检测（索引19相关）
- [ ] **P1**: 实现 `ukeire` 的"安全打牌"比例特征（拆面子检测）

### 低优先级

- [ ] **P2**: 七对子/国士无双向听特征
- [ ] **P2**: 断幺九检测
- [ ] **P2**: 场风/自风/三元牌役牌检测

---

## 四、附录：特征索引对照表

| 索引 | 名称 | 说明 |
|------|------|------|
| 13 | 向听数 | riichienv.calculate_shanten |
| 15 | 有效进张数 | 最佳进张数，向听依赖归一化 |
| 16 | 非役字牌倾向 | 无人立直时打字的倾向 |
| 17 | 胡牌概率 | 占位 (0.0) |
| 18 | Speed Reference | 速度参考，归一化 |
| 19 | 行动模式 | 0=平衡, 0.5=进攻, 1=防守 |
