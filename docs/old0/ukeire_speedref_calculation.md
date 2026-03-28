# 麻将进张数（Ukeire）和速度参考（Speed Reference）计算逻辑

## 参考来源

- [cal-shanten-beta/src/scripts/Ukeire.js](https://github.com/garyleung142857/cal-shanten-beta/blob/master/src/scripts/Ukeire.js)
- [mahjong-tile-efficiency](https://github.com/garyleung142857/mahjong-tile-efficiency)

---

## 1. 核心数据结构

### 手牌格式

```javascript
hand = [
  [1,1,1,1,0,0,0,0,0],  // 万子 1-9
  [0,0,0,0,0,0,0,0,0],  // 筒子 1-9
  [0,0,0,0,0,0,0,0,0],  // 索子 1-9
  [0,0,0,0,0,0,0]       // 字牌 东南西北白发中
]
// 每种牌最多4张
```

### FULLSET 常量

```javascript
FULLSET = [
  [4,4,4,4,4,4,4,4,4],  // 万子每种4张
  [4,4,4,4,4,4,4,4,4],  // 筒子
  [4,4,4,4,4,4,4,4,4],  // 索子
  [4,4,4,4,4,4,4]       // 字牌
]
```

---

## 2. 进张数（Ukeire）计算

### 2.1 摸牌阶段（3n+1张手牌）

```javascript
const ukeire1 = (hand) => {
  let ukeireList = []
  let totalUkeire = 0
  const originalShanten = calRule(hand)

  for (let i = 0; i < 4; i++){
    for (let j = 0; j < FULLSET[i].length; j++){
      const remainingCount = FULLSET[i][j] - hand[i][j]  // 剩余张数
      if (remainingCount > 0){
        hand[i][j]++                           // 模拟摸入
        const newShanten = calRule(hand)
        hand[i][j]--                           // 恢复
        if (newShanten < originalShanten){     // 能降低向听
          ukeireList.push(tileNames[i][j])
          totalUkeire += remainingCount
        }
      }
    }
  }
  return {
    ukeireList,
    totalUkeire,  // Acceptance / Waiting 总数
    shanten: originalShanten,
  }
}
```

**Acceptance / Waiting** = `totalUkeire`，即所有能降低向听数的牌的剩余数量之和

**Acceptance List** = `ukeireList`，能进张的牌的列表

### 2.2 打牌阶段（3n+2张手牌）

```javascript
const ukeire2 = (hand) => {
  const originalShanten = calRule(hand)
  let bestUkeire = 0

  for (let i = 0; i < 4; i++){
    for (let j = 0; j < FULLSET[i].length; j++){
      if(hand[i][j] > 0){
        hand[i][j]--                          // 模拟打出
        const newUkeire = ukeire1(hand)
        if(newUkeire.shanten == originalShanten && newUkeire.totalUkeire > bestUkeire){
          bestUkeire = newUkeire.totalUkeire
        }
        hand[i][j]++                          // 恢复
      }
    }
  }
  return {
    best: bestUkeire,  // 最佳进张数
    shanten: originalShanten,
  }
}
```

---

## 3. 速度参考（Speed Reference）计算

### 3.1 核心公式

```javascript
const speedRef = (ukeire, avgNextUkeire, leftTurns) => {
  if (ukeire == 0 || avgNextUkeire == 0){
    return 0
  } else {
    const leftCount = 120                    // 剩余牌堆估算
    const p2 = ukeire / leftCount           // 摸到进张牌的概率
    const p1 = avgNextUkeire / leftCount    // 摸到改进牌的概率
    const q2 = 1 - p2
    const q1 = 1 - p1
    // 在leftTurns次摸牌中至少进步一次的概率
    const result = 1 - (p2 * Math.pow(q1, leftTurns) - p1 * Math.pow(q2, leftTurns)) / (q1 - q2)
    return result * 100                     // 转为百分比
  }
}
```

### 3.2 参数说明

| 参数 | 含义 | 来源 |
|------|------|------|
| `ukeire` | 当前进张总数 | `ukeire1().totalUkeire` |
| `avgNextUkeire` | 进步后的平均进张数 | 见下方计算 |
| `leftCount` | 剩余牌堆估算值 | 固定 120 |
| `leftTurns` | 剩余摸牌次数 | 1向听=10, 2向听=3 |

### 3.3 avgNextUkeire 的计算

在 `analyze1` 函数中：

```javascript
let nextShantenTiles = 0
let nextShantenUkeire = 0

for (每张未进张牌){
  hand[i][j]++
  const newUkeire = ukeire2(hand)  // 计算打这张后的最佳进张
  hand[i][j]--

  if (newUkeire.shanten < originalShanten){
    // 这是能降低向听的进张牌
    nextShantenTiles += remainingCount
    nextShantenUkeire += remainingCount * newUkeire.best
  }
}

const avgNextUkeire = nextShantenUkeire / nextShantenTiles
```

---

## 4. 完整分析函数 analyze1

```javascript
const analyze1 = (hand) => {
  const thisUkeire = ukeire1(hand)
  const originalShanten = thisUkeire.shanten

  let totalTiles = 0
  let totalUkeire = 0
  let ukeireImprovement = {}   // Improvement List
  let ukeireList = {}          // Acceptance List
  let nextShantenTiles = 0
  let nextShantenUkeire = 0

  for (每种牌){
    if (remainingCount > 0){
      hand[i][j]++
      const newUkeire = ukeire2(hand)
      hand[i][j]--

      if (newUkeire.shanten == originalShanten){
        // 不能进张，但可能改进
        totalTiles += remainingCount
        totalUkeire += remainingCount * newUkeire.best
        if (newUkeire.best > thisUkeire.totalUkeire){
          ukeireImprovement[tileNames[i][j]] = newUkeire.best
        }
      } else if (newUkeire.shanten < originalShanten){
        // 能进张
        ukeireList[tileNames[i][j]] = newUkeire.best
        nextShantenTiles += remainingCount
        nextShantenUkeire += remainingCount * newUkeire.best
      }
    }
  }

  const avgNextUkeire = nextShantenUkeire / nextShantenTiles
  let speed = null
  if (originalShanten == 1){
    speed = speedRef(thisUkeire.totalUkeire, avgNextUkeire, 10)
  } else if (originalShanten == 2){
    speed = speedRef(thisUkeire.totalUkeire, avgNextUkeire, 3)
  }

  return {
    shanten: originalShanten,
    improvedUkeire: ukeireImprovement,    // Improvement List
    ukeire: thisUkeire.totalUkeire,       // Acceptance / Waiting
    ukeireList: ukeireList,               // Acceptance List
    avgWithImprovement: totalUkeire / totalTiles,
    avgNextUkeire: avgNextUkeire,
    speedRef: speed                        // 百分比
  }
}
```

---

## 5. 返回值说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `shanten` | number | 当前向听数 |
| `ukeire` | number | Acceptance / Waiting（总进张数） |
| `ukeireList` | object | Acceptance List `{tileName: bestUkeire数}` |
| `improvedUkeire` | object | Improvement List `{tileName: 改进后的bestUkeire}` |
| `avgNextUkeire` | number | 进步后的平均进张数 |
| `avgWithImprovement` | number | 包含改进牌在内的平均进张 |
| `speedRef` | number | Speed Reference（百分比），仅1-2向听时有值 |

---

## 6. Speed Reference 的含义

| 向听级别 | leftTurns | 含义 |
|----------|-----------|------|
| 1-向听 | 10 | 在10次摸牌内和牌的概率 |
| 2-向听 | 3 | 在3次摸牌内达到听牌的概率 |

---

## 7. 排序函数 sortFunc

```javascript
const sortFunc = (a, b) => {
  if (a.shanten == b.shanten){
    if (a.speedRef == null || b.speedRef == null || a.speedRef == b.speedRef){
      if (a.ukeire == b.ukeire){
        if (a.avgWithImprovement == b.avgWithImprovement){
          return a.avgNextUkeire > b.avgNextUkeire ? -1 : 1
        }
        return a.avgWithImprovement > b.avgWithImprovement ? -1 : 1
      }
      return a.ukeire > b.ukeire ? -1 : 1
    }
    return a.speedRef > b.speedRef ? -1 : 1
  }
  return a.shanten > b.shanten ? 1 : -1
}
```

排序优先级：
1. **向听数**：升序（越小越好）
2. **Speed Reference**：降序
3. **Ukeire**：降序
4. **AvgWithImprovement**：降序
5. **AvgNextUkeire**：降序

---

## 8. 关键公式速查

```
进张数 = Σ (每张进张牌的剩余数量)

avgNextUkeire = nextShantenUkeire / nextShantenTiles

p2 = ukeire / 120
p1 = avgNextUkeire / 120
q1 = 1 - p1
q2 = 1 - p2

speedRef = (1 - (p2 * q1^n - p1 * q2^n) / (q1 - q2)) * 100
其中 n = 10 (1-向听) 或 n = 3 (2-向听)
```
