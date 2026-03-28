# Mortal 项目深度分析报告

## 项目概述

Mortal（凡夫）是一个基于深度强化学习的日本麻将AI，由 Equim-chan 开发。这是一个混合 Rust + Python 项目，核心游戏逻辑和批量推理用 Rust 编写（libriichi），神经网络模型用 Python 编写（mortal）。项目采用 AGPL-3.0 开源协议。

---

## 一、项目架构总览

### 1.1 技术栈

| 层次 | 技术选型 | 说明 |
|------|----------|------|
| 游戏引擎 | Rust | libriichi crate，负责麻将规则、状态管理、批量对局 |
| Python绑定 | pyo3 | Rust向Python暴露接口 |
| 神经网络 | Python + PyTorch | 模型定义、训练、数据增强 |
| 通信层 | TCP Socket | 在线训练时的参数分发与数据回收 |
| 并行计算 | rayon (Rust) + PyTorch DataLoader | 多线程数据加载 |

### 1.2 目录结构

```
Mortal/
├── libriichi/          # Rust核心库
│   └── src/
│       ├── agent/         # AI代理实现（mortal/akochan/tsumogiri/mjai_log）
│       ├── algo/          # 算法模块（向听数/和牌/点值计算/单人牌效）
│       ├── arena/         # 对局管理（batch game, 1v3, 2v2）
│       ├── dataset/       # 数据集加载（gameplay, grp, invisible）
│       ├── state/         # 玩家状态与观察编码
│       ├── mjai/          # mjai协议事件解析
│       └── ...
├── mortal/             # Python训练/推理模块
│   ├── model.py           # 神经网络模型定义
│   ├── train.py          # 主训练流程
│   ├── train_grp.py      # GRP(Grand Rank Predictor)训练
│   ├── dataloader.py     # 数据加载迭代器
│   ├── engine.py         # Python端推理引擎
│   ├── server.py         # 在线训练参数服务器
│   ├── client.py         # 在线训练工作客户端
│   ├── player.py         # 测试/训练对战玩家
│   └── ...
└── docs/               # 文档
```

---

## 二、模型设计详解

### 2.1 模型架构概览

Mortal使用**三模块架构**：Brain（编码器）+ DQN（价值评估）+ AuxNet（辅助排名预测）。GRP（Grand Rank Predictor）作为独立模块用于奖励计算。

#### Brain（编码器）

负责将麻将观察状态编码为隐向量（latent representation）。

```python
# 核心结构
class Brain(nn.Module):
    def __init__(self, *, conv_channels, num_blocks, is_oracle=False, version=1):
        # 1. ResNet编码器
        self.encoder = ResNet(in_channels, conv_channels, num_blocks)
        # 2. 版本相关变换
        # version 1: 输出 mu, logsig（VAE风格）
        # version 2/3/4: 直接输出 actv(phi)，即1024维隐向量
```

**文件来源**：`mortal/model.py` 第108-187行

关键组件：

1. **ResBlock**：残差块，包含Conv1d + BatchNorm + Mish/ReLU + ChannelAttention
2. **ChannelAttention**：通道注意力机制（avg + max pooling拼接）
3. **ResNet**：多层ResBlock堆叠，输出固定维度(32*34=1088)的特征向量，再接Linear(1088->1024)

**Brain与DQN的对齐方式**：Brain输出的1024维向量直接作为DQN的输入。

```python
# Brain输出维度 (mortal/model.py 第92-106行)
layers += [
    nn.Conv1d(conv_channels, 32, kernel_size=3, padding=1),  # 32通道
    nn.Flatten(),
    nn.Linear(32 * 34, 1024),  # 32*34=1088 → 1024
]
# v1返回512维(via latent_net)，v2/3/4返回1024维

# DQN接收1024维输入 (mortal/model.py 第221-231行)
def forward(self, phi, mask):
    # phi: Brain输出的1024维向量
    v = self.v_head(phi)  # 1024 → hidden_size (512或256)
    a = self.a_head(phi)   # 1024 → hidden_size → 46
    q = (v + a - a.mean()).masked_fill(~mask, -torch.inf)
    return q
```

#### DQN（价值网络）

接收Brain的输出（phi），输出每个动作的Q值。

```python
class DQN(nn.Module):
    def forward(self, phi, mask):
        # 动作空间共46个（37种打出 + 1立直 + 3吃 + 1碰 + 1杠 + 1和 + 1流局 + 1跳过）
        # Q(s,a) = V(s) + A(s,a) - mean(A(s,a'))
        v = self.v_head(phi)
        a = self.a_head(phi)
        q = (v + a - a.mean()).masked_fill(~mask, -torch.inf)
        return q
```

**文件来源**：`mortal/model.py` 第197-231行

#### GRP（Grand Rank Predictor，排名预测器）

**独立模块**，用于预测游戏最终排名。注意：**GRP与Brain/DQN是分开独立训练的**。

```python
class GRP(nn.Module):
    def __init__(self, hidden_size=64, num_layers=2):
        self.rnn = nn.GRU(input_size=GRP_SIZE, hidden_size=64, num_layers=2)
        # GRP_SIZE = 7，对应7维特征
        # 输入序列：[grand_kyoku, honba, kyotaku, s[0], s[1], s[2], s[3]]
```

**文件来源**：`mortal/model.py` 第233-286行

**GRP的输入特征**（`libriichi/src/dataset/grp.rs` 第134-145行）：

```rust
let mut kyoku_info = array_vec!([_; GRP_SIZE]);
kyoku_info.push(grand_kyoku as f64);      // 索引0: 局次(0-11)
kyoku_info.push(honba as f64);            // 索引1: 本场数
kyoku_info.push(kyotaku as f64);          // 索引2: 场供
kyoku_info.extend(scores.iter().map(|&score| score as f64 / 10000.));
// 索引3-6: s[0], s[1], s[2], s[3] - 四位玩家的分数/10000（归一化）
```

**s[0]~s[3]的含义**：代表**四位玩家在当前局的分数**（除以10000归一化后的值）。玩家0是起始时的亲家（东一局）。

**核心思想**：GRP接收每局开始时的状态序列（每行7维：[局次, 本场, 场供, 玩家0分, 玩家1分, 玩家2分, 玩家3分]），预测24种排名排列的概率分布。

#### AuxNet（辅助排名预测）

与Brain/DQN一起训练，共享phi特征，输出4维排名预测。

```python
class AuxNet(nn.Module):
    def __init__(self, dims=None):
        self.net = nn.Linear(1024, sum(dims), bias=False)  # 输出4维
```

**文件来源**：`mortal/model.py` 第188-195行

### 2.2 观察空间编码（Observation Encoding）

这是理解Mortal的关键。`libriichi/src/state/obs_repr.rs`定义了如何将麻将状态编码为神经网络输入。

#### 通道(Channel)的含义

在Mortal的代码中，"通道(channel)"有两种不同含义：

1. **观察编码中的"通道"**：指2D编码数组的**行索引**（每行34维，对应34种麻将牌）
2. **卷积网络中的通道**：指`nn.Conv1d`的输入/输出通道数

```rust
// libriichi/src/state/obs_repr.rs 第18-25行
struct ObsEncoderContext<'a> {
    state: &'a PlayerState,
    arr: Simple2DArray<34, f32>,  // 34通道 = 34种麻将牌
    mask: Array1<bool>,
    idx: usize,  // 当前编码到的行索引
}
```

编码按**行**组织，每行34维（34种牌：万子1-9、筒子1-9、索子1-9、文字1-7）。

#### 版本演进

| 版本 | 观察shape | 主要变化 |
|------|-----------|----------|
| v1 | (938, 34) | 基础版本，使用VAE潜在空间 |
| v2 | (942, 34) | 增加RBF编码 |
| v3 | (934, 34) | 优化编码效率 |
| v4 | (1012, 34) | **加入单人牌效表(SP Table)**，最复杂 |

#### v4版本的编码细节

观察编码按行组织，每行34维（对应34种麻将牌）。**但这里存在一个重要的设计选择**。

##### Simple2DArray的fill机制与数据冗余

关键代码在 `libriichi/src/array.rs`：

```rust
// 第29-31行：fill会填充整行所有COLS列
pub fn fill(&mut self, row: usize, value: T) {
    self.fill_rows(row, 1, value);
}

// 第37-39行：实际填充 [row*COLS, (row+1)*COLS) 的所有元素
pub fn fill_rows(&mut self, row: usize, n_rows: usize, value: T) {
    self.arr[row * COLS..(row + n_rows) * COLS].fill(value);
}
```

这意味着：**所有特征都被"广播"到整行34列**，即使该特征只需要1列。

##### 分数编码示例

```rust
// obs_repr.rs 第149-152行
for &score in &state.scores {
    let v = score.clamp(0, 100_000) as f32 / 100000.;
    self.arr.fill(self.idx, v);  // 整行34列都填充相同的分数值！
    self.idx += 1;
}
```

假设分数是 `[30000, 25000, 25000, 20000]`（归一化后 `[0.3, 0.25, 0.25, 0.2]`）：

**实际存储**（虽然只有4行，但每行34列都存储相同值）：
```
第0行: [0.3, 0.3, 0.3, 0.3, ... x34]  ← 30000分被复制到34列
第1行: [0.25, 0.25, 0.25, 0.25, ... x34]  ← 25000分被复制到34列
第2行: [0.25, 0.25, 0.25, 0.25, ... x34]
第3行: [0.2, 0.2, 0.2, 0.2, ... x34]
```

**问题**：
- 信息密度低：34列中只有第0列有效
- 存储冗余：每个标量值被复制34次
- 计算浪费：后续卷积需要对34列都进行处理

**好处**：
- 编码接口统一：所有特征都是 `(N, 34)` 形状
- 代码简洁：不需要区分"标量特征"和"麻将牌特征"
- Conv1d 可以用同样的方式处理所有特征

##### 编码细节（修正版）

```
===== 基础特征 =====
手牌: 4行 × 34列 (每种牌最多4张，每行只有1列有效)
赤宝牌指示: 3行 × 34列 (每行只有1列有效)
分数: 4行 × 34列 (每行只有1列有效，但fill了整行)
排名: 4行 × 34列 (one-hot，每行只有1列有效)
局数: 4行 × 34列 (one-hot，每行只有1列有效)
本场: 3行 × 34列 (RBF编码，每行只有1列有效，但fill了整行)
场供: 3行 × 34列 (RBF编码，每行只有1列有效，但fill了整行)
宝牌指示: 7行 × 34列 (每行只有1-4列有效)
自风/场风: 2行 × 34列 (每行只有1列有效)

===== 舍牌特征 =====
自舍牌: 6张 × 4行 = 24行 × 34列
自舍牌(倒序): 18张 × 4行 = 72行 × 34列
他家舍牌: 3家 × (6张+18张) × 8行 = 576行 × 34列

===== 状态特征 =====
剩余牌数: 1行 × 34列 (每行只有1列有效，但fill了整行)
自摸损失: 3行 × 34列
未见宝牌: 4行 × 34列
舍牌概要: 4家 × 7行 = 28行 × 34列
副露概要: 4家 × 5行 = 20行 × 34列
暗杠概要: 4家 × 1行 = 4行 × 34列
牌山见过: 34行 × 34列 (每个tile一行)
他家最后舍牌: 3家 × 3行 = 9行 × 34列
他家立直声明: 3行 × 34列
他家流局听牌: 3行 × 34列

===== 行动相关 =====
向听数: 6行 × 34列 (one-hot)
舍牌候选: 5行 × 34列
立直/吃/碰/杠/和/流局/跳过: 各1-3行 × 34列

===== v4独有：单人牌效表(SP Table) =====
最大期望值: 2行 × 34列
必需牌编码: 2×34行 × 34列
最优打牌: 2行 × 34列
弃牌表: 3 × 17行 × 34列
自摸表: 3 × 17行 × 34列

##### v4 SP Table详解

**为什么v4最强？** 因为它将**专家知识**（单人牌效计算）编码进了观察空间，让神经网络可以直接"看到"每个打牌的期望值。

**SP Table的编码逻辑** (`libriichi/src/state/obs_repr.rs` 第564-624行)：

```rust
// 1. 获取单人牌效计算结果
if let Ok(SinglePlayerTables { max_ev_table }) = state.single_player_tables() {
    // 2. 编码最大期望值（2通道）
    let max_ev = max_ev_table.first()
        .and_then(|c| c.exp_values.first().copied())
        .unwrap_or_default();
    self.encode_ev(max_ev);  // 填充2个值：max_ev/100000, max_ev/30000

    // 3. 编码必需牌
    // 对于每个候选打牌，标记其听牌需要的牌
    if cans.can_discard {
        for candidate in &max_ev_table {
            let discard_tid = candidate.tile.deaka().as_usize();
            for r in &candidate.required_tiles {
                let required_tid = r.tile.deaka().as_usize();
                // shanten_down=true: 从索引34开始编码
                // shanten_down=false: 从索引0开始编码
                self.arr.assign(self.idx + 34 + discard_tid, required_tid, 1.);
            }
        }
        self.idx += 2 * 34;  // 2*34通道
    }

    // 4. 编码弃牌表和自摸表
    // 每个候选打牌有17个值：[听牌率, 自摸率, 期望值] × 17巡
    for candidate in candidates {
        for (turn, ...) in enumerate(...) {
            // t步的听牌率、自摸率、期望值
            self.arr.assign(idx + 0, tid, tenpai_prob);
            self.arr.assign(idx + 17, tid, win_prob);
            self.arr.assign(idx + 34, tid, exp_value);
        }
    }
}
```

**SP Table的实际数据示例**：

假设手牌是 `45678m 34789p 3344z`，最优打 `4m`：

```
候选打牌: 4m (向听数不变)
├── 必需牌: 2m×4, 5m×4, ... (14种牌，共57张)
├── 听牌率: [0%, 8%, 15%, 22%, 28%, ...] (17巡)
├── 自摸率: [0%, 0.5%, 1.2%, 2.1%, 3.2%, ...] (17巡)
└── 期望值: [0, 120, 380, 780, 1350, ...] (17巡)

候选打牌: 7m (向听数+1，变差)
├── 必需牌: ... (可能更多)
├── 听牌率: ...
└── 期望值: ...
```

**v4网络的感知能力**：

神经网络可以"看到"：
1. 每种打牌的**当前期望值**（第一列）
2. 每种打牌的**长期听牌/自摸潜力**（后16列）
3. **不同打牌之间的EV差异**（让网络学习何时该追求高EV，何时该保守）

### 2.3 Oracle（神谕）观察

当`is_oracle=True`时，模型额外接收**不可观察信息**的编码：
- 剩余牌山（每种牌的剩余数量）
- 岭上牌
- 宝牌指示和里宝牌指示
- **他家手牌**（完全信息）

Oracle observation shape: v1=(211, 34), v2/v3/v4=(217, 34)

---

## 三、训练方法论

### 3.1 离线训练（Offline Training）

#### 数据来源

使用mjai协议格式的`.json.gz`压缩日志文件，包含完整对局记录。

#### 数据增强

```python
# mortal/dataloader.py
if self.enable_augmentation:
    yield from self.load_files(not self.augmented_first)
```

增强方式（`Event::augment` in Rust）：
- 牌面变换（万子/筒子/索子之间的对称性）
- 座位旋转

#### 损失函数设计（文件来源：mortal/train.py）

**1. DQN Loss（第235行）**
```python
dqn_loss = 0.5 * mse(q, q_target_mc)  # 主损失：Q-Learning损失
```

**2. CQL Loss（第236-238行）**
```python
cql_loss = 0
if not online:
    cql_loss = q_out.logsumexp(-1).mean() - q.mean()
```
> 仅离线训练时使用，源自CQL论文，用于缓解Q值过估计。

**3. AuxNet Loss / next_rank_loss（第240-241行）**
```python
next_rank_logits, = aux_net(phi)
next_rank_loss = ce(next_rank_logits, player_ranks)  # CrossEntropyLoss
```

**总损失**：
```python
loss = sum((
    dqn_loss,
    cql_loss * min_q_weight,
    next_rank_loss * next_rank_weight,
))
```

**4. GRP Loss（文件来源：mortal/train_grp.py 第178行）**
```python
loss = F.cross_entropy(logits, labels)  # GRP独立训练，使用交叉熵损失
```

| Loss名称 | 定义文件 | 计算位置 | 损失类型 | 用途 |
|----------|----------|----------|----------|------|
| `dqn_loss` | `mortal/train.py:235` | `train_batch()` | MSE | 主Q-Learning损失 |
| `cql_loss` | `mortal/train.py:238` | `train_batch()` | CQL公式 | Q值过估计正则化 |
| `next_rank_loss` | `mortal/train.py:241` | `train_batch()` | CrossEntropy | 辅助排名预测 |
| `grp_loss` | `mortal/train_grp.py:178` | `train()` | CrossEntropy | GRP排名预测 |

#### 目标值计算

```python
# 使用蒙特卡洛回报
q_target_mc = gamma ** steps_to_done * kyoku_rewards
```

其中`kyoku_rewards`由**RewardCalculator**计算：
1. GRP网络预测当前排名概率分布
2. 结合排名点值[90, 45, 0, -135]计算期望得分
3. 相邻状态的期望得分差作为即时奖励

### 3.2 在线训练（Online Training）

#### 架构

```
┌─────────────────┐      ┌─────────────────┐
│   Parameter     │      │   Client Workers│
│   Server       │◄────►│   (self-play)   │
│  (server.py)   │      │  (client.py)    │
└─────────────────┘      └─────────────────┘
```

#### 流程

1. Server启动，监听worker连接
2. Trainer定期提交新参数到Server
3. Worker从Server获取最新参数
4. Worker运行自对战（1个Mortal vs 3个Baseline）
5. Worker将对局日志提交到Server的buffer
6. Trainer从buffer drain数据用于训练
7. 循环

### 3.3 GRP训练

**GRP与Brain/DQN是分开独立训练的**，使用不同的训练脚本和优化器。

**文件来源**：`mortal/train_grp.py`

```python
# train_grp.py 第83-182行
def train():
    grp = GRP(**cfg['network']).to(device)
    optimizer = optim.AdamW(grp.parameters())  # 独立优化器

    for inputs, rank_by_players in train_data_loader:
        logits = grp.forward_packed(inputs)
        labels = grp.get_label(rank_by_player)  # 将排名映射到24种排列
        loss = F.cross_entropy(logits, labels)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
```

| 对比项 | GRP | Brain/DQN + AuxNet |
|--------|-----|---------------------|
| 训练脚本 | `train_grp.py` | `train.py` |
| 优化器 | 独立 | 与AuxNet共享 |
| 损失函数 | `F.cross_entropy` | `mse + CQL + CrossEntropy` |
| 输入 | 每局7维状态序列 | 每步34xN观察 |
| 输出 | 24种排名排列 | 46个动作Q值 |

**训练顺序**：通常先独立训练GRP（用于奖励计算），再训练Brain/DQN。

### 3.4 训练配置示例

```toml
[control]
version = 4
batch_size = 512
opt_step_every = 1  # 每步优化

[resnet]
conv_channels = 192
num_blocks = 40

[env]
gamma = 1  # 无折扣因子（麻将为片段游戏）
pts = [90, 45, 0, -135]  # 排名奖励（用于计算期望得分）

[cql]
min_q_weight = 5  # CQL损失权重

[aux]
next_rank_weight = 0.2  # 排名预测损失权重

[optim]
weight_decay = 0.1
max_grad_norm = 0  # 无梯度裁剪
```

---

## 四、Rust核心模块解析

### 4.1 状态管理（PlayerState）

`libriichi/src/state/player_state.rs`是核心数据结构，存储从特定玩家视角看到的所有游戏状态。

#### PlayerState完整字段解析

**文件来源**：`libriichi/src/state/player_state.rs` 第24-140行

```rust
pub struct PlayerState {
    // ===== 基础标识 =====
    player_id: u8,              // 玩家ID (0-3)，0=东家

    // ===== 手牌相关 =====
    tehai: [u8; 34],          // 手牌（不含赤宝牌），索引0-8=万子，9-17=筒子，18-26=索子，27-33=文字
    akas_in_hand: [bool; 3],   // 手牌中的赤宝牌 [5mr, 5pr, 5sr]
    tehai_len_div3: u8,        // 手牌张数/3，用于向听数计算
    doras_owned: [u8; 4],      // 拥有的宝牌数（含自摸宝牌）
    doras_seen: u8,            // 已见宝牌总数

    // ===== 牌状态追踪 =====
    waits: [bool; 34],         // 听牌时的待牌（和牌后有效）
    dora_factor: [u8; 34],    // 每种牌的宝牌倍率
    tiles_seen: [u8; 34],     // 已见过的牌总数（含其他人摸到/打出的）
    akas_seen: [bool; 3],     // 已见赤宝牌

    // ===== 舍牌策略相关 =====
    keep_shanten_discards: [bool; 34],  // 保持向听的舍牌候选
    next_shanten_discards: [bool; 34],  // 下一向听的舍牌候选
    forbidden_tiles: [bool; 34],         // 禁打牌（绝对不听牌）
    discarded_tiles: [bool; 34],         // 舍牌列表（用于振听检测）

    // ===== 风向/局信息 =====
    bakaze: Tile,               // 场风 (E/S/W/N)
    jikaze: Tile,               // 自风 (E/S/W/N)，根据player_id和bakaze计算
    kyoku: u8,                  // 局数 (0=E1, 1=E2, ..., 7=W4)
    honba: u8,                  // 本场数 (连续立直/荒牌次数)
    kyotaku: u8,                // 场供（立直宣言棒）
    oya: u8,                    // 亲家相对位置 (0=自己, 1=下家, 2=对家, 3=上家)
    rank: u8,                   // 当前排名 (0=第1, 1=第2, 2=第3, 3=第4)
    is_all_last: bool,          // 是否南入/西入（最后一局）

    // ===== 分数 =====
    scores: [i32; 4],           // 相对分数，scores[0]=自己的分数

    // ===== 舍牌（河） =====
    kawa: [TinyVec<[Option<KawaItem>; 24]>; 4],  // 4家舍牌历史
    last_tedashis: [Option<Sutehai>; 4],         // 每家最后舍牌
    riichi_sutehais: [Option<Sutehai>; 4],        // 每家立直时舍牌

    // ===== 副露/杠 =====
    fuuro_overview: [ArrayVec<[ArrayVec<[Tile; 4]>; 4]>; 4],  // 副露概要
    ankan_overview: [ArrayVec<[Tile; 4]>; 4],     // 暗杠概要
    intermediate_kan: ArrayVec<[Tile; 4]>,         // 中途杠（含加杠）
    intermediate_chi_pon: Option<ChiPon>,           // 中途吃碰
    minkans: ArrayVec<[u8; 4]>,                   // 明杠
    chis: ArrayVec<[u8; 4]>,                      // 吃
    pons: ArrayVec<[u8; 4]>,                      // 碰
    kans_on_board: u8,                             // 场上的杠数（4个会流局）

    // ===== 特殊状态 =====
    riichi_declared: [bool; 4],    // 各家是否已宣布立直
    riichi_accepted: [bool; 4],     // 各家是否已接受立直（需支付1000点）
    can_w_riichi: bool,            // 能否宣言和琪（双报）
    is_w_riichi: bool,             // 是否已宣言和琪
    at_rinshan: bool,              // 是否在岭上牌摸打阶段
    at_ippatsu: bool,              // 是否处于一発巡内
    at_furiten: bool,              // 是否处于振听状态
    is_menzen: bool,               // 是否门清（无副露）
    chankan_chance: Option<()>,    // 抢杠机会

    // ===== 向听数/动作 =====
    shanten: i8,                   // 向听数 (-1=听牌, 0=一向听, 8=两向听)
    has_next_shanten_discard: bool, // 是否有下一向听的舍牌
    last_cans: ActionCandidate,    // 当前可行动作
    last_self_tsumo: Option<Tile>,  // 最后自摸的牌
    last_kawa_tile: Option<Tile>,  // 最后舍牌
    ankan_candidates: ArrayVec<[Tile; 3]>,  // 暗杠候选
    kakan_candidates: ArrayVec<[Tile; 3]>,   // 加杠候选

    // ===== 局相关 =====
    dora_indicators: ArrayVec<[Tile; 5]>,  // 宝牌指示牌
    tiles_left: u8,                 // 剩余牌数
    at_turn: u8,                    // 当前巡目
    to_mark_same_cycle_furiten: Option<()>,  // 同巡振听标记
}
```

#### ActionCandidate（行动候选）详解

**文件来源**：`libriichi/src/state/action.rs`

```rust
pub struct ActionCandidate {
    // ===== 核心动作 =====
    pub can_discard: bool,           // 能否舍牌
    pub can_riichi: bool,            // 能否立直
    pub can_chi_low: bool,          // 能否吃（下家舍牌，低种）
    pub can_chi_mid: bool,          // 能否吃（中种）
    pub can_chi_high: bool,         // 能否吃（高种）
    pub can_pon: bool,               // 能否碰
    pub can_daiminkan: bool,         // 能否大明杠
    pub can_ankan: bool,             // 能否暗杠
    pub can_kakan: bool,             // 能否加杠
    pub can_agari: bool,             // 能否和牌（自摸/荣和）
    pub can_ryukyoku: bool,          // 能否流局
    pub can_pass: bool,              // 能否跳过（吃/碰/杠后）

    // ===== 目标相关 =====
    pub target_actor: u8,            // 吃/碰/杠的目标玩家ID
}
```

**注意**：`can_agari`是一个布尔值而非方法，用于表示能否和牌（自摸或荣和）。

### 4.2 游戏对局（BatchGame）

`libriichi/src/arena/game.rs`实现批量对局，支持同时运行多局游戏。

#### Game结构详解

**文件来源**：`libriichi/src/arena/game.rs` 第28-55行

```rust
struct Game {
    length: u8,                // 半庄战=8，东风战=4
    seed: (u64, u64),         // 随机种子
    indexes: [Index; 4],       // 玩家到Agent的映射

    oracle_obs_versions: [Option<u32>; 4],  // 各玩家的Oracle版本
    invisible_state_cache: [Option<Array2<f32>>; 4],  // 缓存的不可观察状态

    last_reactions: [EventExt; 4],  // 每局缓存的反应

    board: BoardState,         // 牌山和状态
    kyoku: u8,                // 当前局
    honba: u8,                // 本场数
    kyotaku: u8,              // 场供
    scores: [i32; 4],         // 当前分数
    game_log: Vec<Vec<EventExt>>,  // 对局日志

    kyoku_started: bool,       // 局是否已开始
    ended: bool,               // 比赛是否结束
    in_renchan: bool,         // 是否连庄
}
```

#### Index映射结构

```rust
struct Index {
    agent_idx: usize,    // Agent在agents数组中的索引 (game → agent)
    player_id_idx: usize,  // 玩家ID在agent中的索引 (agent → game)
}
```

#### 核心循环：poll与commit

```rust
impl Game {
    fn poll(&mut self, agents: &mut [Box<dyn BatchAgent>]) -> Result<()> {
        if !self.kyoku_started {
            // 检查是否结束（东4局后 或 西入后有人30k+ 且 亲家第一）
            if self.kyoku >= self.length + 4 || /* ... */ {
                self.ended = true;
                return Ok(());
            }
            // 初始化新局
            self.board.init_from_seed(self.seed);
            self.kyoku_started = true;
        }

        let poll = self.board.poll(reactions)?;
        match poll {
            Poll::InGame => {
                // 让所有可行动的Agent设置场景（编码观察）
                for (player_id, state) in ctx.player_states.iter().enumerate() {
                    agents[idx.agent_idx].set_scene(...)?;
                }
            }
            Poll::End => {
                // 局结束，处理得分/连庄/流局等
            }
        }
    }

    fn commit(&mut self, agents: &mut [Box<dyn BatchAgent>]) -> Result<Option<GameResult>> {
        // 收集所有Agent的反应并执行
        for (player_id, state) in ctx.player_states.iter().enumerate() {
            self.last_reactions[player_id] = agents[idx.agent_idx].get_reaction(...)?;
        }
    }
}
```

#### 批量推理优化（MortalBatchAgent）

**文件来源**：`libriichi/src/agent/mortal.rs` 第114-159行

```rust
impl MortalBatchAgent {
    fn evaluate(&mut self) -> Result<()> {
        // 等待所有特征编码完成
        mem::take(&mut self.wg).wait();

        let start = Instant::now();
        self.last_batch_size = sync_fields.states.len();

        // 批量调用Python引擎
        (self.actions, self.q_values, self.masks_recv, self.is_greedy) =
            Python::with_gil(|py| {
                let states: Vec<_> = sync_fields.states
                    .drain(..)
                    .map(|v| PyArray2::from_owned_array(py, v))
                    .collect();
                let masks: Vec<_> = sync_fields.masks
                    .drain(..)
                    .map(|v| PyArray1::from_owned_array(py, v))
                    .collect();

                self.engine.bind_borrowed(py)
                    .call_method1("react_batch", (states, masks, invisible_states))
                    .context("failed to execute react_batch")?
                    .extract()
            })?;

        self.last_eval_elapsed = start.elapsed();
        Ok(())
    }
}
```

### 4.3 单人牌效计算（Single Player Calculator）

`libriichi/src/algo/sp/`实现了麻将的单人牌效计算，用于v4版本的观察编码。

#### SPCalculator核心功能

**文件来源**：`libriichi/src/algo/sp/`

给定手牌，计算：
- **向听数**：当前手牌距离听牌还差几张
- **每种打牌的候选**：打出某张牌后的状态
- **听牌概率**：`tenpai_probs[t]` = t巡后听牌的概率
- **自摸概率**：`win_probs[t]` = t巡后自摸的概率
- **期望值**：`exp_values[t]` = t巡后的期望得分

#### Candidate：每种打牌后的状态

**文件来源**：`libriichi/src/algo/sp/candidate.rs` 和 `calc.rs`

每种Candidate代表**打出某张牌后的完整后续状态**：

```rust
pub struct Candidate {
    pub tile: Tile,                      // 打出的牌
    pub shanten_down: bool,              // 是否降向听（变好）

    // 各巡概率/期望值（最多17巡，因为牌山最多剩17张）
    pub tenpai_probs: ArrayVec<[f32; MAX_TSUMOS_LEFT]>,   // t巡后听牌概率
    pub win_probs: ArrayVec<[f32; MAX_TSUMOS_LEFT]>,       // t巡后自摸概率
    pub exp_values: ArrayVec<[f32; MAX_TSUMOS_LEFT]>,     // t巡后期望得分

    pub required_tiles: ArrayVec<[RequiredTile; 34]>,  // 听牌需要的牌及剩余枚数
    pub num_required_tiles: u8,                    // 听牌需要的总枚数
}
```

**"打出某张牌后的状态"具体指**：

1. **手牌变化**：移除该牌，手牌变为13张（3n+2形态）
2. **剩余牌山概率分布**：每种牌剩余多少张
3. **向听数变化**：打出后是向听不变还是变差
4. **听牌概率曲线**：`tenpai_probs[0..17]` = 第0..16巡后听牌的概率
5. **自摸概率曲线**：`win_probs[0..17]` = 第0..16巡后自摸的概率
6. **期望得分曲线**：`exp_values[0..17]` = 第0..16巡后期望得分
7. **必需牌列表**：`required_tiles` = 听牌需要的每种牌及其剩余枚数

#### SPCalculatorState的计算逻辑

**文件来源**：`libriichi/src/algo/sp/calc.rs` 第169-637行

```rust
impl<const MAX_TSUMO: usize> SPCalculatorState<'_, MAX_TSUMO> {
    fn calc(&mut self, can_discard: bool, cur_shanten: i8) -> Vec<Candidate> {
        if cur_shanten <= SHANTEN_THRES {
            // 3向听以下：计算完整概率和期望值
            let mut candidates = if can_discard {
                self.analyze_discard(cur_shanten)  // 分析每种打牌
            } else {
                self.analyze_draw(cur_shanten)    // 分析摸牌（已听牌时）
            };
            candidates.sort_by(...);  // 按EV或自摸概率排序
            candidates
        } else {
            // 4向听以上：只计算受入枚数（简化计算）
            self.analyze_discard_simple(cur_shanten)
        }
    }

    fn analyze_discard(&mut self, shanten: i8) -> Vec<Candidate> {
        // 1. 获取所有可打牌
        let discard_tiles = self.state.get_discard_tiles(shanten, tehai_len_div3);

        for DiscardTile { tile, shanten_diff } in discard_tiles {
            if shanten_diff == 0 {
                // 向听不变：计算完整概率
                self.state.discard(tile);
                let required_tiles = self.state.get_required_tiles(...);
                let values = self.draw(shanten);  // 递归计算摸牌后的期望值
                self.state.undo_discard(tile);

                candidates.push(Candidate::from(RawCandidate {
                    tile,
                    tenpai_probs: &values.tenpai_probs,
                    win_probs: &values.win_probs,
                    exp_values: &values.exp_values,
                    required_tiles,
                    shanten_down: false,
                }));
            } else if shanten_diff == 1 && shanten < SHANTEN_THRES {
                // 向听变差：但计算特殊处理（向听落とし）
                // ...
            }
        }
        candidates
    }

    fn draw(&mut self, shanten: i8) -> Rc<Values<MAX_TSUMO>> {
        // 计算摸到各种牌后的期望值（递归）
        if self.sup.calc_tegawari {
            self.draw_with_tegawari(shanten)  // 考虑手变化
        } else {
            self.draw_without_tegawari(shanten)
        }
    }
}
```

#### 期望值计算示例

```rust
// calc.rs 第639-758行 - get_score()
fn get_score(&self, win_tile: Tile) -> Option<[f32; 4]> {
    // 计算和牌得分，返回 [基础分, +1翻分, +2翻分, +3翻分]
    // 考虑：符数、番数、宝牌、里宝牌概率、立直、一发、海底等
}
```

**期望得分计算考虑的因素**：
- 符数（Fu）
- 番数（Han）- 断幺九、一杯口、混全带等
- 宝牌加成
- 里宝牌概率分布
- 役满（国士、 四暗刻等）
- 立直/一发自摸/海底摸月等额外番

pub struct RequiredTile {
    pub tile: Tile,       // 必需的牌
    pub count: u8,        // 剩余张数
}
```

#### Tile编码（34种牌）

**文件来源**：`libriichi/src/tile.rs`

麻将牌用0-38的整数编码：
```
0-8:   一万～九万 (1-9m)
9-17:  一筒～九筒 (1-9p)
18-26: 一索～九索 (1-9s)
27-33: 东南西北白发中 (7z)
34-38: 赤宝牌 (5mr, 5pr, 5sr)
```

`deaka()`: 将赤宝牌转为普通牌（5mr→5m）
`as_usize()`: 转为数组索引

#### 辅助数据结构

**KawaItem（舍牌项）**：`libriichi/src/state/item.rs`

```rust
pub struct KawaItem {
    pub chi_pon: Option<ChiPon>,  // 被吃/碰的牌信息
    pub kan: ArrayVec<[Tile; 4]>, // 杠的牌
    pub sutehai: Sutehai,          // 舍牌本身
}

pub struct Sutehai {
    pub tile: Tile,        // 舍出的牌
    pub is_dora: bool,    // 是否为宝牌
    pub is_tedashi: bool, // 是否为摸切（false=手切）
    pub is_riichi: bool,  // 是否为立直宣言牌
}

pub struct ChiPon {
    pub consumed: [Tile; 2],  // 吃/碰消耗的牌
    pub target_tile: Tile,    // 目标牌
}
```

#### Kawa编码细节

**文件来源**：`libriichi/src/state/obs_repr.rs` 第714-773行

舍牌编码使用两种通道数：

```rust
const SELF_KAWA_ITEM_CHANNELS: usize = 4;  // 自家舍牌
const KAWA_ITEM_CHANNELS: usize = 8;        // 他家舍牌

fn encode_self_kawa(&mut self, item: Option<&KawaItem>) {
    // 4通道：kan(1) + tile(1) + is_aka(1) + is_dora(1)
}

fn encode_kawa(&mut self, item: Option<&KawaItem>) {
    // 8通道：chi_pon(2) + kan(1) + tile(1) + is_aka(1) + is_dora(1) + is_tedashi(1) + is_riichi(1)
}
```
```

---

## 五、数据流分析

### 5.1 训练数据生成流程

#### 完整数据流图

```
                    .json.gz 人类对局日志文件
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  GameplayLoader::load_events() [Rust - gameplay.rs]                    │
│                                                                      │
│  1. 解析mjai事件流（start_game, start_kyoku, dahai, tsumo, hora...） │
│  2. 逐事件更新PlayerState::update()                                    │
│  3. 每回合调用encode_obs()生成(1012, 34)观察张量                      │
│  4. 记录动作label、mask（合法动作）、at_kyoku等                        │
│  5. Invisible::new() 计算不可观察信息（牌山、他家手牌）                │
│  6. Grp::load_events() 提取每局开始的7维状态序列                      │
└──────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  FileDatasetsIter::populate_buffer() [Python - dataloader.py]        │
│                                                                      │
│  输入：GameplayLoader返回的原始数据                                    │
│                                                                      │
│  1. 调用RewardCalculator（包含预加载的GRP模型）                         │
│                                                                      │
│  2. GRP.forward() 计算每时刻的排名概率分布                            │
│     └─ calc_rank_prob(player_id, grp_feature, rank_by_player)         │
│                                                                      │
│  3. calc_delta_pt() 计算相邻状态的奖励差                               │
│     └─ reward[t] = E[pt | t时刻] - E[pt | t+1时刻]                  │
│                                                                      │
│  4. 构建训练样本:                                                     │
│     - obs[i]: 第i步的观察 (N, 34)                                    │
│     - actions[i]: 实际采取的动作                                      │
│     - masks[i]: 合法动作掩码 (46,)                                    │
│     - steps_to_done[i]: 到游戏结束的步数                               │
│     - kyoku_rewards[i]: 第i步的奖励                                  │
│     - player_ranks[i]: 第i步时预计的最终排名                         │
└──────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│  DataLoader → GPU Training [train.py]                                 │
│                                                                      │
│  for obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks │
│      phi = brain(obs)                    # (B, 1024)                │
│      q_out = dqn(phi, masks)            # (B, 46)                 │
│                                                                      │
│      q_target_mc = gamma ** steps_to_done * kyoku_rewards            │
│                                                                      │
│      dqn_loss = 0.5 * mse(q[actions], q_target_mc)                  │
│      cql_loss = q_out.logsumexp(-1).mean() - q.mean()              │
│      next_rank_loss = cross_entropy(aux_net(phi), player_ranks)    │
│                                                                      │
│      loss = dqn_loss + cql_loss * 5 + next_rank_loss * 0.2         │
│      loss.backward()                                                 │
└──────────────────────────────────────────────────────────────────────┘
```

#### RewardCalculator详解

**文件来源**：`mortal/reward_calculator.py`

```python
class RewardCalculator:
    def calc_delta_pt(self, player_id, grp_feature, rank_by_player):
        # 1. 用GRP计算当前时刻的排名概率矩阵
        # matrix[kyoku_idx, player_id, rank] = P(该玩家在该局结束时排名为rank)
        matrix = self.calc_grp(grp_feature)  # shape: (num_kyoku, 4, 4)

        # 2. 计算期望得分
        # pts = [90, 45, 0, -135] - 排名对应得分
        # exp_pts[kyoku_idx] = P @ pts = 该时刻的期望排名得分
        exp_pts = rank_prob @ self.pts

        # 3. 奖励 = 相邻时刻期望得分差
        # reward[t] = exp_pts[t] - exp_pts[t+1]
        reward = exp_pts[1:] - exp_pts[:-1]
        return reward
```

### 5.2 在线训练（Self-Play Continual Training）

**是的，自战对局数据用于继续训练！** 这是一个经典的self-play强化学习循环。

#### 在线训练完整流程

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           Trainer Process                                      │
│                                                                             │
│  train.py:                                                                │
│    while True:                                                             │
│        # 1. 定期保存模型参数                                              │
│        submit_param(mortal, dqn, is_idle=True)                             │
│                                                                             │
│        # 2. 收集Worker的对局数据                                           │
│        drain()  → 从Server获取新对局日志                                  │
│                                                                             │
│        # 3. 用新数据继续训练                                              │
│        train_epoch(new_data)                                                │
│                                                                             │
│        # 4. 定期测试                                                      │
│        test_play() → 与baseline对战，评估性能                              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ drain()
                                    │ submit_param()
┌───────────────────────────────────┴───────────────────────────────────────────┐
│                            Parameter Server (server.py)                        │
│                                                                             │
│  State:                                                                  │
│    buffer_dir: /tmp/buffer  - Worker提交的原始对局                       │
│    drain_dir: /tmp/drain   - Trainer待训练的清洗后对局                    │
│    capacity: 1600          - Buffer容量上限                               │
│                                                                             │
│  消息处理:                                                                │
│    - handle_get_param():     Worker获取最新参数                          │
│    - handle_submit_replay(): Worker提交对局到buffer                     │
│    - handle_submit_param():  Trainer提交新参数                           │
│    - handle_drain():         Trainer取走buffer数据                       │
└─────────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ submit_replay()
                                    │ get_param()
┌───────────────────────────────────┴───────────────────────────────────────────┐
│                          Worker Process (client.py)                            │
│                                                                             │
│  client.py:                                                               │
│    while True:                                                             │
│        # 1. 获取最新模型参数                                               │
│        rsp = get_param()                                                   │
│        mortal.load_state_dict(rsp['mortal'])                                 │
│        dqn.load_state_dict(rsp['dqn'])                                      │
│                                                                             │
│        # 2. 运行1v3自战                                                  │
│        rankings, file_list = train_play(mortal, dqn)                        │
│                                                                             │
│        # 3. 提交对局日志                                                  │
│        submit_replay(logs)                                                  │
│                                                                             │
│  train_play() 内部:                                                        │
│    ┌────────────────────────────────────────────────────────────────────┐   │
│    │  OneVsThree.py_vs_py()                                           │   │
│    │                                                                     │   │
│    │  challenger: 当前训练的Mortal (epsilon-greedy exploration)       │   │
│    │  champion:   固定Baseline (之前最好的模型)                        │   │
│    │                                                                     │   │
│    │  运行 N局 (默认2000局)                                            │   │
│    │  每个seed产生4个game (4个split，challenger轮流在4个位置)           │   │
│    │                                                                     │   │
│    │  返回 rankings: [1st次数, 2nd次数, 3rd次数, 4th次数]             │   │
│    └────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 自战探索策略

**文件来源**：`mortal/player.py` 的 `TrainPlayer` 类

```python
class TrainPlayer:
    def __init__(self):
        # Boltzmann探索
        self.boltzmann_epsilon = 0.005   # 5%概率随机
        self.boltzmann_temp = 0.05       # 温度参数
        self.top_p = 1.0                # nucleus sampling阈值

# engine.py 中的实现
if self.boltzmann_epsilon > 0:
    is_greedy = torch.bernoulli(1 - boltzmann_epsilon)  # epsilon-greedy
    logits = (q_out / boltzmann_temp).masked_fill(~masks, -torch.inf)
    sampled = sample_top_p(logits, top_p)  # nucleus sampling
    actions = torch.where(is_greedy, q_out.argmax(-1), sampled)
```

#### 在线训练监控指标

```python
# client.py 中的移动平均监控
pts = np.array([90, 45, 0, -135])  # 天凤凤桌计分
history_window = 50  # 监控最近50局

rankings, file_list = train_play(mortal, dqn)
avg_rank = rankings @ [1,2,3,4] / rankings.sum()   # 平均排名
avg_pt = rankings @ pts / rankings.sum()           # 平均pt

history.append(rankings)
ma_avg_rank = sum(history) @ [1,2,3,4] / sum
ma_avg_pt = sum(history) @ pts / sum
```

#### 在线训练的特点

| 方面 | 说明 |
|------|------|
| **Self-Play** | Mortal与Baseline对战，而非与自身对战 |
| **固定对手** | Baseline是之前保存的最佳模型，不更新 |
| **探索策略** | 使用Boltzmann/epsilon-greedy平衡exploitation-exploration |
| **异步架构** | Trainer和Worker独立运行，通过Server通信 |
| **数据再利用** | 收集的对局数据保存到buffer，用于后续训练 |

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          Trainer Process (train.py)                        │
│                                                                          │
│  1. 定期调用submit_param()将mortal/dqn参数发送到Server                  │
│  2. 调用drain()从Server获取Worker提交的对局日志                            │
│  3. 用新对局数据继续训练                                                  │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ submit_param()
                                    │ drain()
┌────────────────────────────────────┴────────────────────────────────────┐
│                          Server Process (server.py)                       │
│                                                                          │
│  State:                                                                  │
│    - buffer_dir: Worker提交的对局日志缓存                                  │
│    - drain_dir: Trainer待读取的对局日志                                   │
│    - mortal_param / dqn_param: 最新模型参数                               │
│    - param_version: 参数版本号（递增）                                     │
│                                                                          │
│  消息类型:                                                               │
│    - get_param: Worker获取最新参数                                       │
│    - submit_replay: Worker提交对局日志到buffer                           │
│    - submit_param: Trainer提交新参数                                    │
│    - drain: Trainer读取buffer数据到drain目录                             │
└─────────────────────────────────────────────────────────────────────────┘
                                    ▲
                                    │ submit_replay()
                                    │ get_param()
┌────────────────────────────────────┴────────────────────────────────────┐
│                        Worker Process (client.py)                        │
│                                                                          │
│  while True:                                                             │
│    1. get_param() - 获取最新模型参数                                      │
│    2. train_play() - 运行1v3自战，产生对局日志                           │
│    3. submit_replay() - 提交对局日志                                      │
│                                                                          │
│  train_play()内部:                                                        │
│    └─ OneVsThree.py_vs_py(challenger=Mortal, champion=Baseline)         │
│         └─ BatchGame.run() - Rust批量运行N局对战                         │
│              └─ MortalBatchAgent.set_scene() / get_reaction()              │
│                   └─ Python: engine.react_batch() → 神经网络推理         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 5.3 推理数据流

```
用户请求（TCP Socket / HTTP）
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Rust: mjai协议解析                                                     │
│  - 解析JSON事件流                                                       │
│  - 提取当前游戏状态                                                      │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Rust: BoardState + PlayerState                                         │
│  - 维护4个玩家的状态                                                    │
│  - 跟踪舍牌、副露、立直等                                               │
│  - 调用encode_obs()生成观察张量                                         │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Rust: MortalBatchAgent::set_scene()                                   │
│  - 使用rayon并行编码多个游戏的观察                                      │
│  - 将ndarray转换为PyArray，构造Python调用参数                            │
│  - 等待WaitGroup同步                                                   │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Python: MortalEngine.react_batch()                                     │
│                                                                      │
│  obs_batch = [np.array((1012, 34)), ...]  # 34种牌 x 1012个特征行     │
│  masks_batch = [np.array(46, bool), ...]                             │
│                                                                      │
│  phi = brain(obs)           # GPU: (batch, 1024)                      │
│  q_out = dqn(phi, masks)   # GPU: (batch, 46)                      │
│  actions = q_out.argmax(-1)  # GPU: (batch,)                         │
│                                                                      │
│  return actions.tolist(), q_out.tolist(), ...                        │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Rust: MortalBatchAgent::get_reaction()                               │
│  - 接收Python返回的动作ID                                               │
│  - 转换为mjai Event（dahai, reach, chi, pon, hora...）            │
│  - 更新BoardState                                                     │
└─────────────────────────────────────────────────────────────────────────┘
    │
    ▼
响应（TCP Socket / HTTP）
```

### 5.4 对局核心流程（BoardState poll/commit循环）

**文件来源**：`libriichi/src/arena/game.rs` 和 `board.rs`

```rust
impl Game {
    fn poll(&mut self, agents: &mut [Box<dyn BatchAgent>]) -> Result<()> {
        if !self.kyoku_started {
            // 检查是否结束（东4后 / 西入后有人30k+且亲家第一）
            if self.kyoku >= self.length + 4 || /* 结束条件 */ {
                self.ended = true;
                return Ok(());
            }
            // 初始化新局
            self.board.init_from_seed(self.seed);  // 洗牌
            self.board.haipai();                  // 发牌，广播StartKyoku
            self.kyoku_started = true;
        }

        // 让所有可行动的Agent编码观察
        for (player_id, state) in ctx.player_states.iter().enumerate() {
            if state.last_cans().can_act() {
                agents[idx.agent_idx].set_scene(...)?;
            }
        }
        Ok(())
    }

    fn commit(&mut self, agents: &mut [Box<dyn BatchAgent>]) -> Result<Option<GameResult>> {
        // 收集所有Agent的反应
        for (player_id, state) in ctx.player_states.iter().enumerate() {
            if state.last_cans().can_act() {
                self.last_reactions[player_id] = agents[idx.agent_idx].get_reaction(...)?;
            }
        }

        // 执行反应（更新BoardState）
        self.board.poll(self.last_reactions)?;

        // 如果对局结束，返回GameResult
        if self.ended {
            return Some(GameResult { ... });
        }
        Ok(None)
    }
}
```

### 5.5 OneVsThree 1v3对战模式

**文件来源**：`libriichi/src/arena/one_vs_three.rs`

```
Seed分配 (seed_start, seed_count):
┌─────────────────────────────────────────────────────────────────┐
│ Seed 0: [0, 0, 0, 0] - 4个player都是challenger                │
│ Seed 1: [1, 1, 1, 1] - 4个player都是challenger                │
│ ...                                                              │
│ Seed N: [N%4, N%4, N%4, N%4]                                    │
└─────────────────────────────────────────────────────────────────┘

但是champion分配是固定的：
┌─────────────────────────────────────────────────────────────────┐
│ split A: challenger在位置0，champion在位置1,2,3                  │
│ split B: challenger在位置1，champion在位置0,2,3                  │
│ split C: challenger在位置2，champion在位置0,1,3                  │
│ split D: challenger在位置3，champion在位置0,1,2                  │
└─────────────────────────────────────────────────────────────────┘

因此每个seed实际产生4个game（4个split），
每个split中challenger轮流在4个位置。

最终 ranking[0] = challenger获得第1名的次数
     ranking[1] = challenger获得第2名的次数
     ...
```

---

## 六、在线训练系统

### 6.1 Parameter Server架构

```python
# server.py
class State:
    buffer_dir: str      # 收集的对局日志
    drain_dir: str       # 即将用于训练的日志
    capacity: int        # buffer容量上限
    mortal_param: Dict   # Brain参数
    dqn_param: Dict      # DQN参数
    param_version: int   # 参数版本号
```

### 6.2 客户端自对战

```python
# client.py
# 1. 获取最新参数
rsp = server.get_param()
mortal.load_state_dict(rsp['mortal'])
dqn.load_state_dict(rsp['dqn'])

# 2. 运行自对战
rankings, file_list = train_player.train_play(mortal, dqn, device)

# 3. 提交对局
server.submit_replay(logs)

# 4. 移动平均监控
ma_avg_rank = sum(last_N_rankings) @ [1,2,3,4] / sum
```

---

## 七、关键设计决策

### 7.1 片段游戏特性

麻将对局是**无折扣因子的片段游戏**（episodic task without discount）：
- `gamma = 1`
- 使用蒙特卡洛回报而非TD学习
- 每局的最终排名决定奖励

### 7.2 排名预测作为辅助任务

GRP模块预测最终排名，其输出用于计算**即时奖励的期望值**：

```
reward[t] = E[最终排名得分 | t时刻状态] - E[最终排名得分 | t+1时刻状态]
```

这解决了稀疏奖励问题。

### 7.3 行动掩码与Q值

DQN输出所有46个动作的Q值，但使用掩码只考虑合法动作：

```python
q = (v + a - a_mean).masked_fill(~mask, -torch.inf)
```

### 7.4 CQL（Conservative Q-Learning）

```python
cql_loss = q_out.logsumexp(-1).mean() - q.mean()
```

CQL通过惩罚高Q值来缓解过估计问题。

### 7.5 版本迭代

| 版本 | 关键变化 |
|------|----------|
| v1 | VAE latent space，双头输出mu/logsig |
| v2 | 去掉VAE，直接输出隐向量 |
| v3 | 优化观察编码，减少冗余 |
| v4 | **引入单人牌效表(SP Table)**作为特征 |

---

## 八、使用建议

### 8.1 如何使用Mortal进行推理

```python
from mortal.engine import MortalEngine
from mortal.model import Brain, DQN
import torch

# 加载模型
brain = Brain(version=4, conv_channels=192, num_blocks=40)
dqn = DQN(version=4)
state = torch.load('mortal.pth')
brain.load_state_dict(state['mortal'])
dqn.load_state_dict(state['current_dqn'])

# 创建引擎
engine = MortalEngine(
    brain=brain,
    dqn=dqn,
    is_oracle=False,
    version=4,
    device=torch.device('cuda:0'),
)

# 批量推理
obs_batch = [...]  # 34xN numpy数组列表
mask_batch = [...] # 46 bool列表
actions = engine.react_batch(obs_batch, mask_batch, None)
```

### 8.2 如何训练自己的模型

1. **准备数据**：收集mjai格式的`.json.gz`对局日志
2. **配置**：修改`config.example.toml`
3. **预训练GRP**：`python train_grp.py`
4. **训练主模型**：`python train.py`
5. **在线微调**（可选）：启动server + client进行自对战训练

### 8.3 训练周期估算

根据配置：
- `save_every = 400`：每400步保存一次
- `test_every = 20000`：每20000步测试一次

一个典型的训练运行可能需要数天到数周，取决于：
- GPU性能
- 数据集大小
- batch_size

---

## 九、项目优点

1. **混合架构**：Rust处理高性能游戏逻辑，Python处理灵活的训练
2. **精心设计的观察编码**：v4版本的SP Table设计非常巧妙
3. **离线+在线混合训练**：既能利用人类对局数据，又能自对战提升
4. **模块化设计**：GRP独立训练，辅助任务提升学习效率
5. **生产级代码质量**：完整的测试、类型安全、错误处理

---

## 十、参考资料

- 项目地址：https://github.com/Equim-chan/Mortal
- 文档：https://mortal.ekyu.moe
- 权重说明：https://gist.github.com/Equim-chan/cf3f01735d5d98f1e7be02e94b288c56
