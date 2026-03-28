# 项目概述：Keqing1 Mahjong AI

## 项目简介

Keqing1 是一个基于深度学习的日式麻将 AI 项目。项目名称"Keqing"来自原神中的角色琴（Jean的谐音）。

核心目标是训练一个能够在 mjai 协议下对战的麻将 AI，具有以下特点：
- 基于多任务学习（策略 + 价值评估）
- 支持监督学习和强化学习
- 可与 mjai 协议的其他 bot 对战

## 技术架构

```
┌─────────────────────────────────────────────────────────────┐
│                        用户接口层                            │
├─────────────────────────────────────────────────────────────┤
│  mjai Bot API  │  CLI 工具  │  训练脚本  │  Web 可视化      │
├─────────────────────────────────────────────────────────────┤
│                        AI 核心层                             │
├─────────────────────────────────────────────────────────────┤
│  MjaiPolicyBot  │  RuleBot  │  MultiTaskModel (策略+价值)  │
├─────────────────────────────────────────────────────────────┤
│                      数据处理层                              │
├─────────────────────────────────────────────────────────────┤
│  牌谱转换器  │  数据集构建  │  特征提取  │  验证工具         │
├─────────────────────────────────────────────────────────────┤
│                      环境模拟层                              │
├─────────────────────────────────────────────────────────────┤
│  mahjong_env  │  legal_actions  │  replay  │  state         │
└─────────────────────────────────────────────────────────────┘
```

## 核心模块

### 1. Bot 模块 (`src/bot/`)

| 文件 | 功能 |
|------|------|
| `mjai_bot.py` | mjai 协议 AI bot，核心推理逻辑 |
| `rule_bot.py` | 基于规则的 bot，用于对比基准 |
| `create_mjai_bot.py` | 创建 mjai Docker 格式 bot |
| `features.py` | 状态特征提取 |

**关键类**: `MjaiPolicyBot`
- 加载模型 checkpoint
- 处理 mjai 协议消息
- 枚举合法动作
- 选择最优动作

### 2. 模型模块 (`src/model/`)

| 文件 | 功能 |
|------|------|
| `policy_value.py` | 多任务模型（策略头 + 价值头） |
| `encoder.py` | 输入编码器 |
| `vocab.py` | 动作词汇表构建 |

**模型架构**: MultiTaskModel
```
输入特征 → 共享编码器 → 策略头 (policy head)
                        → 价值头 (value head)
```

### 3. 训练模块 (`src/train/`)

| 文件 | 功能 |
|------|------|
| `train_v2.py` | v2 版本训练（主训练脚本） |
| `train_sl.py` | 监督学习训练 |
| `train_rl.py` | 强化学习训练 |
| `dataset.py` | 数据集加载 |
| `eval_offline.py` | 离线评估 |
| `eval_battle.py` | 对战评估 |

### 4. 麻将环境 (`src/mahjong_env/`)

| 文件 | 功能 |
|------|------|
| `state.py` | 游戏状态管理 |
| `legal_actions.py` | 合法动作枚举 |
| `replay.py` | 牌谱回放与样本构建 |
| `tiles.py` | 牌定义与转换 |
| `types.py` | 类型定义 |

### 5. 数据转换 (`src/convert/`)

| 文件 | 功能 |
|------|------|
| `libriichi_bridge.py` | tenhou6 → mjai-jsonl |
| `batch_converter.py` | 批量转换 |
| `link_converter.py` | 链接格式转换 |
| `validate_mjai.py` | mjai 格式验证 |

### 6. 工具脚本 (`src/tools/`)

| 文件 | 功能 |
|------|------|
| `mjai_jsonl_to_tenhou6.py` | mjai-jsonl → tenhou6 |
| `convert_one.py` | 单文件转换 |
| `convert_dir.py` | 目录批量转换 |
| `run_pipeline.py` | 完整训练流程 |
| `fetch_majsoul_links.py` | 获取雀魂牌谱链接 |
| `majsoul_paipu_to_tenhou6_by_browser.py` | 浏览器爬取牌谱 |

## 数据格式

### tenhou6 JSON 格式
天凤六麻对战记录格式，包含完整对战信息。

```json
{
  "name": ["玩家1", "玩家2", "玩家3", "玩家4"],
  "log": [
    {
      0: [round_info, scores, dora, ura, ...],
      1: {...},
      ...
    }
  ]
}
```

### mjai-jsonl 格式
麻将AI协议格式，每行一个 JSON 事件。

```json
{"type": "start_game", "names": ["A", "B", "C", "D"]}
{"type": "start_kyoku", "bakaze": "E", "kyoku": 1, ...}
{"type": "tsumo", "actor": 0, "pai": "5mr"}
{"type": "dahai", "actor": 0, "pai": "3m", "tsumogiri": false}
...
```

## 训练流程

```
1. 数据采集
   └─ 天凤/雀魂牌谱 → tenhou6 JSON

2. 数据转换
   └─ tenhou6 JSON → mjai-jsonl → tenhou6 JSON (验证)

3. 样本构建
   └─ mjai-jsonl → supervised samples (actor, state, action, value)

4. 模型训练
   ├─ 监督学习 (SL)
   └─ 强化学习 (RL)

5. 模型导出
   └─ checkpoint → mjai bot bundle

6. 对战评估
   └─ mjai bot vs RuleBot / 其他 bot
```

## 依赖技术

- **深度学习**: PyTorch
- **麻将逻辑**: 自研库 (mahjong-env)
- **数据格式**: JSON, JSONL
- **协议**: mjai (Mahjong AI Protocol)
- **牌谱来源**: 天凤 (tenhou.net), 雀魂 (majsoul)

## Checkpoint 历史

| 日期 | Checkpoint | 说明 |
|------|------------|------|
| 03-20 | checkpoint_03202215 | v2 训练版本，包含 bot 打包 |
| 03-21 | checkpoint_03210037 | 往返转换 bug 修复，100% 匹配 |

## 相关文档

- [IMPL_SUMMARY.md](IMPL_SUMMARY.md) - 本次实现总结
- [ROADMAP.md](ROADMAP.md) - 后续计划
