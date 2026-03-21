# Riichi Mahjong MVP

基于强化学习的立直麻将（Mjai协议）策略机器人项目。

## 📋 项目概述

这是一个完整的麻将AI训练和推理系统，使用RiichiEnv作为游戏环境，支持监督学习和强化学习训练。

**核心功能：**
- 🤖 基于策略-价值网络的麻将AI机器人
- 📊 监督学习（SL）和强化学习（RL）训练流程
- 🔄 数据格式转换（支持天凤/雀魂/Mjai格式）
- 🎮 Gym风格的强化学习环境接口
- 🧪 完整的测试套件

## 🏗️ 项目架构

```
keqing1/
├── src/                          # 源代码目录
│   ├── bot/                      # 机器人模块
│   │   ├── mjai_bot.py          # Mjai协议策略机器人
│   │   ├── rule_bot.py          # 基于规则的机器人
│   │   ├── features.py          # 特征工程
│   │   └── create_mjai_bot.py   # 机器人工厂
│   │
│   ├── model/                    # 模型定义
│   │   ├── encoder.py           # ResNet编码器
│   │   ├── policy_value.py      # 多任务模型（策略+价值）
│   │   └── vocab.py             # 动作词汇表
│   │
│   ├── mahjong_env/             # 麻将环境封装
│   │   ├── state.py             # 游戏状态管理
│   │   ├── legal_actions.py     # 合法动作枚举
│   │   ├── replay.py            # 回放管理
│   │   ├── tiles.py             # 牌面定义
│   │   └── types.py             # 类型定义
│   │
│   ├── train/                   # 训练模块
│   │   ├── train_sl.py         # 监督学习训练
│   │   ├── train_rl.py         # 强化学习训练
│   │   ├── train_v2.py         # 训练流程V2
│   │   ├── dataset.py          # 数据集类
│   │   ├── data_stats.py       # 数据统计分析
│   │   ├── eval_offline.py     # 离线评估
│   │   └── eval_battle.py      # 对战评估
│   │
│   ├── convert/                 # 数据转换模块
│   │   ├── libriichi_bridge.py # Libriichi桥接
│   │   ├── batch_converter.py  # 批量转换
│   │   ├── link_converter.py   # 链接转换
│   │   └── validate_mjai.py    # Mjai数据验证
│   │
│   ├── tools/                   # 工具脚本
│   │   ├── train_large.py      # 大规模训练
│   │   ├── run_pipeline.py     # 完整训练流程
│   │   ├── replay_bot.py       # 回放分析机器人
│   │   ├── debug_bot_policy.py # 策略调试
│   │   ├── convert_dir.py      # 目录批量转换
│   │   ├── convert_one.py      # 单文件转换
│   │   ├── fetch_majsoul_links.py    # 雀魂链接抓取
│   │   ├── majsoul_paipu_to_tenhou6_by_browser.py    # 格式转换
│   │   ├── mjai_jsonl_to_tenhou6.py    # Mjai到天凤格式转换
│   │   ├── validate_riichi.py   # 立直验证
│   │   └── ... (其他工具脚本)
│   │
│   ├── main_bot.py             # 机器人主入口
│   └── riichi.py               # 立直库封装
│
├── tests/                       # 测试套件
│   ├── test_bot_protocol.py    # 机器人协议测试
│   ├── test_convert_pipeline.py # 转换流程测试
│   ├── test_link_converter.py  # 链接转换测试
│   ├── test_obs_features.py    # 特征测试
│   ├── test_replay_log.py      # 回放日志测试
│   ├── test_value_target.py    # 价值目标测试
│   └── test_view_filter.py     # 视图过滤测试
│
├── configs/                     # 配置文件
│   ├── mvp.yaml                # MVP配置
│   └── v2.yaml                 # V2配置
│
├── docs/                       # 文档目录
│   ├── checkpoint/             # 开发阶段记录
│   │   └── phase4_riichienv_visualization.md
│   ├── visualization_demo.ipynb # 可视化演示
│   └── JUPYTER_SETUP.md       # Jupyter配置指南
│
├── keqing1.code-workspace      # IDE工作区配置
├── open_trae.sh                # IDE快速启动脚本
├── IDE_QUICK_START.md          # IDE使用指南
├── test_riichienv_import.py    # RiichiEnv测试脚本
├── pyproject.toml              # 项目配置
├── uv.lock                     # UV锁文件
└── .gitignore                  # Git忽略规则
```

## 🔧 核心模块详解

### 1. Bot模块 (`bot/`)

**MjaiPolicyBot**
- 基于策略-价值网络的智能机器人
- 支持MJAI协议交互
- 包含规则机器人的降级策略
- 支持检查点加载和JSON格式模型导出

**RuleBot**
- 基于规则的简单机器人
- 作为策略机器人的备选方案

**Features**
- 特征工程模块
- 状态向量化和特征提取
- OBS_DIM定义观察维度

### 2. Model模块 (`model/`)

**ResNetEncoder**
- 残差网络编码器
- 输入：游戏观察（270维）
- 输出：256维隐藏表示
- 5层残差结构

**MultiTaskModel**
- 多任务学习模型
- 策略头：预测动作概率
- 价值头：预测游戏结果
- 辅助头：预测排名

**Vocab**
- 动作词汇表管理
- 动作到token的映射
- 词汇表构建

### 3. Mahjong Env模块 (`mahjong_env/`)

**GameState**
- 游戏状态管理
- 事件应用和状态更新
- 观察空间定义

**LegalActions**
- 合法动作枚举
- 基于当前状态的动作过滤

**Replay**
- 游戏回放管理
- 样本构建和提取

### 4. Train模块 (`train/`)

**监督学习流程 (train_sl.py)**
```python
from train.train_sl import train
# 完整的监督学习训练流程
# 包含数据加载、模型训练、验证和检查点保存
```

**数据集类 (dataset.py)**
- SupervisedMjaiDataset
- 支持批次加载和预处理
- 观察维度：270

**评估模块**
- eval_offline.py：离线评估
- eval_battle.py：在线对战评估

### 5. Convert模块 (`convert/`)

**Libriichi Bridge**
- Libriichi库接口封装
- 原始数据到Mjai格式转换

**验证器**
- Mjai JSONL格式验证
- 数据完整性检查

## 🚀 快速开始

### 环境配置

```bash
# 激活虚拟环境
source .venv/bin/activate

# 安装依赖
uv sync
```

### 基础使用

```python
import sys
sys.path.insert(0, 'src')

from riichienv import RiichiEnv
from bot.mjai_bot import MjaiPolicyBot

# 初始化环境
env = RiichiEnv()
obs = env.reset()

# 创建机器人
bot = MjaiPolicyBot(player_id=0, checkpoint_path="artifacts/sl/best.npz")
```

### 训练模型

```bash
# 监督学习训练
cd src
python -m train.train_sl \
    --log_path log.jsonl \
    --config ../configs/mvp.yaml \
    --out_dir artifacts/sl
```

### 运行测试

```bash
# 运行单个测试
pytest tests/test_obs_features.py -v

# 运行所有测试
pytest tests/ -v
```

## 📊 配置文件

### MVP配置 (configs/mvp.yaml)

```yaml
seed: 42
max_seq_len: 512
batch_size: 64
num_epochs: 5
learning_rate: 0.0005
weight_decay: 0.0001
hidden_dim: 256
num_layers: 2
dropout: 0.1
train_split: 0.9
device: "cpu"
policy_adv_beta: 0.5
```

## 🔍 开发指南

### 添加新模块

1. 在相应目录创建模块文件
2. 更新 `__init__.py` 导出
3. 添加单元测试到 `tests/`
4. 更新本文档

### 代码规范

- 使用类型注解（Type Hints）
- 遵循PEP 8
- 添加Docstrings
- 编写单元测试

### 数据流程

```
原始数据 (JSON)
    ↓
Libriichi转换 (libriichi_bridge.py)
    ↓
Mjai格式 (JSONL)
    ↓
验证 (validate_mjai.py)
    ↓
数据集构建 (dataset.py)
    ↓
模型训练 (train_sl.py / train_rl.py)
    ↓
模型检查点 (NPZ/JSON)
    ↓
机器人推理 (mjai_bot.py)
```

## 🛠️ 常用工具

### 数据转换

```bash
# 单文件转换
python -m tools.convert_one input.json output.jsonl

# 目录批量转换
python -m tools.convert_dir input_dir/ output_dir/

# Mjai到天凤格式
python -m tools.mjai_jsonl_to_tenhou6 log.jsonl
```

### 训练和分析

```bash
# 运行完整流程
python -m tools.run_pipeline --config configs/mvp.yaml

# 调试策略
python -m tools.debug_bot_policy --checkpoint artifacts/sl/best.npz
```

### 验证和检查

```bash
# 验证Mjai数据
python -m convert.validate_mjai log.jsonl

# 检查模型束
python -m tools.check_bundle_parity checkpoint.npz
```

## 📦 依赖项

**核心依赖：**
- `riichienv >= 0.4.7` - 麻将环境
- `numpy >= 1.24` - 数值计算
- `pyyaml >= 6.0` - 配置文件

**开发依赖：**
- `pytest` - 单元测试
- `ipython` - 交互式开发

## 🎯 项目状态

✅ **核心功能完整**
- 模型训练和推理 ✓
- 数据转换流程 ✓
- 测试套件 ✓
- RiichiEnv可视化集成 ✓
- IDE快速启动配置 ✓

🔄 **持续开发中**
- 强化学习训练流程优化
- ds4数据集训练
- 模型性能评估
- 文档补充

## 🎮 RiichiEnv 可视化功能

项目已集成RiichiEnv可视化组件，支持对局回放和实时可视化。

### 快速开始

```bash
# 启动Jupyter Notebook
cd docs
jupyter notebook visualization_demo.ipynb
```

### 核心API使用

```python
from riichienv import RiichiEnv
from riichienv.visualizer import GameViewer
from riichienv.agents import RandomAgent

# 创建环境
agent = RandomAgent()
env = RiichiEnv(game_mode="4p-red-half")
obs = env.reset()

# 运行对局
while not env.done():
    actions = {player_id: agent.act(obs) 
               for player_id, obs in obs.items()}
    obs = env.step(actions)

# 可视化
viewer = GameViewer.from_env(env)
viewer.show(step=100, perspective=0, freeze=False)
```

### 可视化文件位置

- `docs/visualization_demo.ipynb` - Jupyter Notebook演示
- `docs/JUPYTER_SETUP.md` - Jupyter配置指南
- `test_riichienv_import.py` - Python测试脚本

详细文档请查看 [JUPYTER_SETUP.md](docs/JUPYTER_SETUP.md)

## 🖥️ IDE快速启动

项目已配置好Trae/VSCod工作区，可快速打开并记住项目状态。

### 打开方式

**方式1: 双击workspace文件**
```
keqing1.code-workspace
```

**方式2: 使用启动脚本**
```bash
./open_trae.sh
```

**方式3: 从应用菜单**
搜索 "Keqing1 Mahjong AI" 并点击

### 相关文件

- `keqing1.code-workspace` - Workspace配置
- `open_trae.sh` - 启动脚本
- `IDE_QUICK_START.md` - IDE使用指南

详细文档请查看 [IDE_QUICK_START.md](IDE_QUICK_START.md)

## 📝 注意事项

1. **路径配置**：所有模块使用 `src/` 作为根路径
2. **检查点格式**：支持NPZ（开发用）和JSON（生产用）格式
3. **测试数据**：部分测试需要实际的日志文件
4. **依赖冲突**：避免与ROS等系统包冲突

## 🤝 贡献指南

欢迎提交Issue和Pull Request！

1. Fork 项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 创建Pull Request

## 📄 许可证

本项目仅供研究和学习使用。

## 🙏 致谢

- [RiichiEnv](https://github.com/smly/RiichiEnv) - 高性能麻将环境
- [Mjai](https://github.com/mjai-jp/mjai) - 麻将AI协议
- 天凤/雀魂社区 - 数据集来源
