# Keqing1 — 立直麻将 AI

基于监督学习的立直麻将策略机器人，使用 Conv1d ResNet 架构，支持 Mjai 协议推理。

## 项目状态

当前稳定运行线仍是 **keqingv2**，当前主迭代训练线为 **keqingv3**。

当前运行时推理入口为**统一 runtime bot 壳**：
- `src/inference/runtime_bot.py::RuntimeBot`
- `keqingv1 / keqingv2 / keqingv3` 通过不同 checkpoint 与 adapter 复用同一套 battle/replay 推理壳

主要特性：
- 完整的 45 动作监督学习框架（dahai×34 + reach/chi/pon/kan/hora/none）
- Meld Value Ranking Loss：value head 学习区分值得副露 vs 不值得副露
- 立直状态三段式 legal_actions（`reached` / `pending_reach` / 普通）
- none/pass 样本自动生成（约占总样本 16%）
- 振听系统、荣和精确判断
- 立直后摸切样本过滤（约 5.5%）
- RuntimeBot 支持 Mjai 协议推理 + beam search value 重排

## 快速开始

### 安装

```bash
uv sync
```

### 本地模式 / 天凤模式启动

本项目现在把“本地单机回放/对战”和“天凤平台 Gateway”明确分开：

```bash
# 本地模式：启动 ReplayUI + 本地 Battle API
uv run python src/main.py local --port 8000

# 兼容旧命令，效果等同于 local
uv run python src/main.py replay --port 8000

# 仅启动天凤 Gateway
uv run python src/main.py tenhou --gateway-port 11600

# 兼容旧命令，效果等同于 tenhou
uv run python src/main.py gateway --gateway-port 11600

# 两者同时启动
uv run python src/main.py serve --port 8000 --gateway-port 11600
```

说明：
- `local/replay` 只提供本地网页回放和本地 battle bot 对战，不启动天凤 Gateway
- `tenhou/gateway` 只提供平台接入，不启动本地网页服务
- 如果只是本地 1v3、4bot、自对战回放，不需要启动天凤 Gateway 端口

### 预处理数据

```bash
# keqingv3 预处理入口（默认读取 configs/keqingv3_preprocess.yaml）
uv run python scripts/preprocess_v3.py
```

当前 keqingv3 预处理默认配置要点：
- 输出目录：`processed_v3_fixaux`
- 默认关闭压缩：`compress_output: false`
- 预处理已内建 recent-window ETA
- 近期 hot path 已集中优化到 `keqingv3.progress_oracle`

如果只想临时覆盖并发或 ETA 窗口：

```bash
uv run python scripts/preprocess_v3.py --workers 18 --eta-window 200
```

注意：
- `processed_v3` 的旧 cache 可能带有修复前的 aux target 覆盖问题
- `keqingv3` 训练应优先使用 `processed_v3_fixaux`

### 训练 keqingv3

```bash
# 默认读取 configs/keqingv3_default.yaml
uv run python scripts/train_keqingv3.py
```

如果要临时改设备或输出目录：

```bash
uv run python scripts/train_keqingv3.py \
  --device cpu \
  --output-dir artifacts/models/keqingv3_debug
```

### 运行测试

```bash
uv run pytest
```

## 项目架构

```text
keqing1/
├── src/
│   ├── keqingv1/                 # v1 基础模型（兼容 action space / 特征定义）
│   │   ├── action_space.py       # 45 个动作的索引映射
│   │   ├── features.py           # 特征编码 (54,34) tile_feat + (48,) scalar
│   │   ├── model.py              # MahjongModel (~1.7M 参数)
│   │   ├── cached_dataset.py     # IterableDataset 流式加载（npz）
│   │   ├── trainer.py            # 训练循环（AMP + 梯度累积）
│   │
│   ├── keqingv2/                 # v2 扩展（Meld Value Ranking Loss）
│   │   ├── cached_dataset.py     # 预编码 rank pair，batch 返回 10-tuple
│   │   └── trainer.py            # rank loss 训练循环
│   │
│   ├── keqingv3/                 # v3 迭代（aux heads + progress oracle）
│   │   ├── features.py
│   │   ├── feature_tracker.py
│   │   ├── progress_oracle.py
│   │   ├── model.py
│   │   ├── cached_dataset.py
│   │   └── trainer.py
│   │
│   ├── mahjong_env/              # 麻将环境（游戏状态、合法动作）
│   │   ├── state.py
│   │   ├── legal_actions.py
│   │   ├── replay.py
│   │   ├── tiles.py
│   │   └── types.py
│   │
│   ├── inference/                # 统一运行时推理壳（battle/replay 共用）
│   │   ├── runtime_bot.py
│   │   ├── keqing_adapter.py
│   │   ├── scoring.py
│   │   ├── default_context.py
│   │   └── review.py
│   │
│   └── gateway/                  # Bot 对战网关
│
├── configs/
│   ├── keqingv2_default.yaml
│   ├── keqingv3_default.yaml
│   └── keqingv3_preprocess.yaml
│
├── artifacts/
│   ├── converted_mjai/           # 转换后的训练数据（ds1-ds13）
│   └── models/
│       ├── keqingv1/
│       ├── keqingv2/
│       └── keqingv3/
│
├── tests/
└── docs/
```

## scripts 分区标签

为降低脚本检索成本，`scripts/` 统一按以下标签理解：

### [data] 数据转换与预处理

- `batch_convert_ds.py`
- `batch_convert_mjai_converter.py`
- `batch_convert_tenhou6.py`
- `download_and_convert.py`
- `extract_tenhou_links.py`
- `mjlog2mjai_parse.py`
- `preprocess_cached.py`
- `preprocess_ds4.py`
- `preprocess_task.py`
- `preprocess_v2.py`
- `preprocess_v3.py`
- `reorganize_ds.py`
- `tenhou_xml_to_json.py`

### [replay] 回放转换与校验

- `verify_roundtrip.py`
- `verify_tenhou6_mjai.py`
- `visualize_replay.py`
- `test_gameviewer.py`

### [selfplay] 自对战与导出流程

- `selfplay.py`
- `run_game_script.py`
- `run_game_and_export.ipynb`

### [devtools] 开发辅助与环境检查

- `build_static_tables.py`
- `download_wait_categories.py`
- `test_riichienv_import.py`

## 训练与数据流（当前）

```text
天凤/雀魂对局记录
      ↓
Mjai JSONL (.mjson)
      ↓ (scripts/preprocess_v3.py + training.preprocess)
缓存样本 (.npz, 当前主线为 processed_v3_fixaux)
      ↓ (scripts/train_keqingv3.py + keqingv3.trainer)
模型 checkpoint (artifacts/models/keqingv3*)
```

当前主线支持 `keqingv1 / keqingv2 / keqingv3` 三条模型分支。

运行时语义：
- battle / replay 统一依赖 `RuntimeBot`
- 模型版本差异主要体现在：
  - checkpoint 参数形状
  - 对应 features / model / adapter
- 不再保留独立的 `keqingv1.bot` 运行时壳

## 注意事项

- `src/` 为包根路径（`pyproject.toml` 配置）
- 训练 checkpoint 保存在 `artifacts/models/keqingv1|keqingv2|keqingv3/`
- GPU 不可用时自动回退到 CPU
- 数据集使用 IterableDataset 流式加载，不会全量加载进内存
- Python 执行用 `uv run python3`，系统 `python3` 无 `riichienv`/`torch`

## 致谢

- [Mortal](https://github.com/Equim-chan/Mortal) — 架构参考
- [Mjai](https://github.com/mjai-jp/mjai) — 麻将 AI 协议
- 天凤/雀魂社区 — 数据集来源
