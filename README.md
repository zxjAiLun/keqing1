# Keqing1

立直麻将 AI 项目。当前仓库已经收口到一条统一运行时主线：

- 规则与状态真源：`src/mahjong_env`
- 在线推理入口：`src/inference/runtime_bot.py::RuntimeBot`
- 本地对战 / 回放服务：`src/replay/server.py`
- 当前训练迭代线：`keqingv3`

当前状态可以简单理解为：

- `keqingv2` 仍是可运行、可对比的稳定基线
- `keqingv3` 是当前主训练线
- battle / replay / selfplay 已不再各自维护一套 bot 壳，而是共用 `RuntimeBot`

## 当前推荐用法

### 1. 安装环境

```bash
uv sync
```

前端回放页面需要单独安装依赖：

```bash
cd src/replay_ui
npm install
```

## 本地运行

### 2. 启动本地回放 + Battle API

```bash
uv run python src/main.py local --port 8000
```

兼容别名：

```bash
uv run python src/main.py replay --port 8000
```

启动后可用：

- Replay / Review 页面
- 本地 bot battle API
- `game-replay` 棋盘回放

### 3. 仅启动天凤 Gateway

```bash
uv run python src/main.py --gateway-port 11600 tenhou
```

兼容别名：

```bash
uv run python src/main.py --gateway-port 11600 gateway
```

### 4. 同时启动本地服务和 Gateway

```bash
uv run python src/main.py --port 8000 --gateway-port 11600 serve
```

### 5. 启动指定房间的多个天凤机器人

例如把两个匿名 `keqingv2` 机器人加入天凤 lobby `L2147` 的四麻东南战（半庄）队列：

```bash
uv run python scripts/launch_tenhou_bots.py --room L2147 --count 2 --bot keqingv2 --start-gateway
```

常用参数：

- `--room L2147`：默认补成 `L2147_9`（四麻东南战 / hanchan）
- `--room 2147_0`：数值房间；三麻可用 `2147_9`
- `--game-type 9`：四麻东南战（半庄，多人匹配）
- `--game-type 1`：四麻东风战（多人匹配）
- `--count 2`：启动两个 bot 客户端
- `--bot keqingv2`：可切换 `keqingv1 / keqingv2 / keqingv3 / rulebase`
- `--name-prefix NoName`：默认以匿名 `NoName` 方式进房
- `--start-gateway`：自动启动 `src/gateway/main.py`
- `--model-path /path/to/best.pth`：显式指定 checkpoint

专属 Tenhou 桥接层支持通过环境变量或 launcher 参数补充握手/认证信息：

- `TENHOU_COOKIE` / `--tenhou-cookie`
- `TENHOU_HELO_JSON` / `--tenhou-helo-json`
- `TENHOU_URI` / `--tenhou-uri`
- `TENHOU_ORIGIN` / `--tenhou-origin`

例如：

```bash
uv run python scripts/launch_tenhou_bots.py \
  --room L2147 \
  --count 1 \
  --bot keqingv2 \
  --start-gateway \
  --tenhou-cookie 'uid=...; other=...' \
  --tenhou-helo-json '{"tid":"..."}'
```

## 训练数据流程

当前 `keqingv3` 的标准数据流程是：

```text
Tenhou / mjlog / 外部牌谱
-> mjai .mjson
-> scripts/preprocess_v3.py
-> processed_v3_fixaux/*.npz
-> scripts/train_keqingv3.py
-> artifacts/models/keqingv3
```

### 5. 预处理

默认配置文件：

- [configs/keqingv3_preprocess.yaml](configs/keqingv3_preprocess.yaml)

默认命令：

```bash
uv run python scripts/preprocess_v3.py
```

当前默认行为：

- 输入目录：`artifacts/converted_mjai/ds1`
- 输出目录：`processed_v3_fixaux`
- `value_strategy = mc_return`
- `compress_output = false`
- 预处理进度会显示：
  - 已运行秒数
  - 预计剩余秒数
  - `skip / empty / error`

常用覆盖参数：

```bash
uv run python scripts/preprocess_v3.py --workers 18 --eta-window 200
uv run python scripts/preprocess_v3.py --output_dir processed_v3_debug
uv run python scripts/preprocess_v3.py --actor_name_filter "玩家名"
```

说明：

- `processed_v3_fixaux` 是当前 `keqingv3` 正式训练应使用的 cache
- 旧 `processed_v3` 可能含有修复前的 aux target 缓存，不建议继续直接训练
- 预处理失败的牌谱会记录到输出目录下的失败日志，便于后续重转

## 训练

### 6. 训练 keqingv3

默认配置文件：

- [configs/keqingv3_default.yaml](configs/keqingv3_default.yaml)

默认命令：

```bash
uv run python scripts/train_keqingv3.py
```

默认配置要点：

- 数据目录：`processed_v3_fixaux/ds1`
- 设备：`cuda`
- 输出目录：`artifacts/models/keqingv3`

常用覆盖：

```bash
uv run python scripts/train_keqingv3.py --smoke
uv run python scripts/train_keqingv3.py --resume artifacts/models/keqingv3/last.pth
uv run python scripts/train_keqingv3.py --output-dir artifacts/models/keqingv3_exp1
uv run python scripts/train_keqingv3.py --device cpu
```

当前 checkpoint 语义：

- `last.pth`：续训入口
- `best.pth`：按总 `val_objective` 选优
- `best_meld.pth`：按 response-window / meld 指标选优

## 自对战与评测

### 7. 运行 selfplay

```bash
uv run python scripts/selfplay.py --model keqingv3 --games 100
```

也可以直接指定权重路径：

```bash
uv run python scripts/selfplay.py \
  --model artifacts/models/keqingv3/best.pth \
  --games 100 \
  --game-format hanchan
```

常用参数：

- `--seat-bots keqingv3 keqingv2 keqingv2 keqingv2`
- `--fixed-seats`
- `--save-games 20`
- `--save-all-games`
- `--export-anomaly-games 20`

默认输出目录：

- `artifacts/selfplay_benchmarks/<model>_<timestamp>`

典型输出：

- `stats.json`
- `progress.jsonl`
- `replays/manifest.json`
- 可选 `anomaly_replays/manifest.json`

## 回放与 Review

### 8. 本地查看 replay / review

启动服务：

```bash
uv run python src/main.py local --port 8000
```

然后访问：

- `http://127.0.0.1:8000/`
- `http://127.0.0.1:8000/game-replay?...`

当前回放链路特点：

- `replay` 与 `battle` 共用统一 runtime 语义
- `hora` 展示优先使用后端重算结果
- replay decision 统计由后端统一归一化，不依赖前端临时猜测

## 目录说明

### 核心代码

- [src/mahjong_env](src/mahjong_env)
  - 游戏状态、合法动作、计分、回放样本语义
- [src/inference](src/inference)
  - 统一运行时推理、beam/value/aux rerank、review 导出
- [src/gateway](src/gateway)
  - live battle 与 bot 驱动
- [src/replay](src/replay)
  - replay API、存储、归一化、展示辅助
- [src/keqingv3](src/keqingv3)
  - 当前主训练线模型、特征、progress oracle、trainer
- [src/training](src/training)
  - 通用 preprocess、cached dataset、训练循环基础设施

### 常用脚本

- [scripts/preprocess_v3.py](scripts/preprocess_v3.py)
- [scripts/train_keqingv3.py](scripts/train_keqingv3.py)
- [scripts/selfplay.py](scripts/selfplay.py)
- [scripts/reconvert_failed_ds1.py](scripts/reconvert_failed_ds1.py)

### 产物目录

- [artifacts/converted_mjai](artifacts/converted_mjai)
  - 训练输入 `.mjson`
- [processed_v3_fixaux](processed_v3_fixaux)
  - 当前 v3 cache 输出
- [artifacts/models](artifacts/models)
  - 训练产物
- [artifacts/replays](artifacts/replays)
  - review / replay 导出

## 测试

### 9. 运行测试

全量：

```bash
uv run pytest
```

常用 focused tests：

```bash
uv run pytest tests/test_battle_rules_checklist.py tests/test_battle_edge_cases.py tests/test_replay_strict_legal.py
```

```bash
uv run pytest tests/test_normal_progress.py tests/test_features_v3.py tests/test_training_updates.py
```

语法检查：

```bash
uv run python -m py_compile src/keqingv3/progress_oracle.py src/training/trainer.py
```

Rust progress-analysis 路径常用验证：

```bash
cd rust/keqing_core
cargo test
cd /media/bailan/DISK1/AUbuntuProject/project/keqing1
uv run python rust/keqing_core/build.py
uv pip install --reinstall rust/keqing_core/target/wheels/keqing_core-*.whl
PYTHONPATH=src uv run pytest tests/test_progress_oracle_rust.py tests/test_normal_progress.py tests/test_features_v3.py
PYTHONPATH=src uv run python scripts/benchmark_progress_oracle_rust.py
```

pre-push / CI 约定：

```bash
bash scripts/verify_python.sh
```

- 本地全局 Git hook 现在应优先调用仓库内的 `scripts/verify_python.sh`
- CI 使用同一脚本，并在运行前执行 `uv sync --locked --group dev`
- 这样可以避免系统 `pytest` / 系统 Python 绕开 `.venv` 导致的依赖缺失

前端构建：

```bash
cd src/replay_ui
npm run build
```

## 当前约束与注意事项

- `mahjong_env` 是规则真源，不要把 `riichienv` 引进 live battle 规则主链
- `RuntimeBot` 是 battle / replay / selfplay 的统一在线 bot 入口
- `tsumo` 的训练特征契约是：
  - `hand` = 决策前 13 张
  - `tsumo_pai` = 当前摸到的牌
- replay strict legality 默认 fail-fast，不会静默把非法 label 塞进 legal set
- `keqingv3` 预处理热路径集中在：
  - `src/keqingv3/progress_oracle.py`
  - `src/keqingv3/feature_tracker.py`
- `src/keqing_core/` + `rust/keqing_core/` 是 `keqingv3` progress-analysis 的 Rust 能力层；
  不应把 battle / replay / selfplay 规则真值迁入这里
- 如果 push 触发全局 hook 环境问题，优先确认测试命令是否通过 `uv run pytest`

## 更多设计说明

- [docs/keqingv3_design.md](docs/keqingv3_design.md)
- [docs/rust_phase2a_status.md](docs/rust_phase2a_status.md)
- [docs/rust_refactor_handoff_2026-04-08.md](docs/rust_refactor_handoff_2026-04-08.md)
- [docs/old/data_pipeline_overview.md](docs/old/data_pipeline_overview.md)

## 致谢

- [Mortal](https://github.com/Equim-chan/Mortal)
- [mjai](https://github.com/mjai-jp/mjai)
- 天凤 / 雀魂相关社区数据与工具链
