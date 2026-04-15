# 变更总结 2026-04-16

本文档记录本次迭代的主要变更。

---

## 1. Xmodel1 新架构

### 新增文件
- `rust/keqing_core/src/xmodel1_export.rs` - Rust 端候选导出
- `rust/keqing_core/src/xmodel1_schema.rs` - Xmodel1 schema 定义
- `rust/keqing_core/tests/xmodel1_schema_test.rs` - schema 测试
- `src/xmodel1/` - Python 训练层
  - `model.py` - 模型定义（共享状态编码器 + candidate scorer）
  - `trainer.py` - 训练循环
  - `cached_dataset.py` - 缓存 dataset
  - `adapter.py` - 推理适配器
  - `features.py` - 特征处理
  - `candidate_quality.py` - 候选质量计算
  - `schema.py` - 数据 schema
- `configs/xmodel1_default.yaml` - 训练配置
- `configs/xmodel1_preprocess.yaml` - 预处理配置
- `scripts/train_xmodel1.py` - 训练脚本
- `scripts/preprocess_xmodel1.py` - 预处理脚本
- `scripts/review_xmodel1.py` - Review 脚本
- `evals/xmodel1/` - 评测模块
- `tests/test_xmodel1_*.py` - 大量测试覆盖

### 设计要点
- Rust 统一负责数据处理与 candidate 级质量生成
- Python 负责模型训练与推理
- 以 discard candidate ranking 为核心
- 目标：教模型学会比较合法候选动作的优劣

---

## 2. Keqingv31 新模型线

### 新增文件
- `src/keqingv31/` - 新模型目录
  - `model.py` - 模型定义
  - `trainer.py` - 训练器
- `scripts/train_keqingv31.py` - 训练脚本
- `configs/keqingv31_default.yaml` - 配置文件

### 定位
- keqingv3 的后续改进线
- 在 v3 统一动作空间上继续增强

---

## 3. Keqingv4 探索线启动

### 新增文件
- `src/keqingv4/__init__.py` - 模型入口
- `configs/keqingv4_default.yaml` - 训练配置
- `configs/keqingv4_exp_b3_probe.yaml` - B3 实验配置
- `configs/keqingv4_preprocess.yaml` - 预处理配置
- 相关测试文件

### 设计方向
- 统一动作空间外壳 + typed consequence-aware decoder 内核
- typed summary 语义（discard/call/special）
- 运行时 typed summaries 缓存

---

## 4. Tenhou 天凤平台对接

### 新增文件
- `src/gateway/riichi_dev_client.py` - 立直开发版客户端
- `src/gateway/tenhou_bot_client.py` - 天凤 bot 客户端
- `src/gateway/tenhou_bridge.py` - 天凤桥接层
- `scripts/launch_tenhou_bots.py` - bot 启动脚本
- `tests/test_riichi_dev_client.py` - 客户端测试
- `tests/test_tenhou_bot_client.py` - Bot 客户端测试
- `tests/test_tenhou_bridge.py` - 桥接层测试
- `tests/test_tenhou_public_state_sync.py` - 状态同步测试

---

## 5. Rust Semantic Core 迁移

### 进展
- `rust/keqing_core/src/lib.rs` - 新增导出
- `rust/keqing_core/src/scoring_pool.rs` - 积分池
- `rust/keqing_core/src/py_module.rs` - Python 模块接口

### 接管内容
- `legal_actions.rs` - 合法动作结构生成
- hora 候选编排与 shape precheck
- snapshot/state-side hora 输入准备
- 136 tile allocation
- cost->deltas 规则结算
- HoraResult 外围结果拼装

### 测试覆盖
- `tests/test_rust_state_core_parity.py` - 状态核心 parity（7 passed）
- `tests/test_rust_legal_actions_parity.py` - 法律动作 parity（126 passed）
- `tests/test_rust_hora_shape_parity.py` - hora shape parity（39 passed）

---

## 6. 3n+1/3n+2 Rust Native Seams

### 优化内容
- `progress_summary.rs` - 进度摘要 Rust 实现
- `progress_batch.rs` - 批量进度处理
- 移除 Python summary callbacks
- 推进 3n+2 discard pruning 到 Rust
- 减少 3n+2 candidate fanout
- 切换到 delta helpers

### 提交记录
- `655d3ed` - 完成 native 3n+2 seam，移除 Python summary callbacks
- `9a99628` - 移动完整 3n+1 summary path 到 native seam
- `6626b1a` - 移动 3n+2 final candidate ranking 到 native seam
- `ebcea67` - 推送 3n+2 discard pruning 到 native candidate seam
- `6131c02` - 减少 progress summaries 前的 3n+2 candidate fanout
- `2db1de2` - 切换主路径到 delta helpers，减少 3n+1 progress reanalysis

---

## 7. Preprocess 优化

### 优化内容
- `c49a37d` - 缓存 score setup，削减 preprocess legality 成本
- `79b31c2` - 缓存 tracker-derived tile stats，削减 snapshot 构建
- `94dd804` - 完成 table-driven shanten replacement
- `9ce98a5` - 移除 v3 encoding 中的二阶 progress heuristics

---

## 8. 其他改进

### 配置更新
- `AGENTS.md` - 代理编排文档更新
- `README.md` - 文档重构

### Gateway 更新
- `src/gateway/main.py` - 重构
- `src/gateway/responder.py` - 小幅更新

### Inference 更新
- `src/inference/keqing_adapter.py` - 适配器增强
- `src/inference/scoring.py` - 评分增强
- `src/inference/bot_registry.py` - Bot 注册表

### Mahjong Env 更新
- `src/mahjong_env/legal_actions.py` - 法律动作
- `src/mahjong_env/scoring.py` - 评分大幅更新
- `src/mahjong_env/state.py` - 状态更新

---

## 9. 测试覆盖

新增测试文件（部分）：
- `tests/test_bot_model_version.py`
- `tests/test_gateway_lobby_join.py`
- `tests/test_model_v4_shapes.py`
- `tests/test_training_v4_smoke.py`
- `tests/test_xmodel1_adapter.py`
- `tests/test_xmodel1_bot_registry.py`
- `tests/test_xmodel1_cached_dataset.py`
- `tests/test_xmodel1_dataset.py`
- `tests/test_xmodel1_eval_runner.py`
- `tests/test_xmodel1_inference_adapter.py`
- `tests/test_xmodel1_model_shapes.py`
- `tests/test_xmodel1_model_smoke.py`
- `tests/test_xmodel1_preprocess.py`
- `tests/test_xmodel1_replay_integration.py`
- `tests/test_xmodel1_review_export.py`
- `tests/test_xmodel1_runtime_adapter.py`
- `tests/test_xmodel1_rust_contract.py`
- `tests/test_xmodel1_schema.py`
- `tests/test_xmodel1_train_smoke.py`
- `tests/test_xmodel1_training_smoke.py`

---

## 统计

- 变更文件数：99
- 新增行数：~8600+
- 删除行数：~400

---

## 后续关注

1. Xmodel1 baseline 训练稳定性验证
2. Keqingv4 typed summary 向 Rust 迁移
3. Rust semantic core 继续侵入剩余 Python 语义层
4. meld continuation 语义闭环验证
