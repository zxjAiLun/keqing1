# v5model 后续版本改进 SPEC

本文件记录当前版本（v5）已知问题和下一版本（v6）的改进方向。
训练中请勿修改，等当前训练结束后参考执行。

---

## 已确认不改（v5 现状）

- `split_files` 按文件（局）划分 train/val ✅ 正确，无需改动
- 模型参数量维持 ~1.7M（hidden_dim=256, num_res_blocks=4）
  - 原因：RTX 4060 Laptop 8G 显存，当前已 98% 利用率、87°C，扩参数无意义
- 训练/推理 shanten 计算路径不对齐（推理走 fallback）—— 下版本修
- style 特征（scalar 16-19）训练时全为 0，4 个 slot 空置 —— 下版本决策是否启用

---

## v6 优先改进项

### 1. 数据质量过滤（高优先级）
- **问题**：CSV 可能含低段位对局，低质量数据压低 acc 上限
- **方案**：`download_and_convert.py` 加过滤，只保留凤凰桌（四人最高桌）对局
- **判断依据**：tenhou.net URL 中 `tw=` 参数或房间标识

### 2. 训练/推理 shanten 对齐
- **问题**：`features.py:255-267`，训练时 shanten/waits 由 `replay.py` 注入 snap（精确），
  推理时走 fallback（`HandEvaluator`），两路计算方式不同
- **方案**：统一推理时也用 libriichi PlayerState 计算，或确保 fallback 结果与训练一致

### 3. steps_per_epoch 动态计算
- **问题**：`configs/v5_default.yaml` 中 `steps_per_epoch: 5000` 硬编码，
  数据量变化后 cosine decay 计划失准
- **方案**：训练开始时跑一遍数据集统计实际 steps，写入 cfg 再传给 trainer

### 4. value_loss_weight 调参
- **当前（keqingv1）**：`value_loss_weight=0.5`（已从 1.5 降低，原值过高压制 policy 学习）
- **建议**：如 acc 稳定提升可进一步降到 0.2~0.3
- **前提**：先确认 value MSE 趋势正常（持续下降）再调

### 5. global avg pool 位置信息丢失
- **问题**：`model.py:123` `x.mean(dim=-1)` 把 34 个牌位压平，丢失牌种间相对关系
- **候选方案**：
  - 改为 flatten 后接 Linear（参数量会增加，需评估显存）
  - 最后一层用更大 kernel（如 kernel_size=5）再 pool
- **注意**：改结构需重新训练，不能续训旧 checkpoint

### 6. style 特征启用
- **现状**：`scalar[16-19]` 为 style 参数，训练时全为 0，模型未学习风格调控
- **方案**：推理时按需传入非零 style 值，观察模型行为是否有意义
- **前提**：需要先评估当前模型对 style=0 之外的输入的泛化能力（可能需要训练时加扰动）

---

## GPU 约束备忘

- 硬件：RTX 4060 Laptop，8188MiB 显存
- 当前训练状态：显存占用 ~1094MiB，GPU 利用率 98%，温度 87°C
- 瓶颈是算力，不是显存 —— 显存有余量但 GPU 已满载
- 下版本若扩参数，需先评估 batch_size 能否维持，或改用梯度累积

---

## 已完成修复（keqingv1 训练链路）

- `replay.py` call 样本 `waits_count` 改用副露+打牌后状态（`waits_after_cnt_vt`）
- `trainer.py` mask 填充值统一为 `-1e9`
- `keqingv1/features.py` ch 24-55 改为相对 slot 编码，消除绝对 pid 不一致
- `keqingv1/features.py` scalar[12] 改为自家舍牌数（原为 min 各家）
- `keqing_default.yaml` `value_loss_weight` 1.5 → 0.5
- `keqingv1/features.py` scalar[10/21/22] 冗余清零，scalar[13-16/28-35] 改为相对 slot
- `keqingv1/features.py` ch 8-16 重构（删 presence，保留立直宣言+手切），ch 14-16 改为他家副露 presence
- `keqingv1/features.py` ch 18 改为 dora 实际牌（累加），ch 19-22 空闲
- `keqingv1/features.py` ch 57 改为最新立直家立直后其他家打过的牌
- `_DORA_NEXT` 提升为模块级常量

**待完成（下一步）：** 删除所有空闲槽，C_TILE 64→54，N_SCALAR 64→48，重新映射所有索引

---

## 数据扩充目标

- 当前：12870 个 .mjson（ds1-ds13）
- 目标：3 万局以上再考虑扩大模型容量
- 数据来源：nodocchi.moe 导出，正在后台下载更多牌谱
- 建议新数据存 ds14 起，保持旧数据不变以便对比实验
