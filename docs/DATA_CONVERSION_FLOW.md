# 数据转换流程说明

## 概述

本文档详细说明从 Tenhou 6 格式数据到模型训练数据的完整转换链路。

## 数据格式

### 原始格式: Tenhou 6 JSON
- **位置**: `dataset/tenhou6/{dataset_name}/`
- **文件类型**: `.json`
- **格式说明**: 天凤录制的标准格式，包含对局的所有动作记录

### 中间格式: MJAI JSONL
- **位置**: `artifacts/converted/{dataset_name}/`
- **文件类型**: `.jsonl` (JSON Lines)
- **格式说明**: 统一的麻将动作序列格式，便于后续处理

### 训练格式: Supervised Samples
- **使用方式**: 直接加载到内存进行训练
- **格式说明**: Python 对象，包含状态、动作、价值等标签

## 转换链路详解

### 第一阶段: 数据格式转换

**输入**: Tenhou 6 JSON
```json
{
  "name": ["玩家A", "玩家B", "玩家C", "玩家D"],
  "log": [
    [
      [0, 0, 0],           // 回合信息: [场风, 本场, 供托]
      [25000, 25000, 25000, 25000],  // 起始得分
      [19],                // 宝牌指示牌
      [],                  // 里宝牌
      [1,2,3,4,5,6,7,8,9,19,20,21,22,23,24,25,26,27,28],  // 东家手牌
      [11, 21, 31, 41],    // 东家摸牌序列
      [12, 22],            // 东家舍牌序列
      // ... 其他三家数据
    ]
  ]
}
```

**输出**: MJAI JSONL (每行一个事件)
```json
{"type": "start_game", "names": ["玩家A", "玩家B", "玩家C", "玩家D"]}
{"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0, ...}
{"type": "tsumo", "actor": 0, "pai": "5m"}
{"type": "dahai", "actor": 0, "pai": "1m", "tsumogiri": false}
...
{"type": "hora", "actor": 0, "target": 2, ...}
{"type": "end_kyoku"}
```

### 第二阶段: 训练样本构建

**输入**: MJAI JSONL
**处理函数**: `build_supervised_samples()`

**输出**: 训练样本列表
```python
{
    "state": {
        "tehais": [[...], [...], [...], [...]],  # 各家手牌
        "dora": ["5m"],                           # 宝牌
        "kyotaku": 0,                             # 累计供托
        ...
    },
    "action": {
        "type": "dahai",
        "pai": "1m",
        "tsumogiri": false
    },
    "value": 0.85,  # 胜率估计
    "actor": 0
}
```

## 转换实现

### 1. libriichi CLI (推荐)
- **实现语言**: Rust
- **位置**: `third_party/Mortal/target/release/deps/libriichi.so`
- **优点**: 速度快，转换准确
- **调用方式**:
```bash
libriichi convert --input input.json --output output.jsonl --format mjai
```

### 2. Python 回退方案
- **实现位置**: `src/convert/libriichi_bridge.py`
- **函数**: `_fallback_convert_tenhou_json_to_mjai()`
- **触发条件**: libriichi 二进制文件不存在或调用失败
- **优点**: 无需额外依赖，纯 Python 实现
- **缺点**: 速度较慢，部分复杂动作可能解析不完整

### 3. riichienv 包
- **功能**: 牌山格式转换 (MPSZ ↔ MJAI ↔ TID)
- **不支持**: 天凤格式解析
- **适用场景**: 格式验证、格式互转

## 优化策略

### 1. 数据预处理 (推荐)
```bash
# 预先转换所有数据
./preprocess_ds4.sh ds4 8

# 然后训练
./train_from_converted.sh ds4
```

**优点**:
- 避免训练时等待数据转换
- 可以多次复用转换后的数据
- 支持并行预处理

**缺点**:
- 需要额外磁盘空间存储转换后的数据
- 首次需要完整转换时间

### 2. 增量转换
```bash
# 跳过已转换的文件
python preprocess_ds4.py \
    --raw-dir ../dataset/tenhou6/ds4/ \
    --out-dir ../artifacts/converted/ds4 \
    --libriichi-bin ../third_party/Mortal/target/release/deps/libriichi.so \
    --skip-existing  # 默认行为
```

### 3. 并行转换
```bash
# 使用多线程加速
python preprocess_ds4.py \
    --raw-dir ../dataset/tenhou6/ds4/ \
    --out-dir ../artifacts/converted/ds4 \
    --num-workers 16
```

## 数据存储位置

### 原始数据 (不可修改)
```
dataset/tenhou6/
├── ds2/          # 原始 JSON 文件
├── ds3/          # 原始 JSON 文件
└── ds4/          # 原始 JSON 文件 (当前使用)
```

### 转换后数据 (可重新生成)
```
artifacts/converted/
├── train/        # 训练集转换结果
├── train_ds2/    # ds2 转换结果
├── train_ds3/    # ds3 转换结果
└── ds4/          # ds4 转换结果 (当前使用)
```

### 训练输出
```
artifacts/sl/
├── ds2/          # ds2 训练模型
├── ds3/          # ds3 训练模型
└── ds4/          # ds4 训练模型 (当前使用)
```

## 常见问题

### Q1: 为什么转换后的数据比原始数据大？
- 原始数据使用紧凑的整数编码
- 转换后使用可读的字符串编码
- 这是正常的，大小差异约 2-3 倍

### Q2: 可以删除转换后的数据吗？
- 可以，但每次训练都需要重新转换
- 建议保留以便重复使用

### Q3: 如何验证转换是否正确？
```bash
# 查看转换后的文件
head -20 artifacts/converted/ds4/xxx.jsonl

# 验证文件数量
find artifacts/converted/ds4/ -name "*.jsonl" | wc -l

# 运行预处理脚本会自动验证
./preprocess_ds4.sh ds4
```

### Q4: libriichi 和 Python 回退方案哪个更好？
- **libriichi**: 速度快，结果准确，优先使用
- **Python 回退**: 作为备用方案，速度慢但无需额外依赖
- 预处理脚本会优先尝试 libriichi，失败时自动使用回退方案

### Q5: riichienv 包能直接解析天凤格式吗？
- **不能**。riichienv 只提供牌山格式转换（MPSZ ↔ MJAI ↔ TID）
- 天凤格式解析由 `libriichi_bridge.py` 的回退方案实现

## 最佳实践

1. **首次使用**: 运行完整预处理流程
   ```bash
   ./preprocess_ds4.sh ds4 8
   ```

2. **后续训练**: 直接使用已转换的数据
   ```bash
   ./train_from_converted.sh ds4
   ```

3. **增量添加数据**: 只转换新文件
   ```bash
   # 将新文件放到 dataset/tenhou6/ds4/
   ./preprocess_ds4.sh ds4 8
   # 自动跳过已存在的文件
   ```

4. **调试模式**: 查看详细转换信息
   ```bash
   cd src
   ../.venv/bin/python -m train.train_sl \
       --raw-dir ../dataset/tenhou6/ds4/ \
       --converted-dir ../artifacts/converted/ds4 \
       --config ../configs/v2.yaml \
       --view-mode all
   ```
