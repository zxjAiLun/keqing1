# 数据转换流程说明

## 概述

从天凤牌谱链接到 v5model 训练数据的完整链路。

```
CSV（nodocchi.moe 牌谱链接）
    └─→ [download_and_convert.py] 下载 XML + 转换
            ├─→ tenhou_xml_to_json.py → Tenhou6 JSON（临时）
            └─→ convlog → .mjson (mjai JSONL)
                    └─→ [MjaiIterableDataset] → (tile_feat, scalar, mask, action, value)
                            └─→ MahjongModel 训练
```

也可以从已有的 Tenhou6 JSON 直接转换：

```
Tenhou6 JSON
    └─→ convlog → .mjson (mjai JSONL)
            └─→ [MjaiIterableDataset] → ...
```

---

## 数据位置

| 阶段 | 路径 | 格式 |
|------|------|------|
| 牌谱链接 CSV | `dataset/links/csv/` | `.csv`（nodocchi.moe 导出）|
| 原始天凤数据 | `dataset/tenhou6/{ds}/` | `.json`（Tenhou6 格式）|
| 转换后 mjai | `artifacts/converted_mjai/{ds}/` | `.mjson`（每行一个事件）|
| 模型输出 | `artifacts/models/modelv5/` | `.pth` checkpoint |

当前已有数据集：`ds1` ~ `ds13`（共 12870 个 `.mjson` 文件）

---

## 第一阶段：Tenhou6 → mjai JSONL

天凤格式事件流 → 标准 mjai 事件格式（每行一个 JSON 事件）。

### mjai 事件格式示例

```json
{"type": "start_game", "names": ["A", "B", "C", "D"]}
{"type": "start_kyoku", "bakaze": "E", "kyoku": 1, "honba": 0, "kyotaku": 0, "oya": 0, "dora_marker": "5m", "tehais": [[...], [...], [...], [...]]}
{"type": "tsumo", "actor": 0, "pai": "3m"}
{"type": "dahai", "actor": 0, "pai": "9p", "tsumogiri": false}
{"type": "chi", "actor": 1, "target": 0, "pai": "6s", "consumed": ["5s", "7s"]}
{"type": "hora", "actor": 2, "target": 0, "pai": "1m", "uradora_markers": [], "deltas": [-3900, 0, 3900, 0], "ura_markers": []}
{"type": "end_kyoku"}
{"type": "end_game"}
```

### 转换工具

- **推荐**：`third_party/Mortal` 的 Rust `libriichi` CLI（速度快，结果准确）
- **备用**：`src/convert/` 下的 Python 实现

---

## 第二阶段：mjai JSONL → 训练样本

由 `src/v5model/dataset.py` 的 `MjaiIterableDataset` 完成，流式处理，不预加载全量数据。

### 处理流程

1. 按行读取 `.mjson` 文件中的事件
2. 维护 `GameState`，逐事件 `apply_event`
3. 在每个决策点（tsumo/dahai/chi/pon/kan/hora/ryukyoku）：
   - 调用 `features.encode()` 生成 `tile_feat (128,34)` + `scalar (16,)`
   - 调用 `build_legal_mask()` 生成合法动作 mask `(45,)`
   - 将实际动作映射到 action index（0-44）
4. value target：局末累计得分变化 / 30000（所有该局样本共享同一个 value）

### 样本结构

```python
(
    tile_feat,    # np.ndarray (128, 34), float32
    scalar,       # np.ndarray (16,),    float32
    mask,         # np.ndarray (45,),    bool
    action_idx,   # int,  0-44
    value,        # float, [-1, 1]
)
```

---

## 注意事项

- **赤宝牌**：弃牌动作不区分赤/普通（统一归并到牌种索引），但特征通道 ch56-58 单独标注手牌中的赤五
- **chi 动作**：只区分 Low/Mid/High（对齐 Mortal），不区分使用赤五还是普通五
- **hora**：荣和与自摸统一为 `HORA_IDX=42`，不做区分
- **IterableDataset**：没有 `len()`，DataLoader 无法预估 total batch 数

---

## 常用命令

```bash
# 验证单个 mjson 文件格式
head -5 artifacts/converted_mjai/ds1/game_0001.mjson

# 统计文件数量
find artifacts/converted_mjai/ -name '*.mjson' | wc -l

# 启动训练（验证管道）
uv run python -m train.train_v5 --config configs/v5_default.yaml

# 全量训练
uv run python -m train.train_v5 \
  --data_dirs $(echo artifacts/converted_mjai/ds{1..13}) \
  --output_dir artifacts/models/modelv5
```
