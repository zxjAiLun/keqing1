# 2026-03-21 实现总结：mjai-log 与 tenhou6 往返转换

## 概述

本次工作修复了 `mjai-jsonl` ↔ `tenhou6` 往返转换中的三个关键 bug，实现了 **1948/1948 测试用例 100% 匹配**。

## 问题背景

天凤（tenhou.net）牌谱有两种常用格式：
1. **tenhou6 JSON** - 天凤六麻格式，结构化的对战记录
2. **mjai-jsonl** - 麻将AI协议格式，便于AI训练

需要实现两种格式的往返转换，确保信息不丢失。

## 修复的 Bug

### Bug 1: `libriichi_bridge.py` 中红5牌转换错误

**位置**: [libriichi_bridge.py#L69](file:///home/bailan/project/keqing1/src/convert/libriichi_bridge.py#L69)

**问题**:
```python
# 修复前
def _dora_indicator_code(tile_code: str) -> str:
    if tile_code in ("5mr", "5pr", "5sr"):
        return tile_code[0] + tile_code[2]  # "5mr" → "5r" (错误!)
    return tile_code
```
将红5万 `"5mr"` 错误转换为 `"5r"`，导致无效的 mjai tile。

**修复**:
```python
# 修复后
def _dora_indicator_code(tile_code: str) -> str:
    return tile_code
```
红5牌（如 `"5mr"`）已经是有效的 mjai tile，无需转换。

---

### Bug 2: `mjai_jsonl_to_tenhou6.py` 中 dora 指示牌转换错误

**位置**: [mjai_jsonl_to_tenhou6.py#L47](file:///home/bailan/project/keqing1/src/tools/mjai_jsonl_to_tenhou6.py#L47)

**问题**:
```python
# 修复前
def dora_indicator_code(tile: str) -> int:
    if len(tile) == 3 and tile[0] == "5" and tile[2] == "r" and tile[1] in SUIT_BASE:
        return SUIT_BASE[tile[1]] + 5  # "5pr" → 22 (错误! 应为 52)
    return mjai_tile_to_tenhou_code(tile)
```
将红5宝牌错误转换为普通5，丢失了红5信息。

天凤牌码对照：
- 普通 5m = 15, 5p = 25, 5s = 35
- **红5** 5mr = **51**, 5pr = **52**, 5sr = **53**

**修复**:
```python
# 修复后
def dora_indicator_code(tile: str) -> int:
    if len(tile) == 3 and tile[0] == "5" and tile[2] == "r" and tile[1] in SUIT_BASE:
        return 50 + {"m": 1, "p": 2, "s": 3}[tile[1]]  # "5mr" → 51, "5pr" → 52, "5sr" → 53
    return mjai_tile_to_tenhou_code(tile)
```

---

### Bug 3: 多初始 dora 指示牌未完全输出

**位置**: [libriichi_bridge.py#L133](file:///home/bailan/project/keqing1/src/convert/libriichi_bridge.py#L133)

**问题**: 天凤数据中某些局一开始就有多个 dora（如 `[31, 24]`），但只输出了第一个 dora。

**修复**:
```python
# 在 start_kyoku 事件后添加
for extra_dora in dora_indicators[1:]:
    events.append({"type": "dora", "dora_marker": _dora_indicator_code(_tenhou_tile_to_mjai(extra_dora))})
```

## 实现原理

### 转换流程

```
tenhou6 JSON ──→ mjai-jsonl ──→ tenhou6 JSON
    ↑                │                ↑
    │                │                │
tenhou6格式    libriichi_bridge   mjai_jsonl_to_tenhou6
    ↑                │                │
    └────────────────┴────────────────┘
              往返转换
```

### 关键转换函数

1. **`_tenhou_tile_to_mjai()`**: tenhou 码 → mjai tile（如 31 → "1p", 52 → "5pr"）
2. **`mjai_tile_to_tenhou_code()`**: mjai tile → tenhou 码（如 "5pr" → 52）
3. **`dora_indicator_code()`**: dora 指示牌专用转换（处理红5）

### 测试结果

```
=== 总结: 1948/1948 匹配 (0 失败) ===
```

## 文件清单

| 文件 | 功能 |
|------|------|
| `src/convert/libriichi_bridge.py` | tenhou6 → mjai-jsonl 转换 |
| `src/tools/mjai_jsonl_to_tenhou6.py` | mjai-jsonl → tenhou6 转换 |
| `src/convert/validate_mjai.py` | mjai 格式验证 |
| `src/tools/convert_one.py` | 单文件转换工具 |
| `src/tools/convert_dir.py` | 批量转换工具 |
| `src/convert/batch_converter.py` | 批量转换核心逻辑 |

## 麻将牌码对照表

### 天凤牌码（tenhou）
| 牌 | 码 |
|---|---|
| 1-9m | 11-19 |
| 1-9p | 21-29 |
| 1-9s | 31-39 |
| E/S/W/N/P/F/C | 41-47 |
| 红5mr | **51** |
| 红5pr | **52** |
| 红5sr | **53** |

### mjai tile 格式
| 格式 | 示例 | 说明 |
|------|------|------|
| 字牌 | "E", "S", "W", "N" | 直接用字母 |
| 数牌 | "1m", "2p", "3s" | 数字+花色 |
| 红5 | "5mr", "5pr", "5sr" | 数字+r+花色首字母 |
