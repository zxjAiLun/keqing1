# Project Memory — keqing1

## playwithyou（天凤个室 NoName + Mortal）事件格式铁律
原生 libriichi `Bot`（Rust，`third_party/Mortal/libriichi/src/mjai/event.rs`）对 mjai 事件校验极严，
格式错一个字段就会在开局瞬间抛 Rust 异常 → 该 bot 线程崩溃 → 全桌 AI 同时掉线。

- `start_game` 必须带 `names: [String; 4]`（**长度恰好 4**）。发 `[]` 会报
  "invalid length 0, expected an array of length 4"。
  → `responder.Taikyoku` 从 `state.names`（UN 消息解析的 4 名字）取，不足补 `""`、截断到 4。
- `start_kyoku`：`kyoku` 是 `BoundedU8<1,4>`（**1 索引**），天凤 seed[0] 是 0 索引 → 必须 `(seed[0]%4)+1`；
  `scores` 是 `[i32;4]` **原始点数**（如 25000），不是 `*100`。
  → `responder.Init` 已修正。dora_marker 必须是合法牌字符串。
- 任何 `react()` 调用点都已包 try/except：单条坏事件只记 traceback 并返回
  `{"type":"none","actor":seat}`，避免一个 bot 崩溃连累全桌。

## 环境注意
- 本机 shell 是 Windows（win32）。`libriichi.so` 是 Linux 共享库，Windows 下无法 import，
  故真实模型 E2E 只能在该项目的 Linux/WSL 运行时跑（用户实际环境能加载，已验证"进桌并开局"）。
- `keqing_core` 等 Rust 扩展在本机新 venv 未编译时，相关单测会报
  "Rust replay state snapshot capability is not available"——属环境性失败，非代码回归。
- 回归单测 `tests/test_responder_playwithyou.py`（纯 Python，不依赖 torch/libriichi）守住上述格式铁律。
