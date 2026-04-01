# Static Tables

这里放运行时可直接加载的静态统计表编译产物。

当前约定：

- 原始来源：`dataset/mahjong_heratu_data/csv`
- 编译脚本：`scripts/build_static_tables.py`
- 默认 bundle：`data/static/tables/mahjong_book_stats.json`

这层的目标是：

- 不在热路径直接读 CSV
- 先离线编译成结构化静态表
- 再由 `src/static_tables/` 提供统一 lookup 接口
