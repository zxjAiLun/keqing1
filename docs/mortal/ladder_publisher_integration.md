# Ladder Publisher Integration

> 更新：2026-08-03（Round 5：compact snapshot / retention / telemetry；review 收口：race-safe ownership / 幂等 contract）
> 关联：`scripts/mortal/publish_ladder_snapshot.py`、`src/replay/ladder.py`、
> `scripts/mortal/four_player_native.py`、`scripts/mortal/selfplay_native.py`

## 职责边界

`publish_ladder_snapshot.py` 是训练线到 Live Ladder 的**独立生产者**：

```text
训练线持续产生 mjai 日志
→ batch 边界调用 publisher
→ 完整报告在隐藏 staging 构建
→ 提取紧凑快照 → 校验 → manifest → 原子 materialize
→ registry 条件原子切换
→ runtime 30s 轮询自动读取
```

**快照是紧凑运行产物，不是原始牌谱备份。** 默认快照只保留：

- 必需（runtime/API 消费）：`account_summary.json`、`account_ledger.jsonl`、`rating_curve.csv`
- 建议（人工查看）：`account_summary.csv`、`account_summary.md`、`per_game_results.csv`、`detailed_stats.json`、`detailed_stats.md`
- `manifest.json`（schema `keqing.ladder.snapshot.v1`）

**不保留 `account_logs/`**：这些是 build 阶段为计算详细统计而生成的派生副本，
原始完整 mjai 日志仍归训练线所有（`keqing-data/ladder/logs/`），runtime 不读取 `account_logs`。

## 训练线 runner 接入（opt-in）

native runner 现在支持显式 opt-in 的 Live Ladder 发布。**默认不发布**，
避免历史评测、数据生成和普通 smoke 意外切换外部赛季 registry。

发布器始终接收**完整日志目录**，而不是最近新增的增量目录。每次发布都会重新构建
累计 Pt/Rating 快照，并通过 publisher 的 staging、校验和原子 registry switch
更新在线数据面。

### Windows 用法（单四模型 native run）

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-win"
uv run --no-sync python scripts/mortal/four_player_native.py `
  --model K0_70k=artifacts/mortal_training/checkpoints/mortal_default_70k_promoted_candidate.pth `
  --model ext_mortal=artifacts/external_mortal_20240308_best_min.pth `
  --model candidate_a=artifacts/experiments/<run>/checkpoints/mortal.pth `
  --model candidate_b=artifacts/experiments/<run>/checkpoints/mortal.pth `
  --output-dir artifacts/experiments/<run>/league_run `
  --device cuda --require-cuda --seat-mode random `
  --games 1000 --native-batch-games 250 --progress-every 25 `
  --ladder-registry E:/AUbuntuProject/keqing-data/ladder/registries/dev-live.json `
  --ladder-publish-every-games 25 `
  --ladder-snapshot-root E:/AUbuntuProject/keqing-data/ladder/seasons/dev-live/snapshots
```

也可以只按时间触发：

```text
--ladder-publish-every-seconds 600
```

两个 cadence 可以同时提供，任一条件满足就发布。若只传 `--ladder-registry` 而不传
cadence，则只在 runner 最终完成时发布一次。`--ladder-publish-best-effort` 只在希望
发布失败不阻塞本轮实验时使用；正式联赛默认 fail fast。

纯 selfplay runner `scripts/mortal/selfplay_native.py` 使用同一组 `--ladder-*` 参数。
`--defer-reports` 不会关闭周期发布；它只延迟 runner 本地的最终统计文件。

### runner 接入约束

- registry 必须是外部动态赛季，且 `status` 不能是 `completed`；仓库内的
  `final-balanced-2026-07.json` 是历史赛季，不用于发布。
- registry 中的 `model_id/account_id` 必须覆盖日志中的归一化模型标签；
  publisher 会通过 `ladder.validate_snapshot` 拒绝未注册账号。
- 传给 publisher 的每个 `--log-dir` 必须是该赛季截至当前的完整日志集合。
  多目录赛季应直接调用 `publish_ladder_snapshot.py`，重复传入所有完整目录，
  并按需要加 `--interleave-log-dirs`。
- 周期发布按 native batch 完成后触发；不会在 batch 中间读取半成品日志。
- 发布失败默认让 native runner 失败，防止用户误以为在线天梯已经更新。
  只有显式 `--ladder-publish-best-effort` 才会记录错误并继续。
- publisher 不参与训练 loss、checkpoint 选择或 promotion gate；它只是赛后
  Pt/Rating 数据面的派生输出。

### 手动多目录发布

```powershell
$env:UV_PROJECT_ENVIRONMENT = ".venv-win"
uv run --no-sync python scripts/mortal/publish_ladder_snapshot.py `
  --registry E:/AUbuntuProject/keqing-data/ladder/registries/dev-live.json `
  --log-dir E:/AUbuntuProject/keqing-data/leagues/run_a/logs `
  --log-dir E:/AUbuntuProject/keqing-data/leagues/run_b/logs `
  --interleave-log-dirs `
  --mortal-root third_party/Mortal `
  --snapshot-root E:/AUbuntuProject/keqing-data/ladder/seasons/dev-live/snapshots
```

这条命令的输入必须是截至当前的完整目录集合；不能只把最新 25 局传进去，
否则 Pt/Rating 会从初始值重新计算。

## 发布流程（Round 5 后）

```text
创建隐藏 staging：.<snapshot_id>.<pid>.<uuid>.staging
→ staging/build 运行完整 build_platform_account_report
→ 从 build 提取线上所需产物到 staging/snapshot（紧凑快照）
→ validate_snapshot（三件套完整性 + 注册表账号一致性）
→ 写 manifest（含 telemetry 与 source_fingerprint）
→ os.replace 将 snapshot staging 原子改名为最终 snapshot
→ conditional_switch_registry（per-registry 锁 + 唯一 tmp + os.replace）
→ enforce_retention（成功切换后才清理旧快照）
→ 清理完整 build staging
```

失败语义：

- 构建/校验失败：只清理隐藏 staging，不生成最终快照，旧 registry 保持可用；
- registry 切换失败：仅当本次调用确实 materialize 该目录且 registry 尚未切换时，
  才删除该孤儿快照；**绝不删除其他调用创建的快照**；
- registry 已切换后：任何后续异常（含 retention 扫描故障）都降级为
  `retention_errors` 返回，**绝不删除当前线上快照**；
- retention 清理失败：记录 `retention_errors` 返回，不把已成功的发布标记为失败。

## 并发所有权

- 最终快照 ID 天然唯一：`<YYYYmmdd-HHMMSS>-<microseconds>-<uuid8>`，
  两个 publisher 同秒启动也不会争用同一最终目录；
- `materialized_by_this_publish` / `registry_switched` 两个状态明确发布所有权：
  - registry 尚未切换：失败时允许删除本次自己 materialize 的快照；
  - registry 已切换：任何后续异常都不得删除当前快照。

## 幂等与指纹

发布前读取当前 registry 指向快照的 manifest，当以下条件**全部一致**时直接返回
`{"skipped_unchanged": true, "registry_switched": false}`，不重建：

- season contract（注册表模型/账号/状态，除 report_dir）
- `build_contract_version`（report 派生逻辑契约版本，算法更新时递增）
- `source_fingerprint`（**builder 真实处理顺序**下的源日志路径 + 大小 + mtime_ns）
- `rank_points`
- `platform_model_label`
- `keep_account_logs`
- `preserve_log_dir_order` / `interleave_log_dirs`

注意：

- `source_fingerprint` 复用 `iter_log_files` 的实际顺序，`[A,B]` 与 `[B,A]`
  在 preserve/interleave 模式下产生不同指纹，不会被误判为 unchanged；
- **dry-run 始终真正构建并校验**，不命中 skip；
- 命中 skip 前会重新 `validate_snapshot` 当前快照；三件套缺失或含非法 UTF-8 时
  忽略 skip 并完整重建自愈；
- 任何源日志增加、大小或 mtime 变化都会产生新快照。

`snapshot_total_bytes` 为递归统计的整个快照目录字节数（含 manifest.json、
`account_logs/` 等全部文件），与 manifest 一致。

## 快照历史保留

```text
--retain-snapshots N    默认 24；N = 0 表示无限保留（禁用自动清理）
```

- 只在 **registry 成功切换之后** 清理；
- 保留最近 N 个**有效快照**（有合法 manifest 且 season_id 一致）；
- **绝不删除**：当前 registry 指向的快照、manifest 记录的 previous snapshot
  （回滚点）、无 manifest 的目录、season_id 不一致的目录、隐藏 staging、
  snapshots 根目录中的其他人工文件。

## 调试开关

```text
--keep-account-logs     仅在诊断时使用：在快照中保留 account_logs/ 派生副本
```

## 已知成本与建议

25 场 cadence 在大赛季下仍有**累计重建成本**：
N 次发布累计处理约 N^2/2 场等价工作量（6000 场 / 每 25 场 ≈ 240 次发布
≈ 120 倍最终数据量）。当前先通过 manifest telemetry
（`build_duration_seconds` / `snapshot_total_bytes` / `source_total_bytes`）
观察真实构建耗时，再决定是否实现增量状态机。

## 数据所有权边界

- 一个 registry 对应一套完整赛季日志集合；
- 多个并行 runner 共同组成同一赛季时，不能让各自把单目录发布到同一 registry
  （最后发布者会覆盖赛季视图）。使用手动多目录发布（见上文）；
- 或为每个实验建立独立动态 season。
