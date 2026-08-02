# Mortal 训练线与 Live Ladder 发布

## 目的

native runner 现在支持显式 opt-in 的 Live Ladder 发布。默认不发布，避免历史评测、数据生成和普通 smoke 意外切换外部赛季 registry。

发布器始终接收完整日志目录，而不是最近新增的增量目录。每次发布都会重新构建累计 Pt/Rating 快照，并通过 publisher 的 staging、校验和原子 registry switch 更新在线数据面。

## Windows 用法

对单个四模型 native run：

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

两个 cadence 可以同时提供，任一条件满足就发布。若只传 `--ladder-registry` 而不传 cadence，则只在 runner 最终完成时发布一次。`--ladder-publish-best-effort` 只在希望发布失败不阻塞本轮实验时使用；正式联赛默认 fail fast。

纯 selfplay runner `scripts/mortal/selfplay_native.py` 使用同一组 `--ladder-*` 参数。`--defer-reports` 不会关闭周期发布；它只延迟 runner 本地的最终统计文件。

## 约束

- registry 必须是外部动态赛季，且 `status` 不能是 `completed`；仓库内的 `final-balanced-2026-07.json` 是历史赛季，不用于发布。
- registry 中的 `model_id/account_id` 必须覆盖日志中的归一化模型标签；publisher 会通过 `ladder.validate_snapshot` 拒绝未注册账号。
- 传给 publisher 的每个 `--log-dir` 必须是该赛季截至当前的完整日志集合。多目录赛季应直接调用 `publish_ladder_snapshot.py`，重复传入所有完整目录，并按需要加 `--interleave-log-dirs`。
- 周期发布按 native batch 完成后触发；不会在 batch 中间读取半成品日志。
- 发布失败默认让 native runner 失败，防止用户误以为在线天梯已经更新。只有显式 `--ladder-publish-best-effort` 才会记录错误并继续。
- publisher 不参与训练 loss、checkpoint 选择或 promotion gate；它只是赛后 Pt/Rating 数据面的派生输出。

## 手动多目录发布

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

这条命令的输入必须是截至当前的完整目录集合；不能只把最新 25 局传进去，否则 Pt/Rating 会从初始值重新计算。
