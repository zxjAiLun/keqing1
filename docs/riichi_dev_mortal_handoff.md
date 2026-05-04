# Riichi.dev Mortal Handoff

更新时间：2026-05-04

这份文档给接手 agent 用，记录 Mortal 接入 riichi.dev ranked 过程中已经确认的问题、修复口径和排查命令。不要重新从“模型是不是乱打”开始猜，先按下面证据链定位。

## 当前部署状态

pearl 部署目录：

```text
~/keqing1_deploy
```

ranked bot 在 tmux 中运行：

```bash
ssh pearl 'tmux ls'
ssh pearl 'pgrep -af "[r]iichi_dev_client.py"'
```

当前启动命令：

```bash
cd ~/keqing1_deploy
PYTHONUNBUFFERED=1 .venv/bin/python src/gateway/riichi_dev_client.py \
  --mode online \
  --queue ranked \
  --bot-name mortal \
  --model-path artifacts/mortal_serving/mortal.pth \
  --project-root . \
  --device cpu \
  --audit-log-path logs/riichi_dev/ranked-mortal.jsonl \
  --verbose
```

用 tmux 拉起：

```bash
ssh pearl 'cd ~/keqing1_deploy && tmux kill-session -t mortal_ranked 2>/dev/null || true; mkdir -p logs/riichi_dev; tmux new-session -d -s mortal_ranked "PYTHONUNBUFFERED=1 .venv/bin/python src/gateway/riichi_dev_client.py --mode online --queue ranked --bot-name mortal --model-path artifacts/mortal_serving/mortal.pth --project-root . --device cpu --audit-log-path logs/riichi_dev/ranked-mortal.jsonl --verbose 2>&1 | tee -a logs/riichi_dev/ranked-mortal.stdout.log"'
```

运行日志：

```text
~/keqing1_deploy/logs/riichi_dev/ranked-mortal.stdout.log
~/keqing1_deploy/logs/riichi_dev/ranked-mortal.jsonl
```

实时看日志：

```bash
ssh pearl 'tail -f ~/keqing1_deploy/logs/riichi_dev/ranked-mortal.stdout.log'
```

## 核心架构口径

riichi.dev 在线对战只应该响应 `request_action`。其它 WebSocket 消息是信息流，Mortal 通过每次 `request_action` 携带的 base64 `Observation` 消费 `obs.new_events()`。

Mortal 决策路径：

```text
request_action
-> Observation.deserialize_from_base64(...)
-> for event in obs.new_events(): Mortal.react(event)
-> obs.select_action_from_mjai(mortal_response)
-> legality guard
-> normalize MJAI wire payload
-> websocket.send(...)
```

不要维护第二套 event cursor。`obs.new_events()` 已经是 per-seat unseen delta stream。

## Wire Action 规则

`dahai` 必须带 `tsumogiri`：

```json
{"type":"dahai","actor":1,"pai":"8m","tsumogiri":false}
```

判断 `tsumogiri` 的实际含义：

- 打刚摸到的牌：`true`
- 手切：`false`
- 如果手里有同名牌又摸到同名牌，两者都可能合法，但仍应显式发送布尔值。

`chi` / `pon` / `daiminkan` 必须带 `target`。如果 `riichienv.Action.to_mjai()` 没带 target，网关从当前 `obs.new_events()` 中最后一条 `dahai` 的 actor 补。

`chi` 的 `consumed` 形状在 echo 对比时不能要求完全一致。服务器 observation 可能 echo 另一种等价 chi 组合，例如本地发：

```json
{"type":"chi","actor":0,"pai":"4p","consumed":["5pr","6p"],"target":3}
```

服务器 echo：

```json
{"type":"chi","actor":0,"pai":"4p","consumed":["3p","5pr"],"target":3}
```

这种不是 illegal。echo 比较对 `chi` 只看 `type + actor + pai + target`。

## Legality Guard

Mortal 原始输出不能直接信任为线上 wire action。当前网关先用 `obs.select_action_from_mjai()` 转成 `riichienv.Action`，再在合法动作集合内确认。

如果 Mortal 输出非法，日志会出现：

```text
mortal legality guard fallback
```

这表示模型输出被兜底成合法动作，不应导致 WebSocket 断开。排查时先 grep 这条日志，再看 fallback 的 `candidate`、`fallback` 和 `legal_sample`。

## Hand Boundary 同步

riichi.dev 的 observation 有时会在一个 `obs.new_events()` batch 里跨小局，例如上一局尾部 `ryukyoku/end_kyoku` 加下一局 `start_kyoku`。网关会在小局边界 reset native Mortal，并在喂事件前丢弃 stale pre-`start_kyoku` 事件。

这类日志已经降级为 debug。不要把正常 `end_kyoku/start_kyoku` 误判成 bug。

## Illegal / Chombo 处理

服务端当前规则：

- illegal / malformed action：罚满贯，`ryukyoku` reason 类似 `Error: Illegal Action by Player N`
- action timeout：服务器 fallback 摸切，不罚满贯
- websocket disconnect：服务器 fallback 摸切，不罚满贯

因此：

- 看到 `ryukyoku` + `Error: Illegal Action by Player N`：先看 N 是否等于我方 seat。
- 如果 offender 不是我方，客户端必须继续下一局，不能退出。
- 如果 offender 是我方，必须中止，不要继续把本地 Mortal 状态污染下去。

已经修复过的 bug：旧代码只要看到 observation 里有 `Illegal Action` 就退出，不分 offender。典型 case：

```text
replay: cd921685-fe52-4f0a-81dc-02892487be6f
server event: {"deltas":[4000,-8000,2000,2000],"reason":"Error: Illegal Action by Player 1","type":"ryukyoku"}
pearl audit: seat=0
结论：Player 1 被罚，不是我方。不能断开。
```

相关代码：

```text
src/gateway/riichi_dev_client.py
_server_illegal_action_offenders(...)
_server_illegal_action_event(..., seat=self.seat)
```

相关测试：

```bash
uv run pytest tests/test_riichi_dev_client.py -q
```

## Echo Mismatch / Timeout Fallback

每次发送 tracked action 后，下一次 `request_action` 的 `obs.new_events()` 必须 echo 我方上一次动作。否则说明服务器没有采用我们刚才发出的动作。

典型日志：

```text
riichi.dev appears to have applied its timeout/disconnect fallback instead of the previous action
expected={"actor": 1, "pai": "8m", "tsumogiri": false, "type": "dahai"}
observed={"actor": 1, "pai": "6m", "tsumogiri": true, "type": "dahai"}
```

这个含义是：本地 websocket send 完成了，但服务器推进出来的是我方摸切 fallback，不是我们发的动作。常见原因是服务端没收到、连接中断、或服务端 action timeout。它不是 illegal，因为没有罚满贯。

当前正确处理：主动中止当前客户端。riichi.dev ranked 不支持断线重连回同一局；继续发送会让 Mortal 以为自己打了 `8m`，但服务器状态已经是 `6m`，后续必然状态错乱。

2026-05-01 10:13:15 pearl case：

```text
10:13:14.565 send: {"type":"dahai","actor":1,"pai":"8m","tsumogiri":false}
decision_latency_ms=40.103
send_latency_ms=0.262
10:13:15.327 observed: {"actor":1,"pai":"6m","tsumogiri":true,"type":"dahai"}
classification=server_timeout_or_disconnect_fallback
```

结论：不是模型慢，40ms 远小于 2500ms deadline；这是服务端未采用该动作后的 fallback。

## Reconnect 规则

不要在 active game state-risk 后自动重连并继续。riichi.dev ranked 目前不能 reconnect 回原来的对局。

允许自动重连的场景：

- ranked 排队阶段还没 `start_game`
- 已经 `end_game` 后连接关闭

不允许自动重连并继续的场景：

- active game echo mismatch
- active game server illegal by our seat
- active game stale action deadline

当前代码会对 active state-risk 直接退出，让外部人工或 supervisor 再重新拉起新 ranked session。

## Debug Playbook

先看 stdout 的错误类型：

```bash
ssh pearl 'grep -n "ERROR\\|WARNING\\|Illegal\\|echo mismatch\\|timeout/disconnect fallback\\|server closed" ~/keqing1_deploy/logs/riichi_dev/ranked-mortal.stdout.log | tail -n 80'
```

查某个服务器时间附近的本地日志。注意 pearl 日志时间是服务器本地 CST；riichi.dev 显示时间可能差 1 小时：

```bash
ssh pearl 'grep -n "2026-05-01 10:1[0-9]" ~/keqing1_deploy/logs/riichi_dev/ranked-mortal.stdout.log | tail -n 120'
```

看审计 jsonl 中的结构化记录：

```bash
ssh pearl 'grep -n "echo_mismatch\\|server_illegal_action\\|agent_error\\|send_result" ~/keqing1_deploy/logs/riichi_dev/ranked-mortal.jsonl | tail -n 80'
```

如果要查某个 request 序号附近，先找 `echo_mismatch` 或 `server_illegal_action`，再按行号前后展开：

```bash
ssh pearl 'grep -n "echo_mismatch" ~/keqing1_deploy/logs/riichi_dev/ranked-mortal.jsonl | tail -n 5'
ssh pearl 'sed -n "63880,63892p" ~/keqing1_deploy/logs/riichi_dev/ranked-mortal.jsonl'
```

查线上进程：

```bash
ssh pearl 'pgrep -af "[r]iichi_dev_client.py" || true; tmux ls 2>/dev/null || true'
```

确认网络/代理：

```bash
ssh pearl 'tail -n 20 ~/keqing1_deploy/logs/riichi_dev/ranked-mortal.stdout.log | grep "websocket diagnostics"'
```

pearl 期望是直连：

```text
proxy_env={}
explicit_ws_proxy=None
manual_http_connect_proxy=False
```

## 本地验证

修改 `src/gateway/riichi_dev_client.py` 后至少跑：

```bash
uv run pytest tests/test_riichi_dev_client.py -q
uv run python -m py_compile src/gateway/riichi_dev_client.py
```

同步到 pearl 后也检查语法：

```bash
rsync -az src/gateway/riichi_dev_client.py pearl:~/keqing1_deploy/src/gateway/riichi_dev_client.py
ssh pearl 'cd ~/keqing1_deploy && .venv/bin/python -m py_compile src/gateway/riichi_dev_client.py'
```

如果正在 active game，不要直接 kill。先确认：

```bash
ssh pearl 'pgrep -af "[r]iichi_dev_client.py"; tail -n 20 ~/keqing1_deploy/logs/riichi_dev/ranked-mortal.stdout.log'
```

如果进程已经因 state-risk 退出，可以直接重启 tmux。

## 不要重复踩的坑

- 不要把别家的 `Illegal Action by Player N` 当成我方错误。
- 不要把服务器 timeout/disconnect fallback 当成模型非法动作。
- 不要在 active game echo mismatch 后继续发送动作。
- 不要维护第二套 observation event cursor。
- 不要省略 `dahai.tsumogiri`。
- 不要要求 `chi.consumed` echo 完全一致。
- 不要让训练中的 `artifacts/mortal_training/mortal.pth` 直接作为线上 serving 权重；线上用 `artifacts/mortal_serving/mortal.pth`，替换后需要重启进程。
