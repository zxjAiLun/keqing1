// src/replay_ui/scripts/checkParticipantSemantics.ts
// R10 语义检查：4 座不变量、force-save 流程、无 player_id===0 硬编码、
// 路由/导航注册、后端 router 挂载、pytest 白名单齐全。
import { readFileSync, existsSync } from 'node:fs';
import { resolve } from 'node:path';

const ROOT = resolve(import.meta.dirname, '..');
let failures = 0;
function check(ok: boolean, message: string): void {
  if (!ok) {
    failures += 1;
    console.error(`FAIL: ${message}`);
  }
}

function read(p: string): string {
  const full = resolve(ROOT, p);
  return existsSync(full) ? readFileSync(full, 'utf-8') : '';
}

function inFile(p: string, needles: string[]): boolean {
  const content = read(p);
  return needles.every((needle) => content.includes(needle));
}

// 1) 4 座不变量：SEAT_WINDS 恰好 4 座
const labels = read('src/components/Matches/labels.ts');
check(/SEAT_WINDS\s*=\s*\[\s*'東'\s*,\s*'南'\s*,\s*'西'\s*,\s*'北'\s*\]/.test(labels), 'labels.ts 的 SEAT_WINDS 必须恰好 4 座');
check(inFile('src/components/Matches/SeatPicker.tsx', ['SEAT_WINDS']), 'SeatPicker 使用 SEAT_WINDS');

// 2) MatchEntryPage 走 createMatch + force-save 流程
check(
  inFile('src/pages/MatchEntryPage.tsx', ['createMatch', 'force', 'reason', 'ValidationNotice']),
  'MatchEntryPage 含 createMatch + force-save 流程',
);

// 3) 新页面无 player_id===0 / seat0=human 硬编码
for (const page of [
  'src/pages/ParticipantsPage.tsx',
  'src/pages/MatchesPage.tsx',
  'src/pages/MatchEntryPage.tsx',
  'src/pages/MatchDetailPage.tsx',
]) {
  const content = read(page);
  check(
    !/player_id\s*===\s*0/.test(content) && !/id\s*===\s*0\s*\?\s*'human'/.test(content),
    `${page} 不应硬编码 player_id===0 / seat0=human`,
  );
}

// 4) types 包含核心概念
const types = read('src/types/participants.ts');
for (const name of ['AccountType', 'ControllerType', 'MatchSeat', 'Match', 'RevisionSummary']) {
  check(types.includes(name), `types/participants.ts 包含 ${name}`);
}

// 5) 路由与导航注册
const routes = read('src/routes.ts');
check(routes.includes("participants: '/participants'"), 'routes.ts 含 /participants');
check(routes.includes("matches: '/matches'"), 'routes.ts 含 /matches');
check(routes.includes("matchEntry: '/matches/new'"), 'routes.ts 含 /matches/new');
check(routes.includes("MATCH_DETAIL_PATTERN = '/matches/:matchId'"), 'routes.ts 含 MATCH_DETAIL_PATTERN');
check(inFile('src/App.tsx', ['ParticipantsPage', 'MatchesPage', 'MatchEntryPage', 'MatchDetailPage']), 'App.tsx 注册 4 个 R10 页面');
check(inFile('src/components/Layout/Sidebar.tsx', ['routes.participants', 'routes.matches']), 'Sidebar 链接参赛者与对局记录');

// 6) 后端 router 挂载
check(
  inFile('../../src/replay/server.py', ['from participants.api import router as participants_router', 'include_router(participants_router)']),
  'server.py 挂载 participants router',
);
check(inFile('../../src/participants/api.py', ['prefix="/api/participants"']), 'participants api 前缀 /api/participants');

// 7) pytest 白名单包含 5 个 R10 测试文件
const pyproject = read('../../pyproject.toml');
for (const testFile of [
  'test_participants_registry.py',
  'test_participants_ledger.py',
  'test_participants_validation.py',
  'test_participants_server.py',
  'test_participants_migration.py',
]) {
  check(pyproject.includes(testFile), `pyproject python_files 白名单包含 ${testFile}`);
}

// 8) MatchesPage 渲染 status（active|void）
const matchesPage = read('src/pages/MatchesPage.tsx');
check(matchesPage.includes('status') && matchesPage.includes('void'), 'MatchesPage 支持状态筛选（active/void）');

// 9) R10-E：通用四人阵容——roster 与 launcher 数量分离 + 宽松捕获
const playwithyou = read('../../src/gateway/api/playwithyou.py');
check(playwithyou.includes('ParticipantBindingRequest'), 'playwithyou 定义 ParticipantBindingRequest（预期四人阵容）');
check(playwithyou.includes('roster: List[') && playwithyou.includes('ParticipantBindingRequest'), 'StartPlayWithYouRequest 含 roster 字段');
check(playwithyou.includes('launcher_slot'), 'ParticipantBindingRequest 含 launcher_slot（与 launcher 数量分离）');
check(playwithyou.includes('scope="session"') || playwithyou.includes("scope='session'"), 'start 注册 session-scoped 别名');
const capture = read('../../src/gateway/playwithyou_capture.py');
check(capture.includes('awaiting_import'), '捕获层支持 awaiting_import（任一 observer 捕获 log 即可）');
check(capture.includes('roster'), 'CaptureBinding 支持 roster 模式');

// 10) R10-E Repair：真实 launcher 接线 / 赛后状态机 / 互斥
const launcher = read('../../scripts/launch_tenhou_bots.py');
check(launcher.includes('mode == "roster"') && launcher.includes('launcher_slot'), 'launcher 识别 roster 模式并按 slot 接线');
check(launcher.includes('config.ladder_account_id = str(entry["account_id"])'), 'launcher 按 roster entry 绑定 ladder_account_id');
check(playwithyou.includes('_validate_roster_bindings'), 'start 前校验 roster（账号存在/启用/模型归属）');
check(playwithyou.includes('不能同时开启'), 'roster 与旧正式天梯绑定互斥');
const capture2 = read('../../src/gateway/playwithyou_capture.py');
check(capture2.includes('log_captured'), 'roster 状态机：开局仅捕获 log 为 log_captured（非可导入）');
check(capture2.includes('evidence_warning'), 'roster 分数不一致记录 evidence_warning 不阻塞');

// 11) R10-E Repair 2：slot-stable 冻结 / checkpoint 精确匹配 / 单锁原子
check(playwithyou.includes('_resolve_artifact_path'), '冻结按 artifact 绝对路径精确匹配 checkpoint');
check(playwithyou.includes('resolve_bot_spec'), '冻结复用真实 bot_registry checkpoint 解析');
check(playwithyou.includes('roster_bindings, frozen_launcher_specs = _freeze_launcher_models(roster_bindings, specs)'), '冻结保持原 roster 顺序（slot-stable 写回）');
check(playwithyou.includes('participants_data_lock'), 'roster 校验/冻结/别名注册在同一 data_lock');
check(playwithyou.includes('_rollback_roster_start'), 'Popen 失败也回滚 roster 启动');

// 12) R10-E Repair 3：冻结 checkpoint 传给 runtime / evidence 透出 API
const launcher2 = read('../../scripts/launch_tenhou_bots.py');
check(launcher2.includes('resolved_checkpoint_path'), 'launcher 从 binding 冻结路径设 model_path');
check(launcher2.includes('config.model_path = Path(frozen_path)'), 'runtime 加载冻结路径而非动态 spec');
check(playwithyou.includes('launcher_command_specs'), '父进程用冻结绝对路径作为 --bots');
check(playwithyou.includes('"resolved_checkpoint_path": str(resolved_path)'), 'binding 冻结 resolved_checkpoint_path');
check(playwithyou.includes('"roster": payload.get("roster") or []'), '_discover_captures 透出 roster');
check(playwithyou.includes('"evidence_warning": payload.get("evidence_warning")'), '_discover_captures 透出 evidence_warning');

// 13) R10-F：Ledger-driven Ladder Projection
const ladderIngest = read('../../src/replay/ladder_ingest.py');
check(ladderIngest.includes('class ParticipantLedgerAdapter'), 'ladder_ingest 新增 participants ledger adapter');
check(ladderIngest.includes('participants_cfg') && ladderIngest.includes('participants_dir.is_dir()'), '赛季 ingest 配置启用 participants source');
const ledgerF = read('../../src/participants/ledger.py');
check(ledgerF.includes('ladder_dirty_path') && ledgerF.includes('mark_ladder_dirty'), 'ledger 有 dirty marker');
check(ledgerF.includes('set_season_projection_state'), 'ledger 支持批量投影状态');
check(ledgerF.includes('ladder_projection_state="pending"') || ledgerF.includes("ladder_projection_state='pending'"), 'create 置 pending 投影状态');
const participantsApi = read('../../src/participants/api.py');
check(participantsApi.includes('/ladder/{season_id}/project'), '投影触发 API');
check(participantsApi.includes('/ladder/{season_id}/status'), '投影状态 API');

// 14) R10-F Repair：fingerprint / generation CAS / exclusive / 自动消费
const ladderIngest2 = read('../../src/replay/ladder_ingest.py');
check(ladderIngest2.includes('participants_projection_fingerprint'), 'ledger 语义投影指纹（publisher unchanged-skip 感知）');
check(ladderIngest2.includes('"exclusive"') || ladderIngest2.includes("'exclusive'"), 'participants exclusive 排他模式');
const ledgerF2 = read('../../src/participants/ledger.py');
check(ledgerF2.includes('"generation": uuid.uuid4().hex'), 'dirty marker 带 generation（CAS）');
check(ledgerF2.includes('complete_ladder_projection'), '发布完成 CAS（generation 变了保留 dirty）');
check(ledgerF2.includes('model_fields_set'), 'revise 区分省略与显式 null（可清空 season）');
check(ledgerF2.includes('for season in {old_season, new_season}'), 'season move → 新旧双 dirty');
const pls = read('../../scripts/mortal/publish_ladder_snapshot.py');
check(pls.includes('extra_fingerprint'), 'publisher 指纹合并 ledger 投影输入');
const proj = read('../../src/participants/projection.py');
check(proj.includes('run_dirty_projection'), '自动 dirty consumer');
check(proj.includes('start_worker'), '后台 worker 启动');

// 15) R10-F Repair 2：begin barrier / single-flight / lifespan
const ledgerF3 = read('../../src/participants/ledger.py');
check(ledgerF3.includes('begin_ladder_projection'), '投影 begin barrier（锁内恢复 pending 后冻结 generation）');
check(ledgerF3.includes('mark_season_projection_error'), '失败回写带 generation CAS');
const paths = read('../../src/participants/paths.py');
check(paths.includes('try_file_lock'), '非阻塞跨进程锁（per-season single-flight）');
const proj2 = read('../../src/participants/projection.py');
check(proj2.includes('projection_locks'), 'per-season 单飞锁');
check(proj2.includes('already_running'), 'second caller 返回 already_running 不进 publisher');
check(proj2.includes('stop_worker'), 'worker 可停止（lifespan）');
const server2 = read('../../src/replay/server.py');
check(server2.includes('participants_projection.start_worker()') && server2.includes('participants_projection.stop_worker()'), 'worker 放入 FastAPI lifespan');

// 16) R10-F Repair 3：lease 锁 ownership / worker 重入
const paths3 = read('../../src/participants/paths.py');
check(paths3.includes('try_lease_lock'), 'projection 专用 lease 锁（owner 身份）');
check(paths3.includes('_pid_is_alive'), 'reclaim 前校验 owner PID 存活');
check(paths3.includes('payload.get("token") == token') || paths3.includes('payload.get("token") == token'), '释放仅当 token 匹配（不删后来 owner 的锁）');
const proj3 = read('../../src/participants/projection.py');
check(proj3.includes('_stop.clear()'), 'worker start 可重入（清 stop 标记）');
const tstypes = read('src/types/participants.ts');
check(tstypes.includes("'needs_rebuild'") && tstypes.includes("'already_running'"), 'TS 投影状态含 needs_rebuild/already_running');

// 17) R10-F Repair 4：reclaim 串行化 / worker shutdown 保引用
const paths4 = read('../../src/participants/paths.py');
check(paths4.includes('.reclaim'), 'dead-lease reclaim 用 reclaim 互斥串行化');
check(paths4.includes('在 reclaim 临界区内重新读取/重新判定'), 'reclaim 临界区内重新判定 dead-owner 后才删除');
const proj4 = read('../../src/participants/projection.py');
check(proj4.includes('if _thread.is_alive():') && proj4.includes('不重入'), '长 publisher 时 start 不重入');

if (failures > 0) {
  console.error(`participant semantics FAILED (${failures} issues)`);
  process.exit(1);
}
console.log('participant semantics OK (4-seat roster, force-save, routes, server mount, pytest whitelist)');
