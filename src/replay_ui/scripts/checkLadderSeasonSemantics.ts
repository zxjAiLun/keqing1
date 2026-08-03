// src/replay_ui/scripts/checkLadderSeasonSemantics.ts
// 第七轮：赛季选择与 readiness 语义的结构化静态检查（轻量，不引入 Vitest）。
//
// 检查：
// 1. 三页不再出现 seasons[0]（不再依赖数组第一项）；
// 2. 三页使用 default_season_id 语义（catalog hook）；
// 3. canonical navigate 使用 { replace: true }；
// 4. unknown query season 不 fallback（urlSeasonId 优先）；
// 5. 三页使用共享赛季 catalog（useLadderSeasonCatalog）；
// 6. 三页使用 LadderSeasonNotice；
// 7. 类型包含 readiness 与 default source；
// 8. selector 存在无默认占位状态（"选择赛季…"）。
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const SRC = resolve(import.meta.dirname, '../src');

const PAGES = [
  'pages/LadderPage.tsx',
  'pages/LadderAccountPage.tsx',
  'pages/LadderModelPage.tsx',
];
const HOOK = 'hooks/useLadderSeasonCatalog.ts';
const TYPES = 'types/ladder.ts';
const NOTICE = 'components/Ladder/LadderSeasonNotice.tsx';

function read(rel: string): string {
  return readFileSync(resolve(SRC, rel), 'utf8');
}

let failures = 0;
function check(ok: boolean, message: string): void {
  if (!ok) {
    failures += 1;
    console.error(`FAIL: ${message}`);
  }
}

// 1. 三页不再出现 seasons[0]
for (const page of PAGES) {
  const src = read(page);
  check(!/seasons\s*\[\s*0\s*\]/.test(src), `${page} 不应再依赖 seasons[0]`);
}

// 2. 三页使用 default_season_id / catalog
for (const page of PAGES) {
  const src = read(page);
  check(/useLadderSeasonCatalog/.test(src), `${page} 应使用共享赛季目录`);
  check(/defaultSeasonId/.test(src), `${page} 应引用 defaultSeasonId`);
}

// 3. canonical navigate 使用 { replace: true }
const hook = read(HOOK);
check(/navigate\([^)]*,\s*\{\s*replace:\s*true\s*\}\)/.test(hook), 'useLadderSeasonCatalog 规范化导航应使用 replace: true');

// 4. unknown query season 不 fallback（urlSeasonId 优先）
check(/if \(urlSeasonId\) return urlSeasonId/.test(hook), 'useLadderSeasonCatalog 应严格优先 URL 赛季');

// 5. 三页使用共享 season catalog（重复项 2 已覆盖，这里显式）
for (const page of PAGES) {
  const src = read(page);
  check(/useLadderSeasonCatalog/.test(src), `${page} 应使用共享 season catalog`);
}

// 6. 三页使用 LadderSeasonNotice
for (const page of PAGES) {
  const src = read(page);
  check(/LadderSeasonNotice/.test(src), `${page} 应使用 LadderSeasonNotice`);
}

// 7. 类型包含 readiness 与 default source
const types = read(TYPES);
check(/interface LadderSeasonReadiness/.test(types), 'types 应包含 LadderSeasonReadiness');
check(/LadderReadinessState/.test(types), 'types 应包含 LadderReadinessState');
check(/default_season_id/.test(types), 'types 应包含 default_season_id');
check(/default_source/.test(types), 'types 应包含 default_source');

// 8. selector 存在无默认占位状态
const ladderPage = read('pages/LadderPage.tsx');
check(/选择赛季…/.test(ladderPage), 'LadderPage selector 应有"选择赛季…"无默认占位');

// 9. LadderSeasonNotice 展示 message 与 detail 分离
const notice = read(NOTICE);
check(/readiness\.message/.test(notice), 'LadderSeasonNotice 应展示 readiness.message');
check(/readiness\.detail/.test(notice), 'LadderSeasonNotice 应展示 readiness.detail（运维文本）');

if (failures > 0) {
  console.error(`ladder season semantics FAILED (${failures} issues)`);
  process.exit(1);
}
console.log(`ladder season semantics OK (${PAGES.length} pages, ${HOOK}, ${NOTICE}, ${TYPES})`);
