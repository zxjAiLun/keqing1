// src/replay_ui/scripts/checkLadderLiveSemantics.ts
// Ladder 三页 Live 一致性的静态语义检查（轻量，不引入 Vitest）。
//
// 检查：
// 1. 三页（LadderPage / LadderAccountPage / LadderModelPage）均使用共享 useVisibleLiveQuery Hook；
// 2. 三页不再各自注册 setInterval；
// 3. ladderApi 的 fetch 使用 cache: 'no-store'；
// 4. ladderApi 四个方法均支持 AbortSignal；
// 5. 三页均使用共享 LadderSnapshotStatus 组件；
// 6. 轮询间隔仍为 30_000ms（useVisibleLiveQuery 默认 intervalMs）。
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const SRC = resolve(import.meta.dirname, '../src');

const PAGES = [
  'pages/LadderPage.tsx',
  'pages/LadderAccountPage.tsx',
  'pages/LadderModelPage.tsx',
];
const HOOK = 'hooks/useVisibleLiveQuery.ts';
const API = 'api/ladderApi.ts';
const STATUS = 'components/Ladder/LadderSnapshotStatus.tsx';

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

// 1. 三页均使用共享 Hook
for (const page of PAGES) {
  const src = read(page);
  check(
    /useVisibleLiveQuery/.test(src),
    `${page} 应使用 useVisibleLiveQuery`,
  );
}

// 2. 三页不再各自注册 setInterval
for (const page of PAGES) {
  const src = read(page);
  check(
    !/setInterval/.test(src),
    `${page} 不应再自行注册 setInterval`,
  );
}

// 3. ladderApi 使用 cache: 'no-store'
const api = read(API);
check(api.includes("cache: 'no-store'"), 'ladderApi 应使用 cache: "no-store"');

// 4. ladderApi 四个方法均支持 AbortSignal
for (const method of ['listSeasons', 'getLadder', 'getAccount', 'getModel']) {
  const re = new RegExp(`${method}:\\s*\\([^)]*signal`);
  check(re.test(api), `ladderApi.${method} 应支持 AbortSignal`);
}

// 5. 三页均使用共享 snapshot status 组件
for (const page of PAGES) {
  const src = read(page);
  check(
    /LadderSnapshotStatus/.test(src),
    `${page} 应使用 LadderSnapshotStatus 组件`,
  );
}

// 6. 轮询间隔仍为 30 秒（hook 默认 30_000）
const hook = read(HOOK);
check(
  /intervalMs\s*=\s*30_000/.test(hook),
  'useVisibleLiveQuery 默认轮询间隔应为 30_000ms',
);

if (failures > 0) {
  console.error(`ladder live semantics FAILED (${failures} issues)`);
  process.exit(1);
}
console.log(`ladder live semantics OK (${PAGES.length} pages, ${STATUS}, ${HOOK}, ${API})`);
