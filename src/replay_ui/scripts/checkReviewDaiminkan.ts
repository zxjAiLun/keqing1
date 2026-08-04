// src/replay_ui/scripts/checkReviewDaiminkan.ts
// 回归检查：大明杠（及其余副露动作）的 pre/post 状态重建与副露牌面校验。
//
// 复现场景（南4局 1本场 / step 491）：
//   自家 P0 已有 3 组副露，准备大明杠 6m；snapshot 为动作后（暗手剩 1 张、
//   副露已含 daiminkan）。
//
// 验收：
//   A. 动作前：4 张暗手（含 3 张 6m），3 组副露
//   B. 动作后：1 张暗手，4 组副露，新大明杠显示 4 张 6m
//   C. pre/post 来回切换无状态漂移
//   chi/pon/ankan/kakan 可逆语义 + 副露校验
import { entryToBattleState } from '../src/utils/replayAdapter.ts';
import { buildMeldDisplayTiles, validateMeldTiles } from '../src/components/BattleBoard/seatLayout.ts';
import type { Action, DecisionLogEntry, MeldEntry } from '../src/types/replay.ts';

let failures = 0;
function check(ok: boolean, message: string): void {
  if (!ok) {
    failures += 1;
    console.error(`FAIL: ${message}`);
  }
}

const NAMES = ['P0', 'P1', 'P2', 'P3'];

function meld(type: MeldEntry['type'], pai: string, consumed: string[], target: number): MeldEntry {
  return { type, pai, consumed, target };
}

function daiminkanEntry(): DecisionLogEntry {
  return {
    step: 491,
    bakaze: 'N',
    kyoku: 4,
    honba: 1,
    oya: 1,
    scores: [20000, 20000, 20000, 20000],
    reached: [false, false, false, false],
    dora_markers: [],
    hand: ['1p'], // snapshot 为动作后：暗手只剩 1 张
    discards: { 0: [], 1: [{ pai: '6m', tsumogiri: false }], 2: [], 3: [] },
    melds: {
      0: [
        meld('chi', '3s', ['1s', '2s'], 1),
        meld('pon', '5p', ['5p', '5p'], 2),
        meld('pon', '9m', ['9m', '9m'], 3),
        meld('daiminkan', '6m', ['6m', '6m', '6m'], 1),
      ],
      1: [], 2: [], 3: [],
    },
    actor_to_move: 0,
    tsumo_pai: null,
    last_discard: { actor: 1, pai: '6m' },
    is_obs: false,
    chosen: { type: 'daiminkan', actor: 0, pai: '6m', consumed: ['6m', '6m', '6m'], target: 1 },
    candidates: [
      { action: { type: 'daiminkan', actor: 0, pai: '6m', consumed: ['6m', '6m', '6m'], target: 1 }, logit: 1 },
      { action: { type: 'none', actor: 0 }, logit: 0 },
    ],
  };
}

// --- A/B：大明杠 pre/post ------------------------------------------------

const entry = daiminkanEntry();
const pre = entryToBattleState(entry, NAMES, 0, 'pre');
const post = entryToBattleState(entry, NAMES, 0, 'post');

check(pre.hand.length === 4, `动作前暗手应为 4 张，得到 ${pre.hand.length}`);
check(pre.hand.filter((t) => t === '6m').length === 3, `动作前暗手应含 3 张 6m，得到 ${pre.hand}`);
check(pre.melds[0].length === 3, `动作前副露应为 3 组，得到 ${pre.melds[0].length}`);
check(pre.actor_to_move === 0, `动作前 actor_to_move 应为 0`);
check(pre.tsumo_pai === null, `动作前 tsumo_pai 应为 null`);

check(post.hand.length === 1, `动作后暗手应为 1 张，得到 ${post.hand.length}`);
check(post.melds[0].length === 4, `动作后副露应为 4 组，得到 ${post.melds[0].length}`);

// 新大明杠显示 4 张 6m
const kanMeld = post.melds[0][3];
check(kanMeld.type === 'daiminkan', `动作后第 4 组应为 daiminkan`);
const kanTiles = buildMeldDisplayTiles(0, kanMeld);
check(kanTiles.length === 4, `daiminkan 应显示 4 张，得到 ${kanTiles.length}`);
check(kanTiles.every((t) => t.tile === '6m'), `daiminkan 四张都应是 6m`);

// --- C：pre/post 往返无漂移 ----------------------------------------------

const preAgain = entryToBattleState(entry, NAMES, 0, 'pre');
const postAgain = entryToBattleState(entry, NAMES, 0, 'post');
check(
  preAgain.hand.length === pre.hand.length && preAgain.melds[0].length === pre.melds[0].length,
  `pre 往返后状态漂移`,
);
check(
  postAgain.hand.length === post.hand.length && postAgain.melds[0].length === post.melds[0].length,
  `post 往返后状态漂移`,
);

// --- obs 变体：board_phase=before_action 且 snapshot 为动作后，pre 也需回退 ---

const obsEntry: DecisionLogEntry = {
  ...daiminkanEntry(),
  is_obs: true,
  board_phase: 'before_action',
};
const obsPre = entryToBattleState(obsEntry, NAMES, 0, 'pre');
check(obsPre.hand.length === 4 && obsPre.hand.filter((t) => t === '6m').length === 3,
  `obs before_action pre 也应回退为 4 张暗手含 3 张 6m`);

// --- chi/pon 可逆 ----------------------------------------------------------

function singleMeldEntry(action: Action, hand: string[], melds: MeldEntry[]): DecisionLogEntry {
  return {
    ...daiminkanEntry(),
    hand,
    melds: { 0: melds, 1: [], 2: [], 3: [] },
    chosen: action,
    candidates: [],
  };
}

const chiEntry = singleMeldEntry(
  { type: 'chi', actor: 0, pai: '3s', consumed: ['1s', '2s'], target: 1 },
  ['9p'],
  [meld('chi', '3s', ['1s', '2s'], 1)],
);
const chiPre = entryToBattleState(chiEntry, NAMES, 0, 'pre');
check(chiPre.hand.length === 3 && chiPre.hand.includes('1s') && chiPre.hand.includes('2s'),
  `chi pre 应恢复 2 张 consumed（手 ${chiPre.hand}）`);
check(chiPre.melds[0].length === 0, `chi pre 应移除该组副露`);

const ponEntry = singleMeldEntry(
  { type: 'pon', actor: 0, pai: '5p', consumed: ['5p', '5p'], target: 2 },
  ['9p'],
  [meld('pon', '5p', ['5p', '5p'], 2)],
);
const ponPre = entryToBattleState(ponEntry, NAMES, 0, 'pre');
check(ponPre.hand.length === 3 && ponPre.hand.filter((t) => t === '5p').length === 2,
  `pon pre 应恢复 2 张 5p（手 ${ponPre.hand}）`);

// --- ankan 可逆 ------------------------------------------------------------

const ankanEntry = singleMeldEntry(
  { type: 'ankan', actor: 0, consumed: ['7m', '7m', '7m', '7m'] },
  ['9p'],
  [meld('ankan', '', ['7m', '7m', '7m', '7m'], 0)],
);
const ankanPre = entryToBattleState(ankanEntry, NAMES, 0, 'pre');
check(ankanPre.hand.length === 5 && ankanPre.hand.filter((t) => t === '7m').length === 4,
  `ankan pre 应恢复 4 张 7m（手 ${ankanPre.hand}）`);
check(ankanPre.last_discard === null, `ankan pre last_discard 应为 null`);

// --- kakan 可逆 ------------------------------------------------------------

const kakanEntry = singleMeldEntry(
  { type: 'kakan', actor: 0, pai: '6p', consumed: ['6p', '6p', '6p', '6p'], target: 2 },
  ['9p'],
  [meld('kakan', '6p', ['6p', '6p', '6p', '6p'], 2)],
);
const kakanPre = entryToBattleState(kakanEntry, NAMES, 0, 'pre');
check(kakanPre.melds[0].length === 1 && kakanPre.melds[0][0].type === 'pon',
  `kakan pre 应回退为 pon（得到 ${kakanPre.melds[0][0]?.type}）`);
check(kakanPre.hand.length === 2 && kakanPre.hand.filter((t) => t === '6p').length === 1,
  `kakan pre 应把加杠的 6p 放回手（手 ${kakanPre.hand}）`);

// --- 副露牌面校验 ----------------------------------------------------------

check(validateMeldTiles(meld('daiminkan', '6m', ['6m', '6m', '6m'], 1)) === true, `合法 daiminkan 应通过校验`);
check(validateMeldTiles(meld('daiminkan', '6m', ['6m', '6m'], 1)) === false, `consumed 数量错误的 daiminkan 应失败`);
check(validateMeldTiles(meld('chi', '3s', ['1s', ''], 1)) === false, `含空牌的副露应失败`);
check(validateMeldTiles(meld('ankan', '', ['7m', '7m', '7m', '7m'], 0)) === true, `合法 ankan 应通过校验`);

if (failures > 0) {
  console.error(`review daiminkan regression FAILED (${failures} issues)`);
  process.exit(1);
}
console.log('review daiminkan regression OK (pre/post reconstruction, rewind semantics, meld validation)');
