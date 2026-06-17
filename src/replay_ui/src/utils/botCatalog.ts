import type { BotType } from '../types/bot';

export interface BotCatalogEntry {
  value: BotType;
  label: string;
  shortLabel: string;
  badge: string;
  description: string;
}

export const BOT_CATALOG: BotCatalogEntry[] = [
  {
    value: 'mortal',
    label: 'Mortal gui_mortal.pth',
    shortLabel: 'gui_mortal',
    badge: '当前训练导出',
    description: 'artifacts/mortal_serving/gui_mortal.pth。当前 GUI 默认使用的 Mortal checkpoint。',
  },
  {
    value: '70k',
    label: 'Mortal 70k.pth',
    shortLabel: '70k',
    badge: '70k 锚点',
    description: 'artifacts/mortal_serving/70k.pth。固定 70k 训练步锚点权重。',
  },
  {
    value: 't1_71000',
    label: 'Mortal T1@71000',
    shortLabel: 'T1@71000',
    badge: 'baseline',
    description: 'artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth。基于 70k + v4 teacher CE 的 71000 step baseline。',
  },
  {
    value: 'weak_mortal',
    label: 'Mortal v4',
    shortLabel: 'v4',
    badge: '外部 v4',
    description: 'artifacts/model_v4_20240308_best_min.pth。外部 v4 参考权重。',
  },
  {
    value: 'rulebase',
    label: 'rulebase',
    shortLabel: '基线',
    badge: 'Baseline',
    description: '规则基线，用于兼容对照和快速 sanity check。',
  },
];

export const GUI_BOT_CATALOG: BotCatalogEntry[] = BOT_CATALOG.filter((entry) => entry.value !== 'rulebase');

export const DEFAULT_BOT_TYPE: BotType = 'mortal';

export const BOT_CHECKPOINT_DEFAULTS: Record<BotType, string> = {
  mortal: 'artifacts/mortal_serving/gui_mortal.pth',
  '70k': 'artifacts/mortal_serving/70k.pth',
  t1_71000: 'artifacts/experiments/teacher_transfer_2026_05/T1_teacher_ce_01/mortal.pth',
  weak_mortal: 'artifacts/model_v4_20240308_best_min.pth',
  rulebase: '',
};

export function getBotCatalogEntry(botType: BotType): BotCatalogEntry {
  return BOT_CATALOG.find((entry) => entry.value === botType) ?? BOT_CATALOG[0];
}
