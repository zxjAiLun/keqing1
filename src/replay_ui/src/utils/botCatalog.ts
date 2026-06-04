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
    label: 'mortal',
    shortLabel: '主线',
    badge: 'Mainline',
    description: 'Mortal 原生 Brain+DQN checkpoint，当前用于打牌、牌谱 review 和工具化主线。',
  },
  {
    value: '70k',
    label: '70k',
    shortLabel: '70k',
    badge: 'Anchor',
    description: '70k 标准锚点权重，用于和当前主线或实验分支做稳定对照。',
  },
  {
    value: 'weak_mortal',
    label: 'weak mortal',
    shortLabel: 'weak',
    badge: 'Reference',
    description: '本地 model_v4_20240308_best_min 权重，作为 weak mortal 参考模型。',
  },
  {
    value: 'rulebase',
    label: 'rulebase',
    shortLabel: '基线',
    badge: 'Baseline',
    description: '规则基线，用于兼容对照和快速 sanity check。',
  },
];

export const DEFAULT_BOT_TYPE: BotType = 'mortal';

export const BOT_CHECKPOINT_DEFAULTS: Record<BotType, string> = {
  mortal: 'artifacts/mortal_serving/gui_mortal.pth',
  '70k': 'artifacts/mortal_serving/70k.pth',
  weak_mortal: 'artifacts/mortal_serving/weak_mortal.pth',
  rulebase: '',
};

export function getBotCatalogEntry(botType: BotType): BotCatalogEntry {
  return BOT_CATALOG.find((entry) => entry.value === botType) ?? BOT_CATALOG[0];
}
