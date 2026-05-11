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
    value: 'rulebase',
    label: 'rulebase',
    shortLabel: '基线',
    badge: 'Baseline',
    description: '规则基线，用于兼容对照和快速 sanity check。',
  },
];

export const DEFAULT_BOT_TYPE: BotType = 'mortal';

export const BOT_CHECKPOINT_DEFAULTS: Record<BotType, string> = {
  mortal: 'artifacts/mortal_training/mortal.pth',
  rulebase: '',
};

export function getBotCatalogEntry(botType: BotType): BotCatalogEntry {
  return BOT_CATALOG.find((entry) => entry.value === botType) ?? BOT_CATALOG[0];
}
