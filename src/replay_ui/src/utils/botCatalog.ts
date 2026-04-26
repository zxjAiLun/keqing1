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
    value: 'xmodel1',
    label: 'xmodel1',
    shortLabel: '主线',
    badge: 'Mainline',
    description: '当前训练窗口主线，GUI 默认模型。',
  },
  {
    value: 'keqingv4',
    label: 'keqingv4',
    shortLabel: '备线',
    badge: 'Backup',
    description: '当前备线与 Rust 语义收敛路径。',
  },
  {
    value: 'rulebase',
    label: 'rulebase',
    shortLabel: '基线',
    badge: 'Baseline',
    description: '规则基线，用于兼容对照和快速 sanity check。',
  },
];

export const DEFAULT_BOT_TYPE: BotType = 'xmodel1';

export const BOT_CHECKPOINT_DEFAULTS: Record<BotType, string> = {
  xmodel1: 'artifacts/models/xmodel1/best.pth',
  keqingv4: 'artifacts/models/keqingv4/best.pth',
  rulebase: '',
};

export function getBotCatalogEntry(botType: BotType): BotCatalogEntry {
  return BOT_CATALOG.find((entry) => entry.value === botType) ?? BOT_CATALOG[0];
}
