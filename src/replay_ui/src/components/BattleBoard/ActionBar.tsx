// src/replay_ui/src/components/BattleBoard/ActionBar.tsx
import type { Action } from "../../types/battle";

interface ActionBarProps {
  legalActions: Action[];
  onAction: (action: Action) => void;
  disabled?: boolean;
}

export function ActionBar({ legalActions, onAction, disabled }: ActionBarProps) {
  const otherActions = legalActions.filter(
    (a) => a.type !== "dahai" && a.type !== "none"
  );
  const hasNone = legalActions.some((a) => a.type === "none");
  const hasReach = legalActions.some((a) => a.type === "reach");

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: 8,
        padding: 12,
        borderRadius: 'var(--radius-md)',
        background: 'var(--card-bg)',
        border: '1px solid var(--border)',
        backdropFilter: 'blur(12px)',
        transition: 'background var(--transition), border-color var(--transition)',
      }}
    >
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, justifyContent: 'center' }}>
        {hasReach && (
          <button
            style={{
              padding: '6px 16px',
              borderRadius: 'var(--radius-sm)',
              border: 'none',
              background: 'linear-gradient(135deg, var(--gold) 0%, #c49843 100%)',
              color: '#1a1a1a',
              fontWeight: 700,
              fontSize: 13,
              cursor: disabled ? 'not-allowed' : 'pointer',
              opacity: disabled ? 0.5 : 1,
              transition: 'opacity var(--transition)',
            }}
            disabled={disabled}
            onClick={() => {
              const reachAction = legalActions.find((a) => a.type === "reach");
              if (reachAction) onAction(reachAction);
            }}
          >
            立直
          </button>
        )}
        {otherActions.map((action, i) => (
          <button
            key={i}
            style={{
              ...actionBtnStyle(action.type),
              cursor: disabled ? 'not-allowed' : 'pointer',
              opacity: disabled ? 0.5 : 1,
            }}
            disabled={disabled}
            onClick={() => onAction(action)}
          >
            {actionLabel(action)}
          </button>
        ))}
        {hasNone && (
          <button
            style={{
              padding: '6px 16px',
              borderRadius: 'var(--radius-sm)',
              border: '1px solid var(--border)',
              background: 'var(--card-bg)',
              color: 'var(--text-secondary)',
              fontSize: 13,
              cursor: disabled ? 'not-allowed' : 'pointer',
              opacity: disabled ? 0.5 : 1,
              transition: 'opacity var(--transition), background var(--transition)',
            }}
            disabled={disabled}
            onClick={() => {
              const noneAction = legalActions.find((a) => a.type === "none");
              if (noneAction) onAction(noneAction);
            }}
          >
            过
          </button>
        )}
      </div>
    </div>
  );
}

function actionBtnStyle(type: string): React.CSSProperties {
  switch (type) {
    case "pon":
      return { padding: '6px 16px', borderRadius: 'var(--radius-sm)', border: 'none', background: 'linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%)', color: '#fff', fontWeight: 600, fontSize: 13 };
    case "chi":
      return { padding: '6px 16px', borderRadius: 'var(--radius-sm)', border: 'none', background: 'linear-gradient(135deg, var(--success) 0%, #219a52 100%)', color: '#fff', fontWeight: 600, fontSize: 13 };
    case "daiminkan":
    case "ankan":
    case "kakan":
      return { padding: '6px 16px', borderRadius: 'var(--radius-sm)', border: 'none', background: 'linear-gradient(135deg, #8e44ad 0%, #7d3c9e 100%)', color: '#fff', fontWeight: 600, fontSize: 13 };
    case "hora":
      return { padding: '6px 16px', borderRadius: 'var(--radius-sm)', border: 'none', background: 'linear-gradient(135deg, var(--error) 0%, #c0392b 100%)', color: '#fff', fontWeight: 700, fontSize: 13 };
    default:
      return { padding: '6px 16px', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border)', background: 'var(--card-bg)', color: 'var(--text-primary)', fontSize: 13 };
  }
}

function actionLabel(action: Action): string {
  switch (action.type) {
    case "dahai": return action.tsumogiri ? "摸切" : `打 ${action.pai}`;
    case "reach": return "立直";
    case "pon": return `碰 ${action.pai}`;
    case "chi": return `吃 ${action.pai}`;
    case "daiminkan": return `大明杠 ${action.pai}`;
    case "ankan": return `暗杠 ${action.pai}`;
    case "kakan": return `加杠 ${action.pai}`;
    case "hora": return action.target === action.actor ? "自摸" : "荣和";
    default: return action.type;
  }
}
