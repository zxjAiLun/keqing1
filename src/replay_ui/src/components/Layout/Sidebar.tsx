import { useEffect, useState } from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Users,
  BarChart2,
  AlertTriangle,
  Menu,
  X,
  Bot,
  PanelLeftClose,
  PanelLeftOpen,
} from 'lucide-react';
import { ThemeToggle } from './ThemeToggle';
import { Button } from '../ui/Button';
import { t } from '../ui/tokens';
import { useTheme } from '../../context/themeStore';
import { TABLECLOTH_OPTIONS } from '../BattleBoard/tableclothOptions';
import type { TableclothId } from '../BattleBoard/tableclothOptions';

const NAV_ITEMS: Array<{
  path: string;
  icon: typeof LayoutDashboard;
  label: string;
  exact?: boolean;
}> = [
  { path: '/', icon: LayoutDashboard, label: '仪表盘', exact: true },
  { path: '/battle', icon: Users, label: '人机对战' },
  { path: '/bot-battle', icon: Bot, label: '4 Bot 对战' },
  { path: '/review', icon: BarChart2, label: '牌谱分析' },
  { path: '/selfplay-anomalies', icon: AlertTriangle, label: '对局回放' },
];

export function Sidebar() {
  const [isMobile, setIsMobile] = useState(false);
  const [open, setOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(
    () => {
      const stored =
        window.localStorage.getItem('keqing1.sidebar.collapsed')
        ?? window.localStorage.getItem('keqing.sidebar.collapsed');
      return stored === 'true';
    },
  );
  const [tablecloth, setTablecloth] = useState<TableclothId>(() => {
    const stored =
      window.localStorage.getItem('keqing1.tablecloth')
      ?? window.localStorage.getItem('keqing.tablecloth');
    if (stored && TABLECLOTH_OPTIONS.some((o) => o.id === stored)) {
      return stored as TableclothId;
    }
    return 'default';
  });

  useEffect(() => {
    window.localStorage.setItem('keqing1.sidebar.collapsed', String(collapsed));
    window.localStorage.setItem('keqing.sidebar.collapsed', String(collapsed));
  }, [collapsed]);

  useEffect(() => {
    const update = () => setIsMobile(window.innerWidth < 1024);
    update();
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  useEffect(() => {
    document.documentElement.style.setProperty('--mobile-shell-offset', isMobile ? '56px' : '0px');
    return () => document.documentElement.style.setProperty('--mobile-shell-offset', '0px');
  }, [isMobile]);

  const updateTablecloth = (next: TableclothId) => {
    setTablecloth(next);
    window.localStorage.setItem('keqing1.tablecloth', next);
    window.localStorage.setItem('keqing.tablecloth', next);
    window.dispatchEvent(new StorageEvent('storage', { key: 'keqing1.tablecloth', newValue: next }));
    window.dispatchEvent(new StorageEvent('storage', { key: 'keqing.tablecloth', newValue: next }));
  };

  // ── Table Cloth RGB Editor ─────────────────────────────────────────────────
  const { tableCloth, setTableCloth } = useTheme();

  const rgbPreview = `rgb(${tableCloth.r},${tableCloth.g},${tableCloth.b})`;

  const handleRgbChange = (channel: 'r' | 'g' | 'b', raw: string) => {
    const val = parseInt(raw, 10);
    if (isNaN(val)) return;
    setTableCloth({ ...tableCloth, [channel]: Math.max(0, Math.min(255, val)) });
  };

  const tableClothEditor = (
    <div style={{ display: 'grid', gap: 6, width: '100%' }}>
      <div style={{ fontSize: 11, color: 'var(--sidebar-text-muted)' }}>桌布颜色</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
        {/* 颜色预览 */}
        <div
          style={{
            width: 28,
            height: 20,
            borderRadius: 4,
            background: rgbPreview,
            border: '1px solid rgba(255,255,255,0.15)',
            flexShrink: 0,
            boxShadow: '0 1px 4px rgba(0,0,0,0.3)',
            transition: 'background 0.15s',
          }}
        />
        {/* RGB 数值输入 */}
        {(['r', 'g', 'b'] as const).map((ch) => (
          <div key={ch} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
            <span style={{ fontSize: 9, fontWeight: 700, color: ch === 'r' ? '#f87171' : ch === 'g' ? '#4ade80' : '#60a5fa', textTransform: 'uppercase' }}>
              {ch}
            </span>
            <input
              type="number"
              min={0}
              max={255}
              value={tableCloth[ch]}
              onChange={(e) => handleRgbChange(ch, e.target.value)}
              style={{
                width: 38,
                padding: '2px 4px',
                borderRadius: 4,
                border: '1px solid var(--border)',
                background: 'var(--card-bg)',
                color: 'var(--text-primary)',
                fontSize: 11,
                fontFamily: 'Menlo, monospace',
                textAlign: 'center',
              }}
            />
          </div>
        ))}
      </div>
    </div>
  );

  // ── Tablecloth selector (shared between mobile/desktop) ──────────────────
  const tableclothSelector = (
    <div style={{ display: 'grid', gap: 8, width: '100%' }}>
      <div style={{ fontSize: 11, color: 'var(--sidebar-text-muted)' }}>桌布</div>
      <div style={{ display: 'flex', gap: 5, flexWrap: 'wrap' }}>
        {TABLECLOTH_OPTIONS.map((opt) => (
          <button
            key={opt.id}
            onClick={() => updateTablecloth(opt.id)}
            title={opt.color}
            style={{
              padding: '3px 8px',
              borderRadius: 5,
              border: `1px solid ${tablecloth === opt.id ? 'rgba(212,168,83,0.5)' : 'rgba(255,255,255,0.08)'}`,
              background: tablecloth === opt.id ? 'rgba(212,168,83,0.12)' : 'rgba(255,255,255,0.04)',
              color: tablecloth === opt.id ? 'rgba(212,168,83,0.95)' : 'var(--sidebar-text-muted)',
              fontSize: 11,
              cursor: 'pointer',
              transition: 'all var(--transition)',
              fontWeight: tablecloth === opt.id ? 600 : 400,
            }}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );

  // ── Nav Link ─────────────────────────────────────────────────────────────
  const navLink = ({ path, icon: Icon, label, exact }: (typeof NAV_ITEMS)[number]) => (
    <NavLink
      key={path}
      to={path}
      end={exact}
      onClick={() => setOpen(false)}
      className={({ isActive }) =>
        `nav-link-item${isActive ? ' nav-link-active' : ''}`
      }
      style={{ transition: 'background var(--transition), color var(--transition)' }}
      title={label}
    >
      <Icon size={18} />
      {!collapsed && <span>{label}</span>}
    </NavLink>
  );

  // ── Logo ────────────────────────────────────────────────────────────────
  const logo = (
    <div
      style={{
        width: 36,
        height: 36,
        borderRadius: 'var(--radius-md)',
        background: 'linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%)',
        boxShadow: '0 4px 12px var(--accent-shadow)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        flexShrink: 0,
      }}
    >
      <span style={{ color: 'var(--btn-primary-text)', fontWeight: 700, fontSize: 14 }}>麻</span>
    </div>
  );

  // ─────────────────────────────────────────────────────────────────────
  if (isMobile) {
    return (
      <>
        {/* Mobile header bar */}
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            height: 56,
            zIndex: 120,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 14px',
            background: 'var(--sidebar-bg)',
            borderBottom: '1px solid var(--sidebar-border)',
            backdropFilter: 'blur(12px)',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => setOpen((v) => !v)}
              title="菜单"
              style={{ width: 36, height: 36, padding: 0 }}
            >
              {open ? <X size={18} /> : <Menu size={18} />}
            </Button>
            <div>
              <div style={{ fontSize: 14, fontWeight: 800, color: 'var(--text-primary)' }}>Keqing1</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>xmodel1 主线工作台</div>
            </div>
          </div>
          <ThemeToggle />
        </div>

        {/* Backdrop */}
        {open && (
          <div
            onClick={() => setOpen(false)}
            style={{
              position: 'fixed',
              inset: 0,
              background: 'rgba(0,0,0,0.35)',
              zIndex: 118,
            }}
          />
        )}

        {/* Drawer panel */}
        <aside
          style={{
            position: 'fixed',
            top: 56,
            left: 0,
            bottom: 0,
            width: 268,
            maxWidth: '82vw',
            background: 'var(--sidebar-bg)',
            borderRight: '1px solid var(--sidebar-border)',
            backdropFilter: 'blur(12px)',
            transform: open ? 'translateX(0)' : 'translateX(-100%)',
            transition: 'transform 0.22s ease',
            zIndex: 119,
            display: 'flex',
            flexDirection: 'column',
          }}
        >
          {/* Header */}
          <div
            style={{
              padding: 16,
              borderBottom: '1px solid var(--sidebar-border)',
              display: 'flex',
              alignItems: 'center',
              gap: 12,
            }}
          >
            {logo}
            <div>
              <div style={{ fontSize: 14, fontWeight: 800, color: 'var(--sidebar-text)' }}>Keqing1</div>
              <div style={{ fontSize: 11, color: 'var(--sidebar-text-muted)' }}>xmodel1 主线工作台</div>
            </div>
          </div>

          {/* Nav */}
          <nav style={{ flex: 1, padding: 12, display: 'flex', flexDirection: 'column', gap: 2 }}>
            {NAV_ITEMS.map(navLink)}
          </nav>

          {/* Footer */}
          <div
            style={{
              padding: 16,
              borderTop: '1px solid var(--sidebar-border)',
              display: 'grid',
              gap: 12,
            }}
          >
            {tableclothSelector}
            {tableClothEditor}
            <div style={{ fontSize: 11, color: 'var(--sidebar-text-muted)' }}>v2.0</div>
            <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
              <ThemeToggle />
            </div>
          </div>
        </aside>
      </>
    );
  }

  // ── Desktop ────────────────────────────────────────────────────────────
  return (
    <aside
      style={{
        width: collapsed ? t.sidebar.widthCollapsed : t.sidebar.width,
        background: 'var(--sidebar-bg)',
        borderRight: '1px solid var(--sidebar-border)',
        backdropFilter: 'blur(12px)',
        transition: `width 0.18s ease, background var(--transition), border-color var(--transition)`,
        display: 'flex',
        flexDirection: 'column',
        flexShrink: 0,
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div
        style={{
          padding: collapsed ? '16px 8px' : '16px',
          borderBottom: '1px solid var(--sidebar-border)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: collapsed ? 'center' : 'space-between',
          gap: 8,
        }}
      >
        {!collapsed && (
          <div style={{ display: 'flex', alignItems: 'center', gap: 12, minWidth: 0 }}>
            {logo}
            <div>
              <div style={{ fontSize: 14, fontWeight: 800, color: 'var(--sidebar-text)' }}>Keqing1</div>
              <div style={{ fontSize: 11, color: 'var(--sidebar-text-muted)' }}>xmodel1 主线工作台</div>
            </div>
          </div>
        )}
        <Button
          variant="secondary"
          size="sm"
          onClick={() => setCollapsed((v) => !v)}
          title={collapsed ? '展开侧栏' : '收起侧栏'}
          style={{ width: 34, height: 34, padding: 0, flexShrink: 0 }}
        >
          {collapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
        </Button>
      </div>

      {/* Nav */}
      <nav
        style={{
          flex: 1,
          padding: 12,
          display: 'flex',
          flexDirection: 'column',
          gap: 2,
          overflowY: 'auto',
        }}
      >
        {NAV_ITEMS.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            end={item.exact}
            className={({ isActive }) =>
              `nav-link-item${collapsed ? ' justify-center' : ''}${isActive ? ' nav-link-active' : ''}`
            }
            style={{
              display: 'flex',
              alignItems: 'center',
              gap: 12,
              padding: '10px 12px',
              borderRadius: 'var(--radius-sm)',
              color: 'var(--sidebar-text)',
              textDecoration: 'none',
              fontSize: 14,
              fontWeight: 500,
              transition: 'background var(--transition), color var(--transition)',
              justifyContent: collapsed ? 'center' : 'flex-start',
            }}
            title={item.label}
          >
            <item.icon size={18} />
            {!collapsed && <span>{item.label}</span>}
          </NavLink>
        ))}
      </nav>

      {/* Footer */}
      <div
        style={{
          padding: collapsed ? '10px 6px' : '12px 10px',
          borderTop: '1px solid var(--sidebar-border)',
          display: 'flex',
          flexDirection: 'column',
          gap: collapsed ? 6 : 10,
        }}
      >
        {!collapsed && tableclothSelector}
        {!collapsed && tableClothEditor}
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: collapsed ? 'center' : 'space-between',
          }}
        >
          {!collapsed && (
            <span style={{ fontSize: 11, color: 'var(--sidebar-text-muted)' }}>v2.0</span>
          )}
          <ThemeToggle />
        </div>
      </div>
    </aside>
  );
}
