import { useEffect, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { LayoutDashboard, Users, BarChart2, AlertTriangle, Menu, X, Bot, PanelLeftClose, PanelLeftOpen } from 'lucide-react';
import { ThemeToggle } from './ThemeToggle';

const NAV_ITEMS = [
  { path: '/', icon: LayoutDashboard, label: '仪表盘', exact: true },
  { path: '/battle', icon: Users, label: '人机对战' },
  { path: '/bot-battle', icon: Bot, label: '4 Bot 对战' },
  { path: '/review', icon: BarChart2, label: '牌谱分析' },
  { path: '/selfplay-anomalies', icon: AlertTriangle, label: '对局回放' },
];

const TABLECLOTH_OPTIONS = [
  { id: 'default', label: '默认' },
  { id: 'green', label: '浅绿' },
  { id: 'blue', label: '浅蓝' },
  { id: 'beige', label: '米黄' },
] as const;

export function Sidebar() {
  const location = useLocation();
  const [isMobile, setIsMobile] = useState(false);
  const [open, setOpen] = useState(false);
  const [collapsed, setCollapsed] = useState(false);
  const [tablecloth, setTablecloth] = useState<(typeof TABLECLOTH_OPTIONS)[number]['id']>('default');

  useEffect(() => {
    const saved = window.localStorage.getItem('keqing.sidebar.collapsed');
    setCollapsed(saved === 'true');
    const storedTablecloth = window.localStorage.getItem('keqing.tablecloth');
    if (storedTablecloth && TABLECLOTH_OPTIONS.some((opt) => opt.id === storedTablecloth)) {
      setTablecloth(storedTablecloth as (typeof TABLECLOTH_OPTIONS)[number]['id']);
    }
  }, []);

  useEffect(() => {
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

  useEffect(() => {
    setOpen(false);
  }, [location.pathname]);

  const updateTablecloth = (next: (typeof TABLECLOTH_OPTIONS)[number]['id']) => {
    setTablecloth(next);
    window.localStorage.setItem('keqing.tablecloth', next);
    window.dispatchEvent(new StorageEvent('storage', { key: 'keqing.tablecloth', newValue: next }));
  };

  const tableclothControls = (
    <div style={{ display: 'grid', gap: 8, width: '100%' }}>
      <div style={{ fontSize: 11, color: 'var(--sidebar-text-muted)' }}>桌布</div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        {TABLECLOTH_OPTIONS.map((opt) => (
          <button
            key={opt.id}
            onClick={() => updateTablecloth(opt.id)}
            style={{
              padding: '4px 8px',
              borderRadius: 8,
              border: `1px solid ${tablecloth === opt.id ? 'var(--accent)' : 'var(--border)'}`,
              background: tablecloth === opt.id ? 'var(--accent-bg)' : 'var(--card-bg)',
              color: tablecloth === opt.id ? 'var(--accent)' : 'var(--text-secondary)',
              fontSize: 11,
              cursor: 'pointer',
            }}
          >
            {opt.label}
          </button>
        ))}
      </div>
    </div>
  );

  if (isMobile) {
    return (
      <>
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
            <button
              onClick={() => setOpen(v => !v)}
              style={{
                width: 36,
                height: 36,
                borderRadius: 8,
                border: '1px solid var(--border)',
                background: 'var(--card-bg)',
                color: 'var(--text-primary)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                cursor: 'pointer',
              }}
              title="菜单"
            >
              {open ? <X size={18} /> : <Menu size={18} />}
            </button>
            <div>
              <div style={{ fontSize: 14, fontWeight: 800, color: 'var(--text-primary)' }}>Keqing</div>
              <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>立直麻将工作台</div>
            </div>
          </div>
          <ThemeToggle />
        </div>

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

        <aside
          className="flex flex-col"
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
          }}
        >
          <div className="p-4" style={{ borderBottom: '1px solid var(--sidebar-border)' }}>
            <div className="flex items-center gap-3">
              <div
                className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
                style={{
                  background: 'linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%)',
                  boxShadow: '0 4px 12px var(--accent-shadow)',
                  transition: 'background var(--transition), box-shadow var(--transition)',
                }}
              >
                <span
                  className="text-white font-bold text-sm"
                  style={{ color: 'var(--btn-primary-text)' }}
                >
                  麻
                </span>
              </div>
              <div>
                <h1 className="text-base font-bold" style={{ color: 'var(--sidebar-text)', transition: 'color var(--transition)' }}>
                  Keqing
                </h1>
                <p className="text-xs" style={{ color: 'var(--sidebar-text-muted)', transition: 'color var(--transition)' }}>
                  立直麻将工作台
                </p>
              </div>
            </div>
          </div>

          <nav className="flex-1 p-3">
            {NAV_ITEMS.map(({ path, icon: Icon, label, exact }) => (
              <NavLink
                key={path}
                to={path}
                end={exact}
                className={({ isActive }) =>
                  `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium mb-1 nav-link-item ${
                    isActive ? 'nav-link-active' : ''
                  }`
                }
                style={{ transition: 'background var(--transition), color var(--transition)' }}
              >
                <Icon size={18} />
                <span>{label}</span>
              </NavLink>
            ))}
          </nav>

          <div
            className="p-4 flex items-center justify-between"
            style={{ borderTop: '1px solid var(--sidebar-border)' }}
          >
            <div style={{ display: 'grid', gap: 10, width: '100%' }}>
              {tableclothControls}
              <div className="text-xs" style={{ color: 'var(--sidebar-text-muted)', transition: 'color var(--transition)' }}>
                v2.0
              </div>
              <div style={{ display: 'flex', justifyContent: 'flex-end' }}>
                <ThemeToggle />
              </div>
            </div>
          </div>
        </aside>

        <style>{`
          .nav-link-item {
            color: var(--sidebar-text);
          }
          .nav-link-item:hover {
            background: var(--sidebar-hover-bg);
          }
          .nav-link-active {
            background: var(--sidebar-active-bg) !important;
            color: var(--sidebar-active-text) !important;
          }
        `}</style>
      </>
    );
  }

  return (
    <aside
      className="flex flex-col h-full"
      style={{
        width: collapsed ? 72 : 224,
        background: 'var(--sidebar-bg)',
        borderRight: '1px solid var(--sidebar-border)',
        backdropFilter: 'blur(12px)',
        transition: 'background var(--transition), border-color var(--transition), width 0.18s ease',
      }}
    >
      <div className="p-4" style={{ borderBottom: '1px solid var(--sidebar-border)', display: 'flex', alignItems: 'center', justifyContent: collapsed ? 'center' : 'space-between' }}>
        {!collapsed && (
          <div className="flex items-center gap-3" style={{ minWidth: 0 }}>
            <div
              className="w-9 h-9 rounded-xl flex items-center justify-center flex-shrink-0"
              style={{
                background: 'linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%)',
                boxShadow: '0 4px 12px var(--accent-shadow)',
              }}
            >
              <span className="text-white font-bold text-sm" style={{ color: 'var(--btn-primary-text)' }}>麻</span>
            </div>
            <div>
              <h1 className="text-base font-bold" style={{ color: 'var(--sidebar-text)' }}>Keqing</h1>
              <p className="text-xs" style={{ color: 'var(--sidebar-text-muted)' }}>立直麻将工作台</p>
            </div>
          </div>
        )}
        <button
          onClick={() => setCollapsed(v => !v)}
          style={{
            width: 34,
            height: 34,
            borderRadius: 8,
            border: '1px solid var(--border)',
            background: 'var(--card-bg)',
            color: 'var(--text-primary)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            cursor: 'pointer',
            flexShrink: 0,
          }}
          title={collapsed ? '展开侧栏' : '收起侧栏'}
        >
          {collapsed ? <PanelLeftOpen size={17} /> : <PanelLeftClose size={17} />}
        </button>
      </div>

      <nav className="flex-1 p-3">
        {NAV_ITEMS.map(({ path, icon: Icon, label, exact }) => (
          <NavLink
            key={path}
            to={path}
            end={exact}
            className={({ isActive }) =>
              `flex items-center ${collapsed ? 'justify-center' : 'gap-3'} px-3 py-2.5 rounded-lg text-sm font-medium mb-1 nav-link-item ${
                isActive ? 'nav-link-active' : ''
              }`
            }
            style={{ transition: 'background var(--transition), color var(--transition)' }}
            title={label}
          >
            <Icon size={18} />
            {!collapsed && <span>{label}</span>}
          </NavLink>
        ))}
      </nav>

      <div
        className="p-4 flex items-center justify-between"
        style={{ borderTop: '1px solid var(--sidebar-border)', justifyContent: collapsed ? 'center' : 'space-between' }}
      >
        {collapsed ? (
          <ThemeToggle />
        ) : (
          <div style={{ display: 'grid', gap: 10, width: '100%' }}>
            {tableclothControls}
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div className="text-xs" style={{ color: 'var(--sidebar-text-muted)', transition: 'color var(--transition)' }}>
                v2.0
              </div>
              <ThemeToggle />
            </div>
          </div>
        )}
      </div>

      <style>{`
        .nav-link-item {
          color: var(--sidebar-text);
        }
        .nav-link-item:hover {
          background: var(--sidebar-hover-bg);
        }
        .nav-link-active {
          background: var(--sidebar-active-bg) !important;
          color: var(--sidebar-active-text) !important;
        }
      `}</style>
    </aside>
  );
}
