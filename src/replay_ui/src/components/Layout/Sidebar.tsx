import { NavLink } from 'react-router-dom';
import { LayoutDashboard, Users, BarChart2 } from 'lucide-react';
import { ThemeToggle } from './ThemeToggle';

const NAV_ITEMS = [
  { path: '/', icon: LayoutDashboard, label: '仪表盘', exact: true },
  { path: '/battle', icon: Users, label: '人机对战' },
  { path: '/review', icon: BarChart2, label: '牌谱分析' },
];

export function Sidebar() {
  return (
    <aside
      className="flex flex-col h-full"
      style={{
        width: 224,
        background: 'var(--sidebar-bg)',
        borderRight: '1px solid var(--sidebar-border)',
        backdropFilter: 'blur(12px)',
        transition: 'background var(--transition), border-color var(--transition)',
      }}
    >
      {/* Logo */}
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
              立直麻将
            </p>
          </div>
        </div>
      </div>

      {/* 导航 */}
      <nav className="flex-1 p-3">
        {NAV_ITEMS.map(({ path, icon: Icon, label, exact }) => (
          <NavLink
            key={path}
            to={path}
            end={exact}
            className={({ isActive }) =>
              `flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium mb-0.5 nav-link-item ${
                isActive
                  ? 'nav-link-active'
                  : ''
              }`
            }
            style={{ transition: 'background var(--transition), color var(--transition)' }}
          >
            <Icon size={18} />
            <span>{label}</span>
          </NavLink>
        ))}
      </nav>

      {/* 底部：主题切换 + 版本 */}
      <div
        className="p-4 flex items-center justify-between"
        style={{ borderTop: '1px solid var(--sidebar-border)' }}
      >
        <div className="text-xs" style={{ color: 'var(--sidebar-text-muted)', transition: 'color var(--transition)' }}>
          v2.0
        </div>
        <ThemeToggle />
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
