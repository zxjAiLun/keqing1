import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';

export function MainLayout() {
  return (
    <div
      className="flex"
      style={{
        minHeight: '100dvh',
        background: 'var(--page-bg)',
        transition: 'background var(--transition)',
      }}
    >
      <Sidebar />
      <main
        className="flex-1 overflow-hidden"
        style={{ background: 'var(--page-bg)', transition: 'background var(--transition)' }}
      >
        <Outlet />
      </main>
    </div>
  );
}
