import { useNavigate } from 'react-router-dom';
import { Activity, BarChart2, Users } from 'lucide-react';
import { PageHeader, PageShell, SectionTitle } from '../components/Layout/PageScaffold';

const entryItems = [
  { label: '牌谱 Review', path: '/review', icon: BarChart2, note: '选择多个 Mortal checkpoint，生成 NAGA 风格权重对比' },
  { label: '人机对战', path: '/battle', icon: Users, note: '使用明确 checkpoint 的 Mortal 权重本地实战' },
];

function WorkbenchPanel({
  title,
  children,
}: {
  title: string;
  children: React.ReactNode;
}) {
  return (
    <section className="card" style={{ padding: 12 }}>
      <SectionTitle title={title} />
      {children}
    </section>
  );
}

export function DashboardPage() {
  const navigate = useNavigate();

  return (
    <PageShell width={1120}>
      <PageHeader
        eyebrow="Workspace"
        title="工作台总览"
        description="8000 端口的统一 GUI。只保留牌谱 Review 和人机对战，其它 demo 型工具不再作为 GUI 主入口展示。"
        actions={
          <button
            onClick={() => navigate('/review')}
            className="btn-primary"
            style={{ height: 34, padding: '0 16px', fontSize: 13 }}
          >
            进入牌谱 Review
          </button>
        }
      />

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: 12,
          alignItems: 'start',
        }}
      >
        <WorkbenchPanel title="常用入口">
          <div style={{ display: 'grid', gap: 6 }}>
            {entryItems.map((item) => {
              const Icon = item.icon;
              return (
                <button
                  key={item.path}
                  onClick={() => navigate(item.path)}
                  style={{
                    height: 44,
                    border: '1px solid var(--border)',
                    borderRadius: 7,
                    background: 'var(--card-bg)',
                    color: 'var(--text-primary)',
                    display: 'grid',
                    gridTemplateColumns: '24px 1fr auto',
                    alignItems: 'center',
                    gap: 8,
                    padding: '0 10px',
                    textAlign: 'left',
                    cursor: 'pointer',
                  }}
                >
                  <Icon size={16} style={{ color: 'var(--accent)' }} />
                  <span style={{ minWidth: 0 }}>
                    <span style={{ display: 'block', fontSize: 13, fontWeight: 700 }}>{item.label}</span>
                    <span style={{ display: 'block', fontSize: 11, color: 'var(--text-muted)' }}>{item.note}</span>
                  </span>
                  <span style={{ color: 'var(--text-muted)', fontSize: 16 }}>›</span>
                </button>
              );
            })}
          </div>
        </WorkbenchPanel>

        <div style={{ display: 'grid', gap: 12 }}>
          <WorkbenchPanel title="运行状态">
            <div style={{ display: 'grid', gap: 8, fontSize: 12 }}>
              {[
                ['服务', 'src/main.py · local'],
                ['HTTP', '127.0.0.1:8000'],
                ['GUI', 'replay_ui/dist'],
                ['Review', '多 Mortal checkpoint 对比'],
              ].map(([label, value]) => (
                <div
                  key={label}
                  style={{
                    display: 'flex',
                    justifyContent: 'space-between',
                    gap: 12,
                    borderBottom: '1px solid var(--border)',
                    paddingBottom: 6,
                  }}
                >
                  <span style={{ color: 'var(--text-muted)' }}>{label}</span>
                  <span style={{ color: 'var(--text-primary)', fontWeight: 700 }}>{value}</span>
                </div>
              ))}
            </div>
          </WorkbenchPanel>

          <WorkbenchPanel title="最近内容">
            <div style={{ display: 'grid', gap: 7, fontSize: 12, color: 'var(--text-secondary)' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                <Activity size={14} style={{ color: 'var(--accent)' }} />
                最近牌谱会在上传后从 Review 页进入。
              </div>
              <div>GUI 入口已收敛到 Review 和人机对战，避免把临时 demo 工具混入主流程。</div>
            </div>
          </WorkbenchPanel>
        </div>
      </div>
    </PageShell>
  );
}
