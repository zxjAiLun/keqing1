import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { BarChart2, History, Users } from 'lucide-react';
import { replayApi } from '../api/replayApi';
import { PageHeader, PageShell, SectionTitle } from '../components/Layout/PageScaffold';
import type { ReviewHistoryItem } from '../types/replay';
import { buildReviewHistoryPath } from './ReviewHistoryPage';

const entryItems = [
  { label: '牌谱 Review', path: '/review', icon: BarChart2 },
  { label: '历史 Review', path: '/review-history', icon: History },
  { label: '人机对战', path: '/battle', icon: Users },
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
  const [history, setHistory] = useState<ReviewHistoryItem[]>([]);

  useEffect(() => {
    replayApi.listReviewHistory().then((items) => setHistory(items.slice(0, 6))).catch(() => setHistory([]));
  }, []);

  return (
    <PageShell width={1120}>
      <PageHeader
        title="工作台总览"
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
                  <span style={{ minWidth: 0, fontSize: 13, fontWeight: 700 }}>{item.label}</span>
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

          <WorkbenchPanel title="最近 Review">
            <div style={{ display: 'grid', gap: 5 }}>
              {history.map((item) => (
                <button
                  key={`${item.replay_id}-${item.player_id}`}
                  type="button"
                  onClick={() => navigate(buildReviewHistoryPath(item))}
                  style={{
                    minHeight: 34,
                    border: '1px solid var(--border)',
                    borderRadius: 5,
                    background: 'var(--card-bg)',
                    color: 'var(--text-primary)',
                    display: 'grid',
                    gridTemplateColumns: '132px 1fr auto',
                    gap: 8,
                    alignItems: 'center',
                    padding: '4px 7px',
                    textAlign: 'left',
                    cursor: 'pointer',
                    fontSize: 11,
                  }}
                >
                  <span style={{ fontFamily: 'Menlo, Consolas, monospace' }}>{item.created_at}</span>
                  <span style={{ fontWeight: 700 }}>{item.player_name || `P${item.player_id}`}</span>
                  <span>{item.models.length} 模型</span>
                </button>
              ))}
              {history.length === 0 && <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>暂无记录</div>}
            </div>
          </WorkbenchPanel>
        </div>
      </div>
    </PageShell>
  );
}
