import { useNavigate } from 'react-router-dom';
import { BarChart2, Users, Bot, AlertTriangle } from 'lucide-react';
import { MetricCard, PageHeader, PageShell, SectionTitle } from '../components/Layout/PageScaffold';

interface QuickStartCardProps {
  onClick: () => void;
  icon: React.ReactNode;
  title: string;
  description: React.ReactNode;
  gradient: string;
  glow: string;
}

function QuickStartCard({ onClick, icon, title, description, gradient, glow }: QuickStartCardProps) {
  return (
    <button
      onClick={onClick}
      style={{
        flex: '1 1 200px',
        minHeight: 164,
        padding: '22px 24px',
        borderRadius: 'var(--radius-lg)',
        border: 'none',
        background: gradient,
        color: '#fff',
        cursor: 'pointer',
        textAlign: 'left',
        boxShadow: glow,
        transition: 'transform 0.2s, box-shadow 0.2s',
      }}
      onMouseEnter={e => {
        e.currentTarget.style.transform = 'translateY(-2px)';
        e.currentTarget.style.boxShadow = glow.replace('0 4px', '0 8px');
      }}
      onMouseLeave={e => {
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.boxShadow = glow;
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
        {icon}
        <span style={{ fontWeight: 700, fontSize: 15 }}>{title}</span>
      </div>
      <div style={{ fontSize: 13, opacity: 0.88, lineHeight: 1.6 }}>{description}</div>
    </button>
  );
}

export function DashboardPage() {
  const navigate = useNavigate();

  return (
    <PageShell width={980}>
      <PageHeader
        eyebrow="Workspace"
        title="麻将工作台"
        description="统一管理牌谱分析、实时对战、4 Bot 对战和 selfplay 对局回放。当前 GUI 支持 xmodel1、keqingv4、mortal 和 rulebase。"
      />

      <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 28 }}>
        <MetricCard label="核心入口" value={4} />
        <MetricCard label="当前主线" value="xmodel1" tone="success" />
        <MetricCard label="当前备线" value="keqingv4" tone="warning" />
      </div>

      <div style={{ marginBottom: 28 }}>
        <SectionTitle title="快速开始" description="每个入口都直接进入对应主流程，不再让首页只当一个中转站。" />
        <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            <QuickStartCard
              onClick={() => navigate('/review')}
              icon={<BarChart2 size={20} />}
              title="牌谱分析"
              description={<>上传天凤链接或 mjai JSON。<br />默认按 xmodel1 主线跑谱，也可切到 keqingv4 / mortal / rulebase。</>}
              gradient="linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%)"
              glow="0 4px 12px var(--accent-shadow)"
            />
            <QuickStartCard
              onClick={() => navigate('/battle')}
              icon={<Users size={20} />}
              title="人机对战"
              description={<>与 AI Bot 进行实战练习。<br />默认对手为 xmodel1，兼容支持 keqingv4 / mortal / rulebase。</>}
              gradient="linear-gradient(135deg, var(--success) 0%, #219a52 100%)"
              glow="0 4px 12px rgba(39,174,96,0.3)"
            />
            <QuickStartCard
              onClick={() => navigate('/bot-battle')}
              icon={<Bot size={20} />}
              title="4 Bot 对战"
              description={<>观看 4 个 AI 自动对战。<br />适合比较主线、备线和规则基线的兼容表现。</>}
              gradient="linear-gradient(135deg, #8e44ad 0%, #7d3c9e 100%)"
              glow="0 4px 12px rgba(142,68,173,0.3)"
            />
            <QuickStartCard
              onClick={() => navigate('/selfplay-anomalies')}
              icon={<AlertTriangle size={20} />}
              title="对局回放"
              description={<>浏览 selfplay 保存的对局与异常抽样。<br />直接跳转决策或牌桌视图。</>}
              gradient="linear-gradient(135deg, #c0392b 0%, #e67e22 100%)"
              glow="0 4px 12px rgba(192,57,43,0.28)"
            />
          </div>
      </div>

      <div className="card">
        <SectionTitle title="近期活动" description="后续可以接入最近回放、最近对战和对局回放摘要。" />
          <div style={{ color: 'var(--text-muted)', fontSize: 13, padding: '12px 0' }}>
            暂无活动记录，开始使用功能后将显示统计信息
          </div>
      </div>
    </PageShell>
  );
}
