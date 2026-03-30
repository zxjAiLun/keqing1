import { useNavigate } from 'react-router-dom';
import { BarChart2, Users, Bot } from 'lucide-react';

interface QuickStartCardProps {
  onClick: () => void;
  icon: React.ReactNode;
  title: string;
  description: string;
  gradient: string;
  glow: string;
}

function QuickStartCard({ onClick, icon, title, description, gradient, glow }: QuickStartCardProps) {
  return (
    <button
      onClick={onClick}
      style={{
        flex: '1 1 200px',
        padding: '20px 24px',
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
      <p style={{ fontSize: 13, opacity: 0.85, lineHeight: 1.5 }}>{description}</p>
    </button>
  );
}

export function DashboardPage() {
  const navigate = useNavigate();

  return (
    <div style={{ minHeight: '100%', padding: '24px' }}>
      <div style={{ maxWidth: 860, margin: '0 auto' }}>

        <h1 style={{ fontSize: 22, fontWeight: 700, color: 'var(--text-primary)', marginBottom: 6 }}>
          仪表盘
        </h1>
        <p style={{ fontSize: 14, color: 'var(--text-muted)', marginBottom: 28 }}>
          欢迎使用 Keqing 立直麻将分析系统
        </p>

        {/* 快速开始 */}
        <div style={{ marginBottom: 28 }}>
          <h2 style={{
            fontSize: 14, fontWeight: 700, color: 'var(--text-secondary)',
            marginBottom: 12, textTransform: 'uppercase', letterSpacing: '0.5px'
          }}>
            快速开始
          </h2>
          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            <QuickStartCard
              onClick={() => navigate('/review')}
              icon={<BarChart2 size={20} />}
              title="牌谱分析"
              description="上传天凤链接或 JSONL<br />分析 Bot 与玩家的决策差异"
              gradient="linear-gradient(135deg, var(--accent) 0%, var(--accent-hover) 100%)"
              glow="0 4px 12px var(--accent-shadow)"
            />
            <QuickStartCard
              onClick={() => navigate('/battle')}
              icon={<Users size={20} />}
              title="人机对战"
              description="与 AI Bot 进行对战练习<br />支持立直、吃碰杠等操作"
              gradient="linear-gradient(135deg, var(--success) 0%, #219a52 100%)"
              glow="0 4px 12px rgba(39,174,96,0.3)"
            />
            <QuickStartCard
              onClick={() => navigate('/bot-battle')}
              icon={<Bot size={20} />}
              title="4 Bot 对战"
              description="观看 4 个 AI 互相对战<br />支持导出 Mjai/Tenhou6 牌谱"
              gradient="linear-gradient(135deg, #8e44ad 0%, #7d3c9e 100%)"
              glow="0 4px 12px rgba(142,68,173,0.3)"
            />
          </div>
        </div>

        {/* 近期统计 */}
        <div className="card">
          <div className="card-title">近期活动</div>
          <div style={{ color: 'var(--text-muted)', fontSize: 13, padding: '12px 0' }}>
            暂无活动记录，开始使用功能后将显示统计信息
          </div>
        </div>

      </div>
    </div>
  );
}
