// src/replay_ui/src/pages/ReviewPage.tsx
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { CheckCircle2 } from 'lucide-react';
import { UploadForm } from '../components/Upload/UploadForm';
import type { ReplayData } from '../types/replay';
import { MetricCard, PageHeader, PageShell, SectionTitle } from '../components/Layout/PageScaffold';
import { sameReplayAction } from '../utils/tileUtils';

export function ReviewPage() {
  const navigate = useNavigate();
  const [uploadedData, setUploadedData] = useState<ReplayData | null>(null);

  const handleDataLoaded = (data: unknown) => {
    setUploadedData(data as ReplayData);
  };

  const liveMatchCount = uploadedData
    ? uploadedData.log.filter((entry) => !entry.is_obs && sameReplayAction(entry.chosen, entry.gt_action)).length
    : 0;

  return (
    <PageShell width={980}>
      <PageHeader
        eyebrow="Review"
        title="牌谱分析"
        description="支持天凤链接和 mjai JSON 输入。跑谱成功后可以直接切到牌桌回放或决策列表，不需要重复上传。"
      />

      <div className="card">
        <SectionTitle title="跑谱输入" description="推荐先用天凤链接；本地 mjai 文件适合离线调试和异常复现。" />
          <div style={{ marginBottom: 14 }}>
            <UploadForm onDataLoaded={handleDataLoaded} />
          </div>
      </div>

      {uploadedData && (
        <div className="card">
          <div className="flex items-center gap-2 mb-4">
            <CheckCircle2 size={16} style={{ color: 'var(--success)' }} />
            <div className="card-title" style={{ marginBottom: 0 }}>
              数据已加载 · {uploadedData.total_ops} 决策步 / {uploadedData.log.length} 总步
            </div>
          </div>

          <div style={{ display: 'flex', gap: 12, flexWrap: 'wrap', marginBottom: 20 }}>
            <MetricCard label="总决策数" value={uploadedData.total_ops} />
            <MetricCard label="匹配数" value={liveMatchCount} tone="success" />
            <MetricCard label="小局数" value={uploadedData.kyoku_order?.length ?? 0} />
            <MetricCard
              label="匹配率"
              value={
                uploadedData.total_ops > 0
                  ? ((liveMatchCount / uploadedData.total_ops) * 100).toFixed(1) + '%'
                  : '0%'
              }
              tone="warning"
            />
          </div>

          <SectionTitle title="进入视图" description="牌桌视图适合看局面演进，决策列表适合快速定位差异步。" />
          <div className="flex gap-3" style={{ flexWrap: 'wrap' }}>
            <button
              onClick={() => navigate('/game-replay', { state: { replayData: uploadedData } })}
              className="btn-primary"
              style={{ height: 40, padding: '0 24px', fontSize: 14 }}
            >
              ▶ 牌桌回放
            </button>
            <button
              onClick={() => navigate('/replay', { state: { replayData: uploadedData } })}
              className="btn-primary"
              style={{
                height: 40,
                padding: '0 24px',
                fontSize: 14,
                background: 'var(--success)',
              }}
            >
              ▶ 决策列表
            </button>
          </div>
        </div>
      )}
    </PageShell>
  );
}
