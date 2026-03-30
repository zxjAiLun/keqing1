// src/replay_ui/src/pages/ReviewPage.tsx
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { CheckCircle2 } from 'lucide-react';
import { UploadForm } from '../components/Upload/UploadForm';
import type { ReplayData } from '../types/replay';

export function ReviewPage() {
  const navigate = useNavigate();
  const [uploadedData, setUploadedData] = useState<ReplayData | null>(null);

  const handleDataLoaded = (data: unknown) => {
    setUploadedData(data as ReplayData);
  };

  return (
    <div style={{ minHeight: '100%', padding: '24px' }}>
      <div style={{ maxWidth: 860, margin: '0 auto' }}>

        {/* 上传卡片 */}
        <div className="card">
          <div className="card-title">跑谱 Review</div>
          <div style={{ marginBottom: 14 }}>
            <UploadForm onDataLoaded={handleDataLoaded} />
          </div>
        </div>

        {/* 数据已加载 → 视图选择 */}
        {uploadedData && (
          <div className="card">
            <div className="flex items-center gap-2 mb-4">
              <CheckCircle2 size={16} style={{ color: 'var(--success)' }} />
              <div className="card-title" style={{ marginBottom: 0 }}>
                数据已加载 · {uploadedData.log.length} 步
              </div>
            </div>

            {/* 数据概览 */}
            <div className="flex gap-3 mb-5" style={{ flexWrap: 'wrap' }}>
              {[
                { label: '总决策数', value: uploadedData.total_ops },
                { label: '匹配数', value: uploadedData.match_count },
                { label: '小局数', value: uploadedData.kyoku_order?.length ?? 0 },
                {
                  label: '匹配率',
                  value: uploadedData.total_ops > 0
                    ? ((uploadedData.match_count / uploadedData.total_ops) * 100).toFixed(1) + '%'
                    : '0%',
                },
              ].map((item) => (
                <div
                  key={item.label}
                  className="rounded-lg px-4 py-2 text-center"
                  style={{
                    background: 'var(--card-bg)',
                    border: '1px solid var(--border)',
                    minWidth: 100,
                  }}
                >
                  <div
                    style={{
                      fontSize: 18,
                      fontWeight: 700,
                      color: 'var(--accent)',
                      fontFamily: 'Menlo, monospace',
                    }}
                  >
                    {item.value}
                  </div>
                  <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
                    {item.label}
                  </div>
                </div>
              ))}
            </div>

            {/* 视图选择按钮 */}
            <div className="flex gap-3" style={{ flexWrap: 'wrap' }}>
              <button
                onClick={() => navigate('/game-replay', { state: { replayData: uploadedData } })}
                className="btn-primary"
                style={{ height: 38, padding: '0 24px', fontSize: 14 }}
              >
                ▶ 牌桌回放
              </button>
              <button
                onClick={() => navigate('/replay', { state: { replayData: uploadedData } })}
                className="btn-primary"
                style={{
                  height: 38,
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

      </div>
    </div>
  );
}
