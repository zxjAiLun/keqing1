// src/replay_ui/src/pages/ReviewPage.tsx
import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { GitCompareArrows } from 'lucide-react';
import { UploadForm } from '../components/Upload/UploadForm';
import type { ReplayData } from '../types/replay';
import { PageHeader, PageShell, SectionTitle } from '../components/Layout/PageScaffold';

function inferReportName(path: string, index: number) {
  const normalized = path.replace(/\\/g, '/');
  const file = normalized.split('/').pop() ?? path;
  const match = file.match(/__([^_]+)__p\d+\.json$/);
  if (match) return match[1];
  return file.replace(/\.json$/i, '') || `model-${index + 1}`;
}

export function ReviewPage() {
  const navigate = useNavigate();
  const [uploadedData, setUploadedData] = useState<ReplayData | null>(null);

  const buildReplaySearchForData = (data: ReplayData) => {
    if (!data.replay_id) return '/game-replay';
    const params = new URLSearchParams({
      id: data.replay_id,
      player_id: String(data.player_id ?? 0),
    });
    for (const report of data.teacher_report_paths ?? []) {
      params.append('teacher_reports', report);
    }
    return `/game-replay?${params.toString()}`;
  };

  const handleDataLoaded = (data: unknown) => {
    const replayData = data as ReplayData;
    setUploadedData(replayData);
    if (replayData.replay_id) {
      navigate(buildReplaySearchForData(replayData), { replace: true });
    } else {
      navigate('/game-replay', { state: { replayData }, replace: true });
    }
  };

  const teacherReports = uploadedData?.teacher_report_paths ?? [];

  return (
    <PageShell width={1180}>
        <PageHeader
          eyebrow="Review"
          title="牌谱 Review"
          description="选择 Mortal checkpoint 后运行 review，完成后直接进入牌桌。牌桌中当前模型用紫色显示，其它模型用灰色显示。"
        />

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: 12,
          alignItems: 'start',
        }}
      >
        <section className="card" style={{ padding: 12 }}>
          <SectionTitle title="牌谱输入" description="支持天凤链接和 mjai JSON。上传成功后直接进入牌桌 Review。" />
          <UploadForm onDataLoaded={handleDataLoaded} />
        </section>

        <section className="card" style={{ padding: 12 }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
            <GitCompareArrows size={15} style={{ color: '#8e44ad' }} />
            <div style={{ fontSize: 13, fontWeight: 800, color: 'var(--text-primary)' }}>多模型权重</div>
          </div>
          <div style={{ display: 'grid', gap: 6, marginTop: 10 }}>
            {!uploadedData ? (
              <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                在左侧选择 70k、T1@71000、v4 或 gui_mortal。运行后会在这里显示生成的 report 路径。
              </div>
            ) : (
              (uploadedData.selected_teacher_models ?? teacherReports.map((report, index) => ({
                type: 'mortal' as const,
                label: inferReportName(report, index),
                checkpoint: '',
              }))).map((model, index) => {
                const report = teacherReports[index] ?? '';
                return (
                <div
                  key={`${model.label}-${index}`}
                  style={{
                    display: 'grid',
                    gridTemplateColumns: '72px 1fr 44px',
                    gap: 8,
                    alignItems: 'center',
                    minHeight: 30,
                    border: '1px solid var(--border)',
                    borderRadius: 7,
                    padding: '4px 7px',
                    fontSize: 11,
                  }}
                >
                  <span
                    style={{
                      borderRadius: 5,
                      background: index === 0 ? 'rgba(142,68,173,0.14)' : 'var(--button-bg)',
                      color: index === 0 ? '#8e44ad' : 'var(--text-secondary)',
                      fontWeight: 800,
                      padding: '3px 6px',
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                    }}
                  >
                    {model.label}
                  </span>
                  <span
                    title={report}
                    style={{
                      minWidth: 0,
                      overflow: 'hidden',
                      textOverflow: 'ellipsis',
                      whiteSpace: 'nowrap',
                      color: 'var(--text-secondary)',
                      fontFamily: '"Menlo", "Consolas", monospace',
                    }}
                  >
                    {report || model.checkpoint}
                  </span>
                  <span style={{ color: 'var(--success)', textAlign: 'right' }}>已生成</span>
                </div>
                );
              })
            )}
          </div>
        </section>
      </div>

    </PageShell>
  );
}
