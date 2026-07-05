// src/replay_ui/src/pages/ReviewPage.tsx
import { useNavigate } from 'react-router-dom';
import { History } from 'lucide-react';
import { UploadForm } from '../components/Upload/UploadForm';
import type { ReplayData } from '../types/replay';
import { PageHeader, PageShell, SectionTitle } from '../components/Layout/PageScaffold';

export function ReviewPage() {
  const navigate = useNavigate();

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
    if (replayData.replay_id) {
      navigate(buildReplaySearchForData(replayData), { replace: true });
    } else {
      navigate('/game-replay', { state: { replayData }, replace: true });
    }
  };

  return (
    <PageShell width={1180}>
      <PageHeader
        title="牌谱 Review"
        actions={(
          <button
            type="button"
            onClick={() => navigate('/review-history')}
            className="btn-secondary"
            style={{ height: 32, display: 'inline-flex', alignItems: 'center', gap: 6 }}
          >
            <History size={14} />
            历史 Review
          </button>
        )}
      />

      <section className="card" style={{ padding: 12 }}>
          <SectionTitle title="牌谱输入" />
          <UploadForm onDataLoaded={handleDataLoaded} />
      </section>

    </PageShell>
  );
}
