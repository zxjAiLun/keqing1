import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { useNavigate } from 'react-router-dom';
import { BookOpen, Loader2, MonitorPlay, RefreshCw } from 'lucide-react';
import { replayApi } from '../api/replayApi';
import { PageHeader, PageShell, SectionTitle } from '../components/Layout/PageScaffold';
import type { BehaviorReplayCase } from '../types/replay';

const KIND_LABELS: Record<string, string> = {
  '80k_dealer_call_bad': '80k 亲家坏副露',
  '80k_rank2_call_bad': '80k 二位坏副露',
  '70k_dealer_call_good': '70k 亲家好副露',
  '70k_rank2_call_good': '70k 二位好副露',
};

function fmtTime(ts: number | null): string {
  if (!ts) return '-';
  return new Date(ts * 1000).toLocaleString();
}

function fmtMargin(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : '-';
}

export function BehaviorCasebookPage() {
  const navigate = useNavigate();
  const [cases, setCases] = useState<BehaviorReplayCase[]>([]);
  const [casebookPath, setCasebookPath] = useState('');
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const [selectedKind, setSelectedKind] = useState<string>('80k_dealer_call_bad');
  const [loading, setLoading] = useState(true);
  const [openingCaseId, setOpeningCaseId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    setError(null);
    replayApi.listBehaviorCasebook()
      .then((payload) => {
        setCases(payload.cases ?? []);
        setCasebookPath(payload.casebook_dir);
        setUpdatedAt(payload.updated_at);
        if (payload.cases?.length && !payload.cases.some((item) => item.case_kind === selectedKind)) {
          setSelectedKind(payload.cases[0].case_kind);
        }
        setLoading(false);
      })
      .catch((e) => {
        setError(String(e));
        setLoading(false);
      });
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const kinds = useMemo(() => {
    const counts = new Map<string, number>();
    for (const item of cases) counts.set(item.case_kind, (counts.get(item.case_kind) ?? 0) + 1);
    return Array.from(counts.entries()).sort(([a], [b]) => a.localeCompare(b));
  }, [cases]);

  const visibleCases = useMemo(
    () => cases.filter((item) => item.case_kind === selectedKind),
    [cases, selectedKind],
  );

  const openCase = async (item: BehaviorReplayCase) => {
    setOpeningCaseId(item.case_id);
    setError(null);
    try {
      const result = await replayApi.importBehaviorCase(item.case_id);
      navigate(
        `/game-replay?id=${encodeURIComponent(result.replay_id)}`
        + `&player_id=${result.player_id}`
        + `&focus_event_index=${result.focus_event_index}`
        + (result.focus_replay_step !== null ? `&focus_step=${result.focus_replay_step}` : ''),
      );
    } catch (e) {
      setError(String(e));
    } finally {
      setOpeningCaseId(null);
    }
  };

  if (loading) {
    return (
      <div style={{ minHeight: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 12 }}>
        <Loader2 size={28} className="animate-spin" style={{ color: 'var(--accent)' }} />
        <span style={{ color: 'var(--text-muted)' }}>加载行为 casebook...</span>
      </div>
    );
  }

  return (
    <PageShell width={1180}>
      <PageHeader
        eyebrow="Mortal Diagnostics"
        title="行为回放 Casebook"
        description="从 70k/80k L3 诊断中抽取代表性副露案例，点击后导入为普通 replay 并自动跳到关键 call。"
        actions={(
          <button onClick={load} className="btn-secondary" style={actionButtonStyle}>
            <RefreshCw size={15} />
            刷新
          </button>
        )}
      />

      {error && (
        <div className="card" style={{ marginBottom: 16, color: 'var(--error)', fontSize: 14 }}>
          {error}
        </div>
      )}

      <div className="card" style={{ marginBottom: 16 }}>
        <SectionTitle
          title="Casebook"
          description={`路径：${casebookPath || '-'} · 更新时间：${fmtTime(updatedAt)}`}
        />
        {cases.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontSize: 14 }}>
            未找到 casebook manifest。先运行 `scripts/mortal/export_behavior_replay_cases.py` 生成案例。
          </div>
        ) : (
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
            {kinds.map(([kind, count]) => {
              const active = kind === selectedKind;
              return (
                <button
                  key={kind}
                  onClick={() => setSelectedKind(kind)}
                  style={{
                    ...kindButtonStyle,
                    borderColor: active ? 'var(--accent)' : 'var(--border)',
                    background: active ? 'rgba(52, 152, 219, 0.12)' : 'var(--card-bg)',
                    color: active ? 'var(--accent)' : 'var(--text-primary)',
                  }}
                >
                  {KIND_LABELS[kind] ?? kind}
                  <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>{count}</span>
                </button>
              );
            })}
          </div>
        )}
      </div>

      {visibleCases.length > 0 && (
        <div style={{ display: 'grid', gap: 10 }}>
          {visibleCases.map((item) => (
            <div key={item.case_id} className="card" style={caseCardStyle}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap', marginBottom: 6 }}>
                    <BookOpen size={16} style={{ color: 'var(--accent)' }} />
                    <span style={{ fontWeight: 800, color: 'var(--text-primary)' }}>{item.case_id}</span>
                    <span style={pillStyle}>{item.model_label}</span>
                    <span style={pillStyle}>{item.action_type}</span>
                    <span style={item.houjuu ? dangerPillStyle : item.agari ? successPillStyle : pillStyle}>{item.outcome}</span>
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)', display: 'flex', gap: 10, flexWrap: 'wrap' }}>
                    <span>actor {item.actor}</span>
                    <span>turn {item.turn}</span>
                    <span>focus {item.focus_event_index}</span>
                    <span>margin {fmtMargin(item.margin)}</span>
                    <span>shanten {item.shanten ?? '-'}</span>
                  </div>
                  <div style={{ marginTop: 8, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {item.slice_tags.map((tag) => (
                      <span key={tag} style={tagStyle}>{tag}</span>
                    ))}
                  </div>
                </div>
                <button
                  onClick={() => openCase(item)}
                  className="btn-primary"
                  style={openButtonStyle}
                  disabled={openingCaseId === item.case_id}
                >
                  {openingCaseId === item.case_id ? <Loader2 size={15} className="animate-spin" /> : <MonitorPlay size={15} />}
                  牌桌回放
                </button>
              </div>
            </div>
          ))}
        </div>
      )}
    </PageShell>
  );
}

const actionButtonStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  height: 36,
  padding: '0 14px',
};

const kindButtonStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  gap: 8,
  minHeight: 36,
  padding: '0 12px',
  borderRadius: 8,
  border: '1px solid var(--border)',
  cursor: 'pointer',
};

const caseCardStyle: CSSProperties = {
  padding: 14,
};

const pillStyle: CSSProperties = {
  padding: '3px 8px',
  borderRadius: 999,
  border: '1px solid var(--border)',
  color: 'var(--text-secondary)',
  fontSize: 12,
};

const successPillStyle: CSSProperties = {
  ...pillStyle,
  color: 'var(--success)',
  borderColor: 'rgba(34, 197, 94, 0.35)',
};

const dangerPillStyle: CSSProperties = {
  ...pillStyle,
  color: 'var(--error)',
  borderColor: 'rgba(239, 68, 68, 0.35)',
};

const tagStyle: CSSProperties = {
  ...pillStyle,
  fontSize: 11,
  padding: '2px 7px',
};

const openButtonStyle: CSSProperties = {
  display: 'inline-flex',
  alignItems: 'center',
  justifyContent: 'center',
  gap: 6,
  height: 38,
  minWidth: 112,
};
