import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { useNavigate } from 'react-router-dom';
import { BookOpen, GitCompare, Loader2, MonitorPlay, RefreshCw } from 'lucide-react';
import { replayApi } from '../api/replayApi';
import { PageHeader, PageShell, SectionTitle } from '../components/Layout/PageScaffold';
import type { BehaviorReplayCase, PairedBehaviorReplayCase } from '../types/replay';

const SINGLE_KIND_LABELS: Record<string, string> = {
  '80k_dealer_call_bad': '80k 亲家副露漂移',
  '80k_rank2_call_bad': '80k 二位副露漂移',
  '70k_dealer_call_good': '70k 亲家副露参照',
  '70k_rank2_call_good': '70k 二位副露参照',
};

const PAIRED_KIND_LABELS: Record<string, string> = {
  paired_dealer_call_divergence: '亲家 CALL 首分叉',
  paired_rank2_call_divergence: '二位 CALL 首分叉',
  paired_call_divergence: 'PASS/CALL 首分叉',
  paired_discard_divergence: '打牌首分叉',
  paired_reach_divergence: '立直首分叉',
  paired_first_divergence: '其他首分叉',
};

function fmtTime(ts: number | null | undefined): string {
  if (!ts) return '-';
  return new Date(ts * 1000).toLocaleString();
}

function fmtMargin(value: number): string {
  return Number.isFinite(value) ? value.toFixed(3) : '-';
}

function shortAction(action: { type?: string; pai?: string; actor?: number; target?: number } | null | undefined): string {
  if (!action) return '-';
  const parts = [action.type ?? 'unknown'];
  if (action.pai) parts.push(action.pai);
  if (typeof action.actor === 'number') parts.push(`a${action.actor}`);
  if (typeof action.target === 'number') parts.push(`t${action.target}`);
  return parts.join(' ');
}

function outcomePair(item: PairedBehaviorReplayCase): string {
  const left = item.downstream_summary?.left_kyoku_outcome ?? '-';
  const right = item.downstream_summary?.right_kyoku_outcome ?? '-';
  return `${left} / ${right}`;
}

export function BehaviorCasebookPage() {
  const navigate = useNavigate();
  const [cases, setCases] = useState<BehaviorReplayCase[]>([]);
  const [pairedCases, setPairedCases] = useState<PairedBehaviorReplayCase[]>([]);
  const [casebookPath, setCasebookPath] = useState('');
  const [pairedCasebookPath, setPairedCasebookPath] = useState('');
  const [updatedAt, setUpdatedAt] = useState<number | null>(null);
  const [pairedUpdatedAt, setPairedUpdatedAt] = useState<number | null>(null);
  const [selectedSingleKind, setSelectedSingleKind] = useState<string>('80k_dealer_call_bad');
  const [selectedPairedKind, setSelectedPairedKind] = useState<string>('paired_dealer_call_divergence');
  const [loading, setLoading] = useState(true);
  const [openingCaseId, setOpeningCaseId] = useState<string | null>(null);
  const [focusNotice, setFocusNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const load = () => {
    setLoading(true);
    setError(null);
    replayApi.listBehaviorCasebook()
      .then((payload) => {
        setCases(payload.cases ?? []);
        setPairedCases(payload.paired_cases ?? []);
        setCasebookPath(payload.casebook_dir);
        setPairedCasebookPath(payload.paired_casebook_dir ?? '');
        setUpdatedAt(payload.updated_at);
        setPairedUpdatedAt(payload.paired_updated_at ?? null);
        if (payload.cases?.length && !payload.cases.some((item) => item.case_kind === selectedSingleKind)) {
          setSelectedSingleKind(payload.cases[0].case_kind);
        }
        if (payload.paired_cases?.length && !payload.paired_cases.some((item) => item.case_kind === selectedPairedKind)) {
          setSelectedPairedKind(payload.paired_cases[0].case_kind);
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

  const singleKinds = useMemo(() => countKinds(cases), [cases]);
  const pairedKinds = useMemo(() => countKinds(pairedCases), [pairedCases]);

  const visibleSingleCases = useMemo(
    () => cases.filter((item) => item.case_kind === selectedSingleKind),
    [cases, selectedSingleKind],
  );

  const visiblePairedCases = useMemo(
    () => pairedCases.filter((item) => item.case_kind === selectedPairedKind),
    [pairedCases, selectedPairedKind],
  );

  const navigateToReplay = (result: Awaited<ReturnType<typeof replayApi.importBehaviorCase>>) => {
    if (result.focus_resolution !== 'exact') {
      setFocusNotice(result.focus_resolution === 'nearest'
        ? '该案例未精确命中原始事件，已跳到最近的回放决策步。'
        : '该案例没有可定位的回放决策步。');
    }
    navigate(
      `/game-replay?id=${encodeURIComponent(result.replay_id)}`
      + `&player_id=${result.player_id}`
      + `&focus_event_index=${result.focus_event_index}`
      + (result.focus_replay_step !== null ? `&focus_step=${result.focus_replay_step}` : '')
      + `&focus_resolution=${result.focus_resolution}`,
    );
  };

  const openCase = async (item: BehaviorReplayCase) => {
    setOpeningCaseId(item.case_id);
    setError(null);
    try {
      navigateToReplay(await replayApi.importBehaviorCase(item.case_id));
    } catch (e) {
      setError(String(e));
    } finally {
      setOpeningCaseId(null);
    }
  };

  const openPairedCase = async (item: PairedBehaviorReplayCase, side: 'left' | 'right') => {
    const openKey = `${item.case_id}:${side}`;
    setOpeningCaseId(openKey);
    setError(null);
    try {
      navigateToReplay(await replayApi.importPairedBehaviorCase(item.case_id, side));
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
        description="按同 seed / 同座位前缀对齐 70k 与 80k，优先查看首次分叉及 PASS/CALL 漂移。"
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

      {focusNotice && (
        <div className="card" style={{ marginBottom: 16, color: 'var(--text-secondary)', fontSize: 14 }}>
          {focusNotice}
        </div>
      )}

      <div className="card" style={{ marginBottom: 16 }}>
        <SectionTitle
          title="Paired Divergence"
          description={`路径：${pairedCasebookPath || '-'} · 更新时间：${fmtTime(pairedUpdatedAt)}`}
        />
        {pairedCases.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontSize: 14 }}>
            未找到 paired manifest。先运行 `scripts/mortal/export_paired_behavior_divergence_cases.py` 生成配对案例。
          </div>
        ) : (
          <KindButtons
            kinds={pairedKinds}
            selected={selectedPairedKind}
            labels={PAIRED_KIND_LABELS}
            onSelect={setSelectedPairedKind}
          />
        )}
      </div>

      {visiblePairedCases.length > 0 && (
        <div style={{ display: 'grid', gap: 10, marginBottom: 18 }}>
          {visiblePairedCases.map((item) => (
            <div key={item.case_id} className="card" style={caseCardStyle}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
                <div style={{ minWidth: 0, flex: '1 1 620px' }}>
                  <div style={cardTitleRowStyle}>
                    <GitCompare size={16} style={{ color: 'var(--accent)' }} />
                    <span style={caseIdStyle}>{item.case_id}</span>
                    <span style={pillStyle}>{PAIRED_KIND_LABELS[item.case_kind] ?? item.case_kind}</span>
                    <span style={pillStyle}>prefix {item.prefix_match_event_count}</span>
                    <span style={item.downstream_summary?.right_kyoku_outcome === 'houjuu' ? dangerPillStyle : pillStyle}>
                      outcome {outcomePair(item)}
                    </span>
                  </div>
                  <div style={metaRowStyle}>
                    <span>actor {item.actor}</span>
                    <span>turn {item.turn ?? '-'}</span>
                    <span>rank {item.start_rank ?? '-'}</span>
                    <span>{item.divergence_kind}</span>
                    <span>{item.left_model}: {shortAction(item.left_action)}</span>
                    <span>{item.right_model}: {shortAction(item.right_action)}</span>
                  </div>
                  <div style={{ marginTop: 8, display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                    {item.slice_tags.map((tag) => (
                      <span key={tag} style={tagStyle}>{tag}</span>
                    ))}
                  </div>
                </div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', alignItems: 'center' }}>
                  <button
                    onClick={() => openPairedCase(item, 'left')}
                    className="btn-secondary"
                    style={openButtonStyle}
                    disabled={openingCaseId === `${item.case_id}:left`}
                  >
                    {openingCaseId === `${item.case_id}:left` ? <Loader2 size={15} className="animate-spin" /> : <MonitorPlay size={15} />}
                    打开 {item.left_model}
                  </button>
                  <button
                    onClick={() => openPairedCase(item, 'right')}
                    className="btn-primary"
                    style={openButtonStyle}
                    disabled={openingCaseId === `${item.case_id}:right`}
                  >
                    {openingCaseId === `${item.case_id}:right` ? <Loader2 size={15} className="animate-spin" /> : <MonitorPlay size={15} />}
                    打开 {item.right_model}
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}

      <div className="card" style={{ marginBottom: 16 }}>
        <SectionTitle
          title="Single-Case Reference"
          description={`路径：${casebookPath || '-'} · 更新时间：${fmtTime(updatedAt)} · 旧版单边样例仅作漂移/参照查看`}
        />
        {cases.length === 0 ? (
          <div style={{ color: 'var(--text-muted)', fontSize: 14 }}>
            未找到 single-case manifest。先运行 `scripts/mortal/export_behavior_replay_cases.py` 生成案例。
          </div>
        ) : (
          <KindButtons
            kinds={singleKinds}
            selected={selectedSingleKind}
            labels={SINGLE_KIND_LABELS}
            onSelect={setSelectedSingleKind}
          />
        )}
      </div>

      {visibleSingleCases.length > 0 && (
        <div style={{ display: 'grid', gap: 10 }}>
          {visibleSingleCases.map((item) => (
            <div key={item.case_id} className="card" style={caseCardStyle}>
              <div style={{ display: 'flex', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' }}>
                <div style={{ minWidth: 0 }}>
                  <div style={cardTitleRowStyle}>
                    <BookOpen size={16} style={{ color: 'var(--accent)' }} />
                    <span style={caseIdStyle}>{item.case_id}</span>
                    <span style={pillStyle}>{item.model_label}</span>
                    <span style={pillStyle}>{item.action_type}</span>
                    <span style={item.houjuu ? dangerPillStyle : item.agari ? successPillStyle : pillStyle}>{item.outcome}</span>
                  </div>
                  <div style={metaRowStyle}>
                    <span>actor {item.actor}</span>
                    <span>turn {item.turn}</span>
                    <span>focus {item.focus_event_index}</span>
                    <span>margin {fmtMargin(item.margin)}</span>
                    <span>shanten {item.shanten ?? '-'}</span>
                    {item.checkpoint_path && <span>checkpoint {item.checkpoint_path.split('/').pop()}</span>}
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

function countKinds<T extends { case_kind: string }>(items: T[]): Array<[string, number]> {
  const counts = new Map<string, number>();
  for (const item of items) counts.set(item.case_kind, (counts.get(item.case_kind) ?? 0) + 1);
  return Array.from(counts.entries()).sort(([a], [b]) => a.localeCompare(b));
}

function KindButtons({
  kinds,
  selected,
  labels,
  onSelect,
}: {
  kinds: Array<[string, number]>;
  selected: string;
  labels: Record<string, string>;
  onSelect: (kind: string) => void;
}) {
  return (
    <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
      {kinds.map(([kind, count]) => {
        const active = kind === selected;
        return (
          <button
            key={kind}
            onClick={() => onSelect(kind)}
            style={{
              ...kindButtonStyle,
              borderColor: active ? 'var(--accent)' : 'var(--border)',
              background: active ? 'rgba(52, 152, 219, 0.12)' : 'var(--card-bg)',
              color: active ? 'var(--accent)' : 'var(--text-primary)',
            }}
          >
            {labels[kind] ?? kind}
            <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>{count}</span>
          </button>
        );
      })}
    </div>
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

const cardTitleRowStyle: CSSProperties = {
  display: 'flex',
  alignItems: 'center',
  gap: 8,
  flexWrap: 'wrap',
  marginBottom: 6,
};

const caseIdStyle: CSSProperties = {
  fontWeight: 800,
  color: 'var(--text-primary)',
  wordBreak: 'break-all',
};

const metaRowStyle: CSSProperties = {
  fontSize: 12,
  color: 'var(--text-muted)',
  display: 'flex',
  gap: 10,
  flexWrap: 'wrap',
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
