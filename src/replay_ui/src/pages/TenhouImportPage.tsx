// src/replay_ui/src/pages/TenhouImportPage.tsx
// R10-D：天凤链接 → preview → 逐座身份解析 → 确认落账。
import { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { PageHeader, PageShell } from '../components/Layout/PageScaffold';
import { SEAT_WINDS } from '../components/Matches/labels';
import { participantsApi } from '../api/participantsApi';
import { ApiError } from '../api/replayApi';
import { routes } from '../routes';
import type { Account, IntakePreview, SeatResolution, SeatNo } from '../types/participants';

type DraftResolution = {
  seat: SeatNo;
  action: 'assign' | 'create';
  account_id: string;
  alias_id: string;
  display_name: string;
  account_type: 'human' | 'managed_bot' | 'external_bot';
  alias_scope: 'global' | 'session' | 'match' | 'none';
  confidence: 'confirmed' | 'unresolved';
};

const EMPTY_DRAFT: DraftResolution = {
  seat: 0,
  action: 'assign',
  account_id: '',
  alias_id: '',
  display_name: '',
  account_type: 'external_bot',
  alias_scope: 'match',
  confidence: 'confirmed',
};

export function TenhouImportPage() {
  const navigate = useNavigate();
  const [url, setUrl] = useState('');
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [preview, setPreview] = useState<IntakePreview | null>(null);
  const [drafts, setDrafts] = useState<DraftResolution[]>([]);
  const [loading, setLoading] = useState(false);
  const [confirming, setConfirming] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const load = useCallback(async (signal: AbortSignal) => {
    try {
      const accountsResp = await participantsApi.listAccounts(signal);
      setAccounts(accountsResp.accounts);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const runPreview = async () => {
    if (!url.trim()) return;
    setLoading(true);
    setError(null);
    try {
      const result = await participantsApi.intakePreview({ url: url.trim() });
      setPreview(result);
      setDrafts(
        result.seats.map((seat) => {
          const autoCandidate =
            seat.candidates.length === 1 && seat.candidates[0].confidence === 'confirmed'
              ? seat.candidates[0]
              : undefined;
          return {
            ...EMPTY_DRAFT,
            seat: seat.seat,
            account_id: autoCandidate?.account_id ?? '',
            alias_id: autoCandidate?.alias_id ?? '',
            // 消费已有候选别名时不再创建新 alias（提升为 global 需用户显式操作）
            alias_scope: autoCandidate ? 'none' : 'match',
          };
        }),
      );
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  };

  const updateDraft = (seat: SeatNo, patch: Partial<DraftResolution>) => {
    setDrafts((prev) => prev.map((d) => (d.seat === seat ? { ...d, ...patch } : d)));
  };

  const confirmImport = async () => {
    if (!preview) return;
    const resolutions: SeatResolution[] = drafts.map((d) => {
      const base = {
        seat: d.seat,
        action: d.action,
        alias_scope: d.alias_scope,
        confidence: d.confidence,
      };
      if (d.action === 'create') {
        return { ...base, display_name: d.display_name || preview.raw_player_names[d.seat], account_type: d.account_type };
      }
      return { ...base, account_id: d.account_id, alias_id: d.alias_id || undefined };
    });
    if (resolutions.some((r) => r.action === 'assign' && !r.account_id)) {
      setError('仍有座位未指派账号');
      return;
    }
    setConfirming(true);
    setError(null);
    try {
      const resp = await participantsApi.intakeConfirm({ log_id: preview.log_id, resolutions });
      navigate(routes.matchDetail(resp.match.match_id));
    } catch (e) {
      if (e instanceof ApiError && e.status === 409 && e.body && typeof e.body === 'object') {
        const detail = e.body as { error?: string };
        setError(detail.error ?? '重复导入');
      } else {
        setError(e instanceof Error ? e.message : String(e));
      }
    } finally {
      setConfirming(false);
    }
  };

  return (
    <PageShell maxWidth={860}>
      <PageHeader
        title="天凤链接导入"
        description="粘贴天凤牌谱链接 → 预览 → 逐座确认身份 → 写入统一账本（full_replay）"
      />

      <div style={{ display: 'grid', gap: 16, paddingBottom: 24 }}>
        <section style={cardStyle}>
          <div style={{ display: 'flex', gap: 8 }}>
            <input
              value={url}
              onChange={(e) => setUrl(e.target.value)}
              onKeyDown={(e) => e.key === 'Enter' && void runPreview()}
              placeholder="https://tenhou.net/3/?log=20260804gm-xxxx-xxxx&tw=0"
              style={{
                flex: 1, border: '1px solid var(--border)', background: 'var(--page-bg)',
                color: 'var(--text-primary)', borderRadius: 6, padding: '8px 10px', fontSize: 13,
              }}
            />
            <button
              onClick={runPreview}
              disabled={loading || !url.trim()}
              style={{
                border: '1px solid var(--accent)', background: 'var(--accent)', color: '#fff',
                borderRadius: 6, fontSize: 13, fontWeight: 700, padding: '8px 16px', cursor: 'pointer',
              }}
            >
              {loading ? '解析中…' : '解析预览'}
            </button>
          </div>
          {error && <div style={{ color: '#e74c3c', fontSize: 13, marginTop: 8 }}>{error}</div>}
        </section>

        {preview && (
          <>
            {preview.duplicate_match_id && (
              <div style={{ border: '1px solid #e67e22', background: 'rgba(230,126,34,0.08)', borderRadius: 8, padding: 10, fontSize: 13 }}>
                该牌谱已导入为对局 <b>{preview.duplicate_match_id}</b>（防重）。
              </div>
            )}
            <section style={cardStyle}>
              <div style={{ fontWeight: 800, marginBottom: 8 }}>
                {preview.game_length === 'hanchan' ? '半庄' : '东风'} · {preview.occurred_at.slice(0, 10)} · {preview.hand_count} 局 · 完整牌谱
              </div>
              <div style={{ display: 'grid', gap: 6 }}>
                {preview.seats.map((seat) => (
                  <div
                    key={seat.seat}
                    style={{
                      display: 'grid', gridTemplateColumns: '40px 1fr 120px 120px', gap: 6,
                      border: '1px solid var(--border)', borderRadius: 8, padding: '8px 10px', fontSize: 13,
                    }}
                  >
                    <span style={{ fontWeight: 800, color: 'var(--text-muted)' }}>{SEAT_WINDS[seat.seat]}</span>
                    <span style={{ fontWeight: 700 }}>{seat.raw_name}</span>
                    <span style={{ fontVariantNumeric: 'tabular-nums' }}>{preview.final_scores[seat.seat].toLocaleString()}</span>
                    <span>{preview.ranks[seat.seat] + 1}位</span>
                  </div>
                ))}
              </div>
            </section>

            <section style={cardStyle}>
              <div style={{ fontWeight: 800, marginBottom: 10 }}>身份解析</div>
              <div style={{ display: 'grid', gap: 8 }}>
                {drafts.map((draft) => {
                  const seatInfo = preview.seats.find((s) => s.seat === draft.seat)!;
                  return (
                    <div key={draft.seat} style={{ border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px', display: 'grid', gap: 8 }}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                        <span style={{ fontWeight: 800, color: 'var(--text-muted)' }}>{SEAT_WINDS[draft.seat]}</span>
                        <span style={{ fontWeight: 700 }}>{seatInfo.raw_name}</span>
                        {seatInfo.candidates.length > 0 && (
                          <span style={{ fontSize: 11, color: 'var(--text-secondary)' }}>
                            候选：{seatInfo.candidates.map((c) => c.account_id).join(' / ')}
                          </span>
                        )}
                      </div>
                      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
                        <select
                          value={draft.action}
                          onChange={(e) => updateDraft(draft.seat, { action: e.target.value as 'assign' | 'create' })}
                          style={selectStyle}
                        >
                          <option value="assign">指派已有账号</option>
                          <option value="create">新建账号</option>
                        </select>
                        {draft.action === 'assign' ? (
                          <select
                            value={draft.account_id}
                            onChange={(e) => {
                              const accountId = e.target.value;
                              const candidate = seatInfo.candidates.find((c) => c.account_id === accountId);
                              updateDraft(draft.seat, { account_id: accountId, alias_id: candidate?.alias_id ?? '' });
                            }}
                            style={{ ...selectStyle, flex: 1 }}
                          >
                            <option value="">选择账号…</option>
                            {accounts.map((a) => (
                              <option key={a.account_id} value={a.account_id}>{a.display_name}（{a.account_id}）</option>
                            ))}
                          </select>
                        ) : (
                          <>
                            <input
                              value={draft.display_name}
                              onChange={(e) => updateDraft(draft.seat, { display_name: e.target.value })}
                              placeholder={seatInfo.raw_name}
                              style={{ ...selectStyle, flex: 1 }}
                            />
                            <select
                              value={draft.account_type}
                              onChange={(e) => updateDraft(draft.seat, { account_type: e.target.value as DraftResolution['account_type'] })}
                              style={selectStyle}
                            >
                              <option value="human">真人</option>
                              <option value="managed_bot">本地 AI</option>
                              <option value="external_bot">外部 AI</option>
                            </select>
                          </>
                        )}
                        <select
                          value={draft.alias_scope}
                          onChange={(e) => {
                            const scope = e.target.value as DraftResolution['alias_scope'];
                            // 用户显式要新建/提升 alias → 放弃消费已有候选，切换为基于当前账号新建
                            updateDraft(draft.seat, {
                              alias_scope: scope,
                              alias_id: scope !== 'none' ? '' : draft.alias_id,
                            });
                          }}
                          style={selectStyle}
                        >
                          <option value="global">保存为全局别名</option>
                          <option value="session">仅本次会话</option>
                          <option value="match">仅本局</option>
                          <option value="none">不保存别名</option>
                        </select>
                        <select
                          value={draft.confidence}
                          onChange={(e) => updateDraft(draft.seat, { confidence: e.target.value as 'confirmed' | 'unresolved' })}
                          style={selectStyle}
                        >
                          <option value="confirmed">确认</option>
                          <option value="unresolved">暂记 unresolved</option>
                        </select>
                      </div>
                    </div>
                  );
                })}
              </div>
            </section>

            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8 }}>
              <button
                onClick={confirmImport}
                disabled={confirming || Boolean(preview.duplicate_match_id)}
                style={{
                  border: '1px solid var(--accent)', background: 'var(--accent)', color: '#fff',
                  borderRadius: 6, fontSize: 13, fontWeight: 700, padding: '9px 18px', cursor: 'pointer',
                }}
              >
                {confirming ? '导入中…' : '确认导入'}
              </button>
            </div>
          </>
        )}
      </div>
    </PageShell>
  );
}

const cardStyle: React.CSSProperties = {
  background: 'var(--card-bg)',
  border: '1px solid var(--border)',
  borderRadius: 10,
  padding: 14,
};
const selectStyle: React.CSSProperties = {
  border: '1px solid var(--border)', background: 'var(--page-bg)', color: 'var(--text-primary)',
  borderRadius: 4, padding: '5px 8px', fontSize: 13,
};
