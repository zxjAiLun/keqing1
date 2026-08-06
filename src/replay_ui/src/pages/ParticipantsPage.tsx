// src/replay_ui/src/pages/ParticipantsPage.tsx
// R10：参赛者管理 —— 账号 + 模型身份/产物。
import { useCallback, useEffect, useState } from 'react';
import { PageHeader, PageShell } from '../components/Layout/PageScaffold';
import { AccountTable } from '../components/Participants/AccountTable';
import { AccountFormModal } from '../components/Participants/AccountFormModal';
import { ModelFormModal } from '../components/Participants/ModelFormModal';
import { participantsApi } from '../api/participantsApi';
import type { Account, AccountCreate, ModelIdentity } from '../types/participants';

export function ParticipantsPage() {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [identities, setIdentities] = useState<ModelIdentity[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [editing, setEditing] = useState<Account | null>(null);
  const [creating, setCreating] = useState(false);
  const [creatingModel, setCreatingModel] = useState(false);

  const load = useCallback(async (signal: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const [accountsResp, modelsResp] = await Promise.all([
        participantsApi.listAccounts(signal),
        participantsApi.listModels(signal),
      ]);
      setAccounts(accountsResp.accounts);
      setIdentities(modelsResp.identities);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    void load(controller.signal);
    return () => controller.abort();
  }, [load]);

  const handleSaveAccount = async (payload: AccountCreate) => {
    if (editing) {
      const updated = await participantsApi.updateAccount(editing.account_id, {
        display_name: payload.display_name,
        default_controller: payload.default_controller ?? undefined,
        note: payload.note ?? null,
      });
      setAccounts((prev) => prev.map((a) => (a.account_id === updated.account_id ? updated : a)));
    } else {
      const created = await participantsApi.createAccount(payload);
      setAccounts((prev) => [...prev, created]);
    }
  };

  const handleToggle = async (account: Account) => {
    const updated = await participantsApi.updateAccount(account.account_id, { enabled: !account.enabled });
    setAccounts((prev) => prev.map((a) => (a.account_id === updated.account_id ? updated : a)));
  };

  const handleDelete = async (account: Account) => {
    if (!window.confirm(`删除/停用 ${account.display_name}？被对局引用时将软停用。`)) return;
    await participantsApi.deleteAccount(account.account_id);
    await load(new AbortController().signal);
  };

  return (
    <PageShell>
      <PageHeader title="参赛者" description="真人 / 本地 AI / 外部 AI / 仅登记 —— 任意四账号可组成一桌" />
      <div style={{ display: 'grid', gap: 16 }}>
        <section style={cardStyle}>
          <div style={cardHeaderStyle}>
            <span style={{ fontWeight: 800 }}>账号（{accounts.length}）</span>
            <div style={{ display: 'flex', gap: 8 }}>
              <button style={primaryBtn} onClick={() => setCreating(true)}>新建参赛者</button>
              <button style={ghostBtn} onClick={() => setCreatingModel(true)}>新建模型身份</button>
            </div>
          </div>
          {loading ? (
            <div style={{ padding: 24, color: 'var(--text-muted)' }}>加载中…</div>
          ) : error ? (
            <div style={{ padding: 24, color: '#e74c3c' }}>{error}</div>
          ) : (
            <AccountTable
              accounts={accounts}
              identities={identities}
              onEdit={setEditing}
              onToggleEnabled={handleToggle}
              onDelete={handleDelete}
            />
          )}
        </section>

        <section style={cardStyle}>
          <div style={cardHeaderStyle}>
            <span style={{ fontWeight: 800 }}>模型身份（{identities.length}）</span>
          </div>
          <div style={{ display: 'grid', gap: 8 }}>
            {identities.map((identity) => (
              <div key={identity.model_identity_id} style={modelRowStyle}>
                <div>
                  <div style={{ fontWeight: 700 }}>{identity.label}</div>
                  <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                    {identity.model_identity_id} · {identity.kind}
                    {identity.account_id ? ` · ${identity.account_id}` : ' · 全局'}
                  </div>
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                  {identity.artifacts.length > 0
                    ? identity.artifacts.map((a) => `${a.label}${a.is_current ? '（当前）' : ''}`).join(' / ')
                    : '无产物（外部代理或仅身份）'}
                </div>
              </div>
            ))}
            {identities.length === 0 && (
              <div style={{ padding: 16, color: 'var(--text-muted)', textAlign: 'center' }}>暂无模型身份</div>
            )}
          </div>
        </section>
      </div>

      {(creating || editing) && (
        <AccountFormModal
          account={editing}
          onClose={() => { setCreating(false); setEditing(null); }}
          onSave={handleSaveAccount}
        />
      )}
      {creatingModel && (
        <ModelFormModal
          onClose={() => setCreatingModel(false)}
          onSave={async (payload) => {
            const created = await participantsApi.createModel(payload);
            setIdentities((prev) => [...prev, created]);
          }}
        />
      )}
    </PageShell>
  );
}

const cardStyle: React.CSSProperties = {
  background: 'var(--card-bg)',
  border: '1px solid var(--border)',
  borderRadius: 10,
  padding: 14,
};
const cardHeaderStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 10,
};
const primaryBtn: React.CSSProperties = {
  border: '1px solid var(--accent)', background: 'var(--accent)', color: '#fff',
  borderRadius: 5, fontSize: 12, fontWeight: 700, padding: '6px 12px', cursor: 'pointer',
};
const ghostBtn: React.CSSProperties = {
  border: '1px solid var(--border)', background: 'var(--page-bg)', color: 'var(--text-primary)',
  borderRadius: 5, fontSize: 12, padding: '6px 12px', cursor: 'pointer',
};
const modelRowStyle: React.CSSProperties = {
  display: 'flex', alignItems: 'center', justifyContent: 'space-between', gap: 12,
  border: '1px solid var(--border)', borderRadius: 8, padding: '10px 12px',
};
