// src/replay_ui/src/pages/LadderPage.tsx
// 天梯榜：赛季账号排名 + 模型展示性聚合。
import { useEffect, useMemo, useState, type CSSProperties } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { ladderApi } from '../api/ladderApi';
import { PageHeader, PageShell } from '../components/Layout/PageScaffold';
import { useVisibleLiveQuery } from '../hooks/useVisibleLiveQuery';
import { routes, withLadderSeason } from '../routes';
import type { LadderAccountRow, LadderModelSummary, LadderResponse, LadderSeason } from '../types/ladder';
import { fmtPt, fmtRank, fmtRate, fmtRating } from '../utils/ladderFormat';

const SORT_OPTIONS = [
  { value: 'pt', label: '按 PT' },
  { value: 'rating', label: '按 Rating' },
  { value: 'avg_rank', label: '按平均顺位' },
  { value: 'games', label: '按场数' },
];

function fmtUpdatedAt(epochSecs: number | undefined): string {
  if (!epochSecs) return '—';
  return new Date(epochSecs * 1000).toLocaleString();
}

export function LadderPage() {
  const location = useLocation();
  const navigate = useNavigate();
  const [seasons, setSeasons] = useState<LadderSeason[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [seasonsLoaded, setSeasonsLoaded] = useState(false);
  const [sort, setSort] = useState('pt');

  const seasonFromQuery = new URLSearchParams(location.search).get('season');
  const activeSeasonId = seasonFromQuery ?? seasons[0]?.season_id ?? null;

  useEffect(() => {
    ladderApi.listSeasons()
      .then((payload) => {
        setSeasons(payload.seasons);
        setSeasonsLoaded(true);
      })
      .catch((reason) => {
        setError(reason instanceof Error ? reason.message : String(reason));
        setSeasonsLoaded(true);
      });
  }, []);

  // 共享可见性实时查询：页面可见时每 30 秒静默刷新，hidden 跳过，
  // 恢复可见立即刷新，轮询失败保留旧数据，queryKey 变化清空旧实体重新加载。
  const ladderQuery = useVisibleLiveQuery<LadderResponse>({
    enabled: Boolean(activeSeasonId),
    queryKey: `ladder:${activeSeasonId ?? ''}:${sort}`,
    load: useMemo(
      () => (signal: AbortSignal) => {
        if (!activeSeasonId) return Promise.reject(new Error('no active season'));
        return ladderApi.getLadder(activeSeasonId, sort, signal);
      },
      [activeSeasonId, sort],
    ),
  });
  const ladder = ladderQuery.data;
  const loading = ladderQuery.loading;

  const switchSeason = (seasonId: string) => {
    navigate(`${routes.ladder}?season=${encodeURIComponent(seasonId)}`);
  };

  const openAccount = (row: LadderAccountRow) => {
    navigate(withLadderSeason(routes.ladderAccount(row.account_id), activeSeasonId));
  };

  const openModel = (model: LadderModelSummary) => {
    navigate(withLadderSeason(routes.ladderModel(model.model_id), activeSeasonId));
  };

  return (
    <PageShell width={1240}>
      <PageHeader
        eyebrow="Model Ladder"
        title="天梯榜"
        description={ladder?.season.notes || ladder?.season.title || '赛季账号 PT / Rating 排名。'}
        actions={(
          <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
            {seasons.length > 1 && (
              <select
                value={activeSeasonId ?? ''}
                onChange={(event) => switchSeason(event.target.value)}
                style={selectStyle}
              >
                {seasons.map((season) => (
                  <option key={season.season_id} value={season.season_id}>
                    {season.title || season.season_id}
                  </option>
                ))}
              </select>
            )}
            <div style={{ display: 'flex', gap: 4 }}>
              {SORT_OPTIONS.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => setSort(option.value)}
                  style={sortButtonStyle(sort === option.value)}
                >
                  {option.label}
                </button>
              ))}
            </div>
          </div>
        )}
      />

      {error && <div role="alert" style={{ color: 'var(--error)', fontSize: 13, marginBottom: 10 }}>{error}</div>}
      {(error === null && !seasonsLoaded && !ladderQuery.error) || (loading && activeSeasonId) ? (
        <div style={{ color: 'var(--text-muted)', fontSize: 13, padding: 20, textAlign: 'center' }}>加载中...</div>
      ) : null}

      {seasonsLoaded && !activeSeasonId && !error && !ladderQuery.error && (
        <div className="card" style={{ padding: 16, color: 'var(--text-muted)', fontSize: 13 }}>
          暂无已注册赛季。往 <code>configs/ladder/seasons/</code> 添加赛季注册表并生成 platform account 报告后，这里会出现天梯数据。
        </div>
      )}

      {!loading && ladder && (
        <>
          {/* 模型展示性聚合（正式排名仍以账号为主体） */}
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))', gap: 10, marginBottom: 12 }}>
            {ladder.models.map((model) => (
              <button
                key={model.model_id}
                type="button"
                onClick={() => openModel(model)}
                style={modelCardStyle}
              >
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8 }}>
                  <span style={{ fontSize: 13, fontWeight: 800, color: 'var(--text-primary)' }}>{model.model_id}</span>
                  <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{model.accounts} 账号 · {model.games} 场</span>
                </div>
                <div style={{ display: 'flex', gap: 12, marginTop: 6, fontSize: 12, color: 'var(--text-secondary)' }}>
                  <span>均PT <b style={modelValueStyle}>{fmtPt(model.avg_pt)}</b></span>
                  <span>均R <b style={modelValueStyle}>{fmtRating(model.avg_rating)}</b></span>
                  <span>均顺位 <b style={modelValueStyle}>{fmtRank(model.avg_rank)}</b></span>
                </div>
              </button>
            ))}
          </div>

          {/* 账号天梯 */}
          <div style={{ border: '1px solid var(--border)', borderRadius: 7, overflowX: 'auto', background: 'var(--card-bg)' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12, minWidth: 1060 }}>
              <thead>
                <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
                  <th style={thStyle}>排名</th>
                  <th style={thStyle}>账号</th>
                  <th style={thStyle}>模型</th>
                  <th style={{ ...thStyle, textAlign: 'right' }}>PT</th>
                  <th style={{ ...thStyle, textAlign: 'right' }}>距目标</th>
                  <th style={{ ...thStyle, textAlign: 'right' }}>Rating</th>
                  <th style={{ ...thStyle, textAlign: 'right' }}>场数</th>
                  <th style={{ ...thStyle, textAlign: 'right' }}>平均顺位</th>
                  <th style={{ ...thStyle, textAlign: 'right' }}>一位率</th>
                  <th style={{ ...thStyle, textAlign: 'right' }}>四位率</th>
                  <th style={{ ...thStyle, textAlign: 'right' }}>和率</th>
                  <th style={{ ...thStyle, textAlign: 'right' }}>放铳率</th>
                  <th style={{ ...thStyle, textAlign: 'right' }}>副露率</th>
                  <th style={{ ...thStyle, textAlign: 'right' }}>立直率</th>
                </tr>
              </thead>
              <tbody>
                {ladder.accounts.map((row) => (
                  <tr
                    key={row.account_id}
                    onClick={() => openAccount(row)}
                    style={{ borderBottom: '1px solid var(--border)', cursor: 'pointer' }}
                    title={`查看 ${row.display_name} 账号详情`}
                  >
                    <td style={{ ...tdStyle, fontWeight: 800, color: 'var(--text-muted)' }}>{row.rank_position}</td>
                    <td style={{ ...tdStyle, fontWeight: 800, color: 'var(--accent)' }}>{row.display_name}</td>
                    <td style={{ ...tdStyle, color: 'var(--text-secondary)' }}>{row.model_id}</td>
                    <td style={{ ...tdStyle, ...numStyle, fontWeight: 800 }}>{fmtPt(row.pt_current)}</td>
                    <td style={{ ...tdStyle, ...numStyle, color: row.pt_gap > 0 ? 'var(--text-muted)' : 'var(--success)' }}>
                      {row.pt_gap > 0 ? `-${fmtPt(row.pt_gap)}` : '达标'}
                    </td>
                    <td style={{ ...tdStyle, ...numStyle }}>{fmtRating(row.rating)}</td>
                    <td style={{ ...tdStyle, ...numStyle }}>{row.games}</td>
                    <td style={{ ...tdStyle, ...numStyle }}>{fmtRank(row.avg_rank)}</td>
                    <td style={{ ...tdStyle, ...numStyle }}>{fmtRate(row.rank_1_rate)}</td>
                    <td style={{ ...tdStyle, ...numStyle }}>{fmtRate(row.rank_4_rate)}</td>
                    <td style={{ ...tdStyle, ...numStyle }}>{fmtRate(row.agari_rate)}</td>
                    <td style={{ ...tdStyle, ...numStyle }}>{fmtRate(row.houjuu_rate)}</td>
                    <td style={{ ...tdStyle, ...numStyle }}>{fmtRate(row.fuuro_rate)}</td>
                    <td style={{ ...tdStyle, ...numStyle }}>{fmtRate(row.riichi_rate)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {ladder.season.scoring && (
            <div style={{ marginTop: 8, fontSize: 11, color: 'var(--text-muted)' }}>
              {ladder.season.scoring.pt_profile} · PT {ladder.season.scoring.pt_rank_deltas?.join('/')} · 初始 {ladder.season.scoring.pt_initial} → 目标 {ladder.season.scoring.pt_target} · {ladder.season.scoring.rank_name}
            </div>
          )}

          {/* 快照状态：数据更新时间 / snapshot ID / 已计入场数 */}
          <div style={{ marginTop: 6, fontSize: 11, color: 'var(--text-muted)' }}>
            快照 {ladder.season.snapshot_id || '—'} · 更新 {fmtUpdatedAt(ladder.season.updated_at)} · 已计入 {ladder.season.games ?? '—'} 场
            <span style={{ marginLeft: 8 }}>（页面可见时每 30 秒自动刷新）</span>
          </div>
        </>
      )}
    </PageShell>
  );
}

const selectStyle: CSSProperties = {
  height: 30,
  border: '1px solid var(--border)',
  borderRadius: 6,
  background: 'var(--card-bg)',
  color: 'var(--text-primary)',
  fontSize: 12,
  padding: '0 8px',
};

const sortButtonStyle = (active: boolean): CSSProperties => ({
  height: 30,
  padding: '0 10px',
  borderRadius: 6,
  border: `1px solid ${active ? 'var(--accent)' : 'var(--border)'}`,
  background: active ? 'rgba(52, 152, 219, 0.12)' : 'var(--card-bg)',
  color: active ? 'var(--accent)' : 'var(--text-secondary)',
  fontSize: 12,
  fontWeight: active ? 800 : 600,
  cursor: 'pointer',
});

const modelCardStyle: CSSProperties = {
  border: '1px solid var(--border)',
  borderRadius: 7,
  background: 'var(--card-bg)',
  padding: '10px 12px',
  textAlign: 'left',
  cursor: 'pointer',
};

const modelValueStyle: CSSProperties = { color: 'var(--text-primary)' };

const thStyle: CSSProperties = {
  padding: '7px 8px',
  fontSize: 11,
  fontWeight: 700,
  whiteSpace: 'nowrap',
};

const tdStyle: CSSProperties = {
  padding: '7px 8px',
  color: 'var(--text-primary)',
  whiteSpace: 'nowrap',
};

const numStyle: CSSProperties = {
  textAlign: 'right',
  fontFamily: 'Menlo, Consolas, monospace',
};
