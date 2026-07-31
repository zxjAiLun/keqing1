// src/replay_ui/src/pages/LadderModelPage.tsx
// 模型详情：账号横向对比 + 可选联赛聚合。
import { useEffect, useState, type CSSProperties } from 'react';
import { useLocation, useNavigate, useParams } from 'react-router-dom';
import { ladderApi } from '../api/ladderApi';
import { PageHeader, PageShell, SectionTitle } from '../components/Layout/PageScaffold';
import { routes, withLadderSeason } from '../routes';
import type { LadderAccountRow, LadderModelDetail, LadderSeason } from '../types/ladder';
import { fmtPt, fmtRank, fmtRate, fmtRating } from '../utils/ladderFormat';

function leagueNum(entry: Record<string, unknown>, key: string): number | null {
  const value = entry[key];
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}

function leagueRankCounts(entry: Record<string, unknown>): number[] | null {
  const value = entry['rank_counts'];
  return Array.isArray(value) && value.length === 4 ? value.map((item) => Number(item)) : null;
}

export function LadderModelPage() {
  const { modelId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  const [seasons, setSeasons] = useState<LadderSeason[]>([]);
  const [detail, setDetail] = useState<LadderModelDetail | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [seasonsLoaded, setSeasonsLoaded] = useState(false);

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

  useEffect(() => {
    if (!activeSeasonId || !modelId) return;
    let cancelled = false;
    const load = async () => {
      setLoading(true);
      setError(null);
      try {
        const payload = await ladderApi.getModel(activeSeasonId, modelId);
        if (!cancelled) {
          setDetail(payload);
          setLoading(false);
        }
      } catch (reason) {
        if (!cancelled) {
          setError(reason instanceof Error ? reason.message : String(reason));
          setDetail(null);
          setLoading(false);
        }
      }
    };
    void load();
    return () => { cancelled = true; };
  }, [activeSeasonId, modelId]);

  const model = detail?.model;
  const backToLadder = () => navigate(withLadderSeason(routes.ladder, activeSeasonId));
  const openAccount = (row: LadderAccountRow) => {
    navigate(withLadderSeason(routes.ladderAccount(row.account_id), activeSeasonId));
  };

  return (
    <PageShell width={1180}>
      <PageHeader
        eyebrow="Model Profile"
        title={model?.model_id ?? modelId ?? '模型详情'}
        description={detail ? `${detail.season.title || detail.season.season_id} · ${model?.checkpoint || '未登记 checkpoint'}` : undefined}
        actions={(
          <button type="button" onClick={backToLadder} className="btn-secondary" style={actionButtonStyle}>
            返回天梯
          </button>
        )}
      />

      {error && <div role="alert" style={{ color: 'var(--error)', fontSize: 13, marginBottom: 10 }}>{error}</div>}
      {((!seasonsLoaded && !error) || (loading && activeSeasonId && modelId)) && (
        <div style={{ color: 'var(--text-muted)', fontSize: 13, padding: 20, textAlign: 'center' }}>加载中...</div>
      )}

      {seasonsLoaded && !error && !detail && !(loading && activeSeasonId && modelId) && (
        <div className="card" style={{ padding: 16, color: 'var(--text-muted)', fontSize: 13 }}>
          未找到模型数据。请从天梯榜的模型卡片进入，或确认 ?season= 参数与模型 ID。
        </div>
      )}

      {!loading && detail && model && (
        <>
          {/* 模型汇总（展示性聚合，不另算 Rating） */}
          {model.summary && (
            <section className="card" style={{ padding: 14, marginBottom: 12 }}>
              <div style={{ display: 'flex', gap: 22, flexWrap: 'wrap', fontSize: 12 }}>
                {[
                  ['账号数', String(model.summary.accounts)],
                  ['总场数', String(model.summary.games)],
                  ['平均 PT', fmtPt(model.summary.avg_pt)],
                  ['平均 Rating', fmtRating(model.summary.avg_rating)],
                  ['平均顺位', fmtRank(model.summary.avg_rank)],
                  ['平均顺位列PT', model.summary.avg_rank_pt === null ? '—' : model.summary.avg_rank_pt.toFixed(2)],
                ].map(([label, value]) => (
                  <div key={label}>
                    <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>{label}</div>
                    <div style={{ color: 'var(--text-primary)', fontWeight: 800, fontSize: 16, fontFamily: 'Menlo, Consolas, monospace' }}>{value}</div>
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* 账号横向对比 */}
          <section className="card" style={{ padding: 12, marginBottom: 12 }}>
            <SectionTitle title="账号对比" description="点击账号查看详情。" />
            <div style={{ overflowX: 'auto' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 12 }}>
                <thead>
                  <tr style={{ color: 'var(--text-muted)', borderBottom: '1px solid var(--border)', textAlign: 'left' }}>
                    <th style={thStyle}>账号</th>
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
                  {model.accounts.map((row) => (
                    <tr
                      key={row.account_id}
                      onClick={() => openAccount(row)}
                      style={{ borderBottom: '1px solid var(--border)', cursor: 'pointer' }}
                      title={`查看 ${row.display_name} 账号详情`}
                    >
                      <td style={{ ...tdStyle, fontWeight: 800, color: 'var(--accent)' }}>{row.display_name}</td>
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
          </section>

          {/* 联赛聚合（可选） */}
          {detail.league_summary && (
            <section className="card" style={{ padding: 12 }}>
              <SectionTitle
                title="联赛聚合"
                description={`${detail.league_summary.schema || ''} · 共 ${detail.league_summary.games_total ?? '—'} 场 · lineups: ${(detail.league_summary.lineups ?? []).join(', ') || '—'}`}
              />
              <div style={{ display: 'flex', gap: 20, flexWrap: 'wrap', fontSize: 12 }}>
                {leagueRankCounts(detail.league_summary.model) && (
                  <div>
                    <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>顺位分布</div>
                    <div style={{ color: 'var(--text-primary)', fontWeight: 700, fontFamily: 'Menlo, Consolas, monospace' }}>
                      {leagueRankCounts(detail.league_summary.model)?.join(' / ')}
                    </div>
                  </div>
                )}
                {[
                  ['场数', leagueNum(detail.league_summary.model, 'games')],
                  ['局数', leagueNum(detail.league_summary.model, 'rounds')],
                  ['平均顺位', leagueNum(detail.league_summary.model, 'avg_rank')],
                  ['平均顺位列PT', leagueNum(detail.league_summary.model, 'avg_rank_pt')],
                ].map(([label, value]) => (
                  <div key={label}>
                    <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>{label}</div>
                    <div style={{ color: 'var(--text-primary)', fontWeight: 700, fontFamily: 'Menlo, Consolas, monospace' }}>
                      {value === null ? '—' : typeof value === 'number' && !Number.isInteger(value) ? value.toFixed(3) : value}
                    </div>
                  </div>
                ))}
                {[
                  ['和率', fmtRate(leagueNum(detail.league_summary.model, 'agari_rate'))],
                  ['放铳率', fmtRate(leagueNum(detail.league_summary.model, 'houjuu_rate'))],
                  ['副露率', fmtRate(leagueNum(detail.league_summary.model, 'fuuro_rate'))],
                  ['立直率', fmtRate(leagueNum(detail.league_summary.model, 'riichi_rate'))],
                  ['流局率', fmtRate(leagueNum(detail.league_summary.model, 'ryukyoku_rate'))],
                  ['被飞率', fmtRate(leagueNum(detail.league_summary.model, 'tobi_rate'))],
                ].map(([label, value]) => (
                  <div key={label}>
                    <div style={{ color: 'var(--text-muted)', fontSize: 11 }}>{label}</div>
                    <div style={{ color: 'var(--text-primary)', fontWeight: 700, fontFamily: 'Menlo, Consolas, monospace' }}>{value}</div>
                  </div>
                ))}
              </div>
            </section>
          )}
        </>
      )}
    </PageShell>
  );
}

const actionButtonStyle: CSSProperties = {
  height: 32,
  display: 'inline-flex',
  alignItems: 'center',
  gap: 6,
  padding: '0 12px',
};

const thStyle: CSSProperties = {
  padding: '6px 8px',
  fontSize: 11,
  fontWeight: 700,
  whiteSpace: 'nowrap',
};

const tdStyle: CSSProperties = {
  padding: '6px 8px',
  color: 'var(--text-primary)',
  whiteSpace: 'nowrap',
};

const numStyle: CSSProperties = {
  textAlign: 'right',
  fontFamily: 'Menlo, Consolas, monospace',
};
