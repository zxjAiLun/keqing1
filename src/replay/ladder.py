"""Model ladder & account profile data loaders.

Reads versioned season registries from ``configs/ladder/seasons/*.json`` and
the platform account reports produced by
``scripts/mortal/build_platform_account_report.py``.

The ladder never reads raw replay logs: it only consumes the aggregated
``account_summary.json`` / ``rating_curve.csv`` / ``account_ledger.jsonl``
artifacts, so the GUI never loads thousands of hanchans at once.
"""

from __future__ import annotations

import csv
import json
import threading
from pathlib import Path
from typing import Any

SEASON_SCHEMA = "keqing.ladder.season.v1"
REPORT_SCHEMA = "keqing.mortal.platform_account_report.v1"

LADDER_SORTS = ("pt", "rating", "avg_rank", "games")


class LadderError(Exception):
    """Base class for ladder data failures."""


class SeasonRegistryError(LadderError):
    """Season registry file is missing or invalid."""


class SeasonNotFoundError(LadderError):
    """Requested season does not exist."""


class SeasonDataError(LadderError):
    """Season report artifacts are missing or invalid."""


class AccountNotFoundError(LadderError):
    """Requested account does not exist in the season."""


class ModelNotFoundError(LadderError):
    """Requested model does not exist in the season."""


_json_cache: dict[str, tuple[float, Any]] = {}
_cache_lock = threading.Lock()


def _read_json_cached(path: Path) -> Any:
    key = str(path)
    mtime = path.stat().st_mtime
    with _cache_lock:
        cached = _json_cache.get(key)
        if cached and cached[0] == mtime:
            return cached[1]
    data = json.loads(path.read_text(encoding="utf-8"))
    with _cache_lock:
        _json_cache[key] = (mtime, data)
    return data


# ---------------------------------------------------------------------------
# Season registry
# ---------------------------------------------------------------------------

def _load_registry_file(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SeasonRegistryError(f"赛季注册表无法解析: {path.name}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != SEASON_SCHEMA:
        raise SeasonRegistryError(f"赛季注册表 schema 无效: {path.name}")
    return raw


def list_season_configs(configs_dir: Path) -> list[dict[str, Any]]:
    """Parse every season registry file, sorted by season_id."""
    if not configs_dir.exists():
        return []
    seasons = [_load_registry_file(path) for path in sorted(configs_dir.glob("*.json"))]
    seasons.sort(key=lambda item: str(item.get("season_id", "")))
    return seasons


def get_season_config(configs_dir: Path, season_id: str) -> dict[str, Any]:
    for season in list_season_configs(configs_dir):
        if season.get("season_id") == season_id:
            return season
    raise SeasonNotFoundError(f"season 不存在: {season_id}")


def _registry_models(season: dict[str, Any]) -> list[dict[str, Any]]:
    models = season.get("models")
    if not isinstance(models, list):
        return []
    return [model for model in models if isinstance(model, dict)]


def _registry_account_index(season: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """account_id -> {display_name, model_id, checkpoint}"""
    index: dict[str, dict[str, Any]] = {}
    for model in _registry_models(season):
        model_id = str(model.get("model_id", ""))
        checkpoint = model.get("checkpoint")
        accounts = model.get("accounts")
        if not isinstance(accounts, list):
            continue
        for account in accounts:
            if not isinstance(account, dict):
                continue
            account_id = str(account.get("account_id", ""))
            if not account_id:
                continue
            index[account_id] = {
                "display_name": str(account.get("display_name") or account_id),
                "model_id": model_id,
                "checkpoint": checkpoint,
            }
    return index


# ---------------------------------------------------------------------------
# Report artifacts
# ---------------------------------------------------------------------------

def _report_dir(project_root: Path, season: dict[str, Any]) -> Path:
    raw_dir = str(season.get("report_dir") or "")
    if not raw_dir:
        raise SeasonDataError(f"赛季 {season.get('season_id')} 未配置 report_dir")
    report_dir = (project_root / raw_dir).resolve()
    if not report_dir.exists():
        raise SeasonDataError(f"赛季报告目录不存在: {raw_dir}")
    return report_dir


def _load_account_summary(report_dir: Path) -> dict[str, Any]:
    summary_path = report_dir / "account_summary.json"
    if not summary_path.exists():
        raise SeasonDataError("赛季缺少 account_summary.json，请先运行 build_platform_account_report.py")
    report = _read_json_cached(summary_path)
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise SeasonDataError("account_summary.json schema 无效")
    return report


def _summary_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows = report.get("accounts")
    if not isinstance(rows, list):
        return []
    return [row for row in rows if isinstance(row, dict)]


def _enrich_account_row(row: dict[str, Any], registry_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    account_id = str(row.get("account_id", ""))
    registry = registry_index.get(account_id, {})
    games = int(row.get("games") or 0)
    rank_1 = int(row.get("rank_1") or 0)
    rank_4 = int(row.get("rank_4") or 0)
    pt_current = float(row.get("pt_current") or 0.0)
    pt_target = float(row.get("pt_target") or 0.0)
    return {
        "account_id": account_id,
        "display_name": registry.get("display_name", account_id),
        "model_id": registry.get("model_id") or str(row.get("model_label", "")),
        "checkpoint": registry.get("checkpoint"),
        "games": games,
        "rank_name": row.get("rank_name"),
        "pt_current": pt_current,
        "pt_target": pt_target,
        "pt_gap": pt_target - pt_current,
        "rating": float(row.get("rating") or 0.0),
        "rank_1": rank_1,
        "rank_2": int(row.get("rank_2") or 0),
        "rank_3": int(row.get("rank_3") or 0),
        "rank_4": rank_4,
        "rank_1_rate": (rank_1 / games) if games else None,
        "rank_4_rate": (rank_4 / games) if games else None,
        "avg_rank": row.get("avg_rank"),
        "avg_rank_pt": row.get("avg_rank_pt"),
        "agari_rate": row.get("agari_rate"),
        "houjuu_rate": row.get("houjuu_rate"),
        "fuuro_rate": row.get("fuuro_rate"),
        "riichi_rate": row.get("riichi_rate"),
        "agari_rate_after_fuuro": row.get("agari_rate_after_fuuro"),
        "houjuu_rate_after_fuuro": row.get("houjuu_rate_after_fuuro"),
        "agari_rate_after_riichi": row.get("agari_rate_after_riichi"),
        "houjuu_rate_after_riichi": row.get("houjuu_rate_after_riichi"),
        "avg_point_per_agari": row.get("avg_point_per_agari"),
        "total_delta_score": row.get("total_delta_score"),
    }


def _summarize_models(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """展示性模型聚合：对账号表现按场数加权，不另算一套 Rating。"""
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(str(row["model_id"]), []).append(row)
    summaries: list[dict[str, Any]] = []
    for model_id, items in by_model.items():
        games = sum(int(item["games"]) for item in items)

        def _weighted(key: str) -> float | None:
            values = [
                (float(item[key]), int(item["games"]))
                for item in items
                if item.get(key) is not None and int(item["games"]) > 0
            ]
            total_games = sum(count for _, count in values)
            if not values or total_games == 0:
                return None
            return sum(value * count for value, count in values) / total_games

        summaries.append({
            "model_id": model_id,
            "accounts": len(items),
            "games": games,
            "avg_pt": _weighted("pt_current"),
            "avg_rating": _weighted("rating"),
            "avg_rank": _weighted("avg_rank"),
            "avg_rank_pt": _weighted("avg_rank_pt"),
        })
    summaries.sort(key=lambda item: (item["avg_pt"] is None, -(item["avg_pt"] or 0.0)))
    return summaries


# ---------------------------------------------------------------------------
# Public loaders
# ---------------------------------------------------------------------------

def _season_public(season: dict[str, Any], report: dict[str, Any] | None = None) -> dict[str, Any]:
    registry_models = _registry_models(season)
    payload: dict[str, Any] = {
        "season_id": season.get("season_id"),
        "title": season.get("title"),
        "status": season.get("status"),
        "games_expected": season.get("games_expected"),
        "notes": season.get("notes"),
        "models": [model.get("model_id") for model in registry_models],
        "accounts": sum(len(model.get("accounts") or []) for model in registry_models),
    }
    if report is not None:
        payload["games"] = report.get("games")
        payload["report_schema"] = report.get("schema")
        scoring = report.get("scoring")
        if isinstance(scoring, dict):
            payload["scoring"] = {
                "pt_profile": scoring.get("pt_profile"),
                "pt_rank_deltas": scoring.get("pt_rank_deltas"),
                "pt_initial": scoring.get("pt_initial"),
                "pt_target": scoring.get("pt_target"),
                "rating_initial": scoring.get("rating_initial"),
                "rating_formula": scoring.get("rating_formula"),
                "rank_name": scoring.get("rank_name"),
            }
    return payload


def list_seasons(project_root: Path, configs_dir: Path) -> list[dict[str, Any]]:
    seasons: list[dict[str, Any]] = []
    for season in list_season_configs(configs_dir):
        entry = _season_public(season)
        try:
            report = _load_account_summary(_report_dir(project_root, season))
        except SeasonDataError:
            report = None
        entry["data_ready"] = report is not None
        if report is not None:
            entry["games"] = report.get("games")
        seasons.append(entry)
    return seasons


def load_ladder(project_root: Path, configs_dir: Path, season_id: str, sort: str = "pt") -> dict[str, Any]:
    if sort not in LADDER_SORTS:
        sort = "pt"
    season = get_season_config(configs_dir, season_id)
    report = _load_account_summary(_report_dir(project_root, season))
    registry_index = _registry_account_index(season)
    rows = [_enrich_account_row(row, registry_index) for row in _summary_rows(report)]
    if sort == "pt":
        rows.sort(key=lambda row: row["pt_current"], reverse=True)
    elif sort == "rating":
        rows.sort(key=lambda row: row["rating"], reverse=True)
    elif sort == "avg_rank":
        rows.sort(key=lambda row: (row["avg_rank"] is None, row["avg_rank"] or 0.0))
    else:  # games
        rows.sort(key=lambda row: row["games"], reverse=True)
    for index, row in enumerate(rows, 1):
        row["rank_position"] = index
    return {
        "season": _season_public(season, report),
        "sort": sort,
        "accounts": rows,
        "models": _summarize_models(rows),
    }


def _read_rating_curve(report_dir: Path, account_id: str, max_points: int) -> list[dict[str, Any]]:
    curve_path = report_dir / "rating_curve.csv"
    if not curve_path.exists():
        return []
    points: list[dict[str, Any]] = []
    with curve_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("account_id") != account_id:
                continue
            try:
                points.append({
                    "games": int(float(row.get("games") or 0)),
                    "rating": float(row.get("rating") or 0.0),
                    "pt": float(row.get("pt") or 0.0),
                })
            except (TypeError, ValueError):
                continue
    if max_points > 0 and len(points) > max_points:
        stride = (len(points) - 1) / (max_points - 1)
        sampled = [points[round(i * stride)] for i in range(max_points - 1)]
        sampled.append(points[-1])
        seen: set[int] = set()
        deduped: list[dict[str, Any]] = []
        for point in sampled:
            marker = int(point["games"])
            if marker in seen:
                continue
            seen.add(marker)
            deduped.append(point)
        points = deduped
    return points


def _read_recent_games(report_dir: Path, account_id: str, limit: int) -> list[dict[str, Any]]:
    ledger_path = report_dir / "account_ledger.jsonl"
    if not ledger_path.exists() or limit <= 0:
        return []
    games: list[dict[str, Any]] = []
    with ledger_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("account_id") != account_id:
                continue
            games.append({
                "game_index": row.get("game_index"),
                "rank": row.get("rank"),
                "final_score": row.get("final_score"),
                "score_delta": row.get("score_delta"),
                "pt_delta": row.get("pt_delta"),
                "pt_after": row.get("pt_after"),
                "rating_after": row.get("rating_after"),
                "source_log": row.get("source_log"),
            })
    return list(reversed(games[-limit:]))


def load_account(
    project_root: Path,
    configs_dir: Path,
    season_id: str,
    account_id: str,
    *,
    recent_limit: int = 50,
    curve_max_points: int = 240,
) -> dict[str, Any]:
    season = get_season_config(configs_dir, season_id)
    report_dir = _report_dir(project_root, season)
    report = _load_account_summary(report_dir)
    registry_index = _registry_account_index(season)
    rows = [_enrich_account_row(row, registry_index) for row in _summary_rows(report)]
    match = next((row for row in rows if row["account_id"] == account_id), None)
    if match is None:
        raise AccountNotFoundError(f"account 不存在: {account_id}")
    return {
        "season": _season_public(season, report),
        "account": match,
        "rank_distribution": [match["rank_1"], match["rank_2"], match["rank_3"], match["rank_4"]],
        "curve": _read_rating_curve(report_dir, account_id, curve_max_points),
        "recent_games": _read_recent_games(report_dir, account_id, recent_limit),
    }


def _load_league_summary(project_root: Path, season: dict[str, Any], model_id: str) -> dict[str, Any] | None:
    raw_path = season.get("league_summary")
    if not raw_path:
        return None
    summary_path = (project_root / str(raw_path)).resolve()
    if not summary_path.exists():
        return None
    try:
        payload = _read_json_cached(summary_path)
    except (OSError, json.JSONDecodeError):
        return None
    models = payload.get("models") if isinstance(payload, dict) else None
    if not isinstance(models, dict):
        return None
    entry = models.get(model_id)
    if not isinstance(entry, dict):
        return None
    return {
        "schema": payload.get("schema"),
        "games_total": payload.get("games_total"),
        "lineups": payload.get("lineups"),
        "model": entry,
    }


def load_model(project_root: Path, configs_dir: Path, season_id: str, model_id: str) -> dict[str, Any]:
    season = get_season_config(configs_dir, season_id)
    report_dir = _report_dir(project_root, season)
    report = _load_account_summary(report_dir)
    registry_index = _registry_account_index(season)
    rows = [_enrich_account_row(row, registry_index) for row in _summary_rows(report)]
    model_rows = [row for row in rows if row["model_id"] == model_id]
    if not model_rows:
        raise ModelNotFoundError(f"model 不存在: {model_id}")
    model_rows.sort(key=lambda row: row["pt_current"], reverse=True)
    summary = next((item for item in _summarize_models(rows) if item["model_id"] == model_id), None)
    registry_model = next((m for m in _registry_models(season) if m.get("model_id") == model_id), {})
    return {
        "season": _season_public(season, report),
        "model": {
            "model_id": model_id,
            "checkpoint": registry_model.get("checkpoint"),
            "accounts": model_rows,
            "summary": summary,
        },
        "league_summary": _load_league_summary(project_root, season, model_id),
    }



