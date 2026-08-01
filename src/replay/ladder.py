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
import os
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

def _validate_registry(raw: Any, filename: str) -> dict[str, Any]:
    """校验赛季注册表结构，使其成为账号身份的权威来源。

    任何无效 model/account 都会抛出 SeasonRegistryError，不做静默跳过。
    """
    def _fail(reason: str) -> SeasonRegistryError:
        return SeasonRegistryError(f"赛季注册表 {filename}: {reason}")

    if not isinstance(raw, dict) or raw.get("schema") != SEASON_SCHEMA:
        raise _fail("schema 无效")
    season_id = raw.get("season_id")
    if not isinstance(season_id, str) or not season_id.strip():
        raise _fail("season_id 必须是非空字符串")
    report_dir = raw.get("report_dir")
    if not isinstance(report_dir, str) or not report_dir.strip():
        raise _fail(f"season {season_id}: report_dir 必须是非空字符串")
    models = raw.get("models")
    if not isinstance(models, list):
        raise _fail(f"season {season_id}: models 必须是数组")
    seen_models: set[str] = set()
    seen_accounts: set[str] = set()
    for model_index, model in enumerate(models):
        if not isinstance(model, dict):
            raise _fail(f"season {season_id}: models[{model_index}] 必须是对象")
        model_id = model.get("model_id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise _fail(f"season {season_id}: models[{model_index}] 的 model_id 必须是非空字符串")
        if model_id in seen_models:
            raise _fail(f"season {season_id}: 重复 model_id '{model_id}'")
        seen_models.add(model_id)
        accounts = model.get("accounts")
        if not isinstance(accounts, list):
            raise _fail(f"season {season_id} / model {model_id}: accounts 必须是数组")
        for account_index, account in enumerate(accounts):
            if not isinstance(account, dict):
                raise _fail(f"season {season_id} / model {model_id}: accounts[{account_index}] 必须是对象")
            account_id = account.get("account_id")
            if not isinstance(account_id, str) or not account_id.strip():
                raise _fail(f"season {season_id} / model {model_id}: accounts[{account_index}] 的 account_id 必须是非空字符串")
            if account_id in seen_accounts:
                raise _fail(f"season {season_id}: 重复 account_id '{account_id}'（出现于 model {model_id}）")
            seen_accounts.add(account_id)
    return raw


def _load_registry_file(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SeasonRegistryError(f"赛季注册表无法解析: {path.name}") from exc
    return _validate_registry(raw, path.name)


def read_registry(path: Path) -> dict[str, Any]:
    """读取并校验单个赛季注册表文件（供发布器等生产端复用）。"""
    return _load_registry_file(path)


def list_season_configs(configs_dir: Path) -> list[dict[str, Any]]:
    """Parse every season registry file, sorted by season_id."""
    if not configs_dir.exists():
        return []
    seasons = [_load_registry_file(path) for path in sorted(configs_dir.glob("*.json"))]
    seen_ids: set[str] = set()
    for season in seasons:
        season_id = str(season["season_id"])
        if season_id in seen_ids:
            raise SeasonRegistryError(f"season_id 全局重复: {season_id}")
        seen_ids.add(season_id)
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
# 外部注册表 / 数据根边界（Live Ladder Data Plane）
# ---------------------------------------------------------------------------

def resolve_config_dir(project_root: Path) -> Path:
    """赛季注册表目录。

    默认读取仓库内 ``configs/ladder/seasons``；正式 runtime 通过环境变量
    ``KEQING_LADDER_CONFIG_DIR`` 指向外部动态注册表目录（如 keqing-data）。
    """
    override = os.environ.get("KEQING_LADDER_CONFIG_DIR", "").strip()
    if override:
        return Path(override)
    return project_root / "configs" / "ladder" / "seasons"


def resolve_report_dir(project_root: Path, raw_dir: str) -> Path:
    """解析注册表 report_dir。

    - 绝对路径原样使用（动态赛季通常直接指向 keqing-data 下的快照）；
    - 相对路径默认相对 project_root；设置 ``KEQING_LADDER_DATA_ROOT`` 时相对该数据根。
    """
    path = Path(raw_dir)
    if path.is_absolute():
        return path.resolve()
    data_root = os.environ.get("KEQING_LADDER_DATA_ROOT", "").strip()
    base = Path(data_root) if data_root else project_root
    return (base / raw_dir).resolve()


# ---------------------------------------------------------------------------
# Report artifacts
# ---------------------------------------------------------------------------

def _report_dir(project_root: Path, season: dict[str, Any]) -> Path:
    raw_dir = str(season.get("report_dir") or "")
    if not raw_dir:
        raise SeasonDataError(f"赛季 {season.get('season_id')} 未配置 report_dir")
    report_dir = resolve_report_dir(project_root, raw_dir)
    if not report_dir.exists():
        raise SeasonDataError(f"赛季报告目录不存在: {raw_dir}")
    return report_dir


def _load_account_summary(report_dir: Path) -> dict[str, Any]:
    summary_path = report_dir / "account_summary.json"
    if not summary_path.exists():
        raise SeasonDataError("赛季缺少 account_summary.json，请先运行 build_platform_account_report.py")
    try:
        report = _read_json_cached(summary_path)
    except (OSError, json.JSONDecodeError) as exc:
        raise SeasonDataError(f"account_summary.json 无法解析: {summary_path}") from exc
    if not isinstance(report, dict) or report.get("schema") != REPORT_SCHEMA:
        raise SeasonDataError("account_summary.json schema 无效")
    return report


def _validate_report_accounts(season: dict[str, Any], report: dict[str, Any]) -> list[dict[str, Any]]:
    """校验 report 账号集合与注册表的一致性（注册表为权威身份表）。

    - report 每个账号必须已在注册表声明；
    - report 内不得有重复 account_id；
    - report 的 model_label 非空时必须与注册表 model_id 一致；
    - status == "completed" 时注册表与 report 的账号集合必须完全一致；
    - 未完成赛季允许注册账号尚未出现在 report，但 report 不得出现未注册账号。
    """
    season_id = season.get("season_id")
    rows = report.get("accounts")
    if not isinstance(rows, list):
        raise SeasonDataError(f"season {season_id}: account_summary.json 的 accounts 必须是数组")
    registry_index = _registry_account_index(season)
    seen: set[str] = set()
    validated: list[dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise SeasonDataError(f"season {season_id}: report 账号行必须是对象")
        account_id = row.get("account_id")
        if not isinstance(account_id, str) or not account_id.strip():
            raise SeasonDataError(f"season {season_id}: report 账号行缺少非空 account_id")
        if account_id in seen:
            raise SeasonDataError(f"season {season_id}: report 内重复 account_id '{account_id}'")
        seen.add(account_id)
        registry = registry_index.get(account_id)
        if registry is None:
            raise SeasonDataError(f"season {season_id}: report 账号 '{account_id}' 未在注册表声明")
        model_label = row.get("model_label")
        if isinstance(model_label, str) and model_label.strip() and model_label != registry["model_id"]:
            raise SeasonDataError(
                f"season {season_id}: 账号 '{account_id}' 的 model_label '{model_label}' "
                f"与注册表 model_id '{registry['model_id']}' 不一致"
            )
        validated.append(row)
    if str(season.get("status") or "") == "completed":
        missing = sorted(set(registry_index) - seen)
        if missing:
            raise SeasonDataError(
                f"season {season_id}: completed 赛季注册账号未出现在 report 中: {missing}"
            )
    return validated


def _load_validated_report(project_root: Path, season: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    report = _load_account_summary(_report_dir(project_root, season))
    rows = _validate_report_accounts(season, report)
    return report, rows


SNAPSHOT_REQUIRED_FILES = ("account_summary.json", "account_ledger.jsonl", "rating_curve.csv")


def validate_snapshot(season: dict[str, Any], snapshot_dir: Path) -> list[dict[str, Any]]:
    """校验一个已构建好的快照目录是否满足注册表契约（供发布器复用）。

    除 account_summary.json 外，要求 UI/API 实际消费的 account_ledger.jsonl
    与 rating_curve.csv 存在且可读（零场快照允许内容为空，但文件必须存在）。
    """
    for name in SNAPSHOT_REQUIRED_FILES:
        path = snapshot_dir / name
        if not path.is_file():
            raise SeasonDataError(f"快照缺少必需文件: {name}")
        try:
            with path.open("r", encoding="utf-8") as handle:
                handle.read(1)
        except OSError as exc:
            raise SeasonDataError(f"快照文件不可读: {name}") from exc
    report = _load_account_summary(snapshot_dir)
    return _validate_report_accounts(season, report)


def _summary_mtime(report_dir: Path) -> float | None:
    summary_path = report_dir / "account_summary.json"
    try:
        return summary_path.stat().st_mtime
    except OSError:
        return None


def _attach_snapshot_meta(season_pub: dict[str, Any], report_dir: Path) -> dict[str, Any]:
    season_pub["snapshot_id"] = report_dir.name
    season_pub["updated_at"] = _summary_mtime(report_dir)
    return season_pub


def _enrich_account_row(row: dict[str, Any], registry_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    account_id = str(row.get("account_id", ""))
    # 注册表为权威身份表；账号存在性已由 _validate_report_accounts 保证
    registry = registry_index[account_id]
    games = int(row.get("games") or 0)
    rank_1 = int(row.get("rank_1") or 0)
    rank_4 = int(row.get("rank_4") or 0)
    pt_current = float(row.get("pt_current") or 0.0)
    pt_target = float(row.get("pt_target") or 0.0)
    return {
        "account_id": account_id,
        "display_name": registry.get("display_name") or account_id,
        "model_id": registry["model_id"],
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
            report, _rows = _load_validated_report(project_root, season)
        except SeasonDataError:
            # 报告缺失或与注册表不一致：该赛季标记为未就绪，不影响整个清单
            report = None
        entry["data_ready"] = report is not None
        if report is not None:
            entry["games"] = report.get("games")
            _attach_snapshot_meta(entry, _report_dir(project_root, season))
        seasons.append(entry)
    return seasons


def load_ladder(project_root: Path, configs_dir: Path, season_id: str, sort: str = "pt") -> dict[str, Any]:
    if sort not in LADDER_SORTS:
        sort = "pt"
    season = get_season_config(configs_dir, season_id)
    report, validated_rows = _load_validated_report(project_root, season)
    registry_index = _registry_account_index(season)
    rows = [_enrich_account_row(row, registry_index) for row in validated_rows]
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
        "season": _attach_snapshot_meta(_season_public(season, report), _report_dir(project_root, season)),
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
    if max_points <= 0:
        return []
    if max_points == 1:
        return points[-1:] if points else []
    if len(points) > max_points:
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
    report, validated_rows = _load_validated_report(project_root, season)
    registry_index = _registry_account_index(season)
    rows = [_enrich_account_row(row, registry_index) for row in validated_rows]
    match = next((row for row in rows if row["account_id"] == account_id), None)
    if match is None:
        raise AccountNotFoundError(f"account 不存在: {account_id}")
    return {
        "season": _attach_snapshot_meta(_season_public(season, report), report_dir),
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
    report, validated_rows = _load_validated_report(project_root, season)
    registry_index = _registry_account_index(season)
    rows = [_enrich_account_row(row, registry_index) for row in validated_rows]
    model_rows = [row for row in rows if row["model_id"] == model_id]
    if not model_rows:
        raise ModelNotFoundError(f"model 不存在: {model_id}")
    model_rows.sort(key=lambda row: row["pt_current"], reverse=True)
    summary = next((item for item in _summarize_models(rows) if item["model_id"] == model_id), None)
    registry_model = next((m for m in _registry_models(season) if m.get("model_id") == model_id), {})
    return {
        "season": _attach_snapshot_meta(_season_public(season, report), report_dir),
        "model": {
            "model_id": model_id,
            "checkpoint": registry_model.get("checkpoint"),
            "accounts": model_rows,
            "summary": summary,
        },
        "league_summary": _load_league_summary(project_root, season, model_id),
    }



