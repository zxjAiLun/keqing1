from __future__ import annotations

import json
from pathlib import Path

import pytest

from replay import ladder


def _account_row(account_id: str, model_label: str, *, games: int, pt: float, rating: float, ranks: list[int], avg_rank: float) -> dict:
    return {
        "account_id": account_id,
        "model_label": model_label,
        "games": games,
        "rank_name": "七段",
        "pt_current": pt,
        "pt_target": 2800.0,
        "rating": rating,
        "rank_1": ranks[0],
        "rank_2": ranks[1],
        "rank_3": ranks[2],
        "rank_4": ranks[3],
        "avg_rank": avg_rank,
        "avg_rank_pt": 0.5,
        "agari_rate": 0.21,
        "houjuu_rate": 0.12,
        "fuuro_rate": 0.28,
        "riichi_rate": 0.19,
        "agari_rate_after_fuuro": 0.25,
        "houjuu_rate_after_fuuro": 0.10,
        "agari_rate_after_riichi": 0.30,
        "houjuu_rate_after_riichi": 0.08,
        "avg_point_per_agari": 6500.0,
        "total_delta_score": 1200,
    }


@pytest.fixture()
def season_env(tmp_path: Path) -> dict:
    configs_dir = tmp_path / "configs" / "ladder" / "seasons"
    configs_dir.mkdir(parents=True)
    report_dir = tmp_path / "artifacts" / "season_report"
    report_dir.mkdir(parents=True)

    season = {
        "schema": ladder.SEASON_SCHEMA,
        "season_id": "test-season",
        "title": "Test Season",
        "status": "completed",
        "report_dir": "artifacts/season_report",
        "league_summary": "artifacts/season_report/league_summary.json",
        "games_expected": 4,
        "models": [
            {"model_id": "model_a", "checkpoint": "artifacts/a.pth", "accounts": [
                {"account_id": "model_a@01", "display_name": "A-01"},
                {"account_id": "model_a@02", "display_name": "A-02"},
            ]},
            {"model_id": "model_b", "checkpoint": "artifacts/b.pth", "accounts": [
                {"account_id": "model_b@01", "display_name": "B-01"},
            ]},
        ],
        "notes": "fixture season",
    }
    (configs_dir / "test-season.json").write_text(json.dumps(season), encoding="utf-8")

    report = {
        "schema": ladder.REPORT_SCHEMA,
        "games": 4,
        "scoring": {
            "pt_profile": "houou_7dan_hanchan",
            "pt_rank_deltas": [90, 45, 0, -135],
            "pt_initial": 1400.0,
            "pt_target": 2800.0,
            "rating_initial": 1500.0,
            "rating_formula": "test formula",
            "rank_name": "七段",
        },
        "accounts": [
            _account_row("model_a@01", "model_a", games=3, pt=1600.0, rating=1520.0, ranks=[2, 0, 1, 0], avg_rank=1.667),
            _account_row("model_a@02", "model_a", games=1, pt=1300.0, rating=1470.0, ranks=[0, 0, 0, 1], avg_rank=4.0),
            _account_row("model_b@01", "model_b", games=4, pt=1800.0, rating=1560.0, ranks=[2, 2, 0, 0], avg_rank=1.5),
        ],
    }
    (report_dir / "account_summary.json").write_text(json.dumps(report), encoding="utf-8")

    curve_lines = ["game_index,account_id,model_label,rating,pt,rank_name,games"]
    for i in range(10):
        curve_lines.append(f"{i},model_a@01,model_a,{1500.0 + i},{1400.0 + i * 10},七段,{i + 1}")
    (report_dir / "rating_curve.csv").write_text("\n".join(curve_lines) + "\n", encoding="utf-8")

    ledger_rows = []
    for i in range(5):
        ledger_rows.append({"game_index": i, "account_id": "model_a@01", "rank": (i % 4) + 1,
                            "final_score": 25000 + i * 100, "score_delta": i * 100,
                            "pt_delta": 45.0, "pt_after": 1400.0 + (i + 1) * 45,
                            "rating_after": 1500.0 + i, "source_log": f"/logs/{i}.json.gz"})
        ledger_rows.append({"game_index": i, "account_id": "model_b@01", "rank": ((i + 1) % 4) + 1,
                            "final_score": 24000, "score_delta": -100,
                            "pt_delta": 0.0, "pt_after": 1400.0,
                            "rating_after": 1501.0, "source_log": f"/logs/{i}.json.gz"})
    (report_dir / "account_ledger.jsonl").write_text(
        "\n".join(json.dumps(row) for row in ledger_rows) + "\n", encoding="utf-8")

    league_summary = {
        "schema": "keqing.mortal.model_pool_league.v1",
        "games_total": 4,
        "lineups": ["L1"],
        "models": {
            "model_a": {"games": 4, "avg_rank": 2.1, "agari_rate": 0.2},
            "model_b": {"games": 4, "avg_rank": 1.9, "agari_rate": 0.22},
        },
    }
    (report_dir / "league_summary.json").write_text(json.dumps(league_summary), encoding="utf-8")

    return {"root": tmp_path, "configs": configs_dir, "report_dir": report_dir}


def test_list_seasons_marks_data_ready(season_env):
    seasons = ladder.list_seasons(season_env["root"], season_env["configs"])
    assert len(seasons) == 1
    entry = seasons[0]
    assert entry["season_id"] == "test-season"
    assert entry["title"] == "Test Season"
    assert entry["data_ready"] is True
    assert entry["games"] == 4
    assert entry["accounts"] == 3
    assert entry["models"] == ["model_a", "model_b"]


def test_ladder_pt_sort_and_rank_positions(season_env):
    payload = ladder.load_ladder(season_env["root"], season_env["configs"], "test-season", sort="pt")
    rows = payload["accounts"]
    assert [row["account_id"] for row in rows] == ["model_b@01", "model_a@01", "model_a@02"]
    assert [row["rank_position"] for row in rows] == [1, 2, 3]
    first = rows[0]
    assert first["display_name"] == "B-01"
    assert first["checkpoint"] == "artifacts/b.pth"
    assert first["pt_gap"] == pytest.approx(1000.0)
    assert first["rank_1_rate"] == pytest.approx(0.5)
    assert payload["season"]["scoring"]["pt_profile"] == "houou_7dan_hanchan"


def test_ladder_avg_rank_and_rating_sorts(season_env):
    by_rank = ladder.load_ladder(season_env["root"], season_env["configs"], "test-season", sort="avg_rank")
    assert [row["account_id"] for row in by_rank["accounts"]] == ["model_b@01", "model_a@01", "model_a@02"]
    by_rating = ladder.load_ladder(season_env["root"], season_env["configs"], "test-season", sort="rating")
    assert [row["account_id"] for row in by_rating["accounts"]] == ["model_b@01", "model_a@01", "model_a@02"]
    by_games = ladder.load_ladder(season_env["root"], season_env["configs"], "test-season", sort="games")
    assert by_games["accounts"][0]["account_id"] == "model_b@01"


def test_model_summary_is_games_weighted(season_env):
    payload = ladder.load_ladder(season_env["root"], season_env["configs"], "test-season")
    summaries = {item["model_id"]: item for item in payload["models"]}
    assert [item["model_id"] for item in payload["models"]] == ["model_b", "model_a"]
    model_a = summaries["model_a"]
    assert model_a["accounts"] == 2
    assert model_a["games"] == 4
    # (1600*3 + 1300*1) / 4 = 1525
    assert model_a["avg_pt"] == pytest.approx(1525.0)
    # (1520*3 + 1470*1) / 4 = 1507.5
    assert model_a["avg_rating"] == pytest.approx(1507.5)


def test_account_detail_curve_sampling_and_recent_games(season_env):
    payload = ladder.load_account(
        season_env["root"], season_env["configs"], "test-season", "model_a@01",
        recent_limit=2, curve_max_points=4,
    )
    account = payload["account"]
    assert account["display_name"] == "A-01"
    assert account["checkpoint"] == "artifacts/a.pth"
    assert account["pt_gap"] == pytest.approx(1200.0)
    assert payload["rank_distribution"] == [2, 0, 1, 0]
    curve = payload["curve"]
    assert len(curve) == 4
    assert curve[0]["games"] == 1
    assert curve[-1]["games"] == 10
    recent = payload["recent_games"]
    assert [game["game_index"] for game in recent] == [4, 3]
    assert recent[0]["pt_after"] == pytest.approx(1400.0 + 5 * 45)
    assert recent[0]["source_log"] == "/logs/4.json.gz"


def test_account_not_found(season_env):
    with pytest.raises(ladder.AccountNotFoundError):
        ladder.load_account(season_env["root"], season_env["configs"], "test-season", "ghost@01")


def test_model_detail_includes_accounts_and_league_summary(season_env):
    payload = ladder.load_model(season_env["root"], season_env["configs"], "test-season", "model_a")
    model = payload["model"]
    assert model["checkpoint"] == "artifacts/a.pth"
    assert [row["account_id"] for row in model["accounts"]] == ["model_a@01", "model_a@02"]
    assert model["summary"]["avg_pt"] == pytest.approx(1525.0)
    league = payload["league_summary"]
    assert league is not None
    assert league["schema"] == "keqing.mortal.model_pool_league.v1"
    assert league["model"]["avg_rank"] == 2.1


def test_model_not_found(season_env):
    with pytest.raises(ladder.ModelNotFoundError):
        ladder.load_model(season_env["root"], season_env["configs"], "test-season", "ghost")


def test_season_not_found(season_env):
    with pytest.raises(ladder.SeasonNotFoundError):
        ladder.load_ladder(season_env["root"], season_env["configs"], "ghost-season")


def test_invalid_registry_schema(tmp_path: Path):
    configs_dir = tmp_path / "configs" / "ladder" / "seasons"
    configs_dir.mkdir(parents=True)
    (configs_dir / "bad.json").write_text(json.dumps({"schema": "wrong"}), encoding="utf-8")
    with pytest.raises(ladder.SeasonRegistryError):
        ladder.list_seasons(tmp_path, configs_dir)


def test_missing_report_dir_raises_data_error(tmp_path: Path):
    configs_dir = tmp_path / "configs" / "ladder" / "seasons"
    configs_dir.mkdir(parents=True)
    season = {
        "schema": ladder.SEASON_SCHEMA,
        "season_id": "empty-season",
        "report_dir": "artifacts/missing",
        "models": [],
    }
    (configs_dir / "empty-season.json").write_text(json.dumps(season), encoding="utf-8")
    with pytest.raises(ladder.SeasonDataError):
        ladder.load_ladder(tmp_path, configs_dir, "empty-season")
    seasons = ladder.list_seasons(tmp_path, configs_dir)
    assert seasons[0]["data_ready"] is False


# ---------------------------------------------------------------------------
# 注册表权威性与 report 一致性
# ---------------------------------------------------------------------------

def _season(status: str = "completed", **overrides) -> dict:
    base = {
        "schema": ladder.SEASON_SCHEMA,
        "season_id": "s1",
        "report_dir": "artifacts/report",
        "status": status,
        "models": [
            {"model_id": "m1", "checkpoint": "artifacts/m1.pth", "accounts": [
                {"account_id": "m1@01"}, {"account_id": "m1@02"},
            ]},
            {"model_id": "m2", "checkpoint": "artifacts/m2.pth", "accounts": [
                {"account_id": "m2@01"},
            ]},
        ],
    }
    base.update(overrides)
    return base


def _write_custom_env(
    tmp_path: Path,
    *,
    season: dict,
    report_accounts: list[dict] | None,
    curve_count: int = 0,
) -> dict:
    configs_dir = tmp_path / "configs" / "ladder" / "seasons"
    configs_dir.mkdir(parents=True)
    (configs_dir / f"{season.get('season_id', 'season')}.json").write_text(json.dumps(season), encoding="utf-8")
    report_dir = tmp_path / str(season.get("report_dir", "artifacts/report"))
    report_dir.mkdir(parents=True, exist_ok=True)
    if report_accounts is not None:
        report = {"schema": ladder.REPORT_SCHEMA, "games": len(report_accounts), "accounts": report_accounts}
        (report_dir / "account_summary.json").write_text(json.dumps(report), encoding="utf-8")
    if curve_count and report_accounts:
        account_id = report_accounts[0]["account_id"]
        lines = ["game_index,account_id,model_label,rating,pt,rank_name,games"]
        for i in range(curve_count):
            lines.append(f"{i},{account_id},m1,{1500.0 + i},{1400.0 + i * 10},七段,{i + 1}")
        (report_dir / "rating_curve.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"root": tmp_path, "configs": configs_dir, "report_dir": report_dir}


def _three_account_rows() -> list[dict]:
    return [
        _account_row("m1@01", "m1", games=2, pt=1500.0, rating=1501.0, ranks=[1, 1, 0, 0], avg_rank=1.5),
        _account_row("m1@02", "m1", games=2, pt=1400.0, rating=1499.0, ranks=[1, 0, 1, 0], avg_rank=2.0),
        _account_row("m2@01", "m2", games=2, pt=1300.0, rating=1498.0, ranks=[0, 1, 1, 0], avg_rank=2.5),
    ]


def test_registry_duplicate_model_id(tmp_path: Path):
    season = _season(models=[
        {"model_id": "m1", "accounts": [{"account_id": "m1@01"}]},
        {"model_id": "m1", "accounts": [{"account_id": "m1@02"}]},
    ])
    env = _write_custom_env(tmp_path, season=season, report_accounts=None)
    with pytest.raises(ladder.SeasonRegistryError, match="model_id"):
        ladder.load_ladder(env["root"], env["configs"], "s1")


def test_registry_duplicate_account_id_across_models(tmp_path: Path):
    season = _season(models=[
        {"model_id": "m1", "accounts": [{"account_id": "dup@01"}]},
        {"model_id": "m2", "accounts": [{"account_id": "dup@01"}]},
    ])
    env = _write_custom_env(tmp_path, season=season, report_accounts=None)
    with pytest.raises(ladder.SeasonRegistryError, match="account_id"):
        ladder.load_ladder(env["root"], env["configs"], "s1")


def test_registry_empty_season_id(tmp_path: Path):
    env = _write_custom_env(tmp_path, season=_season(season_id=""), report_accounts=None)
    with pytest.raises(ladder.SeasonRegistryError, match="season_id"):
        ladder.list_seasons(env["root"], env["configs"])


def test_registry_global_duplicate_season_id(tmp_path: Path):
    configs_dir = tmp_path / "configs" / "ladder" / "seasons"
    configs_dir.mkdir(parents=True)
    for name in ("a.json", "b.json"):
        (configs_dir / name).write_text(json.dumps(_season()), encoding="utf-8")
    with pytest.raises(ladder.SeasonRegistryError, match="全局重复"):
        ladder.list_seasons(tmp_path, configs_dir)


def test_report_unregistered_account_rejected(tmp_path: Path):
    rows = _three_account_rows() + [_account_row("ghost@01", "m1", games=1, pt=1.0, rating=1500.0, ranks=[0, 0, 0, 1], avg_rank=4.0)]
    env = _write_custom_env(tmp_path, season=_season(), report_accounts=rows)
    with pytest.raises(ladder.SeasonDataError, match="未在注册表声明"):
        ladder.load_ladder(env["root"], env["configs"], "s1")


def test_completed_season_missing_registry_account_in_report(tmp_path: Path):
    env = _write_custom_env(tmp_path, season=_season(), report_accounts=_three_account_rows()[:2])
    with pytest.raises(ladder.SeasonDataError, match="未出现在 report"):
        ladder.load_ladder(env["root"], env["configs"], "s1")


def test_report_model_label_conflicts_registry(tmp_path: Path):
    rows = _three_account_rows()
    rows[0]["model_label"] = "wrong_model"
    env = _write_custom_env(tmp_path, season=_season(), report_accounts=rows)
    with pytest.raises(ladder.SeasonDataError, match="model_label"):
        ladder.load_ladder(env["root"], env["configs"], "s1")


def test_report_duplicate_account_id_rejected(tmp_path: Path):
    rows = _three_account_rows() + [_account_row("m1@01", "m1", games=1, pt=1.0, rating=1500.0, ranks=[0, 0, 0, 1], avg_rank=4.0)]
    env = _write_custom_env(tmp_path, season=_season(), report_accounts=rows)
    with pytest.raises(ladder.SeasonDataError, match="重复 account_id"):
        ladder.load_ladder(env["root"], env["configs"], "s1")


def test_corrupted_account_summary_raises_data_error(tmp_path: Path):
    env = _write_custom_env(tmp_path, season=_season(), report_accounts=None)
    (env["report_dir"] / "account_summary.json").write_text("{ not valid json", encoding="utf-8")
    with pytest.raises(ladder.SeasonDataError, match="无法解析"):
        ladder.load_ladder(env["root"], env["configs"], "s1")


def test_running_season_allows_registry_accounts_missing_from_report(tmp_path: Path):
    env = _write_custom_env(
        tmp_path,
        season=_season(status="running"),
        report_accounts=_three_account_rows()[:1],
    )
    payload = ladder.load_ladder(env["root"], env["configs"], "s1")
    assert [row["account_id"] for row in payload["accounts"]] == ["m1@01"]


def test_running_season_rejects_unregistered_account(tmp_path: Path):
    rows = [_account_row("ghost@01", "m1", games=1, pt=1.0, rating=1500.0, ranks=[0, 0, 0, 1], avg_rank=4.0)]
    env = _write_custom_env(tmp_path, season=_season(status="running"), report_accounts=rows)
    with pytest.raises(ladder.SeasonDataError, match="未在注册表声明"):
        ladder.load_ladder(env["root"], env["configs"], "s1")


def test_list_seasons_marks_mismatched_report_not_ready(tmp_path: Path):
    good = _season(season_id="good-season", report_dir="artifacts/good")
    bad = _season(season_id="bad-season", report_dir="artifacts/bad")
    configs_dir = tmp_path / "configs" / "ladder" / "seasons"
    configs_dir.mkdir(parents=True)
    for season in (good, bad):
        (configs_dir / f"{season['season_id']}.json").write_text(json.dumps(season), encoding="utf-8")
    good_dir = tmp_path / "artifacts" / "good"
    good_dir.mkdir(parents=True)
    (good_dir / "account_summary.json").write_text(
        json.dumps({"schema": ladder.REPORT_SCHEMA, "games": 3, "accounts": _three_account_rows()}),
        encoding="utf-8",
    )
    bad_dir = tmp_path / "artifacts" / "bad"
    bad_dir.mkdir(parents=True)
    (bad_dir / "account_summary.json").write_text(
        json.dumps({"schema": ladder.REPORT_SCHEMA, "games": 2, "accounts": _three_account_rows()[:2]}),
        encoding="utf-8",
    )
    seasons = {entry["season_id"]: entry for entry in ladder.list_seasons(tmp_path, configs_dir)}
    assert seasons["good-season"]["data_ready"] is True
    assert seasons["bad-season"]["data_ready"] is False


@pytest.mark.parametrize("max_points,expected_games", [
    (0, []),
    (1, [10]),
    (2, [1, 10]),
    (4, [1, 4, 7, 10]),
])
def test_curve_point_boundaries(season_env, max_points, expected_games):
    payload = ladder.load_account(
        season_env["root"], season_env["configs"], "test-season", "model_a@01",
        recent_limit=0, curve_max_points=max_points,
    )
    assert [point["games"] for point in payload["curve"]] == expected_games
