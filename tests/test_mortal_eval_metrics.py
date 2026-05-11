from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.mortal import ab_match
from scripts.mortal import eval_metrics
from scripts.mortal import one_vs_three_smoke
from scripts.mortal import stat_report


def test_summarize_rank_counts_exports_rank_and_pt() -> None:
    summary = eval_metrics.summarize_rank_counts([1, 2, 1, 0])

    assert summary["games"] == 4
    assert summary["rank_counts"] == [1, 2, 1, 0]
    assert summary["avg_rank"] == 2.0
    assert summary["avg_rank_pt"] == 45.0


def test_ab_match_one_vs_three_seat_assignment_rotates_challenger() -> None:
    assert ab_match.seat_assignment(0, seat_mode="one-vs-three") == ["A", "B", "B", "B"]
    assert ab_match.seat_assignment(1, seat_mode="one-vs-three") == ["B", "A", "B", "B"]
    assert ab_match.seat_assignment(2, seat_mode="one-vs-three") == ["B", "B", "A", "B"]
    assert ab_match.seat_assignment(3, seat_mode="one-vs-three") == ["B", "B", "B", "A"]


def test_ab_match_two_vs_two_seat_assignment_alternates() -> None:
    assert ab_match.seat_assignment(0, seat_mode="two-vs-two") == ["A", "B", "A", "B"]
    assert ab_match.seat_assignment(1, seat_mode="two-vs-two") == ["B", "A", "B", "A"]


def test_ab_match_summarize_games_uses_one_based_riichienv_ranks() -> None:
    games = [
        {
            "assignment": ["A", "B", "B", "B"],
            "ranks": [1, 2, 3, 4],
            "scores": [33000, 27000, 22000, 18000],
            "env_step_count": 10,
            "mjai_event_count": 100,
            "fallback_count": 1,
            "wall_time_sec": 2.5,
        },
        {
            "assignment": ["B", "A", "B", "B"],
            "ranks": [1, 4, 2, 3],
            "scores": [33000, 18000, 27000, 22000],
            "env_step_count": 11,
            "mjai_event_count": 101,
            "fallback_count": 2,
            "wall_time_sec": 3.0,
        },
    ]

    summary = ab_match._summarize_games(games)

    assert summary["by_label"]["A"]["rank_counts"] == [1, 0, 0, 1]
    assert summary["by_label"]["A"]["avg_rank"] == 2.5
    assert summary["by_label"]["A"]["avg_score"] == 25500
    assert summary["by_label"]["B"]["seat_count"] == 6
    assert summary["totals"]["games"] == 2
    assert summary["totals"]["fallback_count"] == 3


def test_one_vs_three_smoke_writes_unified_metrics(monkeypatch, tmp_path) -> None:
    class FakeOneVsThree:
        def __init__(self, *, disable_progress_bar, log_dir):
            self.disable_progress_bar = disable_progress_bar
            self.log_dir = log_dir

        def py_vs_py(self, *, challenger, champion, seed_start, seed_count):
            assert challenger == "challenger-engine"
            assert champion == "champion-engine"
            assert seed_start == (10000, 8192)
            assert seed_count == 1
            return [1, 1, 1, 1]

    monkeypatch.setattr(one_vs_three_smoke, "_load_engine", lambda **kwargs: f"{kwargs['name']}-engine")
    monkeypatch.setattr(
        one_vs_three_smoke,
        "write_stat_report",
        lambda **kwargs: {"schema": "keqing.mortal.libriichi.stat.v1"},
    )

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "libriichi.arena":
            return SimpleNamespace(OneVsThree=FakeOneVsThree)
        return real_import(name, globals, locals, fromlist, level)

    real_import = __import__
    monkeypatch.setattr("builtins.__import__", fake_import)

    args = SimpleNamespace(
        challenger=tmp_path / "a.pth",
        champion=tmp_path / "b.pth",
        mortal_root=tmp_path / "Mortal",
        output_dir=tmp_path / "out",
        device="cpu",
        seed_start=10000,
        seed_key=8192,
        seed_count=1,
        enable_amp=False,
        challenger_label="new (x1)",
        champion_label="old (x3)",
    )

    document = one_vs_three_smoke.run(args)

    written = json.loads((tmp_path / "out" / "metrics.json").read_text(encoding="utf-8"))
    assert document["schema"] == eval_metrics.METRICS_SCHEMA
    assert written["schema"] == eval_metrics.METRICS_SCHEMA
    assert written["metrics"]["challenger"]["rank_counts"] == [1, 1, 1, 1]
    assert written["artifacts"]["detailed_stats_json"].endswith("detailed_stats.json")


def test_stat_report_normalizes_and_formats_markdown() -> None:
    class FakeStat:
        game = 4
        round = 40
        oya = 10
        point = 1200
        rank_1 = 1
        rank_2 = 1
        rank_3 = 1
        rank_4 = 1
        tobi = 0
        agari = 8
        houjuu = 4
        fuuro = 12
        fuuro_num = 18
        riichi = 6
        ryukyoku = 2
        yakuman = 0
        nagashi_mangan = 0

        rank_1_rate = 0.25
        rank_2_rate = 0.25
        rank_3_rate = 0.25
        rank_4_rate = 0.25
        tobi_rate = 0.0
        avg_rank = 2.5
        avg_point_per_game = 300.0
        avg_point_per_round = 30.0
        agari_rate = 0.2
        houjuu_rate = 0.1
        fuuro_rate = 0.3
        riichi_rate = 0.15
        ryukyoku_rate = 0.05
        avg_point_per_agari = 6000.0
        avg_point_per_oya_agari = float("nan")
        avg_point_per_ko_agari = 5200.0
        avg_point_per_riichi_agari = 7800.0
        avg_point_per_fuuro_agari = 4300.0
        avg_point_per_dama_agari = 6200.0
        avg_point_per_ryukyoku = 0.0
        avg_agari_jun = 11.0
        avg_riichi_agari_jun = 11.5
        avg_fuuro_agari_jun = 10.8
        avg_dama_agari_jun = 12.0
        avg_houjuu_jun = 11.2
        avg_point_per_houjuu = -5200.0
        avg_point_per_houjuu_to_oya = -7200.0
        avg_point_per_houjuu_to_ko = -4500.0
        chasing_riichi_rate = 0.1
        riichi_chased_rate = 0.2
        agari_rate_after_riichi = 0.5
        houjuu_rate_after_riichi = 0.16
        avg_riichi_jun = 7.8
        avg_riichi_point = 3000.0
        avg_fuuro_num = 1.5
        agari_rate_after_fuuro = 0.3
        houjuu_rate_after_fuuro = 0.14
        avg_fuuro_point = 800.0
        agari_rate_as_oya = 0.2
        agari_as_oya_rate = 0.25
        houjuu_to_oya_rate = 0.25
        yakuman_rate = 0.0
        nagashi_mangan_rate = 0.0

        def total_pt(self, pts):
            return sum(pts)

        def avg_pt(self, pts):
            return sum(pts) / self.game

        def __str__(self):
            return "fake stat"

    metrics = stat_report.stat_to_metrics(FakeStat())
    report = {
        "backend": "fake",
        "log_dir": "logs",
        "rank_pts": [90, 45, 0, -135],
        "players": {"A (x1)": {"player_name": "challenger", **metrics}},
    }

    markdown = stat_report.format_markdown_report(report)

    assert metrics["derived"]["avg_point_per_oya_agari"] is None
    assert "| Games | 4 |" in markdown
    assert "| 1st (rate) | 1 (0.250000) |" in markdown
