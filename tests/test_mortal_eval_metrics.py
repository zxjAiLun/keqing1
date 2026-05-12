from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.mortal import ab_match
from scripts.mortal import audit_grp_on_logs
from scripts.mortal import eval_metrics
from scripts.mortal import experiment_registry
from scripts.mortal import one_vs_three_smoke
from scripts.mortal import prepare_reward_pt_experiments
from scripts.mortal import stat_report


def test_summarize_rank_counts_exports_rank_and_pt() -> None:
    summary = eval_metrics.summarize_rank_counts([1, 2, 1, 0])

    assert summary["games"] == 4
    assert summary["rank_counts"] == [1, 2, 1, 0]
    assert summary["avg_rank"] == 2.0
    assert summary["avg_rank_pt"] == 45.0


def test_rank_point_profiles_and_custom_values() -> None:
    profile, points = eval_metrics.resolve_rank_points(profile="avoid4_strong")
    assert profile == "avoid4_strong"
    assert points == (4.0, 3.0, 2.0, -3.0)
    assert eval_metrics.resolve_rank_points(rank_points="1,2,3,4") == ("custom", (1.0, 2.0, 3.0, 4.0))

    try:
        eval_metrics.parse_rank_points("1,2,3")
    except ValueError as exc:
        assert "length 4" in str(exc)
    else:
        raise AssertionError("expected invalid rank point list to fail")

    try:
        eval_metrics.resolve_rank_points(profile="custom")
    except ValueError as exc:
        assert "requires --rank-points" in str(exc)
    else:
        raise AssertionError("expected custom profile without values to fail")


def test_build_metrics_document_records_rank_point_metadata() -> None:
    document = eval_metrics.build_metrics_document(
        run={"kind": "unit"},
        metrics={},
        rank_points_profile="custom",
        rank_points_values=(1, 2, 3, 4),
    )

    assert document["rank_points_profile"] == "custom"
    assert document["rank_points_values"] == [1.0, 2.0, 3.0, 4.0]


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

    summary = ab_match._summarize_games(games, rank_points=(100, 50, 0, -100))

    assert summary["by_label"]["A"]["rank_counts"] == [1, 0, 0, 1]
    assert summary["by_label"]["A"]["avg_rank"] == 2.5
    assert summary["by_label"]["A"]["avg_rank_pt"] == 0.0
    assert summary["by_label"]["A"]["avg_rank_pt_tenhou_reference"] == -22.5
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
    assert written["rank_points_profile"] == "tenhou_reference"
    assert written["rank_points_values"] == [90.0, 45.0, 0.0, -135.0]
    assert written["metrics"]["challenger"]["rank_counts"] == [1, 1, 1, 1]
    assert written["metrics"]["challenger"]["avg_rank_pt_training_profile"] == 0.0
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
    assert "- Rank point profile: `custom`" in markdown
    assert "| 1st (rate) | 1 (0.250000) |" in markdown


def test_registry_helper_appends_valid_jsonl(tmp_path) -> None:
    path = tmp_path / "registry.jsonl"
    entry = {
        "experiment_id": "R0_base",
        "parent_checkpoint": "artifacts/mortal_training/mortal.pth",
        "reward_profile": "base",
        "pt_table": [6, 4, 2, 0],
        "grp_checkpoint": "artifacts/mortal_training/grp.pth",
        "training_data": "artifacts/mortal_mjai_gz/train/**/*.json.gz",
        "style_data": None,
        "train_steps": 5000,
        "eval_bundle": None,
        "notes": "unit",
    }

    first = experiment_registry.append_registry_entry(path, entry)
    second = experiment_registry.append_registry_entry(path, {**entry, "experiment_id": "R1_avoid4"})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert first["schema"] == experiment_registry.REGISTRY_SCHEMA
    assert second["experiment_id"] == "R1_avoid4"
    assert [row["experiment_id"] for row in rows] == ["R0_base", "R1_avoid4"]


def test_prepare_reward_pt_experiments_dry_run_is_isolated(tmp_path) -> None:
    base_config = tmp_path / "config.toml"
    base_config.write_text(
        """
[control]
state_file = "/tmp/base/mortal.pth"
best_state_file = "/tmp/base/mortal_best.pth"
tensorboard_dir = "/tmp/base/tb_mortal"

[train_play.default]
log_dir = "/tmp/base/train_play"

[test_play]
log_dir = "/tmp/base/test_play"

[dataset]
globs = ["/data/train/**/*.json.gz"]
file_index = "/tmp/base/file_index.pth"

[env]
pts = [6.0, 4.0, 2.0, 0.0]

[grp]
state_file = "/tmp/base/grp.pth"
""".lstrip(),
        encoding="utf-8",
    )

    report = prepare_reward_pt_experiments.write_experiment_configs(
        base_config_path=base_config,
        parent_checkpoint=tmp_path / "parent.pth",
        output_root=tmp_path / "experiments",
        matrix=[("R1_avoid4_strong", "avoid4_strong")],
        target_steps=65000,
        train_steps=5000,
        copy_parent_checkpoint=False,
        dry_run=True,
    )

    row = report["experiments"][0]
    assert row["pt_table"] == [4.0, 3.0, 2.0, -3.0]
    assert row["target_steps"] == 65000
    assert row["config"].endswith("R1_avoid4_strong/config.toml")
    assert not (tmp_path / "experiments").exists()


def test_grp_audit_finalizers_export_calibration_and_reward_variance() -> None:
    calibration = audit_grp_on_logs.finalize_calibration(
        [
            {"count": 2, "confidence_sum": 1.0, "accuracy_sum": 1.0, "true_prob_sum": 0.8},
            *[
                {"count": 0, "confidence_sum": 0.0, "accuracy_sum": 0.0, "true_prob_sum": 0.0}
                for _ in range(9)
            ],
        ]
    )
    errors = audit_grp_on_logs.finalize_profile_errors(
        {
            "base": {
                "count": 2,
                "abs_error_sum": 3.0,
                "sq_error_sum": 5.0,
                "signed_error_sum": 1.0,
                "reward_deltas": [1.0, 3.0],
            }
        }
    )

    assert calibration[0]["avg_confidence"] == 0.5
    assert calibration[0]["accuracy"] == 0.5
    assert errors["base"]["mean_abs_expected_pt_error"] == 1.5
    assert errors["base"]["old_grp_reward_delta_mean"] == 2.0
    assert errors["base"]["old_grp_reward_delta_variance"] == 1.0
