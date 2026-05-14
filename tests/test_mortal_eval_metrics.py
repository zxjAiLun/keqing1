from __future__ import annotations

import json
from types import SimpleNamespace

from scripts.mortal import ab_match
from scripts.mortal import analyze_checkpoint_behavior_slices
from scripts.mortal import audit_grp_on_logs
from scripts.mortal import compare_checkpoint_stat_reports
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
    profile, points = eval_metrics.resolve_rank_points(profile="avoid4_norm")
    assert profile == "avoid4_norm"
    assert points == (15.0 / 7.0, 9.0 / 7.0, 3.0 / 7.0, -27.0 / 7.0)
    assert eval_metrics.resolve_rank_points(profile="mortal_default")[1] == (6.0, 4.0, 2.0, 0.0)
    assert "zero_sum_balanced" not in eval_metrics.RANK_POINT_PROFILES
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

    metrics = stat_report.stat_to_metrics(FakeStat(), rank_pts=[2.5, 1.5, 0.5, -4.5])
    report = {
        "backend": "fake",
        "log_dir": "logs",
        "rank_pts": [90, 45, 0, -135],
        "players": {"A (x1)": {"player_name": "challenger", **metrics}},
    }

    markdown = stat_report.format_markdown_report(report)

    assert metrics["derived"]["avg_point_per_oya_agari"] is None
    assert metrics["derived"]["total_rank_pt"] == 0.0
    assert metrics["derived"]["avg_rank_pt"] == 0.0
    assert "| Games | 4 |" in markdown
    assert "- Rank point profile: `custom`" in markdown
    assert "| 1st (rate) | 1 (0.250000) |" in markdown


def test_compare_checkpoint_stat_reports_builds_delta() -> None:
    def player(*, agari: float, houjuu: float, fuuro: float, riichi: float):
        derived = {key: 0.0 for _, _, key, _ in compare_checkpoint_stat_reports.METRICS}
        derived.update(
            {
                "agari_rate": agari,
                "houjuu_rate": houjuu,
                "fuuro_rate": fuuro,
                "riichi_rate": riichi,
                "agari_rate_after_fuuro": 0.3,
                "houjuu_rate_after_fuuro": 0.1,
                "agari_rate_after_riichi": 0.5,
                "houjuu_rate_after_riichi": 0.14,
            }
        )
        return {"player_name": "champion", "raw": {}, "derived": derived}

    delta = compare_checkpoint_stat_reports.build_delta(
        left_report={"log_dir": "left-logs", "players": {"left": player(agari=0.2, houjuu=0.1, fuuro=0.25, riichi=0.15)}},
        right_report={"log_dir": "right-logs", "players": {"right": player(agari=0.21, houjuu=0.13, fuuro=0.3, riichi=0.18)}},
        left_label="70k",
        right_label="80k",
    )
    markdown = compare_checkpoint_stat_reports.format_markdown(delta)

    assert delta["rows"][0]["delta"] == 0.009999999999999981
    assert any("calls more often" in note for note in delta["diagnosis"])
    assert "| Win rate | 20.00% | 21.00% | +1.00pp |" in markdown


def test_behavior_slice_parser_summarizes_player_rounds(tmp_path) -> None:
    log_path = tmp_path / "game.json.gz"
    events = [
        {"type": "start_game", "names": ["a", "b", "c", "d"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "oya": 0,
            "scores": [30000, 25000, 24000, 21000],
        },
        {"type": "tsumo", "actor": 0, "pai": "1m"},
        {"type": "reach", "actor": 0},
        {"type": "dahai", "actor": 0, "pai": "1m"},
        {"type": "tsumo", "actor": 1, "pai": "2m"},
        {"type": "pon", "actor": 2, "target": 1, "pai": "2m", "consumed": ["2m", "2m"]},
        {"type": "hora", "actor": 2, "target": 0},
        {"type": "end_kyoku"},
    ]
    import gzip

    with gzip.open(log_path, "wt", encoding="utf-8") as handle:
        for event in events:
            handle.write(json.dumps(event) + "\n")

    summary = analyze_checkpoint_behavior_slices.analyze_logs([log_path], model_label="unit")
    all_row = next(row for row in summary["rows"] if row["slice_dim"] == "all")
    rank_1 = next(row for row in summary["rows"] if row["slice_dim"] == "start_rank" and row["slice_value"] == "1")

    assert summary["game_count"] == 1
    assert summary["player_rounds"] == 4
    assert all_row["agari_rate"] == 0.25
    assert all_row["houjuu_rate"] == 0.25
    assert all_row["fuuro_rate"] == 0.25
    assert all_row["riichi_rate"] == 0.25
    assert rank_1["houjuu_rate"] == 1.0


def test_registry_helper_appends_valid_jsonl(tmp_path) -> None:
    path = tmp_path / "registry.jsonl"
    entry = {
        "experiment_id": "R0_mortal_default",
        "parent_checkpoint": "artifacts/mortal_training/mortal.pth",
        "reward_profile": "mortal_default",
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
    assert [row["experiment_id"] for row in rows] == ["R0_mortal_default", "R1_avoid4"]


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
        matrix=[("R1_avoid4_norm", "avoid4_norm")],
        parent_steps=60000,
        target_steps=65000,
        train_steps=5000,
        copy_parent_checkpoint=False,
        dry_run=True,
    )

    row = report["experiments"][0]
    assert row["pt_table"] == [15.0 / 7.0, 9.0 / 7.0, 3.0 / 7.0, -27.0 / 7.0]
    assert row["parent_steps"] == 60000
    assert row["target_steps"] == 65000
    assert row["effective_train_steps"] == 5000
    assert row["config"].endswith("R1_avoid4_norm/config.toml")
    assert not (tmp_path / "experiments").exists()


def test_prepare_reward_pt_experiments_warns_for_raw_scale_profiles(tmp_path) -> None:
    base_config = tmp_path / "config.toml"
    base_config.write_text(
        """
[control]
state_file = "/tmp/base/mortal.pth"
best_state_file = "/tmp/base/mortal_best.pth"
tensorboard_dir = "/tmp/base/tb_mortal"

[dataset]
globs = ["/data/train/**/*.json.gz"]
file_index = "/tmp/base/file_index.pth"

[env]
pts = [6.0, 4.0, 2.0, 0.0]
""".lstrip(),
        encoding="utf-8",
    )

    report = prepare_reward_pt_experiments.write_experiment_configs(
        base_config_path=base_config,
        parent_checkpoint=tmp_path / "parent.pth",
        output_root=tmp_path / "experiments",
        matrix=[("R1_avoid4_raw", "avoid4_raw")],
        parent_steps=60000,
        target_steps=65000,
        train_steps=5000,
        copy_parent_checkpoint=False,
        dry_run=True,
    )

    assert report["experiments"][0]["warning"] == "raw profile changes reward scale as well as utility shape"


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
            "mortal_default": {
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
    assert audit_grp_on_logs.summarize_calibration(calibration) == {"ece": 0.0, "mce": 0.0}
    assert errors["mortal_default"]["mean_abs_expected_pt_error"] == 1.5
    assert errors["mortal_default"]["old_grp_reward_delta_mean"] == 2.0
    assert errors["mortal_default"]["old_grp_reward_delta_variance"] == 1.0


def test_grp_audit_sample_log_files_is_deterministic() -> None:
    files = [f"log_{idx}.json.gz" for idx in range(10)]

    sample_a = audit_grp_on_logs.sample_log_files(files, sample_log_files=4, sample_seed=7)
    sample_b = audit_grp_on_logs.sample_log_files(reversed(files), sample_log_files=4, sample_seed=7)

    assert sample_a == sample_b
    assert sample_a == sorted(sample_a)
    assert audit_grp_on_logs.sample_log_files(files, sample_log_files=20, sample_seed=7) == files
