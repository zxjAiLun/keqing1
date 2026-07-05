from replay.server import _build_runtime_teacher_report, _list_review_history
from replay.storage import ReplayStorage


def test_review_history_groups_reports_by_replay_and_player(tmp_path) -> None:
    storage = ReplayStorage(tmp_path / "replays")
    replay_id = storage.save(
        events=[{"type": "start_game", "names": ["A", "B", "C", "D"]}],
        decisions={
            "log": [{"scores": [25000, 25000, 25000, 25000]}],
            "kyoku_order": [{"bakaze": "E", "kyoku": 1, "honba": 0}],
        },
        bot_type="70k",
        player_names=["A", "B", "C", "D"],
    )
    report_dir = tmp_path / "gui_teacher_reports"
    report_dir.mkdir()
    (report_dir / f"{replay_id}__v4__p1.json").write_text("{}", encoding="utf-8")
    (report_dir / f"{replay_id}__70k.pth__p1.json").write_text("{}", encoding="utf-8")
    (report_dir / f"{replay_id}__T1_71000__p1.json").write_text("{}", encoding="utf-8")

    history = _list_review_history(storage=storage, report_dir=report_dir, project_root=tmp_path)

    assert len(history) == 1
    assert history[0]["replay_id"] == replay_id
    assert history[0]["player_id"] == 1
    assert history[0]["player_name"] == "B"
    assert history[0]["models"] == ["v4", "70k.pth", "T1@71000"]
    assert len(history[0]["teacher_report_paths"]) == 3


def test_runtime_teacher_report_treats_null_ground_truth_as_pass(tmp_path) -> None:
    report = _build_runtime_teacher_report(
        replay_id="replay_test",
        model_type="weak_mortal",
        player_id=1,
        checkpoint=tmp_path / "v4.pth",
        decisions={
            "log": [
                {
                    "step": 4,
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "is_obs": False,
                    "chosen": {"type": "none"},
                    "gt_action": None,
                    "candidates": [
                        {"action": {"type": "none"}, "final_score": 0.2, "prob": 0.6},
                        {"action": {"type": "hora", "actor": 1, "target": 0}, "final_score": 0.1, "prob": 0.4},
                    ],
                }
            ]
        },
    )

    entry = report["review"]["kyokus"][0]["entries"][0]
    assert entry["actual"] == {"type": "none"}
    assert entry["is_equal"] is True
    assert [item["action"]["type"] for item in entry["details"]] == ["none", "hora"]
