import json

import numpy as np
import pytest

from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
)
from xmodel1.schema import XMODEL1_SAMPLE_TYPE_HORA, XMODEL1_SPECIAL_TYPE_HORA


def _export_rust_arrays(tmp_path, events):
    from keqing_core import build_xmodel1_discard_records

    data_dir = tmp_path / "ds1"
    data_dir.mkdir()
    replay_path = data_dir / "sample.mjson"
    replay_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )
    try:
        _count, _manifest_path, produced_npz = build_xmodel1_discard_records(
            data_dirs=[str(data_dir)],
            output_dir=str(tmp_path / "out"),
            smoke=False,
        )
    except RuntimeError:
        pytest.skip("Rust extension unavailable")
    assert produced_npz is True
    return np.load(tmp_path / "out" / "ds1" / "sample.npz", allow_pickle=True)


def _candidate_feat_diff_message(rust_candidate_feat, py_candidate_feat, candidate_tile_id) -> str:
    diff = np.argwhere(rust_candidate_feat != py_candidate_feat)
    lines = [f"candidate_feat drift_count={len(diff)}"]
    for row, cand, slot in diff[:20]:
        lines.append(
            f"row={int(row)} cand={int(cand)} tile={int(candidate_tile_id[row, cand])} "
            f"slot={int(slot)} rust={float(rust_candidate_feat[row, cand, slot]):.6f} "
            f"py={float(py_candidate_feat[row, cand, slot]):.6f}"
        )
    return "\n".join(lines)


def _assert_rust_matches_python_oracle(tmp_path, events, *, replay_id="fixture.mjson", compare_targets=False):
    from xmodel1.preprocess import events_to_xmodel1_arrays

    py_arrays = events_to_xmodel1_arrays(events, replay_id=replay_id)
    assert py_arrays is not None
    with _export_rust_arrays(tmp_path, events) as rust_arrays:
        py_discard_mask = py_arrays["sample_type"] == 0
        rust_discard_mask = rust_arrays["sample_type"] == 0
        rust_tile_ids = rust_arrays["candidate_tile_id"][rust_discard_mask]
        exact_fields = [
            "state_tile_feat",
            "state_scalar",
            "candidate_feat",
            "candidate_flags",
            "candidate_tile_id",
            "candidate_mask",
            "chosen_candidate_idx",
            "candidate_rank_bucket",
            "candidate_hard_bad_flag",
            "special_candidate_type_id",
            "special_candidate_mask",
            "chosen_special_candidate_idx",
            "special_candidate_rank_bucket",
            "special_candidate_hard_bad_flag",
        ]
        for field in exact_fields:
            if field == "candidate_feat":
                rust_field = rust_arrays[field][rust_discard_mask]
                py_field = py_arrays[field][py_discard_mask]
                assert np.array_equal(
                    rust_field,
                    py_field,
                ), _candidate_feat_diff_message(rust_field, py_field, rust_tile_ids)
                continue
            assert np.array_equal(rust_arrays[field][rust_discard_mask], py_arrays[field][py_discard_mask]), field
        assert np.allclose(
            rust_arrays["special_candidate_feat"][rust_discard_mask],
            py_arrays["special_candidate_feat"][py_discard_mask],
            atol=1e-3,
            rtol=0.0,
        ), "special_candidate_feat"
        assert np.allclose(
            rust_arrays["candidate_quality_score"][rust_discard_mask],
            py_arrays["candidate_quality_score"][py_discard_mask],
            atol=1e-6,
            rtol=0.0,
        )
        assert np.allclose(
            rust_arrays["special_candidate_quality_score"][rust_discard_mask],
            py_arrays["special_candidate_quality_score"][py_discard_mask],
            atol=1e-6,
            rtol=0.0,
        )
        assert np.array_equal(
            rust_arrays["event_history"][rust_discard_mask],
            py_arrays["event_history"][py_discard_mask],
        ), "event_history"
        if compare_targets:
            target_fields = [
                "score_delta_target",
                "win_target",
                "dealin_target",
                "pts_given_win_target",
                "pts_given_dealin_target",
            ]
            for field in target_fields:
                assert np.allclose(
                    rust_arrays[field][rust_discard_mask],
                    py_arrays[field][py_discard_mask],
                    atol=1e-6,
                    rtol=0.0,
                ), field
        # Stage 2: opp_tenpai_target 是决策时刻标签 (不依赖终局),每个
        # discard 样本都能对拍,不需要 compare_targets=True 才比。
        # rust_arrays 是 np.load 的 NpzFile,py_arrays 是 dict,分别用各自的 key 检查。
        if "opp_tenpai_target" in rust_arrays.files and "opp_tenpai_target" in py_arrays:
            rust_opp = rust_arrays["opp_tenpai_target"][rust_discard_mask]
            py_opp = py_arrays["opp_tenpai_target"][py_discard_mask]
            assert rust_opp.shape == py_opp.shape, (
                f"opp_tenpai_target shape mismatch: rust={rust_opp.shape} py={py_opp.shape}"
            )
            assert np.array_equal(rust_opp, py_opp), "opp_tenpai_target"


def _assert_candidate_feat_matches_python_oracle(tmp_path, events, *, replay_id: str) -> None:
    from xmodel1.preprocess import events_to_xmodel1_arrays

    py_arrays = events_to_xmodel1_arrays(events, replay_id=replay_id)
    assert py_arrays is not None
    with _export_rust_arrays(tmp_path, events) as rust_arrays:
        py_discard_mask = py_arrays["sample_type"] == 0
        rust_discard_mask = rust_arrays["sample_type"] == 0
        rust_field = rust_arrays["candidate_feat"][rust_discard_mask]
        py_field = py_arrays["candidate_feat"][py_discard_mask]
        rust_tile_ids = rust_arrays["candidate_tile_id"][rust_discard_mask]
        assert np.array_equal(
            rust_field,
            py_field,
        ), _candidate_feat_diff_message(rust_field, py_field, rust_tile_ids)


def test_xmodel1_schema_contract_matches_python_constants():
    from keqing_core import xmodel1_schema_info

    try:
        name, version, max_candidates, candidate_dim, flag_dim = xmodel1_schema_info()
    except RuntimeError:
        # Rust extension might be absent in a fresh environment.
        return

    assert name == XMODEL1_SCHEMA_NAME
    assert version == XMODEL1_SCHEMA_VERSION
    assert max_candidates == XMODEL1_MAX_CANDIDATES
    assert candidate_dim == XMODEL1_CANDIDATE_FEATURE_DIM
    assert flag_dim == XMODEL1_CANDIDATE_FLAG_DIM


def test_xmodel1_discard_record_validation_surface_is_safe():
    from keqing_core import validate_xmodel1_discard_record

    try:
        ok = validate_xmodel1_discard_record(
            0,
            [1] + [0] * (XMODEL1_MAX_CANDIDATES - 1),
            [0] + [-1] * (XMODEL1_MAX_CANDIDATES - 1),
        )
    except RuntimeError:
        return

    assert ok is True


def test_xmodel1_rust_export_surface_writes_manifest(tmp_path):
    from keqing_core import build_xmodel1_discard_records

    data_dir = tmp_path / "ds1"
    data_dir.mkdir()
    (data_dir / "sample.mjson").write_text('{"type":"start_game","names":["A","B","C","D"]}\n', encoding="utf-8")
    try:
        count, manifest_path, produced_npz = build_xmodel1_discard_records(
            data_dirs=[str(data_dir)],
            output_dir=str(tmp_path / "out"),
            smoke=True,
        )
    except RuntimeError:
        return

    assert count == 1
    manifest = json.loads((tmp_path / "out" / "xmodel1_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["file_count"] == 1
    assert manifest["exported_file_count"] in {0, 1}
    assert manifest["exported_sample_count"] >= 0
    assert "processed_file_count" in manifest
    assert "skipped_existing_file_count" in manifest
    assert "ds1" in manifest["shard_file_counts"] or manifest["shard_file_counts"] == {}
    if produced_npz:
        npz_path = tmp_path / "out" / "ds1" / "sample.npz"
        assert npz_path.exists()
        assert manifest["exported_file_count"] == 1
        assert manifest["exported_sample_count"] >= 1
        assert manifest["processed_file_count"] == 1
        assert manifest["skipped_existing_file_count"] == 0
        assert manifest["shard_file_counts"]["ds1"] == 1
        assert manifest["shard_sample_counts"]["ds1"] >= 1
        with np.load(npz_path, allow_pickle=False) as data:
            assert data["schema_name"].item() == XMODEL1_SCHEMA_NAME
            assert int(data["schema_version"].item()) == XMODEL1_SCHEMA_VERSION


def test_xmodel1_rust_export_produces_multiple_candidates_for_real_smoke_record(tmp_path):
    from keqing_core import build_xmodel1_discard_records

    data_dir = tmp_path / "ds1"
    data_dir.mkdir()
    (data_dir / "sample.mjson").write_text(
        "\n".join(
            [
                '{"type":"start_game","names":["A","B","C","D"]}',
                '{"type":"start_kyoku","bakaze":"E","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"dora_marker":"1m","tehais":[["1m","1m","5p","5p","7s","7s","E","E","S","S","W","N","F"],["1s","1s","1s","1s","2s","2s","2s","2s","3s","3s","3s","3s","4s"],["4m","4m","4m","5m","5m","5m","6m","6m","6m","7m","7m","7m","8m"],["4p","4p","4p","5p","5p","5p","6p","6p","6p","7p","7p","7p","8p"]]}',
                '{"type":"tsumo","actor":0,"pai":"5m"}',
                '{"type":"dahai","actor":0,"pai":"5m","tsumogiri":true}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    try:
        _count, _manifest_path, produced_npz = build_xmodel1_discard_records(
            data_dirs=[str(data_dir)],
            output_dir=str(tmp_path / "out"),
            smoke=False,
        )
    except RuntimeError:
        return

    assert produced_npz is True
    npz_path = tmp_path / "out" / "ds1" / "sample.npz"
    assert npz_path.exists()
    with np.load(npz_path, allow_pickle=True) as data:
        assert data["schema_name"].item() == XMODEL1_SCHEMA_NAME
        assert int(data["schema_version"].item()) == XMODEL1_SCHEMA_VERSION
        mask = data["candidate_mask"][0]
        tile_ids = data["candidate_tile_id"][0]
        qualities = data["candidate_quality_score"][0]
        rank_bucket = data["candidate_rank_bucket"][0]
        hard_bad = data["candidate_hard_bad_flag"][0]
        flags = data["candidate_flags"][0]
        assert int(mask.sum()) >= 4
        assert tile_ids[0] >= 0
        assert qualities.shape[0] == XMODEL1_MAX_CANDIDATES
        assert rank_bucket.shape[0] == XMODEL1_MAX_CANDIDATES
        assert hard_bad.shape[0] == XMODEL1_MAX_CANDIDATES
        assert flags.shape == (XMODEL1_MAX_CANDIDATES, XMODEL1_CANDIDATE_FLAG_DIM)
        active_idx = np.where(mask > 0)[0]
        assert len(active_idx) >= 4
        assert np.any(qualities[active_idx] != 0.0)
        assert np.all(rank_bucket[active_idx] >= 0)
        assert np.all(rank_bucket[active_idx] <= 3)


def test_xmodel1_rust_export_matches_python_oracle_on_simple_discard_fixture(tmp_path):
    events = [
        {"type": "start_game", "names": ["A", "B", "C", "D"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
                ["1s"] * 13,
                ["2s"] * 13,
                ["3s"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "4p"},
        {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
    ]
    _assert_rust_matches_python_oracle(tmp_path, events)


def test_xmodel1_rust_export_keeps_hora_as_dedicated_special_sample(tmp_path):
    events = [
        {"type": "start_game", "names": ["A", "B", "C", "D"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "2m", "3m", "1p", "2p", "3p", "1s", "2s", "3s", "4m", "5m", "9s", "9s"],
                ["1s"] * 13,
                ["2s"] * 13,
                ["3s"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "6m"},
        {"type": "hora", "actor": 0, "target": 0, "pai": "6m", "deltas": [1000, -500, -500, 0], "ura_markers": []},
    ]

    from xmodel1.preprocess import events_to_xmodel1_arrays

    py_arrays = events_to_xmodel1_arrays(events, replay_id="hora_fixture.mjson")
    assert py_arrays is not None
    with _export_rust_arrays(tmp_path, events) as rust_arrays:
        py_rows = np.where(py_arrays["sample_type"] == XMODEL1_SAMPLE_TYPE_HORA)[0]
        rust_rows = np.where(rust_arrays["sample_type"] == XMODEL1_SAMPLE_TYPE_HORA)[0]
        assert py_rows.tolist() == [0]
        assert rust_rows.tolist() == [0]
        py_row = int(py_rows[0])
        rust_row = int(rust_rows[0])
        assert int(py_arrays["action_idx_target"][py_row]) == 42
        assert int(rust_arrays["action_idx_target"][rust_row]) == 42
        py_chosen = int(py_arrays["chosen_special_candidate_idx"][py_row])
        rust_chosen = int(rust_arrays["chosen_special_candidate_idx"][rust_row])
        assert py_chosen >= 0
        assert rust_chosen >= 0
        assert int(py_arrays["special_candidate_type_id"][py_row, py_chosen]) == XMODEL1_SPECIAL_TYPE_HORA
        assert int(rust_arrays["special_candidate_type_id"][rust_row, rust_chosen]) == XMODEL1_SPECIAL_TYPE_HORA


@pytest.mark.parametrize(
    ("name", "events"),
    [
        (
            "simple_discard",
            [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "dora_marker": "1m",
                    "tehais": [
                        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
                        ["1s"] * 13,
                        ["2s"] * 13,
                        ["3s"] * 13,
                    ],
                },
                {"type": "tsumo", "actor": 0, "pai": "4p"},
                {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
            ],
        ),
        (
            "riichi",
            [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "dora_marker": "1m",
                    "tehais": [
                        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
                        ["1s"] * 13,
                        ["2s"] * 13,
                        ["3s"] * 13,
                    ],
                },
                {"type": "tsumo", "actor": 0, "pai": "4p"},
                {"type": "reach", "actor": 0},
                {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
                {"type": "reach_accepted", "actor": 0, "scores": [24000, 25000, 25000, 25000], "kyotaku": 1},
            ],
        ),
        (
            "pon_call",
            [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "dora_marker": "1m",
                    "tehais": [
                        ["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"],
                        ["5p", "5p", "7s", "7s", "8s", "8s", "9s", "9s", "E", "E", "S", "S", "W"],
                        ["2s"] * 13,
                        ["3s"] * 13,
                    ],
                },
                {"type": "tsumo", "actor": 0, "pai": "5p"},
                {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
                {"type": "pon", "actor": 1, "target": 0, "pai": "5p", "consumed": ["5p", "5p"]},
                {"type": "dahai", "actor": 1, "pai": "W", "tsumogiri": False},
            ],
        ),
    ],
)
def test_xmodel1_rust_export_candidate_feat_minimal_regressions(tmp_path, name, events):
    _assert_candidate_feat_matches_python_oracle(tmp_path, events, replay_id=f"{name}.mjson")


@pytest.mark.parametrize(
    ("name", "events"),
    [
        (
            "riichi",
            [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "dora_marker": "1m",
                    "tehais": [
                        ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
                        ["1s"] * 13,
                        ["2s"] * 13,
                        ["3s"] * 13,
                    ],
                },
                {"type": "tsumo", "actor": 0, "pai": "4p"},
                {"type": "reach", "actor": 0},
                {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
                {"type": "reach_accepted", "actor": 0, "scores": [24000, 25000, 25000, 25000], "kyotaku": 1},
            ],
        ),
        (
            "pon",
            [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "dora_marker": "1m",
                    "tehais": [
                        ["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"],
                        ["5p", "5p", "7s", "7s", "8s", "8s", "9s", "9s", "E", "E", "S", "S", "W"],
                        ["2s"] * 13,
                        ["3s"] * 13,
                    ],
                },
                {"type": "tsumo", "actor": 0, "pai": "5p"},
                {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
                {"type": "pon", "actor": 1, "target": 0, "pai": "5p", "consumed": ["5p", "5p"]},
                {"type": "dahai", "actor": 1, "pai": "W", "tsumogiri": False},
            ],
        ),
        (
            "aka",
            [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "dora_marker": "1m",
                    "tehais": [
                        ["1m", "2m", "3m", "4m", "5mr", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
                        ["1s"] * 13,
                        ["2s"] * 13,
                        ["3s"] * 13,
                    ],
                },
                {"type": "tsumo", "actor": 0, "pai": "4p"},
                {"type": "dahai", "actor": 0, "pai": "5mr", "tsumogiri": False},
            ],
        ),
        (
            "chi2",
            [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "dora_marker": "1m",
                    "tehais": [
                        ["4m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p", "5p", "6p", "7p", "8p"],
                        ["2m", "3m", "4m", "6m", "7s", "7s", "8s", "8s", "9s", "9s", "E", "E", "S"],
                        ["2s"] * 13,
                        ["3s"] * 13,
                    ],
                },
                {"type": "tsumo", "actor": 0, "pai": "5m"},
                {"type": "dahai", "actor": 0, "pai": "5m", "tsumogiri": True},
                {"type": "chi", "actor": 1, "target": 0, "pai": "5m", "consumed": ["3m", "4m"]},
                {"type": "dahai", "actor": 1, "pai": "S", "tsumogiri": False},
            ],
        ),
        (
            "ankan2",
            [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "dora_marker": "1m",
                    "tehais": [
                        ["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"],
                        ["5p", "5p", "5p", "7s", "7s", "8s", "8s", "9s", "9s", "E", "E", "S", "S"],
                        ["2s"] * 13,
                        ["3s"] * 13,
                    ],
                },
                {"type": "tsumo", "actor": 0, "pai": "5s"},
                {"type": "dahai", "actor": 0, "pai": "5s", "tsumogiri": True},
                {"type": "tsumo", "actor": 1, "pai": "5p"},
                {"type": "ankan", "actor": 1, "consumed": ["5p", "5p", "5p", "5p"]},
                {"type": "tsumo", "actor": 1, "pai": "W", "rinshan": True},
                {"type": "dahai", "actor": 1, "pai": "W", "tsumogiri": True},
            ],
        ),
        (
            "kakan2",
            [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "dora_marker": "1m",
                    "tehais": [
                        ["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"],
                        ["5p", "5p", "7s", "7s", "8s", "8s", "9s", "9s", "E", "E", "S", "S", "W"],
                        ["2s"] * 13,
                        ["3s"] * 13,
                    ],
                },
                {"type": "tsumo", "actor": 0, "pai": "5p"},
                {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
                {"type": "pon", "actor": 1, "target": 0, "pai": "5p", "consumed": ["5p", "5p"]},
                {"type": "dahai", "actor": 1, "pai": "W", "tsumogiri": False},
                {"type": "tsumo", "actor": 1, "pai": "5p"},
                {"type": "kakan", "actor": 1, "pai": "5p", "consumed": ["5p", "5p", "5p"]},
                {"type": "kakan_accepted", "actor": 1, "pai": "5p", "consumed": ["5p", "5p", "5p"], "target": 0},
                {"type": "tsumo", "actor": 1, "pai": "N", "rinshan": True},
                {"type": "dahai", "actor": 1, "pai": "N", "tsumogiri": True},
            ],
        ),
        (
            "daiminkan2",
            [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "dora_marker": "1m",
                    "tehais": [
                        ["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"],
                        ["5p", "5p", "5p", "7s", "7s", "8s", "8s", "9s", "9s", "E", "E", "S", "S"],
                        ["2s"] * 13,
                        ["3s"] * 13,
                    ],
                },
                {"type": "tsumo", "actor": 0, "pai": "5p"},
                {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
                {"type": "daiminkan", "actor": 1, "target": 0, "pai": "5p", "consumed": ["5p", "5p", "5p"]},
                {"type": "tsumo", "actor": 1, "pai": "W", "rinshan": True},
                {"type": "dahai", "actor": 1, "pai": "W", "tsumogiri": True},
            ],
        ),
        (
            "chi_aka2",
            [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "dora_marker": "1m",
                    "tehais": [
                        ["6p", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "5m", "5p", "6s", "7p", "8p"],
                        ["2m", "5pr", "6p", "6m", "7s", "7s", "8s", "8s", "9s", "9s", "E", "E", "S"],
                        ["2s"] * 13,
                        ["3s"] * 13,
                    ],
                },
                {"type": "tsumo", "actor": 0, "pai": "4p"},
                {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
                {"type": "chi", "actor": 1, "target": 0, "pai": "4p", "consumed": ["5pr", "6p"]},
                {"type": "dahai", "actor": 1, "pai": "S", "tsumogiri": False},
            ],
        ),
        (
            "ankan_aka2",
            [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "dora_marker": "1m",
                    "tehais": [
                        ["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"],
                        ["5p", "5p", "5pr", "7s", "7s", "8s", "8s", "9s", "9s", "E", "E", "S", "S"],
                        ["2s"] * 13,
                        ["3s"] * 13,
                    ],
                },
                {"type": "tsumo", "actor": 0, "pai": "5s"},
                {"type": "dahai", "actor": 0, "pai": "5s", "tsumogiri": True},
                {"type": "tsumo", "actor": 1, "pai": "5p"},
                {"type": "ankan", "actor": 1, "pai": "5pr", "consumed": ["5p", "5p", "5pr", "5p"]},
                {"type": "tsumo", "actor": 1, "pai": "W", "rinshan": True},
                {"type": "dahai", "actor": 1, "pai": "W", "tsumogiri": True},
            ],
        ),
        (
            "kakan_aka2",
            [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "dora_marker": "1m",
                    "tehais": [
                        ["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"],
                        ["5p", "5pr", "7s", "7s", "8s", "8s", "9s", "9s", "E", "E", "S", "S", "W"],
                        ["2s"] * 13,
                        ["3s"] * 13,
                    ],
                },
                {"type": "tsumo", "actor": 0, "pai": "5p"},
                {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
                {"type": "pon", "actor": 1, "target": 0, "pai": "5p", "consumed": ["5p", "5pr"]},
                {"type": "dahai", "actor": 1, "pai": "W", "tsumogiri": False},
                {"type": "tsumo", "actor": 1, "pai": "5p"},
                {"type": "kakan", "actor": 1, "pai": "5p", "consumed": ["5p", "5pr", "5p"]},
                {"type": "kakan_accepted", "actor": 1, "pai": "5p", "consumed": ["5p", "5pr", "5p"], "target": 0},
                {"type": "tsumo", "actor": 1, "pai": "N", "rinshan": True},
                {"type": "dahai", "actor": 1, "pai": "N", "tsumogiri": True},
            ],
        ),
        (
            "legacy_kakan2",
            [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "dora_marker": "1m",
                    "tehais": [
                        ["2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"],
                        ["5p", "5p", "7s", "7s", "8s", "8s", "9s", "9s", "E", "E", "S", "S", "W"],
                        ["2s"] * 13,
                        ["3s"] * 13,
                    ],
                },
                {"type": "tsumo", "actor": 0, "pai": "5p"},
                {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
                {"type": "pon", "actor": 1, "target": 0, "pai": "5p", "consumed": ["5p", "5p"]},
                {"type": "dahai", "actor": 1, "pai": "W", "tsumogiri": False},
                {"type": "tsumo", "actor": 1, "pai": "5p"},
                {"type": "kakan", "actor": 1, "pai": "5p", "consumed": ["5p", "5p", "5p"]},
                {"type": "tsumo", "actor": 1, "pai": "N", "rinshan": True},
                {"type": "dahai", "actor": 1, "pai": "N", "tsumogiri": True},
            ],
        ),
    ],
)
def test_xmodel1_rust_export_matches_python_oracle_on_extended_discard_fixtures(tmp_path, name, events):
    _assert_rust_matches_python_oracle(tmp_path, events, replay_id=f"{name}.mjson")


@pytest.mark.parametrize(
    ("name", "events"),
    [
        (
            "multi_ron",
            [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "dora_marker": "1m",
                    "tehais": [
                        ["1m", "2m", "3m", "4m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p", "4p"],
                        ["2m", "3m", "4m", "6p", "7p", "8p", "2s", "3s", "4s", "E", "E", "S", "5m"],
                        ["1m", "1m", "2m", "2m", "3p", "3p", "4p", "4p", "6s", "6s", "E", "E", "5m"],
                        ["7m", "7m", "8m", "8m", "9p", "9p", "1s", "1s", "2s", "2s", "S", "S", "5m"],
                    ],
                },
                {"type": "tsumo", "actor": 0, "pai": "5p"},
                {"type": "dahai", "actor": 0, "pai": "5p", "tsumogiri": True},
                {"type": "tsumo", "actor": 1, "pai": "9s"},
                {"type": "dahai", "actor": 1, "pai": "5m", "tsumogiri": False},
                {"type": "hora", "actor": 2, "target": 1, "pai": "5m", "deltas": [0, -8000, 8000, 0], "scores": [25000, 17000, 33000, 25000]},
                {"type": "hora", "actor": 3, "target": 1, "pai": "5m", "deltas": [0, -3900, 0, 3900], "scores": [25000, 13100, 33000, 28900]},
                {"type": "end_kyoku"},
            ],
        ),
        (
            "chankan",
            [
                {"type": "start_game", "names": ["A", "B", "C", "D"]},
                {
                    "type": "start_kyoku",
                    "bakaze": "E",
                    "kyoku": 1,
                    "honba": 0,
                    "kyotaku": 0,
                    "oya": 0,
                    "scores": [25000, 25000, 25000, 25000],
                    "dora_marker": "1m",
                    "tehais": [
                        ["5m", "5m", "2m", "3m", "4m", "6p", "7p", "8p", "2s", "3s", "4s", "E", "E"],
                        ["1m", "1m", "2m", "2m", "3p", "3p", "4p", "4p", "6s", "6s", "S", "S", "5m"],
                        ["9m", "9m", "9p", "1p", "2p", "3p", "4s", "5s", "6s", "N", "N", "P", "5m"],
                        ["3m"] * 13,
                    ],
                },
                {"type": "tsumo", "actor": 0, "pai": "W"},
                {"type": "dahai", "actor": 0, "pai": "W", "tsumogiri": True},
                {"type": "tsumo", "actor": 1, "pai": "9s"},
                {"type": "dahai", "actor": 1, "pai": "9s", "tsumogiri": True},
                {"type": "tsumo", "actor": 2, "pai": "7m"},
                {"type": "dahai", "actor": 2, "pai": "5m", "tsumogiri": False},
                {"type": "pon", "actor": 0, "target": 2, "pai": "5m", "consumed": ["5m", "5m"]},
                {"type": "dahai", "actor": 0, "pai": "E", "tsumogiri": False},
                {"type": "tsumo", "actor": 0, "pai": "5m"},
                {"type": "kakan", "actor": 0, "pai": "5m", "consumed": ["5m", "5m", "5m"]},
                {"type": "hora", "actor": 1, "target": 0, "pai": "5m", "deltas": [-8000, 8000, 0, 0], "scores": [17000, 33000, 25000, 25000]},
                {"type": "end_kyoku"},
            ],
        ),
    ],
)
def test_xmodel1_rust_export_matches_python_oracle_on_terminal_edge_cases(tmp_path, name, events):
    _assert_rust_matches_python_oracle(
        tmp_path,
        events,
        replay_id=f"{name}.mjson",
        compare_targets=True,
    )


def test_xmodel1_rust_export_emits_nonzero_round_targets(tmp_path):
    events = [
        {"type": "start_game", "names": ["A", "B", "C", "D"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
                ["1s", "1s", "1s", "1s", "2s", "2s", "2s", "2s", "3s", "3s", "3s", "3s", "4s"],
                ["4m", "4m", "4m", "5m", "5m", "5m", "6m", "6m", "6m", "7m", "7m", "7m", "8m"],
                ["4p", "4p", "4p", "5p", "5p", "5p", "6p", "6p", "6p", "7p", "7p", "7p", "8p"],
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "4p"},
        {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
        {"type": "hora", "actor": 0, "target": 0, "pai": "4p", "deltas": [12000, -4000, -4000, -4000], "scores": [37000, 21000, 21000, 21000]},
        {"type": "end_kyoku"},
    ]
    with _export_rust_arrays(tmp_path, events) as data:
        assert np.allclose(data["score_delta_target"], np.array([0.4], dtype=np.float32))
        assert np.allclose(data["win_target"], np.array([1.0], dtype=np.float32))
        assert np.allclose(data["dealin_target"], np.array([0.0], dtype=np.float32))
        assert np.allclose(data["pts_given_win_target"], np.array([0.4], dtype=np.float32))
        assert np.allclose(data["pts_given_dealin_target"], np.array([0.0], dtype=np.float32))


def test_xmodel1_rust_export_emits_dealin_target(tmp_path):
    events = [
        {"type": "start_game", "names": ["A", "B", "C", "D"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
                ["4p", "4p", "4p", "5p", "5p", "6p", "6p", "7p", "7p", "8p", "8p", "9p", "9p"],
                ["2s"] * 13,
                ["3s"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "4p"},
        {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
        {"type": "hora", "actor": 1, "target": 0, "pai": "4p", "deltas": [-8000, 8000, 0, 0], "scores": [17000, 33000, 25000, 25000]},
        {"type": "end_kyoku"},
    ]
    with _export_rust_arrays(tmp_path, events) as data:
        discard_rows = data["sample_type"] == 0
        hora_rows = data["sample_type"] == XMODEL1_SAMPLE_TYPE_HORA
        assert int(discard_rows.sum()) == 1
        assert int(hora_rows.sum()) == 1
        assert np.allclose(data["score_delta_target"][discard_rows], np.array([-8000 / 30000], dtype=np.float32))
        assert np.allclose(data["win_target"][discard_rows], np.array([0.0], dtype=np.float32))
        assert np.allclose(data["dealin_target"][discard_rows], np.array([1.0], dtype=np.float32))
        assert np.allclose(data["pts_given_win_target"][discard_rows], np.array([0.0], dtype=np.float32))
        assert np.allclose(
            data["pts_given_dealin_target"][discard_rows],
            np.array([8000 / 30000], dtype=np.float32),
        )
        assert np.allclose(data["win_target"][hora_rows], np.array([1.0], dtype=np.float32))


def test_xmodel1_rust_export_emits_opp_tenpai_target_with_correct_shape(tmp_path):
    """Stage 2 Rust 迁移:Rust export 必须写出 opp_tenpai_target.npy 并与
    Python reference 在决策时刻 shanten ≤ 0 的定义上一致。

    fixture 构造:actor=0 弃 4p 时,对手 1 手牌 123456789s + 1234p 是清一色+纯连
    形,听牌 wait 4p (shanten=0);对手 2/3 手牌是同花堆不可能 tenpai。期望
    opp_tenpai_target = [1.0, 0.0, 0.0] (顺序 = 下家/对家/上家)。
    """
    events = [
        {"type": "start_game", "names": ["A", "B", "C", "D"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
                ["1s", "2s", "3s", "4s", "5s", "6s", "7s", "8s", "9s", "1p", "2p", "3p", "4p"],
                ["2s"] * 13,
                ["3s"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "4p"},
        {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
    ]
    with _export_rust_arrays(tmp_path, events) as data:
        assert "opp_tenpai_target" in data.files
        arr = data["opp_tenpai_target"]
        assert arr.shape == (1, 3), arr.shape
        assert arr.dtype == np.float32, arr.dtype
        # 对手 1 (1-9s + 1234p) 听牌 wait 4p → 1.0;对手 2/3 非 tenpai → 0.0
        assert np.array_equal(arr, np.array([[1.0, 0.0, 0.0]], dtype=np.float32)), arr


def test_xmodel1_rust_export_opp_tenpai_matches_python_oracle_on_non_tenpai_fixture(tmp_path):
    """Rust vs Python reference 对拍:对手都是不听牌的分散牌型,opp_tenpai
    应当三家全 0.0。同时用 Python `events_to_xmodel1_arrays` 做 cross check。"""
    from xmodel1.preprocess import events_to_xmodel1_arrays

    events = [
        {"type": "start_game", "names": ["A", "B", "C", "D"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
                ["1m", "4m", "7m", "1p", "4p", "7p", "1s", "4s", "7s", "E", "S", "W", "N"],
                ["2m", "5m", "8m", "2p", "5p", "8p", "2s", "5s", "8s", "P", "F", "C", "E"],
                ["3m", "6m", "9m", "3p", "6p", "9p", "3s", "6s", "9s", "S", "W", "N", "P"],
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "4p"},
        {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
    ]
    py_arrays = events_to_xmodel1_arrays(events, replay_id="non_tenpai.mjson")
    assert py_arrays is not None
    with _export_rust_arrays(tmp_path, events) as rust_arrays:
        py_mask = py_arrays["sample_type"] == 0
        rust_mask = rust_arrays["sample_type"] == 0
        rust_opp = rust_arrays["opp_tenpai_target"][rust_mask]
        py_opp = py_arrays["opp_tenpai_target"][py_mask]
        assert rust_opp.shape == py_opp.shape, (rust_opp.shape, py_opp.shape)
        assert np.array_equal(rust_opp, py_opp), (rust_opp, py_opp)
        assert np.all(rust_opp == 0.0)


def test_xmodel1_rust_export_emits_ryukyoku_score_delta_target(tmp_path):
    events = [
        {"type": "start_game", "names": ["A", "B", "C", "D"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
                ["1s"] * 13,
                ["2s"] * 13,
                ["3s"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "4p"},
        {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
        {"type": "ryukyoku", "tenpai_players": [0], "deltas": [1500, -500, -500, -500], "scores": [26500, 24500, 24500, 24500]},
        {"type": "end_kyoku"},
    ]
    with _export_rust_arrays(tmp_path, events) as data:
        assert np.allclose(data["score_delta_target"], np.array([1500 / 30000], dtype=np.float32))
        assert np.allclose(data["win_target"], np.array([0.0], dtype=np.float32))
        assert np.allclose(data["dealin_target"], np.array([0.0], dtype=np.float32))
        assert np.allclose(data["pts_given_win_target"], np.array([0.0], dtype=np.float32))
        assert np.allclose(data["pts_given_dealin_target"], np.array([0.0], dtype=np.float32))
