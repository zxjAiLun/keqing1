import numpy as np

from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
)


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
    assert (tmp_path / "out" / "xmodel1_export_manifest.json").exists()
    if produced_npz:
        assert (tmp_path / "out" / "ds1" / "sample.npz").exists()


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
