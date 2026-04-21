from __future__ import annotations

import pytest

from training.cache_schema import (
    XMODEL1_CANDIDATE_FEATURE_DIM,
    XMODEL1_CANDIDATE_FLAG_DIM,
    XMODEL1_HISTORY_SUMMARY_DIM,
    XMODEL1_MAX_CANDIDATES,
    XMODEL1_MAX_RESPONSE_CANDIDATES,
    XMODEL1_SCHEMA_NAME,
    XMODEL1_SCHEMA_VERSION,
)


def test_xmodel1_python_schema_constants_are_stable():
    assert XMODEL1_SCHEMA_NAME == "xmodel1_discard_v6"
    assert XMODEL1_SCHEMA_VERSION == 6
    assert XMODEL1_MAX_CANDIDATES == 14
    assert XMODEL1_CANDIDATE_FEATURE_DIM == 22
    assert XMODEL1_CANDIDATE_FLAG_DIM == 8
    assert XMODEL1_HISTORY_SUMMARY_DIM == 20
    assert XMODEL1_MAX_RESPONSE_CANDIDATES == 8


def test_xmodel1_rust_schema_info_matches_python_constants():
    keqing_core = pytest.importorskip("keqing_core")
    name, version, max_candidates, candidate_dim, flag_dim = keqing_core.xmodel1_schema_info()
    assert name == XMODEL1_SCHEMA_NAME
    assert version == XMODEL1_SCHEMA_VERSION
    assert max_candidates == XMODEL1_MAX_CANDIDATES
    assert candidate_dim == XMODEL1_CANDIDATE_FEATURE_DIM
    assert flag_dim == XMODEL1_CANDIDATE_FLAG_DIM


def test_xmodel1_rust_validate_accepts_basic_valid_record():
    keqing_core = pytest.importorskip("keqing_core")
    candidate_mask = [1, 1] + [0] * (XMODEL1_MAX_CANDIDATES - 2)
    candidate_tile_id = [4, 27] + [-1] * (XMODEL1_MAX_CANDIDATES - 2)
    assert keqing_core.validate_xmodel1_discard_record(0, candidate_mask, candidate_tile_id) is True


def test_xmodel1_rust_validate_rejects_invalid_padding_tile_id():
    keqing_core = pytest.importorskip("keqing_core")
    candidate_mask = [1] + [0] * (XMODEL1_MAX_CANDIDATES - 1)
    candidate_tile_id = [4, 3] + [-1] * (XMODEL1_MAX_CANDIDATES - 2)
    with pytest.raises(ValueError):
        keqing_core.validate_xmodel1_discard_record(0, candidate_mask, candidate_tile_id)
