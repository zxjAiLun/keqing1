from __future__ import annotations

import pytest

import keqing_core
from keqingrl.env import DiscardOnlyMahjongEnv


def test_native_schema_info_contract_is_strict() -> None:
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")
    previous = keqing_core.is_enabled()
    keqing_core.enable_rust(True)
    try:
        info = keqing_core.require_native_schema()
    finally:
        keqing_core.enable_rust(previous)

    assert info["schema_name"] == "keqingrl_native_boundary"
    assert info["schema_version"] == 1
    assert info["capabilities"]["action_identity_v1"] == "supported"
    assert info["capabilities"]["legal_enumeration_v1"] == "supported"
    assert info["capabilities"]["terminal_resolver_v1"] == "supported"
    assert info["capabilities"]["state_apply_validation_v1"] == "supported"
    assert info["capabilities"]["action_feature_parity_v1"] == "supported"
    assert info["capabilities"]["fixed_seed_eval_gate_v1"] == "supported"
    assert info["capabilities"]["typed_action_feature_api_v1"] == "supported"
    assert info["capabilities"]["nuki"] == "unsupported"
    assert info["runtime_fallback_policy"] == "fail_closed_no_silent_python_fallback"


def test_native_schema_strict_check_rejects_version_mismatch() -> None:
    if not keqing_core.is_available():
        pytest.skip("keqing_core native module is not available")
    previous = keqing_core.is_enabled()
    keqing_core.enable_rust(True)
    try:
        with pytest.raises(RuntimeError, match="schema_version"):
            keqing_core.require_native_schema(schema_version=999)
    finally:
        keqing_core.enable_rust(previous)


def test_keqingrl_env_requires_native_schema(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        keqing_core,
        "require_native_schema",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("native schema unavailable")),
    )

    with pytest.raises(RuntimeError, match="native schema unavailable"):
        DiscardOnlyMahjongEnv()
