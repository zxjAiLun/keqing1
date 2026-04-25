from __future__ import annotations

import pytest

import keqing_core
from keqingrl.env import DiscardOnlyMahjongEnv


_REQUIRED_NATIVE_BOUNDARY_SYMBOLS = (
    "native_schema_info_json_py",
    "action_identity_json_py",
    "decode_action_id_json_py",
    "mjai_events_for_action_json_py",
    "resolve_terminal_action_json_py",
    "validate_replay_state_snapshot_json_py",
    "build_keqingrl_action_features_py",
    "build_keqingrl_action_features_typed_py",
    "keqingrl_action_feature_dim_py",
    "fixed_seed_eval_gate_json_py",
)


def test_native_boundary_required_symbols_exist() -> None:
    previous = keqing_core.is_enabled()
    keqing_core.enable_rust(True)
    try:
        info = keqing_core.require_native_schema()
        native_module = getattr(keqing_core, "_rust_ext", None)
    finally:
        keqing_core.enable_rust(previous)

    assert info["schema_name"] == "keqingrl_native_boundary"
    assert native_module is not None
    missing = [
        symbol
        for symbol in _REQUIRED_NATIVE_BOUNDARY_SYMBOLS
        if not callable(getattr(native_module, symbol, None))
    ]
    assert missing == []


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
