from __future__ import annotations

import importlib.util
from pathlib import Path


_SELFPLAY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "selfplay.py"
_SPEC = importlib.util.spec_from_file_location("selfplay_module", _SELFPLAY_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_resolve_seat_bots = _MODULE._resolve_seat_bots


def test_resolve_seat_bots_supports_rulebase_mix():
    sources, labels, kinds = _resolve_seat_bots(
        "keqingv2",
        ["keqingv2", "rulebase", "rulebase", "keqingv1"],
        None,
        None,
    )

    assert sources[1] == "rulebase"
    assert labels[1] == "rulebase"
    assert kinds == ["keqingv2", "rulebase", "rulebase", "keqingv1"]
