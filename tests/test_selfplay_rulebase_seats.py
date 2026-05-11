from __future__ import annotations

import importlib.util
from pathlib import Path


_SELFPLAY_PATH = Path(__file__).resolve().parents[1] / "scripts" / "selfplay.py"
_SPEC = importlib.util.spec_from_file_location("selfplay_module", _SELFPLAY_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
_resolve_seat_bots = _MODULE._resolve_seat_bots
_resolve_model_path = _MODULE._resolve_model_path
_canonicalize_action_for_server = _MODULE._canonicalize_action_for_server


def test_resolve_model_path_supports_mortal_default(tmp_path, monkeypatch):
    ckpt = tmp_path / "mortal.pth"
    ckpt.write_bytes(b"checkpoint")
    monkeypatch.setattr(_MODULE, "MORTAL_DEFAULT_MODEL", ckpt)

    assert _resolve_model_path("mortal") == str(ckpt)


def test_resolve_seat_bots_supports_mortal_mix(tmp_path, monkeypatch):
    ckpt = tmp_path / "mortal.pth"
    ckpt.write_bytes(b"checkpoint")
    monkeypatch.setattr(_MODULE, "MORTAL_DEFAULT_MODEL", ckpt)

    sources, labels, kinds = _resolve_seat_bots(
        "mortal",
        ["mortal", "rulebase", "mortal", "rulebase"],
        None,
        None,
    )

    assert sources == [str(ckpt), "rulebase", str(ckpt), "rulebase"]
    assert labels == ["mortal", "rulebase", "mortal", "rulebase"]
    assert kinds == ["mortal", "rulebase", "mortal", "rulebase"]


def test_resolve_seat_bots_uses_explicit_seat_models_for_mortal(tmp_path, monkeypatch):
    default_ckpt = tmp_path / "default_mortal.pth"
    default_ckpt.write_bytes(b"default")
    custom = [tmp_path / f"seat_{idx}.pth" for idx in range(4)]
    for path in custom:
        path.write_bytes(b"seat")
    monkeypatch.setattr(_MODULE, "MORTAL_DEFAULT_MODEL", default_ckpt)

    sources, labels, kinds = _resolve_seat_bots(
        "mortal",
        ["mortal", "mortal", "mortal", "mortal"],
        [str(path) for path in custom],
        None,
    )

    assert sources == [str(path) for path in custom]
    assert labels == [path.stem for path in custom]
    assert kinds == ["mortal", "mortal", "mortal", "mortal"]


def test_canonicalize_hora_fills_missing_pai_from_legal(monkeypatch):
    class _LegalAction:
        type = "hora"
        target = 0
        pai = "6m"

    class _Manager:
        def get_snap_with_shanten(self, room, actor):
            return {"actor": actor}

    monkeypatch.setattr(_MODULE, "enumerate_legal_actions", lambda snap, actor: [_LegalAction()])

    action = _canonicalize_action_for_server(
        _Manager(),
        object(),
        0,
        {"type": "hora", "actor": 0, "target": 0},
    )

    assert action == {
        "type": "hora",
        "actor": 0,
        "target": 0,
        "pai": "6m",
        "is_tsumo": True,
    }
