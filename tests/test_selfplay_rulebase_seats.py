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


def test_resolve_seat_bots_supports_rulebase_mix(tmp_path, monkeypatch):
    for model_name in ("xmodel1", "keqingv4"):
        model_dir = tmp_path / model_name
        model_dir.mkdir(parents=True)
        (model_dir / "best.pth").write_bytes(b"checkpoint")
    monkeypatch.setattr(_MODULE, "MODEL_ROOT", tmp_path)

    sources, labels, kinds = _resolve_seat_bots(
        "xmodel1",
        ["xmodel1", "rulebase", "rulebase", "keqingv4"],
        None,
        None,
    )

    assert sources[1] == "rulebase"
    assert labels[1] == "rulebase"
    assert kinds == ["xmodel1", "rulebase", "rulebase", "keqingv4"]


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
