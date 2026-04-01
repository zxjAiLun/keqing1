import importlib.util
import json
from pathlib import Path

import numpy as np


def _load_selfplay_module():
    module_path = Path(__file__).resolve().parents[1] / "scripts" / "selfplay.py"
    spec = importlib.util.spec_from_file_location("test_selfplay_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_selfplay_events_to_npz_delegates_to_training_preprocess(monkeypatch):
    selfplay = _load_selfplay_module()
    captured = {}
    fake_arrays = {
        "tile_feat": np.zeros((1, 54, 34), dtype=np.float16),
        "scalar": np.zeros((1, 48), dtype=np.float16),
        "mask": np.zeros((1, 47), dtype=np.uint8),
        "action_idx": np.array([0], dtype=np.int16),
        "value": np.array([0.0], dtype=np.float16),
    }

    def fake_events_to_cached_arrays(
        events,
        *,
        actor_name_filter=None,
        adapter=None,
        value_strategy="heuristic",
        encode_module="keqingv1.features",
    ):
        captured["events"] = events
        captured["actor_name_filter"] = actor_name_filter
        captured["adapter_type"] = type(adapter).__name__
        captured["value_strategy"] = value_strategy
        captured["encode_module"] = encode_module
        return fake_arrays

    def fake_create_preprocess_adapter(name):
        captured["requested_adapter_name"] = name
        class FakeAdapter:
            pass
        return FakeAdapter()

    monkeypatch.setattr(
        "training.preprocess.create_preprocess_adapter",
        fake_create_preprocess_adapter,
    )
    monkeypatch.setattr(
        "training.preprocess.events_to_cached_arrays",
        fake_events_to_cached_arrays,
    )

    events = [{"type": "start_game", "names": ["p0", "p1", "p2", "p3"]}]
    result = selfplay._events_to_npz(
        events,
        adapter_name="meld_rank",
        value_strategy="mc_return",
        encode_module="keqingv3.features",
    )

    assert result is fake_arrays
    assert captured == {
        "events": events,
        "actor_name_filter": None,
        "requested_adapter_name": "meld_rank",
        "adapter_type": "FakeAdapter",
        "value_strategy": "mc_return",
        "encode_module": "keqingv3.features",
    }


def test_selfplay_save_npz_delegates_to_preprocess_writer(tmp_path, monkeypatch):
    selfplay = _load_selfplay_module()
    path = tmp_path / "sample.npz"
    events = [{"type": "start_game", "names": ["p0", "p1", "p2", "p3"]}]
    captured = {}

    def fake_save_events_to_cache_file(
        out_path,
        event_stream,
        *,
        actor_name_filter=None,
        adapter=None,
        adapter_name="base",
        value_strategy="heuristic",
        encode_module="keqingv1.features",
    ):
        captured["out_path"] = out_path
        captured["events"] = event_stream
        captured["actor_name_filter"] = actor_name_filter
        captured["adapter"] = adapter
        captured["adapter_name"] = adapter_name
        captured["value_strategy"] = value_strategy
        captured["encode_module"] = encode_module
        return True

    monkeypatch.setattr(
        "training.preprocess.save_events_to_cache_file",
        fake_save_events_to_cache_file,
    )

    ok = selfplay._save_npz(
        path,
        events,
        adapter_name="v3_aux",
        value_strategy="mc_return",
        encode_module="keqingv3.features",
    )

    assert ok is True
    assert captured == {
        "out_path": path,
        "events": events,
        "actor_name_filter": None,
        "adapter": None,
        "adapter_name": "v3_aux",
        "value_strategy": "mc_return",
        "encode_module": "keqingv3.features",
    }


def test_build_cache_export_metadata_marks_legacy_wrapper(tmp_path):
    selfplay = _load_selfplay_module()
    npz_dir = tmp_path / "benchmark" / "npz"

    metadata = selfplay._build_cache_export_metadata(
        adapter_name="v3_aux",
        value_strategy="mc_return",
        encode_module="keqingv3.features",
        saved_games=7,
        output_dir=npz_dir,
    )

    assert metadata["format"] == "preprocess_cache_wrapper"
    assert metadata["owner"] == "training.preprocess"
    assert metadata["status"] == "legacy_convenience_export"
    assert metadata["adapter_name"] == "v3_aux"
    assert metadata["value_strategy"] == "mc_return"
    assert metadata["encode_module"] == "keqingv3.features"
    assert metadata["saved_games"] == 7
    assert metadata["recommended_flow"][0] == "运行 selfplay 保存 .mjson"
    assert metadata["recommended_replays_dir"] == str(tmp_path / "benchmark" / "replays")
    assert metadata["recommended_preprocess_output_dir"] == str(
        tmp_path / "benchmark" / "preprocessed_v3_aux"
    )
    assert "--data_dirs" in metadata["recommended_preprocess_command"]
    assert "create_preprocess_adapter('v3_aux')" in metadata["recommended_preprocess_command"]
    assert str(tmp_path / "benchmark" / "replays") in metadata["recommended_preprocess_command"]


def test_build_recommended_preprocess_command_uses_replays_dir_and_adapter(tmp_path):
    selfplay = _load_selfplay_module()

    command = selfplay._build_recommended_preprocess_command(tmp_path / "bench_001", "meld_rank")

    assert "run_preprocess" in command
    assert "create_preprocess_adapter('meld_rank')" in command
    assert f"--data_dirs {tmp_path / 'bench_001' / 'replays'}" in command


def test_write_cache_export_metadata_writes_json_file(tmp_path):
    selfplay = _load_selfplay_module()
    npz_dir = tmp_path / "npz"
    npz_dir.mkdir()
    metadata = {
        "format": "preprocess_cache_wrapper",
        "owner": "training.preprocess",
        "status": "legacy_convenience_export",
    }

    selfplay._write_cache_export_metadata(npz_dir, metadata)

    written = json.loads((npz_dir / "metadata.json").read_text(encoding="utf-8"))
    assert written == metadata
