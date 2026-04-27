from __future__ import annotations

import gzip
from pathlib import Path

from scripts.prepare_mortal_training import (
    MORTAL_TRAINING_DATASET_CONTRACT_VERSION,
    prepare_mortal_training,
    split_relative_paths,
)


def _write_mjson(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_prepare_mortal_training_gzips_dataset_and_writes_config(tmp_path: Path) -> None:
    source_dir = tmp_path / "converted"
    _write_mjson(source_dir / "a" / "one.mjson", '{"type":"start_game"}\n')
    _write_mjson(source_dir / "a" / "two.mjson", '{"type":"start_game","id":2}\n')
    _write_mjson(source_dir / "b" / "three.mjson", '{"type":"start_game","id":3}\n')
    _write_mjson(source_dir / "b" / "four.mjson", '{"type":"start_game","id":4}\n')

    manifest = prepare_mortal_training(
        source_dir=source_dir,
        output_dir=tmp_path / "mortal_gz",
        training_dir=tmp_path / "mortal_training",
        val_ratio=0.25,
        seed=7,
        device="cpu",
    )

    assert manifest["contract_version"] == MORTAL_TRAINING_DATASET_CONTRACT_VERSION
    assert manifest["source_count"] == 4
    assert manifest["train_count"] == 3
    assert manifest["val_count"] == 1
    assert Path(manifest["config_path"]).exists()
    assert Path(manifest["manifest_path"]).exists()

    targets = [tmp_path / "mortal_gz" / item["target"] for item in manifest["files"]]
    assert len(targets) == 4
    assert all(path.suffixes[-2:] == [".json", ".gz"] for path in targets)
    with gzip.open(targets[0], "rt", encoding="utf-8") as handle:
        assert handle.readline().startswith('{"type":"start_game"')

    config_text = Path(manifest["config_path"]).read_text(encoding="utf-8")
    assert "Mortal is the only allowed teacher source" in config_text
    assert "mortal_baseline_required.pth" in config_text
    assert "/train/**/*.json.gz" in config_text
    assert "/val/**/*.json.gz" in config_text


def test_prepare_mortal_training_dry_run_does_not_write_files(tmp_path: Path) -> None:
    source_dir = tmp_path / "converted"
    _write_mjson(source_dir / "one.mjson", '{"type":"start_game"}\n')
    _write_mjson(source_dir / "two.mjson", '{"type":"start_game","id":2}\n')

    manifest = prepare_mortal_training(
        source_dir=source_dir,
        output_dir=tmp_path / "mortal_gz",
        training_dir=tmp_path / "mortal_training",
        val_ratio=0.5,
        dry_run=True,
    )

    assert manifest["dry_run"] is True
    assert "config_text" in manifest
    assert not (tmp_path / "mortal_gz").exists()
    assert not Path(manifest["config_path"]).exists()


def test_split_relative_paths_keeps_at_least_one_train_file(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    files = [source_dir / f"{index}.mjson" for index in range(2)]
    for file in files:
        _write_mjson(file, "{}\n")

    train, val = split_relative_paths(files, source_dir=source_dir, val_ratio=0.99, seed=1)

    assert len(train) == 1
    assert len(val) == 1
