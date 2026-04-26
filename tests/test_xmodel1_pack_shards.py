from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from tests.xmodel1_test_utils import write_xmodel1_v3_npz
REPO_ROOT = "/media/bailan/DISK1/AUbuntuProject/project/keqing1"


def _write_sample_npz(path: Path, n: int) -> None:
    write_xmodel1_v3_npz(path, n=n, state_scalar_dim=56)


def test_pack_xmodel1_shards_merges_small_npz_files(tmp_path: Path):
    input_root = tmp_path / "processed_xmodel1"
    (input_root / "ds1").mkdir(parents=True)
    _write_sample_npz(input_root / "ds1" / "a.npz", 3)
    _write_sample_npz(input_root / "ds1" / "b.npz", 4)
    _write_sample_npz(input_root / "ds1" / "c.npz", 5)
    output_root = tmp_path / "packed"

    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/pack_xmodel1_shards.py",
            "--input_dir",
            str(input_root),
            "--output_dir",
            str(output_root),
            "--max-samples-per-shard",
            "7",
            "--max-files-per-shard",
            "2",
        ],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert "xmodel1 shard pack:" in result.stdout
    shard_files = sorted((output_root / "ds1").glob("*.npz"))
    assert len(shard_files) == 2
    with np.load(shard_files[0], allow_pickle=False) as data0:
        assert int(data0["state_tile_feat"].shape[0]) == 7
    with np.load(shard_files[1], allow_pickle=False) as data1:
        assert int(data1["state_tile_feat"].shape[0]) == 5
    manifest = json.loads((output_root / "xmodel1_export_manifest.json").read_text(encoding="utf-8"))
    assert manifest["export_mode"] == "xmodel1_packed_shards"
    assert manifest["exported_file_count"] == 2
    assert manifest["exported_sample_count"] == 12
    assert manifest["shard_file_counts"] == {"ds1": 2}
    assert manifest["shard_sample_counts"] == {"ds1": 12}
