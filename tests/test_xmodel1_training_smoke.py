from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from training.cache_schema import XMODEL1_CANDIDATE_FEATURE_DIM, XMODEL1_CANDIDATE_FLAG_DIM
from tests.xmodel1_test_utils import write_xmodel1_v3_npz
from xmodel1.cached_dataset import split_cached_files
from xmodel1.model import Xmodel1Model
from xmodel1.trainer import _autocast_enabled, _unpack_batch, build_dataloaders, train


def _write_fixture(path: Path, n: int = 16) -> None:
    payload = write_xmodel1_v3_npz(path, n=n)
    payload["state_tile_feat"][:] = np.random.randn(n, 57, 34).astype(np.float16)
    payload["state_scalar"][:] = np.random.randn(n, 64).astype(np.float16)
    payload["candidate_feat"][:] = np.random.randn(n, 14, XMODEL1_CANDIDATE_FEATURE_DIM).astype(np.float16)
    payload["candidate_tile_id"][:] = np.where(payload["candidate_mask"] > 0, np.arange(14), -1).astype(np.int16)
    payload["candidate_quality_score"][:] = np.random.randn(n, 14).astype(np.float32)
    np.savez(path, **payload)


def test_xmodel1_training_smoke(tmp_path: Path):
    ds1 = tmp_path / "ds1"
    ds1.mkdir()
    _write_fixture(ds1 / "a.npz", n=16)
    _write_fixture(ds1 / "b.npz", n=16)
    train_files, val_files = split_cached_files([ds1], val_ratio=0.25, seed=7)
    train_loader, val_loader = build_dataloaders(
        train_files=train_files,
        val_files=val_files,
        batch_size=4,
        num_workers=0,
        buffer_size=32,
        seed=7,
    )
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=64,
        candidate_feature_dim=XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim=XMODEL1_CANDIDATE_FLAG_DIM,
        hidden_dim=32,
        num_res_blocks=1,
    )
    out_dir = tmp_path / "model_out"
    train(
        model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg={"num_epochs": 1},
        output_dir=out_dir,
        device_str="cpu",
    )
    assert (out_dir / "last.pth").exists()
    assert (out_dir / "best.pth").exists()
    assert (out_dir / "training_summary.json").exists()
    summary = json.loads((out_dir / "training_summary.json").read_text(encoding="utf-8"))
    assert summary["dataset_summary"] is None
    log_path = out_dir / "train_log.jsonl"
    assert log_path.exists()
    rows = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert rows
    assert "step" in rows[-1]
    assert "action_ce" in rows[-1]["train"]
    assert "response_ce" in rows[-1]["train"]
    assert "response_post_ce" in rows[-1]["train"]
    assert "response_post_rank" in rows[-1]["train"]
    assert "final_rank" in rows[-1]["train"]
    assert "final_score_delta" in rows[-1]["train"]
    assert "rank_pt" in rows[-1]["train"]
    assert "action_acc" in rows[-1]["val"]
    assert "final_rank_acc" in rows[-1]["val"]


def test_xmodel1_training_supports_epoch_file_sampling(tmp_path: Path):
    ds1 = tmp_path / "ds1"
    ds1.mkdir()
    for name in ["a.npz", "b.npz", "c.npz", "d.npz"]:
        _write_fixture(ds1 / name, n=8)
    train_files, val_files = split_cached_files([ds1], val_ratio=0.25, seed=7)
    chosen_subsets: list[tuple[str, ...]] = []

    def _make_loader(epoch: int):
        subset = tuple(sorted(path.name for path in train_files[:2])) if epoch == 0 else tuple(sorted(path.name for path in train_files[-2:]))
        chosen_subsets.append(subset)
        subset_paths = [path for path in train_files if path.name in subset]
        train_loader, _ = build_dataloaders(
            train_files=subset_paths,
            val_files=val_files,
            batch_size=4,
            num_workers=0,
            buffer_size=16,
            seed=7 + epoch,
        )
        return train_loader

    _, val_loader = build_dataloaders(
        train_files=train_files[:2],
        val_files=val_files,
        batch_size=4,
        num_workers=0,
        buffer_size=16,
        seed=7,
    )
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=64,
        candidate_feature_dim=XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim=XMODEL1_CANDIDATE_FLAG_DIM,
        hidden_dim=32,
        num_res_blocks=1,
    )
    out_dir = tmp_path / "model_out_sampling"
    train(
        model,
        train_loader=None,
        train_loader_factory=_make_loader,
        val_loader=val_loader,
        cfg={"num_epochs": 2, "steps_per_epoch": 2, "val_steps_per_epoch": 1, "log_interval": 1},
        output_dir=out_dir,
        device_str="cpu",
    )
    assert len(chosen_subsets) == 2
    assert chosen_subsets[0] != chosen_subsets[1]


def test_xmodel1_unpack_batch_promotes_fp16_on_cpu():
    batch = {
        "candidate_feat": torch.ones((2, 3), dtype=torch.float16),
        "chosen_candidate_idx": torch.tensor([0, 1], dtype=torch.int16),
    }
    out = _unpack_batch(batch, torch.device("cpu"))
    assert out["candidate_feat"].dtype == torch.float32
    assert out["chosen_candidate_idx"].dtype == torch.long


def test_xmodel1_unpack_batch_keeps_fp16_for_cuda_amp_path(monkeypatch):
    batch = {
        "candidate_feat": torch.ones((2, 3), dtype=torch.float16),
        "chosen_candidate_idx": torch.tensor([0, 1], dtype=torch.int16),
    }

    def _fake_to(self, device, non_blocking=True):
        return self

    monkeypatch.setattr(torch.Tensor, "to", _fake_to, raising=False)
    out = _unpack_batch(batch, torch.device("cuda"))
    assert out["candidate_feat"].dtype == torch.float16
    assert out["chosen_candidate_idx"].dtype == torch.long


def test_xmodel1_autocast_enabled_uses_cuda_for_val_without_scaler():
    class _DisabledScaler:
        def is_enabled(self):
            return False

    assert _autocast_enabled(device=torch.device("cpu"), is_train=False, scaler=None) is False
    assert _autocast_enabled(device=torch.device("cuda"), is_train=False, scaler=None) is True
    assert _autocast_enabled(device=torch.device("cuda"), is_train=True, scaler=_DisabledScaler()) is False
