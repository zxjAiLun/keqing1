from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from xmodel1.cached_dataset import split_cached_files
from xmodel1.model import Xmodel1Model
from xmodel1.trainer import _autocast_enabled, _unpack_batch, build_dataloaders, train


def _write_fixture(path: Path, n: int = 16) -> None:
    k = 14
    d = 35
    f = 10
    candidate_mask = np.zeros((n, k), dtype=np.uint8)
    candidate_mask[:, :4] = 1
    np.savez(
        path,
        schema_name=np.array("xmodel1_discard_v2", dtype=np.str_),
        schema_version=np.array(2, dtype=np.int32),
        state_tile_feat=np.random.randn(n, 57, 34).astype(np.float16),
        state_scalar=np.random.randn(n, 64).astype(np.float16),
        candidate_feat=np.random.randn(n, k, d).astype(np.float16),
        candidate_tile_id=np.where(candidate_mask > 0, np.arange(k), -1).astype(np.int16),
        candidate_mask=candidate_mask,
        candidate_flags=np.zeros((n, k, f), dtype=np.uint8),
        chosen_candidate_idx=np.zeros((n,), dtype=np.int16),
        action_idx_target=np.zeros((n,), dtype=np.int16),
        candidate_quality_score=np.random.randn(n, k).astype(np.float32),
        candidate_rank_bucket=np.zeros((n, k), dtype=np.int8),
        candidate_hard_bad_flag=np.zeros((n, k), dtype=np.uint8),
        special_candidate_feat=np.zeros((n, 12, 25), dtype=np.float16),
        special_candidate_type_id=np.full((n, 12), -1, dtype=np.int16),
        special_candidate_mask=np.zeros((n, 12), dtype=np.uint8),
        special_candidate_quality_score=np.zeros((n, 12), dtype=np.float32),
        special_candidate_rank_bucket=np.zeros((n, 12), dtype=np.int8),
        special_candidate_hard_bad_flag=np.zeros((n, 12), dtype=np.uint8),
        chosen_special_candidate_idx=np.full((n,), -1, dtype=np.int16),
        score_delta_target=np.zeros((n,), dtype=np.float32),
        win_target=np.zeros((n,), dtype=np.float32),
        dealin_target=np.zeros((n,), dtype=np.float32),
        pts_given_win_target=np.zeros((n,), dtype=np.float32),
        pts_given_dealin_target=np.zeros((n,), dtype=np.float32),
        opp_tenpai_target=np.zeros((n, 3), dtype=np.float32),
        event_history=np.zeros((n, 48, 5), dtype=np.int16),
        sample_type=np.zeros((n,), dtype=np.int8),
        actor=np.zeros((n,), dtype=np.int8),
        event_index=np.arange(n, dtype=np.int32),
        kyoku=np.ones((n,), dtype=np.int8),
        honba=np.zeros((n,), dtype=np.int8),
        is_open_hand=np.zeros((n,), dtype=np.uint8),
    )


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
        candidate_feature_dim=35,
        candidate_flag_dim=10,
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
    assert "special_rank" in rows[-1]["train"]
    assert "reach_pair" in rows[-1]["train"]
    assert "call_pair" in rows[-1]["train"]
    assert "action_acc" in rows[-1]["val"]


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
        candidate_feature_dim=35,
        candidate_flag_dim=10,
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
