from __future__ import annotations

from pathlib import Path

import numpy as np

from xmodel1.cached_dataset import split_cached_files
from xmodel1.model import Xmodel1Model
from xmodel1.trainer import build_dataloaders, train


def _write_fixture(path: Path, n: int = 16) -> None:
    k = 14
    d = 21
    f = 10
    candidate_mask = np.zeros((n, k), dtype=np.uint8)
    candidate_mask[:, :4] = 1
    np.savez(
        path,
        state_tile_feat=np.random.randn(n, 57, 34).astype(np.float16),
        state_scalar=np.random.randn(n, 64).astype(np.float16),
        candidate_feat=np.random.randn(n, k, d).astype(np.float16),
        candidate_tile_id=np.where(candidate_mask > 0, np.arange(k), -1).astype(np.int16),
        candidate_mask=candidate_mask,
        candidate_flags=np.zeros((n, k, f), dtype=np.uint8),
        chosen_candidate_idx=np.zeros((n,), dtype=np.int16),
        candidate_quality_score=np.random.randn(n, k).astype(np.float32),
        candidate_rank_bucket=np.zeros((n, k), dtype=np.int8),
        candidate_hard_bad_flag=np.zeros((n, k), dtype=np.uint8),
        global_value_target=np.zeros((n,), dtype=np.float32),
        score_delta_target=np.zeros((n,), dtype=np.float32),
        win_target=np.zeros((n,), dtype=np.float32),
        dealin_target=np.zeros((n,), dtype=np.float32),
        offense_quality_target=np.zeros((n,), dtype=np.float32),
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
        candidate_feature_dim=21,
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
    assert (out_dir / "train_log.jsonl").exists()
