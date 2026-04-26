from __future__ import annotations

from pathlib import Path

import numpy as np

from training.cache_schema import XMODEL1_CANDIDATE_FEATURE_DIM, XMODEL1_CANDIDATE_FLAG_DIM
from tests.xmodel1_test_utils import write_xmodel1_v3_npz
from xmodel1.model import Xmodel1Model
from xmodel1.cached_dataset import Xmodel1DiscardDataset
from xmodel1.trainer import train


def _write_sample_npz(path: Path) -> None:
    payload = write_xmodel1_v3_npz(path, n=20)
    payload["state_tile_feat"][:] = np.random.randn(20, 57, 34).astype(np.float16)
    payload["state_scalar"][:] = np.random.randn(20, 64).astype(np.float16)
    payload["candidate_feat"][:] = np.random.randn(20, 14, XMODEL1_CANDIDATE_FEATURE_DIM).astype(np.float16)
    payload["candidate_tile_id"][:, 0] = 1
    payload["candidate_tile_id"][:, 1] = 2
    payload["candidate_tile_id"][:, 2] = 3
    np.savez(path, **payload)


def test_xmodel1_train_smoke(tmp_path: Path):
    train_file = tmp_path / "train.npz"
    val_file = tmp_path / "val.npz"
    _write_sample_npz(train_file)
    _write_sample_npz(val_file)

    val_loader = __import__("torch").utils.data.DataLoader(
        Xmodel1DiscardDataset([val_file], shuffle=False, buffer_size=8, seed=1),
        batch_size=4,
        collate_fn=Xmodel1DiscardDataset.collate,
        num_workers=0,
    )
    train_loader = __import__("torch").utils.data.DataLoader(
        Xmodel1DiscardDataset([train_file], shuffle=True, buffer_size=8, seed=1),
        batch_size=4,
        collate_fn=Xmodel1DiscardDataset.collate,
        num_workers=0,
    )
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=64,
        candidate_feature_dim=XMODEL1_CANDIDATE_FEATURE_DIM,
        candidate_flag_dim=XMODEL1_CANDIDATE_FLAG_DIM,
        hidden_dim=64,
        num_res_blocks=2,
        dropout=0.0,
    )
    cfg = {
        "num_epochs": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "warmup_steps": 1,
        "accumulation_steps": 1,
        "steps_per_epoch": 2,
        "val_steps_per_epoch": 1,
        "log_interval": 1,
    }
    output_dir = tmp_path / "artifacts"
    trained = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=output_dir,
        device_str="cpu",
    )
    assert trained is not None
    assert (output_dir / "last.pth").exists()
    assert (output_dir / "best.pth").exists()
    rows = [
        __import__("json").loads(line)
        for line in (output_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows[-1]["step"] == 2
    assert rows[-1]["train"]["num_batches"] == 2
    assert rows[-1]["val"]["num_batches"] == 1
