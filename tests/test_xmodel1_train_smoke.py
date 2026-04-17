from __future__ import annotations

from pathlib import Path

import numpy as np

from xmodel1.model import Xmodel1Model
from xmodel1.cached_dataset import Xmodel1DiscardDataset
from xmodel1.trainer import train


def _write_sample_npz(path: Path) -> None:
    n = 8
    candidate_mask = np.concatenate(
        [np.ones((n, 3), dtype=np.uint8), np.zeros((n, 11), dtype=np.uint8)],
        axis=1,
    )
    candidate_tile_id = np.full((n, 14), -1, dtype=np.int16)
    candidate_tile_id[:, 0] = 1
    candidate_tile_id[:, 1] = 2
    candidate_tile_id[:, 2] = 3
    np.savez(
        path,
        schema_name=np.array("xmodel1_discard_v2", dtype=np.str_),
        schema_version=np.array(2, dtype=np.int32),
        state_tile_feat=np.random.randn(n, 57, 34).astype(np.float16),
        state_scalar=np.random.randn(n, 64).astype(np.float16),
        candidate_feat=np.random.randn(n, 14, 35).astype(np.float16),
        candidate_tile_id=candidate_tile_id,
        candidate_mask=candidate_mask,
        candidate_flags=np.zeros((n, 14, 10), dtype=np.uint8),
        chosen_candidate_idx=np.zeros((n,), dtype=np.int16),
        action_idx_target=np.zeros((n,), dtype=np.int16),
        candidate_quality_score=np.zeros((n, 14), dtype=np.float32),
        candidate_rank_bucket=np.zeros((n, 14), dtype=np.int8),
        candidate_hard_bad_flag=np.zeros((n, 14), dtype=np.uint8),
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
        event_index=np.zeros((n,), dtype=np.int32),
        kyoku=np.ones((n,), dtype=np.int8),
        honba=np.zeros((n,), dtype=np.int8),
        is_open_hand=np.zeros((n,), dtype=np.uint8),
    )


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
        candidate_feature_dim=35,
        candidate_flag_dim=10,
        hidden_dim=64,
        num_res_blocks=2,
        dropout=0.0,
    )
    cfg = {
        "num_epochs": 1,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
        "warmup_steps": 1,
        "steps_per_epoch": 2,
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
