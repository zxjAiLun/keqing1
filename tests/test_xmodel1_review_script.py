from __future__ import annotations

from pathlib import Path
import subprocess

import torch

from xmodel1.model import Xmodel1Model


def _write_sample_npz(path: Path) -> None:
    import numpy as np

    n = 2
    k = 14
    d = 35
    f = 10
    candidate_mask = np.zeros((n, k), dtype=np.uint8)
    candidate_mask[:, :3] = 1
    candidate_tile_id = np.full((n, k), -1, dtype=np.int16)
    candidate_tile_id[:, 0] = 0
    candidate_tile_id[:, 1] = 1
    candidate_tile_id[:, 2] = 27
    special_candidate_type_id = np.full((n, 12), -1, dtype=np.int16)
    special_candidate_type_id[1, 0] = 11
    special_candidate_mask = np.zeros((n, 12), dtype=np.uint8)
    special_candidate_mask[1, 0] = 1
    np.savez(
        path,
        schema_name=np.array("xmodel1_discard_v2", dtype=np.str_),
        schema_version=np.array(2, dtype=np.int32),
        state_tile_feat=np.random.randn(n, 57, 34).astype(np.float16),
        state_scalar=np.random.randn(n, 64).astype(np.float16),
        candidate_feat=np.random.randn(n, k, d).astype(np.float16),
        candidate_tile_id=candidate_tile_id,
        candidate_mask=candidate_mask,
        candidate_flags=np.zeros((n, k, f), dtype=np.uint8),
        chosen_candidate_idx=np.array([0, -1], dtype=np.int16),
        action_idx_target=np.array([0, 44], dtype=np.int16),
        candidate_quality_score=np.random.randn(n, k).astype(np.float32),
        candidate_rank_bucket=np.zeros((n, k), dtype=np.int8),
        candidate_hard_bad_flag=np.zeros((n, k), dtype=np.uint8),
        score_delta_target=np.zeros((n,), dtype=np.float32),
        win_target=np.zeros((n,), dtype=np.float32),
        dealin_target=np.zeros((n,), dtype=np.float32),
        pts_given_win_target=np.zeros((n,), dtype=np.float32),
        pts_given_dealin_target=np.zeros((n,), dtype=np.float32),
        opp_tenpai_target=np.zeros((n, 3), dtype=np.float32),
        event_history=np.zeros((n, 48, 5), dtype=np.int16),
        special_candidate_feat=np.zeros((n, 12, 25), dtype=np.float16),
        special_candidate_type_id=special_candidate_type_id,
        special_candidate_mask=special_candidate_mask,
        special_candidate_quality_score=np.zeros((n, 12), dtype=np.float32),
        special_candidate_rank_bucket=np.zeros((n, 12), dtype=np.int8),
        special_candidate_hard_bad_flag=np.zeros((n, 12), dtype=np.uint8),
        chosen_special_candidate_idx=np.array([-1, 0], dtype=np.int16),
        sample_type=np.array([0, 2], dtype=np.int8),
        actor=np.zeros((n,), dtype=np.int8),
        event_index=np.arange(n, dtype=np.int32),
        kyoku=np.ones((n,), dtype=np.int8),
        honba=np.zeros((n,), dtype=np.int8),
        is_open_hand=np.zeros((n,), dtype=np.uint8),
        replay_id=np.array(["review_fixture", "review_fixture"], dtype=np.str_),
        sample_id=np.array(["review_fixture:0", "review_fixture:1"], dtype=np.str_),
    )


def test_review_xmodel1_script_smoke(tmp_path: Path):
    ckpt = tmp_path / "xmodel1.pth"
    model = Xmodel1Model(
        state_tile_channels=57,
        state_scalar_dim=64,
        candidate_feature_dim=35,
        candidate_flag_dim=10,
        hidden_dim=32,
        num_res_blocks=1,
    )
    torch.save(
        {
            "model": model.state_dict(),
            "cfg": {
                "model_name": "xmodel1",
                "state_tile_channels": 57,
                "state_scalar_dim": 64,
                "candidate_feature_dim": 35,
                "candidate_flag_dim": 10,
                "special_candidate_feature_dim": 25,
                "hidden_dim": 32,
                "num_res_blocks": 1,
            },
            "model_version": "xmodel1",
        },
        ckpt,
    )
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    _write_sample_npz(data_dir / "sample.npz")
    output_path = tmp_path / "review.jsonl"
    result = subprocess.run(
        [
            "uv",
            "run",
            "python",
            "scripts/review_xmodel1.py",
            "--checkpoint",
            str(ckpt),
            "--data",
            str(data_dir),
            "--output",
            str(output_path),
            "--topk",
            "2",
        ],
        cwd="/media/bailan/DISK1/AUbuntuProject/project/keqing1",
        check=True,
        capture_output=True,
        text=True,
    )
    assert output_path.exists()
    assert "exported" in result.stdout
    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    assert '"sample_id": "review_fixture:0"' in lines[0]
    assert '"chosen_action": "dahai:' in lines[0]
    assert '"sample_id": "review_fixture:1"' in lines[1]
    assert '"chosen_action": "none"' in lines[1]
