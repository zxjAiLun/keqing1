from __future__ import annotations

import itertools
import json
from pathlib import Path

import numpy as np
import torch

from mahjong_env.final_rank import final_ranks, tie_break_order
from mahjong_env.replay import build_replay_samples_mc_return
from training.preprocess import Xmodel2PreprocessAdapter, events_to_cached_arrays
from xmodel2.cached_dataset import CachedMjaiDatasetXmodel2, Xmodel2CacheAdapter
from xmodel2.model import Xmodel2Model
from xmodel2.trainer import train


def _events() -> list[dict]:
    return [
        {"type": "start_game", "names": ["A", "B", "C", "D"]},
        {
            "type": "start_kyoku",
            "bakaze": "E",
            "kyoku": 1,
            "honba": 0,
            "kyotaku": 0,
            "oya": 0,
            "scores": [25000, 25000, 25000, 25000],
            "dora_marker": "1m",
            "tehais": [
                ["1m", "2m", "3m", "4m", "5m", "6m", "7m", "8m", "9m", "1p", "1p", "2p", "3p"],
                ["1s"] * 13,
                ["4p", "4p", "2s", "2s", "2s", "3s", "3s", "3s", "4s", "4s", "5s", "5s", "5s"],
                ["3s"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "4p"},
        {"type": "dahai", "actor": 0, "pai": "4p", "tsumogiri": True},
        {"type": "tsumo", "actor": 1, "pai": "4s"},
        {"type": "dahai", "actor": 1, "pai": "4s", "tsumogiri": True},
        {
            "type": "ryukyoku",
            "scores": [35000, 25000, 20000, 20000],
            "tenpai_players": [],
        },
        {"type": "end_kyoku"},
        {"type": "end_game", "scores": [35000, 25000, 20000, 20000]},
    ]


def test_final_rank_tie_break_uses_initial_oya_order():
    assert tie_break_order(3) == (3, 0, 1, 2)
    assert final_ranks([30000, 20000, 20000, 30000], initial_oya=3) == (1, 2, 3, 0)


def test_replay_samples_emit_raw_placement_targets():
    samples = build_replay_samples_mc_return(_events(), strict_legal_labels=True)
    sample = next(s for s in samples if s.actor == 0 and s.label_action.get("type") == "dahai")
    assert sample.score_before_action == 25000
    assert sample.final_score_delta_points_target == 10000
    assert sample.final_rank_target == 0


def test_xmodel2_cache_adapter_preserves_scalar_placement_targets_under_perm(tmp_path: Path):
    arrays = events_to_cached_arrays(
        _events(),
        adapter=Xmodel2PreprocessAdapter(),
        value_strategy="mc_return",
        encode_module="training.state_features",
    )
    assert arrays is not None
    data_root = tmp_path / "processed_xmodel2" / "ds1"
    data_root.mkdir(parents=True)
    cache_path = data_root / "sample.npz"
    np.savez(cache_path, **{k: v for k, v in arrays.items() if not k.startswith("_")})

    dataset = CachedMjaiDatasetXmodel2([cache_path], shuffle=False, seed=7, aug_perms=1, buffer_size=16)
    samples = list(itertools.islice(iter(dataset), 2))
    batch = CachedMjaiDatasetXmodel2.collate(samples)
    adapter = Xmodel2CacheAdapter()
    row_extra = {
        "score_delta_target": np.float32(0.1),
        "win_target": np.float32(1.0),
        "dealin_target": np.float32(0.0),
        "pts_given_win_target": np.float32(0.3),
        "pts_given_dealin_target": np.float32(0.0),
        "opp_tenpai_target": np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        "final_rank_target": np.int8(2),
        "final_score_delta_points_target": np.int32(-4300),
    }
    permuted = adapter.permute_row_extra(row_extra, (1, 0, 2), 7)

    assert batch[-2].dtype == torch.long
    assert batch[-1].dtype == torch.int32
    assert int(permuted["final_rank_target"]) == 2
    assert int(permuted["final_score_delta_points_target"]) == -4300


def test_xmodel2_training_smoke(tmp_path: Path):
    arrays = events_to_cached_arrays(
        _events(),
        adapter=Xmodel2PreprocessAdapter(),
        value_strategy="mc_return",
        encode_module="training.state_features",
    )
    assert arrays is not None
    assert "final_rank_target" in arrays
    assert "final_score_delta_points_target" in arrays
    assert arrays["final_rank_target"].ndim == 1
    assert arrays["final_score_delta_points_target"].dtype == np.int32

    data_root = tmp_path / "processed_xmodel2" / "ds1"
    data_root.mkdir(parents=True)
    cache_path = data_root / "sample.npz"
    np.savez(cache_path, **{k: v for k, v in arrays.items() if not k.startswith("_")})
    train_files = [cache_path]
    val_files = [cache_path]

    model = Xmodel2Model(
        hidden_dim=64,
        num_res_blocks=2,
        dropout=0.0,
    )
    cfg = {
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "num_epochs": 1,
        "batch_size": 8,
        "buffer_size": 16,
        "warmup_steps": 1,
        "steps_per_epoch": 2,
        "log_interval": 1,
        "value_loss_weight": 0.2,
        "win_loss_weight": 0.2,
        "dealin_loss_weight": 0.2,
        "pts_given_win_loss_weight": 0.2,
        "pts_given_dealin_loss_weight": 0.2,
        "opp_tenpai_loss_weight": 0.2,
        "final_rank_loss_weight": 0.3,
        "final_score_delta_loss_weight": 0.05,
        "rank_pt_value_loss_weight": 0.01,
        "rank_bonus": [90.0, 45.0, 0.0, -135.0],
        "rank_bonus_norm": 90.0,
        "rank_score_scale": 0.0,
        "score_norm": 30000.0,
    }

    out_dir = tmp_path / "xmodel2_smoke"
    train(
        model=model,
        cfg=cfg,
        output_dir=out_dir,
        train_files=train_files,
        val_files=val_files,
        seed=7,
        device_str="cpu",
        batch_size=8,
        num_workers=0,
        aug_perms=1,
        buffer_size=16,
    )

    assert (out_dir / "last.pth").exists()
    rows = [
        json.loads(line)
        for line in (out_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    last = rows[-1]
    assert "train_final_rank_loss" in last
    assert "val_final_rank_loss" in last
    assert "train_final_score_delta_loss" in last
    assert "val_rank_pt_loss" in last
    assert "train_final_rank_acc" in last
    assert "val_composed_ev_mean" in last
