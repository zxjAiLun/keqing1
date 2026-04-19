from pathlib import Path
import json

import numpy as np

from keqingv4.cached_dataset import CachedMjaiDatasetV4
from keqingv4.model import KeqingV4Model
from keqingv4.trainer import train
from torch.utils.data import DataLoader
from training.preprocess import KeqingV4PreprocessAdapter, events_to_cached_arrays


def test_keqingv4_training_smoke(tmp_path: Path):
    events = [
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
                ["2s"] * 13,
                ["3s"] * 13,
            ],
        },
        {"type": "tsumo", "actor": 0, "pai": "5s"},
        {"type": "dahai", "actor": 0, "pai": "5s", "tsumogiri": True},
    ]
    arrays = events_to_cached_arrays(
        events,
        adapter=KeqingV4PreprocessAdapter(),
        value_strategy="mc_return",
        encode_module="training.state_features",
    )
    assert arrays is not None
    data_root = tmp_path / "processed_v4" / "ds1"
    data_root.mkdir(parents=True)
    cache_path = data_root / "sample.npz"
    np.savez(cache_path, **{k: v for k, v in arrays.items() if not k.startswith("_")})
    train_files = [cache_path]
    val_files = [cache_path]

    val_loader = DataLoader(
        CachedMjaiDatasetV4(val_files, shuffle=False, seed=7, aug_perms=1, buffer_size=16),
        batch_size=8,
        collate_fn=CachedMjaiDatasetV4.collate,
        num_workers=0,
    )

    model = KeqingV4Model(
        hidden_dim=64,
        num_res_blocks=2,
        action_embed_dim=16,
        context_dim=12,
        dropout=0.0,
    )
    cfg = {
        "learning_rate": 3e-4,
        "weight_decay": 1e-4,
        "num_epochs": 1,
        "batch_size": 8,
        "buffer_size": 16,
        "prefetch_factor": 2,
        "pin_memory": False,
        "persistent_workers": False,
        "warmup_steps": 1,
        "steps_per_epoch": 2,
        "log_interval": 1,
        "value_loss_weight": 0.2,
        "win_loss_weight": 0.2,
        "dealin_loss_weight": 0.2,
        "pts_given_win_loss_weight": 0.3,
        "pts_given_dealin_loss_weight": 0.3,
        "opp_tenpai_loss_weight": 0.25,
        "mc_reg_loss_weight": 0.05,
    }

    out_dir = tmp_path / "v4_smoke"
    train(
        model=model,
        val_loader=val_loader,
        cfg=cfg,
        output_dir=out_dir,
        train_files=train_files,
        seed=7,
        use_cuda=False,
        device_str='cpu',
        aug_perms=1,
        batch_size=8,
        num_workers=0,
        files_per_epoch_ratio=1.0,
    )

    assert (out_dir / "last.pth").exists()
    assert (out_dir / "train_log.jsonl").exists()
    rows = [
        json.loads(line)
        for line in (out_dir / "train_log.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert rows
    last = rows[-1]
    assert "train_typed_rank_loss" in last
    assert "val_typed_rank_loss" in last
    assert "train_mc_reg_loss" in last
    assert "val_composed_ev_mean" in last
    assert "train_reach_opp_rate" in last
    assert "val_special_slice_acc" in last
