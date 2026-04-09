from pathlib import Path

from keqingv3.cached_dataset import split_cached_files
from keqingv4.model import KeqingV4Model
from keqingv4.trainer import train
from torch.utils.data import DataLoader
from keqingv3.cached_dataset import CachedMjaiDatasetV3


def test_keqingv4_training_smoke(tmp_path: Path):
    data_root = Path('processed_v3_fixaux/ds1')
    train_files, val_files = split_cached_files([data_root], val_ratio=0.2, seed=7)
    train_files = train_files[:1]
    val_files = val_files[:1]

    val_loader = DataLoader(
        CachedMjaiDatasetV3(val_files, shuffle=False, seed=7, aug_perms=1, buffer_size=16),
        batch_size=8,
        collate_fn=CachedMjaiDatasetV3.collate,
        num_workers=0,
    )

    model = KeqingV4Model(hidden_dim=64, num_res_blocks=2, action_embed_dim=16, dropout=0.0)
    cfg = {
        'learning_rate': 3e-4,
        'weight_decay': 1e-4,
        'num_epochs': 1,
        'batch_size': 8,
        'buffer_size': 16,
        'prefetch_factor': 2,
        'pin_memory': False,
        'persistent_workers': False,
        'warmup_steps': 1,
        'steps_per_epoch': 2,
        'log_interval': 1,
        'score_loss_weight': 0.3,
        'win_loss_weight': 0.2,
        'dealin_loss_weight': 0.2,
        'offense_loss_weight': 0.2,
        'defense_loss_weight': 0.2,
        'gate_reg_weight': 0.0,
    }

    out_dir = tmp_path / 'v4_smoke'
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

    assert (out_dir / 'last.pth').exists()
    assert (out_dir / 'train_log.jsonl').exists()
