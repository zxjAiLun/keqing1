"""keqingv1 训练包装：复用共享 trainer，保持旧接口兼容。"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import torch

from keqingv1.model import MahjongModel
from training import (
    TaskSpec,
    _ACTION_LABELS,
    _MERGE_MAP,
    _action_type_name,
    build_scheduler,
    load_checkpoint,
    masked_ce_loss,
    save_checkpoint,
    train_model,
)


def _unpack_v1_batch(batch, device: torch.device) -> Dict:
    tile_feat, scalar, mask, action_idx, value_target = batch
    return {
        "tile_feat": tile_feat.to(device),
        "scalar": scalar.to(device),
        "mask": mask.to(device),
        "action_idx": action_idx.to(device),
        "value_target": value_target.to(device),
    }


def _compute_no_extra_loss(model, device: torch.device, batch_data: Dict, is_train: bool, batch_idx: int):
    del model, device, batch_data, is_train, batch_idx
    return torch.tensor(0.0, device=device), {}


_V1_TASK = TaskSpec(
    name="keqingv1_base",
    unpack_batch=_unpack_v1_batch,
    compute_extra_loss=_compute_no_extra_loss,
)


def train(
    model: MahjongModel,
    train_loader,
    val_loader,
    cfg: Dict,
    output_dir: Path,
    resume_path: Optional[Path] = None,
    weights_only: bool = False,
    device_str: str = "cuda",
):
    return train_model(
        model=model,
        train_loader=train_loader,
        train_loader_factory=None,
        val_loader=val_loader,
        task=_V1_TASK,
        cfg=cfg,
        output_dir=output_dir,
        resume_path=resume_path,
        weights_only=weights_only,
        device_str=device_str,
    )
